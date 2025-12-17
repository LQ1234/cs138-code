#!/usr/bin/env python3
"""
pickleio: high-throughput (HT) + low-latency (LL) client/server library.

Client usage example:

    from pickleio import Client
    import numpy as np, time

    client = Client("10.246.224.208", ht_port=34211, ll_port=34212, debug=True)

    client.queue_upload_ht("episode_1", np.random.rand(255, 255, 3))
    client.queue_upload_ht("episode_2", np.random.rand(255, 255, 3))

    print("Uploads queued, waiting for completion...")
    while not client.upload_queue_empty_ht():
        time.sleep(0.1)
    print("All uploads completed.")

    result = client.send_message_ll("ping")
    avg = client.send_message_ll("process_data")
    print(result, avg)
    print(client.get_stats())

Server usage example:

    from pickleio import Server
    import numpy as np

    server = Server(ht_port=34211, ll_port=34212, upload_dir="./uploads", debug=True)
    server.start()

    while True:
        with server.receive_ll(timeout=0.1) as msg:
            if msg is None:
                continue
            print("Received LL message:", msg)
            if msg == "ping":
                server.send_response_ll("pong")
            elif msg == "process_data":
                data = server.get_upload_ht("episode_1")
                if data is None:
                    server.send_response_ll("data has not finished uploading yet")
                else:
                    import numpy as np
                    result = float(np.mean(data))
                    server.send_response_ll({"mean": result})
            else:
                server.send_response_ll("unknown command")

Key HT behavior:
- queue_upload_ht() is non-blocking.
- A background worker:
    - pickles + bz2-compresses the object,
    - streams it via a persistent HT TCP connection,
    - waits for a server ACK that is only sent after the server has fully saved it,
    - retries cleanly on failure up to reconnect_attempts.
- upload_queue_empty_ht():
    - returns True only when all queued jobs are finished.
    - raises RuntimeError if any job permanently failed.

WARNING: As always with pickle, only use between trusted endpoints.
"""
try:
    import numpy as _np
except ImportError:
    _np = None

def _pickleio_reduce_ndarray(arr):
    # Always serialize as C-contiguous bytes
    return (_pickleio_rebuild_ndarray,
            (arr.shape, str(arr.dtype), arr.tobytes(order="C")))

def _pickleio_rebuild_ndarray(shape, dtype_str, data):
    import numpy as np
    arr = np.frombuffer(data, dtype=dtype_str)
    return arr.reshape(shape)

if _np is not None:
    import copyreg
    copyreg.pickle(_np.ndarray, _pickleio_reduce_ndarray)
import bz2
import os
import pickle
import queue
import socket
import struct
import threading
import time
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Low-latency framing: [4-byte len][pickle]
_LL_HEADER = struct.Struct("!I")

# High-throughput framing:
#   upload:
#       [1-byte op=UPLOAD][8-byte name_len][name_bytes][8-byte data_len][data_bytes]
#       server -> client ACK:
#       [1-byte op=ACK][8-byte name_len][name_bytes]
#   hash-check (to avoid duplicate upload):
#       [1-byte op=CHECK][8-byte name_len][name_bytes][8-byte hash_len][hash_bytes]
#       server -> client response:
#       [1-byte op=CHECK_RESP][1-byte exists_flag]
_HT_UPLOAD_OP = 0x01
_HT_ACK_OP = 0x81
_HT_CHECK_OP = 0x02
_HT_CHECK_RESP_OP = 0x82
_HT_NAME_LEN = struct.Struct("!Q")
_HT_DATA_LEN = struct.Struct("!Q")


# =========================
# Shared helpers
# =========================

def _safe_close(sock: Optional[socket.socket]):
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass


def _recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes or return None if connection closed/failed."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (OSError, ConnectionResetError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send_all(sock: socket.socket, data: bytes):
    """Robust sendall with clear error semantics."""
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        try:
            n = sock.send(view[sent:])
        except (OSError, ConnectionResetError) as e:
            raise ConnectionError(f"send failed: {e}") from e
        if n == 0:
            raise ConnectionError("socket connection broken")
        sent += n


# =========================
# Stats helpers
# =========================

@dataclass
class _MinAvgMax:
    min: float = float("inf")
    max: float = float("-inf")
    sum: float = 0.0
    count: int = 0

    def add(self, value: float):
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        self.sum += value
        self.count += 1

    def as_tuple(self) -> Tuple[float, float, float]:
        if self.count == 0:
            return (0.0, 0.0, 0.0)
        avg = self.sum / self.count
        return (self.min, avg, self.max)


@dataclass
class _HTStats:
    total_queued: int = 0
    total_completed: int = 0      # includes successes and permanent failures
    total_succeeded: int = 0
    compression_ratio: _MinAvgMax = field(default_factory=_MinAvgMax)
    upload_speed: _MinAvgMax = field(default_factory=_MinAvgMax)       # MB/s
    upload_time: _MinAvgMax = field(default_factory=_MinAvgMax)        # s
    compression_time: _MinAvgMax = field(default_factory=_MinAvgMax)   # s
    last_error: Optional[str] = None


@dataclass
class _LLStats:
    total_messages: int = 0
    msg_size: _MinAvgMax = field(default_factory=_MinAvgMax)           # bytes
    rtt_ms: _MinAvgMax = field(default_factory=_MinAvgMax)
    upload_time: _MinAvgMax = field(default_factory=_MinAvgMax)        # s
    server_proc_ms: _MinAvgMax = field(default_factory=_MinAvgMax)
    download_time: _MinAvgMax = field(default_factory=_MinAvgMax)      # s
    last_error: Optional[str] = None


# =========================
# Client
# =========================

class Client:
    def __init__(
        self,
        host: str,
        ht_port: int = 34211,
        ll_port: int = 34212,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 0.5,
        debug: bool = False,
    ):
        self.host = host
        self.ht_port = ht_port
        self.ll_port = ll_port
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.debug = debug

        # High-throughput state
        self._ht_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._ht_sock: Optional[socket.socket] = None
        self._ht_lock = threading.Lock()
        self._ht_stats = _HTStats()
        self._ht_stats_lock = threading.Lock()
        self._ht_stop = threading.Event()
        self._ht_worker = threading.Thread(target=self._ht_worker_loop, daemon=True)
        self._ht_worker.start()

        # Low-latency state
        self._ll_sock: Optional[socket.socket] = None
        self._ll_lock = threading.Lock()
        self._ll_stats = _LLStats()
        self._ll_stats_lock = threading.Lock()

    # ---------- Public HT API ----------

    def queue_upload_ht(self, name: str, obj: Any):
        """
        Queue an object for high-throughput upload (non-blocking).

        The worker will:
        - pickle the object,
        - bz2-compress it (level=5),
        - send via HT channel,
        - wait for server ACK (only sent after successful save),
        - retry on failure up to reconnect_attempts.
        """
        with self._ht_stats_lock:
            self._ht_stats.total_queued += 1
        if self.debug:
            print(f"[Client HT] Queued upload: {name}")
        self._ht_queue.put((name, obj))

    def upload_queue_empty_ht(self) -> bool:
        """
        Returns True once all queued HT uploads are completed (success or permanent failure).

        If any upload has permanently failed after all retries, raises RuntimeError
        with the last error message.
        """
        with self._ht_stats_lock:
            # If we hit a fatal error for any job, surface it
            if self._ht_stats.last_error is not None:
                raise RuntimeError(f"HT upload error: {self._ht_stats.last_error}")
            # empty when all queued jobs have been processed
            return (
                self._ht_queue.empty()
                and self._ht_stats.total_completed >= self._ht_stats.total_queued
            )

    # ---------- Public LL API ----------

    def send_message_ll(self, message: Any, timeout: float = 5.0) -> Any:
        """
        Send a low-latency request and wait for the response (blocking).
        Auto-(re)connects; raises if all retries fail.
        """
        start_total = time.monotonic()
        payload = {"msg": message}
        msg_bytes = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
        msg_len = len(msg_bytes)

        for attempt in range(self.reconnect_attempts):
            with self._ll_lock:
                sock = self._ensure_ll_connected_locked()
                if sock is None:
                    # failed to connect; try outside lock
                    pass
                else:
                    try:
                        if self.debug:
                            print(f"[Client LL] Sending message (attempt={attempt+1}): {message}")

                        t0 = time.monotonic()
                        _send_all(sock, _LL_HEADER.pack(msg_len) + msg_bytes)
                        t1 = time.monotonic()

                        header = _recv_all(sock, _LL_HEADER.size)
                        if not header:
                            raise ConnectionError("LL: closed while waiting for header")
                        (resp_len,) = _LL_HEADER.unpack(header)

                        resp_bytes = _recv_all(sock, resp_len)
                        if not resp_bytes:
                            raise ConnectionError("LL: closed while waiting for body")
                        t2 = time.monotonic()

                        resp = pickle.loads(resp_bytes)
                        data = resp.get("data")
                        server_proc_ms = float(resp.get("server_proc_ms", 0.0))

                        upload_time = t1 - t0
                        rtt_ms = (t2 - t0) * 1000.0
                        download_time = max(
                            0.0,
                            (t2 - t0) - upload_time - server_proc_ms / 1000.0
                        )

                        with self._ll_stats_lock:
                            st = self._ll_stats
                            st.total_messages += 1
                            st.msg_size.add(float(msg_len))
                            st.rtt_ms.add(float(rtt_ms))
                            st.upload_time.add(float(upload_time))
                            st.server_proc_ms.add(float(server_proc_ms))
                            st.download_time.add(float(download_time))
                            st.last_error = None

                        if self.debug:
                            print(f"[Client LL] Got response: {data} "
                                  f"(rtt={rtt_ms:.3f} ms, server={server_proc_ms:.3f} ms)")

                        return data

                    except Exception as e:
                        if self.debug:
                            print(f"[Client LL] Error: {e}, reconnecting...")
                        _safe_close(self._ll_sock)
                        self._ll_sock = None
                        with self._ll_stats_lock:
                            self._ll_stats.last_error = str(e)

            time.sleep(self.reconnect_delay)

        total_time = time.monotonic() - start_total
        raise ConnectionError(
            f"LL: failed after {self.reconnect_attempts} attempts over "
            f"{total_time:.2f}s (last_error={self._ll_stats.last_error})"
        )

    # ---------- Public stats API ----------

    def get_stats(self) -> str:
        """
        Return a formatted stats table for HT and LL channels.
        """
        with self._ht_stats_lock:
            ht = self._ht_stats
            cr = ht.compression_ratio.as_tuple()
            us = ht.upload_speed.as_tuple()
            ut = ht.upload_time.as_tuple()
            ct = ht.compression_time.as_tuple()
            ht_err = ht.last_error
            tq, tc, ts = ht.total_queued, ht.total_completed, ht.total_succeeded

        with self._ll_stats_lock:
            ll = self._ll_stats
            ms = ll.msg_size.as_tuple()
            rtt = ll.rtt_ms.as_tuple()
            up = ll.upload_time.as_tuple()
            sp = ll.server_proc_ms.as_tuple()
            dl = ll.download_time.as_tuple()
            ll_err = ll.last_error
            lm = ll.total_messages

        def fmt_row(cols, widths):
            return "  " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

        def fmt_mmm(mn, avg, mx, unit=""):
            return f"{mn:.3f}{unit} / {avg:.3f}{unit} / {mx:.3f}{unit}"

        lines: List[str] = []
        w = [32, 40]

        # HT table
        lines.append("High-throughput upload stats:")
        lines.append(fmt_row(["Metric", "Value"], w))
        lines.append(fmt_row(["-" * w[0], "-" * w[1]], w))
        lines.append(fmt_row(["Total uploads queued", tq], w))
        lines.append(fmt_row(["Total uploads completed", tc], w))
        lines.append(fmt_row(["Total uploads succeeded", ts], w))
        lines.append(fmt_row(
            ["Upload speed (MB/s) min/avg/max", fmt_mmm(*us)], w
        ))
        lines.append(fmt_row(
            ["Compression ratio min/avg/max", fmt_mmm(*cr)], w
        ))
        lines.append(fmt_row(
            ["Upload time (s) min/avg/max", fmt_mmm(*ut)], w
        ))
        lines.append(fmt_row(
            ["Compression time (s) min/avg/max", fmt_mmm(*ct)], w
        ))
        if ht_err:
            lines.append(fmt_row(["Last error", ht_err], w))

        lines.append("")
        # LL table
        lines.append("Low-latency message stats:")
        lines.append(fmt_row(["Metric", "Value"], w))
        lines.append(fmt_row(["-" * w[0], "-" * w[1]], w))
        lines.append(fmt_row(["Total messages sent", lm], w))
        lines.append(fmt_row(
            ["Msg size (bytes) min/avg/max", fmt_mmm(*ms, "")], w
        ))
        lines.append(fmt_row(
            ["RTT (ms) min/avg/max", fmt_mmm(*rtt)], w
        ))
        lines.append(fmt_row(
            ["Upload time (s) min/avg/max", fmt_mmm(*up)], w
        ))
        lines.append(fmt_row(
            ["Server proc (ms) min/avg/max", fmt_mmm(*sp)], w
        ))
        lines.append(fmt_row(
            ["Download time (s) min/avg/max", fmt_mmm(*dl)], w
        ))
        if ll_err:
            lines.append(fmt_row(["Last error", ll_err], w))

        return "\n".join(lines)

    # ---------- Internal: HT worker ----------

    def _ensure_ht_connected_locked(self) -> Optional[socket.socket]:
        if self._ht_sock is not None:
            return self._ht_sock
        for attempt in range(self.reconnect_attempts):
            try:
                s = socket.create_connection((self.host, self.ht_port), timeout=5.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._ht_sock = s
                if self.debug:
                    print(f"[Client HT] Connected to {self.host}:{self.ht_port}")
                return s
            except OSError as e:
                if self.debug:
                    print(f"[Client HT] Connect failed (attempt {attempt+1}): {e}")
                with self._ht_stats_lock:
                    self._ht_stats.last_error = str(e)
                time.sleep(self.reconnect_delay)
        return None

    def _ht_worker_loop(self):
        while not self._ht_stop.is_set():
            try:
                try:
                    name, obj = self._ht_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                success = False
                error_msg = None

                try:
                    if self.debug:
                        print(f"[Client HT] Starting upload job for: {name}")

                    # 1) Serialize (uncompressed pickle)
                    uncompressed = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
                    uncompressed_size = len(uncompressed)

                    # Compute hash of the uncompressed pickle so the server can
                    # detect duplicates without needing the full upload.
                    hash_bytes = hashlib.sha256(uncompressed).digest()

                    # 2) Compress with bz2 (GIL-friendly C code)
                    t_comp_start = time.monotonic()
                    compressor = bz2.BZ2Compressor(5)
                    compressed = compressor.compress(uncompressed) + compressor.flush()
                    t_comp_end = time.monotonic()
                    compressed_size = len(compressed)
                    comp_time = t_comp_end - t_comp_start

                    comp_ratio = (
                        (uncompressed_size / compressed_size)
                        if compressed_size > 0 else 0.0
                    )

                    name_bytes = name.encode("utf-8")

                    # 3) Try to upload with retries
                    for attempt in range(self.reconnect_attempts):
                        with self._ht_lock:
                            sock = self._ensure_ht_connected_locked()
                            if sock is None:
                                # connection attempt failed, will retry
                                pass
                            else:
                                try:
                                    # First, ask the server if it already has
                                    # this content (by hash) for this name.
                                    if self.debug:
                                        print(f"[Client HT] Hash-check attempt {attempt+1} for {name}")

                                    check_header = bytearray()
                                    check_header.append(_HT_CHECK_OP)
                                    check_header += _HT_NAME_LEN.pack(len(name_bytes))
                                    check_header += name_bytes
                                    check_header += _HT_DATA_LEN.pack(len(hash_bytes))
                                    check_header += hash_bytes
                                    _send_all(sock, bytes(check_header))

                                    check_resp = _recv_all(sock, 2)
                                    if not check_resp or check_resp[0] != _HT_CHECK_RESP_OP:
                                        raise ConnectionError("Missing/invalid HT CHECK response")

                                    exists_flag = check_resp[1]

                                    if exists_flag == 1:
                                        # Server already has identical content for this name.
                                        with self._ht_stats_lock:
                                            st = self._ht_stats
                                            st.total_succeeded += 1
                                            st.compression_ratio.add(comp_ratio)
                                            st.compression_time.add(comp_time)
                                            st.last_error = None

                                        if self.debug:
                                            print(f"[Client HT] Skipping upload for {name} "
                                                  f"(server already has identical content)")
                                        success = True
                                        break

                                    if self.debug:
                                        print(f"[Client HT] Upload attempt {attempt+1} for {name} "
                                              f"({compressed_size} bytes compressed)")

                                    t_up_start = time.monotonic()

                                    # frame header for actual upload
                                    header = bytearray()
                                    header.append(_HT_UPLOAD_OP)
                                    header += _HT_NAME_LEN.pack(len(name_bytes))
                                    header += name_bytes
                                    header += _HT_DATA_LEN.pack(compressed_size)

                                    _send_all(sock, bytes(header))
                                    _send_all(sock, compressed)

                                    # wait for ACK (this is effectively polling server status)
                                    op_b = _recv_all(sock, 1)
                                    if not op_b or op_b[0] != _HT_ACK_OP:
                                        raise ConnectionError("Missing/invalid HT ACK (op)")

                                    nlen_b = _recv_all(sock, _HT_NAME_LEN.size)
                                    if not nlen_b:
                                        raise ConnectionError("Missing HT ACK name length")

                                    (ack_nlen,) = _HT_NAME_LEN.unpack(nlen_b)
                                    ack_name_b = _recv_all(sock, ack_nlen)
                                    if not ack_name_b:
                                        raise ConnectionError("Missing HT ACK name")
                                    if ack_name_b.decode("utf-8") != name:
                                        raise ConnectionError("HT ACK name mismatch")

                                    t_up_end = time.monotonic()
                                    upload_time = t_up_end - t_up_start
                                    mb = compressed_size / (1024.0 * 1024.0)
                                    mbps = (mb / upload_time) if upload_time > 0 else 0.0

                                    with self._ht_stats_lock:
                                        st = self._ht_stats
                                        st.total_succeeded += 1
                                        st.compression_ratio.add(comp_ratio)
                                        st.upload_speed.add(mbps)
                                        st.upload_time.add(upload_time)
                                        st.compression_time.add(comp_time)
                                        st.last_error = None

                                    if self.debug:
                                        print(f"[Client HT] Upload {name} succeeded: "
                                              f"{mbps:.3f} MB/s, ratio={comp_ratio:.3f}")

                                    success = True
                                    break

                                except Exception as e:
                                    if self.debug:
                                        print(f"[Client HT] Error during upload of {name}: {e}")
                                    _safe_close(self._ht_sock)
                                    self._ht_sock = None
                                    error_msg = str(e)

                        if success:
                            break
                        time.sleep(self.reconnect_delay)

                    if not success:
                        if error_msg is None:
                            error_msg = f"HT upload {name} failed after retries"
                        if self.debug:
                            print(f"[Client HT] Permanent failure for {name}: {error_msg}")
                        with self._ht_stats_lock:
                            if self._ht_stats.last_error is None:
                                self._ht_stats.last_error = error_msg

                finally:
                    with self._ht_stats_lock:
                        self._ht_stats.total_completed += 1
                    self._ht_queue.task_done()

            except Exception as e:
                # Keep worker alive; record something and continue.
                if self.debug:
                    print(f"[Client HT] Worker loop error: {e}")
                continue

    # ---------- Internal: LL connection ----------

    def _ensure_ll_connected_locked(self) -> Optional[socket.socket]:
        if self._ll_sock is not None:
            return self._ll_sock
        for attempt in range(self.reconnect_attempts):
            try:
                s = socket.create_connection((self.host, self.ll_port), timeout=100000000.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._ll_sock = s
                if self.debug:
                    print(f"[Client LL] Connected to {self.host}:{self.ll_port}")
                return s
            except OSError as e:
                if self.debug:
                    print(f"[Client LL] Connect failed (attempt {attempt+1}): {e}")
                with self._ll_stats_lock:
                    self._ll_stats.last_error = str(e)
                time.sleep(self.reconnect_delay)
        return None

    # ---------- Cleanup ----------

    def close(self):
        self._ht_stop.set()
        _safe_close(self._ht_sock)
        _safe_close(self._ll_sock)


# =========================
# Server
# =========================

@dataclass
class _LLRequest:
    req_id: int
    conn: socket.socket
    addr: Tuple[str, int]
    arrived_at: float
    msg: Any


class _LLContext:
    def __init__(self, server: "Server", req: Optional[_LLRequest]):
        self.server = server
        self.req = req
        self._response_sent = False

    def __enter__(self):
        if self.req is None:
            self.server._current_ll_req = None
            self.server._current_ll_ctx = None
            return None
        self.server._current_ll_req = self.req
        self.server._current_ll_ctx = self
        return self.req.msg

    def __exit__(self, exc_type, exc, tb):
        if self.req is None:
            return False
        if not self._response_sent:
            try:
                self.server.send_response_ll(None)
            except Exception:
                pass
        self.server._current_ll_req = None
        self.server._current_ll_ctx = None
        return False

    def mark_response_sent(self):
        self._response_sent = True


class Server:
    def __init__(
        self,
        ht_port: int = 34211,
        ll_port: int = 34212,
        upload_dir: str = "./uploads",
        debug: bool = False,
    ):
        self.ht_port = ht_port
        self.ll_port = ll_port
        self.upload_dir = upload_dir
        self.debug = debug

        os.makedirs(self.upload_dir, exist_ok=True)

        # HT
        self._ht_sock: Optional[socket.socket] = None
        self._ht_thread: Optional[threading.Thread] = None
        self._uploads: Dict[str, Any] = {}
        self._uploads_lock = threading.Lock()
        # Track hashes of uploads (hash of uncompressed pickle bytes) per name
        self._upload_hashes: Dict[str, bytes] = {}

        # LL
        self._ll_sock: Optional[socket.socket] = None
        self._ll_accept_thread: Optional[threading.Thread] = None
        self._ll_queue: "queue.Queue[_LLRequest]" = queue.Queue()
        self._ll_req_id = 0
        self._ll_lock = threading.Lock()
        self._current_ll_req: Optional[_LLRequest] = None
        self._current_ll_ctx: Optional[_LLContext] = None

        self._stop = threading.Event()

    # ---------- Public API ----------

    def start(self):
        # HT listener
        self._ht_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ht_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ht_sock.bind(("0.0.0.0", self.ht_port))
        self._ht_sock.listen(32)
        self._ht_thread = threading.Thread(target=self._ht_accept_loop, daemon=True)
        self._ht_thread.start()
        if self.debug:
            print(f"[Server HT] Listening on 0.0.0.0:{self.ht_port}")

        # LL listener
        self._ll_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ll_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ll_sock.bind(("0.0.0.0", self.ll_port))
        self._ll_sock.listen(32)
        self._ll_accept_thread = threading.Thread(target=self._ll_accept_loop, daemon=True)
        self._ll_accept_thread.start()
        if self.debug:
            print(f"[Server LL] Listening on 0.0.0.0:{self.ll_port}")

    def get_upload_ht(self, name: str) -> Optional[Any]:
        with self._uploads_lock:
            if name in self._uploads:
                return self._uploads[name]

        path = os.path.join(self.upload_dir, f"{name}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                with self._uploads_lock:
                    self._uploads[name] = obj
                return obj
            except Exception:
                return None
        return None

    def receive_ll(self, timeout: Optional[float] = None) -> _LLContext:
        try:
            req = self._ll_queue.get(timeout=timeout)
        except queue.Empty:
            return _LLContext(self, None)
        return _LLContext(self, req)

    def send_response_ll(self, response: Any):
        req = self._current_ll_req
        ctx = self._current_ll_ctx
        if req is None or ctx is None:
            raise RuntimeError("send_response_ll() called outside of receive_ll() context")

        now = time.monotonic()
        proc_ms = max(0.0, (now - req.arrived_at) * 1000.0)
        payload = {
            "ok": True,
            "data": response,
            "server_proc_ms": proc_ms,
        }
        data = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
        header = _LL_HEADER.pack(len(data))

        if self.debug:
            print(f"[Server LL] Sending response to {req.addr}: {response} "
                  f"(proc={proc_ms:.3f} ms)")

        try:
            _send_all(req.conn, header + data)
        except Exception as e:
            if self.debug:
                print(f"[Server LL] Error sending response to {req.addr}: {e}")
            _safe_close(req.conn)

        ctx.mark_response_sent()

    # ---------- Internal: HT handling ----------

    def _ht_accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._ht_sock.accept()
            except OSError:
                break
            if self.debug:
                print(f"[Server HT] Connection from {addr}")
            t = threading.Thread(target=self._ht_client_loop, args=(conn, addr), daemon=True)
            t.start()

    def _ht_client_loop(self, conn: socket.socket, addr):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while not self._stop.is_set():
                op_b = _recv_all(conn, 1)
                if not op_b:
                    if self.debug:
                        print(f"[Server HT] {addr} closed connection")
                    break
                op = op_b[0]

                # Hash-check request: client wants to know if the server already
                # has identical content (by hash) for this name.
                if op == _HT_CHECK_OP:
                    name_len_b = _recv_all(conn, _HT_NAME_LEN.size)
                    if not name_len_b:
                        break
                    (name_len,) = _HT_NAME_LEN.unpack(name_len_b)

                    name_b = _recv_all(conn, name_len)
                    if not name_b:
                        break
                    name = name_b.decode("utf-8")

                    hash_len_b = _recv_all(conn, _HT_DATA_LEN.size)
                    if not hash_len_b:
                        break
                    (hash_len,) = _HT_DATA_LEN.unpack(hash_len_b)

                    hash_bytes = _recv_all(conn, hash_len)
                    if not hash_bytes:
                        break

                    exists = False

                    # Check in-memory hash first
                    with self._uploads_lock:
                        stored_hash = self._upload_hashes.get(name)
                        if stored_hash is not None and stored_hash == hash_bytes:
                            exists = True

                    # If not in memory, check on disk (and populate hash cache on hit)
                    if not exists:
                        path = os.path.join(self.upload_dir, f"{name}.pkl")
                        if os.path.exists(path):
                            try:
                                with open(path, "rb") as f:
                                    disk_data = f.read()
                                disk_hash = hashlib.sha256(disk_data).digest()
                                if disk_hash == hash_bytes:
                                    exists = True
                                    with self._uploads_lock:
                                        self._upload_hashes[name] = hash_bytes
                            except Exception:
                                exists = False

                    if self.debug:
                        if exists:
                            print(f"[Server HT] Hash-check: '{name}' already present for {addr}")
                        else:
                            print(f"[Server HT] Hash-check: '{name}' not present for {addr}")

                    try:
                        resp = bytearray()
                        resp.append(_HT_CHECK_RESP_OP)
                        resp.append(1 if exists else 0)
                        _send_all(conn, bytes(resp))
                    except Exception as e:
                        if self.debug:
                            print(f"[Server HT] Failed to send hash-check response for '{name}' "
                                  f"to {addr}: {e}")
                        break

                    # Continue loop for next op
                    continue

                if op != _HT_UPLOAD_OP:
                    if self.debug:
                        print(f"[Server HT] {addr} unknown op {op}, closing")
                    break

                # Read name length + name
                name_len_b = _recv_all(conn, _HT_NAME_LEN.size)
                if not name_len_b:
                    break
                (name_len,) = _HT_NAME_LEN.unpack(name_len_b)
                name_b = _recv_all(conn, name_len)
                if not name_b:
                    break
                name = name_b.decode("utf-8")

                # Read data length + data
                data_len_b = _recv_all(conn, _HT_DATA_LEN.size)
                if not data_len_b:
                    break
                (data_len,) = _HT_DATA_LEN.unpack(data_len_b)
                data = _recv_all(conn, data_len)
                if not data:
                    # Connection dropped mid-upload: discard
                    if self.debug:
                        print(f"[Server HT] {addr} upload '{name}' interrupted; discarding")
                    break

                if self.debug:
                    print(f"[Server HT] Received compressed upload '{name}' "
                          f"({data_len} bytes) from {addr}")

                # Decompress & unpickle; if this fails, we do NOT save or ack.
                try:
                    decompressed = bz2.decompress(data)
                    file_hash = hashlib.sha256(decompressed).digest()
                    obj = pickle.loads(decompressed)
                except Exception as e:
                    if self.debug:
                        print(f"[Server HT] Error decoding upload '{name}': {e}")
                    # don't ack; client will treat as failure
                    break

                # Save uncompressed pickle to disk
                path = os.path.join(self.upload_dir, f"{name}.pkl")
                try:
                    with open(path, "wb") as f:
                        pickle.dump(obj, f, protocol=PICKLE_PROTOCOL)
                except Exception as e:
                    if self.debug:
                        print(f"[Server HT] Failed to save '{name}' to disk: {e}")
                    # don't ack; client will retry
                    break

                # Cache in memory and record hash
                with self._uploads_lock:
                    self._uploads[name] = obj
                    self._upload_hashes[name] = file_hash

                # Send ACK only now: this means "upload finished & saved"
                try:
                    if self.debug:
                        print(f"[Server HT] ACK upload '{name}' to {addr}")
                    ack = bytearray()
                    ack.append(_HT_ACK_OP)
                    ack += _HT_NAME_LEN.pack(len(name_b))
                    ack += name_b
                    _send_all(conn, bytes(ack))
                except Exception as e:
                    if self.debug:
                        print(f"[Server HT] Failed to send ACK for '{name}' to {addr}: {e}")
                    break

        finally:
            _safe_close(conn)

    # ---------- Internal: LL handling ----------

    def _ll_accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._ll_sock.accept()
            except OSError:
                break
            if self.debug:
                print(f"[Server LL] Connection from {addr}")
            t = threading.Thread(target=self._ll_client_loop, args=(conn, addr), daemon=True)
            t.start()

    def _ll_client_loop(self, conn: socket.socket, addr):
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while not self._stop.is_set():
                header = _recv_all(conn, _LL_HEADER.size)
                if not header:
                    if self.debug:
                        print(f"[Server LL] {addr} closed connection")
                    break
                (length,) = _LL_HEADER.unpack(header)
                body = _recv_all(conn, length)
                if not body:
                    break

                try:
                    payload = pickle.loads(body)
                    msg = payload.get("msg")
                except Exception:
                    msg = None

                if self.debug:
                    print(f"[Server LL] Received message from {addr}: {msg}")

                with self._ll_lock:
                    self._ll_req_id += 1
                    req_id = self._ll_req_id

                req = _LLRequest(
                    req_id=req_id,
                    conn=conn,
                    addr=addr,
                    arrived_at=time.monotonic(),
                    msg=msg,
                )
                self._ll_queue.put(req)
        finally:
            _safe_close(conn)

    # ---------- Cleanup ----------

    def stop(self):
        self._stop.set()
        _safe_close(self._ht_sock)
        _safe_close(self._ll_sock)
        
    # -------- Get Private IP Address --------
    def get_private_ip(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP