#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shlex
import sys
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image  # required by resize_with_pad's default method

from image_tools import resize_with_pad
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from pickleio import Client

# ---- Constants ----

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

ACTION_HORIZON = 20
DEFAULT_TARGET_FPS = 20.0

# Prompts
DEFAULT_RECORD_PROMPT = "pick up the cube"
DEFAULT_INFER_PROMPT = "pick up the cube"

EPISODE_PREFIX = "episode_"
EPISODE_EXT = ".pkl"

# New: store all episode files in a separate directory
EPISODE_DIR = "episodes"

# Fixed camera indices
MAIN_CAM_INDEX = 1   # main scene camera
WRIST_CAM_INDEX = 0  # wrist camera

# Optional: you can set a lower resolution for better FPS
CAM_WIDTH = 320
CAM_HEIGHT = 240

# Always-on camera window
CAM_WINDOW_NAME = "Cameras (main | wrist) - press q/ESC to quit"


# ---- Helpers for episode directory / naming / file management ----

def _ensure_episode_dir():
    """Ensure the episode storage directory exists."""
    if not os.path.isdir(EPISODE_DIR):
        os.makedirs(EPISODE_DIR, exist_ok=True)


def _episode_filename(idx: int) -> str:
    return os.path.join(EPISODE_DIR, f"{EPISODE_PREFIX}{idx:03d}{EPISODE_EXT}")


def _episode_key(idx: int) -> str:
    """Key used for pickleio uploads and training (no .pkl)."""
    return f"{EPISODE_PREFIX}{idx:03d}"


def _existing_episode_indices() -> List[int]:
    """Return sorted list of episode indices found in EPISODE_DIR."""
    if not os.path.isdir(EPISODE_DIR):
        return []
    indices = []
    for name in os.listdir(EPISODE_DIR):
        if name.startswith(EPISODE_PREFIX) and name.endswith(EPISODE_EXT):
            middle = name[len(EPISODE_PREFIX):-len(EPISODE_EXT)]
            try:
                idx = int(middle)
            except ValueError:
                continue
            indices.append(idx)
    return sorted(indices)


def _next_episode_index() -> int:
    indices = _existing_episode_indices()
    if not indices:
        return 1
    return max(indices) + 1


def _reupload_existing_episodes(client: Client):
    """
    On startup or on-demand, reupload all existing local episodes so that the
    client/server are in sync. Blocks until uploads are finished.
    """
    idxs = _existing_episode_indices()
    if not idxs:
        print("No existing episodes found for reupload.")
        return

    print(f"Found {len(idxs)} existing episodes. Reuploading to server...")
    for idx in idxs:
        filename = _episode_filename(idx)
        try:
            with open(filename, "rb") as f:
                episode_buffer = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load episode '{filename}' for reupload: {e}")
            continue
        key = _episode_key(idx)
        client.queue_upload_ht(key, episode_buffer)
        print(f"  Queued reupload: {filename} -> key='{key}'")

    print("Waiting for all episode uploads to complete...")
    while client.upload_queue_empty_ht() is False:
        time.sleep(0.1)
    print("All existing episodes have been reuploaded.")


# ---- Group-RL naming/helpers ----

def _group_rollout_key(group_num: int, rollout_num: int) -> str:
    # Key used for upload/training
    return f"group_{group_num}_n_{rollout_num}"


def _group_rollout_filename(group_num: int, rollout_num: int) -> str:
    return os.path.join(EPISODE_DIR, f"{_group_rollout_key(group_num, rollout_num)}.pkl")


def _group_json_filename(group_num: int) -> str:
    # Spec says: group_1.json (saved as a json file)
    return f"group_{group_num}.json"


def _existing_group_nums() -> List[int]:
    nums = []
    for name in os.listdir("."):
        if name.startswith("group_") and name.endswith(".json"):
            mid = name[len("group_"):-len(".json")]
            try:
                nums.append(int(mid))
            except ValueError:
                continue
    return sorted(set(nums))


def _next_group_num() -> int:
    nums = _existing_group_nums()
    if not nums:
        return 1
    return max(nums) + 1


# ---- Background camera reader ----

class CameraReader:
    """
    Background camera reader that continuously grabs frames from a fixed index.

    - index: OpenCV camera index (e.g., 0 or 1)
    - latest_frame: last successfully read BGR frame
    """

    def __init__(self, index: int, name: str = "", width: int = None, height: int = None):
        self.index = index
        self.name = name or f"cam{index}"
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._opened_ok = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            print(f"[ERROR] CameraReader({self.name}): failed to open index {self.index}", file=sys.stderr)
            self._running = False
            return

        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._cap = cap
        self._opened_ok = True
        print(f"[INFO] CameraReader({self.name}): started on index {self.index}")

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame

        cap.release()
        print(f"[INFO] CameraReader({self.name}): stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest BGR frame, or None if none available yet."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class CameraDisplay:
    """
    Thread-safe, OpenCV-free camera display using an MJPEG stream in your browser.

    Why:
      - On macOS, cv2.imshow/cv2.waitKey can crash or throw exceptions when called
        from non-main threads.
      - Serving an MJPEG stream is safe to run from a background thread.

    How to view:
      - It auto-opens a browser tab to http://127.0.0.1:<port>/
      - Or open it manually if your environment blocks auto-open.

    Controls:
      - This display does not capture keypresses; use `quit/exit` in the CLI to stop.
    """

    def __init__(
        self,
        cam_main: Optional["CameraReader"],
        cam_wrist: Optional["CameraReader"],
        exit_event: threading.Event,
        fps: float = 30.0,
        window_name: str = "Cameras (main | wrist)",
        host: str = "127.0.0.1",
        port: int = 8008,
        auto_open: bool = True,
    ):
        self.cam_main = cam_main
        self.cam_wrist = cam_wrist
        self.exit_event = exit_event
        self.fps = float(fps)
        self.window_name = window_name

        self.host = host
        self.port = int(port)
        self.auto_open = bool(auto_open)

        self._running = False
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True

        # Build a request handler class that can access `self`
        self_ref = self

        def _get_combo_rgb() -> Optional[np.ndarray]:
            frame_main_rgb = None
            frame_wrist_rgb = None

            if self_ref.cam_main is not None:
                bgr = self_ref.cam_main.get_frame()
                if bgr is not None:
                    frame_main_rgb = _resize_frame_to_224(bgr)

            if self_ref.cam_wrist is not None:
                bgr = self_ref.cam_wrist.get_frame()
                if bgr is not None:
                    frame_wrist_rgb = _resize_frame_to_224(bgr)

            return _combine_two_rgb(frame_main_rgb, frame_wrist_rgb)

        class _Handler:
            # Minimal BaseHTTPRequestHandler replacement defined inside start()
            # to avoid global imports/edits elsewhere.
            def __init__(self, *args, **kwargs):
                from http.server import BaseHTTPRequestHandler

                # Dynamically create a subclass so we can keep everything local here
                class _H(BaseHTTPRequestHandler):
                    def log_message(self, fmt, *a):  # quiet
                        return

                    def do_GET(self):
                        import time as _time
                        import io as _io

                        # Simple landing page
                        if self.path in ("/", "/index.html"):
                            body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{self_ref.window_name}</title>
  <style>
    body {{ margin:0; background:#111; color:#ddd; font-family: sans-serif; }}
    header {{ padding:10px 14px; background:#222; }}
    img {{ display:block; margin:0 auto; max-width:100vw; max-height:calc(100vh - 52px); }}
    .hint {{ opacity:0.7; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <div><b>{self_ref.window_name}</b></div>
    <div class="hint">MJPEG stream. Stop the program with the CLI (quit/exit).</div>
  </header>
  <img src="/stream" />
</body>
</html>"""
                            body_b = body.encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "text/html; charset=utf-8")
                            self.send_header("Content-Length", str(len(body_b)))
                            self.end_headers()
                            self.wfile.write(body_b)
                            return

                        # MJPEG stream
                        if self.path.startswith("/stream"):
                            self.send_response(200)
                            self.send_header("Age", "0")
                            self.send_header("Cache-Control", "no-cache, private")
                            self.send_header("Pragma", "no-cache")
                            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                            self.end_headers()

                            dt = 1.0 / max(1e-6, float(self_ref.fps))

                            while self_ref._running and not self_ref.exit_event.is_set():
                                tick = _time.time()

                                combo = _get_combo_rgb()
                                if combo is None:
                                    _time.sleep(0.01)
                                    continue

                                # Encode to JPEG via Pillow (no OpenCV)
                                from PIL import Image as _Image

                                img = _Image.fromarray(combo)  # combo is RGB uint8
                                buf = _io.BytesIO()
                                img.save(buf, format="JPEG", quality=85)
                                jpg = buf.getvalue()

                                try:
                                    self.wfile.write(b"--frame\r\n")
                                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                                    self.wfile.write(jpg)
                                    self.wfile.write(b"\r\n")
                                except (BrokenPipeError, ConnectionResetError):
                                    break

                                elapsed = _time.time() - tick
                                remain = dt - elapsed
                                if remain > 0:
                                    _time.sleep(remain)
                            return

                        # 404
                        self.send_response(404)
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(b"Not found.\n")

                # Now swap this instance into that handler
                self.__class__ = _H
                _H.__init__(self, *args, **kwargs)

        def _serve():
            from http.server import HTTPServer
            import socket

            # Make the server reusable and not hang on exit
            class _ReuseHTTPServer(HTTPServer):
                allow_reuse_address = True

            try:
                self._server = _ReuseHTTPServer((self.host, self.port), _Handler)
            except OSError as e:
                # If port is in use, try an ephemeral port
                self._server = _ReuseHTTPServer((self.host, 0), _Handler)
                self.port = self._server.server_address[1]
                print(f"[WARN] Port in use; camera stream moved to {self.port} ({e})")

            url = f"http://{self.host}:{self.port}/"
            print(f"[INFO] Camera stream available at: {url}")

            if self.auto_open:
                try:
                    import webbrowser
                    webbrowser.open(url, new=1, autoraise=True)
                except Exception as e:
                    print(f"[WARN] Failed to auto-open browser: {e}")

            # Serve until stop() or exit_event is set
            while self._running and not self.exit_event.is_set():
                self._server.handle_request()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        # Poke the server once so handle_request returns promptly.
        try:
            import socket
            with socket.create_connection((self.host, self.port), timeout=0.2) as s:
                s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        except Exception:
            pass

        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None


# ---- Low-level helpers ----

def _enter_waiter(prompt: str, flag: Dict[str, bool]):
    """Block on input() and set flag['stop'] = True when Enter is pressed."""
    try:
        input(prompt)
    except EOFError:
        pass
    flag["stop"] = True


def _pack_dict_to_array(d: Dict[str, Any]) -> np.ndarray:
    """
    Pack a dict of joint values into a float32 numpy array of shape (6,),
    using JOINT_KEYS order. Missing keys default to 0.0.
    """
    return np.asarray([float(d.get(k, 0.0)) for k in JOINT_KEYS], dtype=np.float32)


def add_actions(episode: List[dict], sent_actions: List[np.ndarray], action_horizon: int = ACTION_HORIZON):
    """
    Add an 'actions' field to every step in the episode.

    'actions' is a (ACTION_HORIZON, 6) float32 array of the next ACTION_HORIZON
    sent_action vectors (including the current one), padded at the end with the
    last available sent_action.

    Mutates the episode in-place.
    """
    n = len(episode)
    if n == 0:
        return

    if len(sent_actions) == 0:
        print("[WARN] add_actions: no sent_actions provided; skipping.", file=sys.stderr)
        return

    if len(sent_actions) < n:
        last = sent_actions[-1]
        sent_actions = list(sent_actions) + [last] * (n - len(sent_actions))
    elif len(sent_actions) > n:
        sent_actions = sent_actions[:n]

    sent_actions = [np.asarray(a, dtype=np.float32) for a in sent_actions]

    for i in range(n):
        end = i + action_horizon
        if end <= n:
            window = sent_actions[i:end]
        else:
            window = sent_actions[i:n]
            last = window[-1]
            pad = [last] * (end - n)
            window = window + pad

        episode[i]["actions"] = np.stack(window, axis=0).astype(np.float32)


def _resize_frame_to_224(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Resize a BGR frame to (224, 224, 3) RGB padded, or return None on failure."""
    if frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = resize_with_pad(frame_rgb, 224, 224)
    frame_resized = np.asarray(frame_resized, dtype=np.uint8)
    if frame_resized.ndim == 4 and frame_resized.shape[0] == 1:
        frame_resized = frame_resized[0]
    return frame_resized


def _combine_two_rgb(
    frame_main_rgb: Optional[np.ndarray],
    frame_wrist_rgb: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Combine two RGB images horizontally, each assumed to be 224x224x3.
    If only one is available, return that one.
    If neither is available, return None.
    """
    if frame_main_rgb is not None and frame_wrist_rgb is not None:
        try:
            return np.hstack([frame_main_rgb, frame_wrist_rgb])
        except Exception:
            return frame_main_rgb
    if frame_main_rgb is not None:
        return frame_main_rgb
    if frame_wrist_rgb is not None:
        return frame_wrist_rgb
    return None


def _parse_int_list_with_ranges(spec: str) -> List[int]:
    """
    Parse "1,3,5,7-10" -> [1,3,5,7,8,9,10]
    Ignores invalid tokens.
    """
    out: List[int] = []
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                ia = int(a.strip())
                ib = int(b.strip())
            except ValueError:
                print(f"[WARN] Ignoring invalid range token '{p}'")
                continue
            if ia <= ib:
                out.extend(range(ia, ib + 1))
            else:
                out.extend(range(ib, ia + 1))
        else:
            try:
                out.append(int(p))
            except ValueError:
                print(f"[WARN] Ignoring invalid token '{p}'")
    # de-dupe preserving order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


# ---- Phase runner (used for recording and teleop) ----

def _run_phase(
    leader: SO101Leader,
    follower: SO101Follower,
    cam_main: Optional[CameraReader],
    cam_wrist: Optional[CameraReader],
    fps: float,
    collect: bool,
    storage: List[dict],
    start_time_ref: List[Any],
    sent_actions_out: Optional[List[np.ndarray]] = None,
    prompt_text: str = DEFAULT_RECORD_PROMPT,
):
    """
    Run a phase at target fps.

    - If collect=False:
        Teleop pass-through (leader -> follower), no recording,
        until Enter pressed.

    - If collect=True:
        Teleop + recording of:
            t,
            leader_action (float32[6]),
            follower_obs (float32[6]),
            image (main, 224x224x3 or None, RGB),
            image_wrist (wrist, 224x224x3 or None, RGB),
            prompt (str),

    Camera frames are taken from the background CameraReader threads.
    """
    dt = 1.0 / fps
    stop_flag = {"stop": False}
    if not collect:
        prompt = "Press enter after setup is complete:"
    else:
        prompt = f"Press enter after episode {start_time_ref[1]} is complete:"
    waiter = threading.Thread(target=_enter_waiter, args=(prompt + " ", stop_flag), daemon=True)
    waiter.start()

    frame_count = 0
    loop_phase_start = time.time()

    while not stop_flag["stop"]:
        tick_start = time.time()

        # Leader action and teleop
        leader_action_raw = leader.get_action()
        sent_action_raw = follower.send_action(leader_action_raw)
        obs_raw = follower.get_observation()

        frame_main_rgb = None
        frame_wrist_rgb = None

        if collect:
            if cam_main is not None:
                frame_main_bgr = cam_main.get_frame()
                if frame_main_bgr is not None:
                    frame_main_rgb = _resize_frame_to_224(frame_main_bgr)

            if cam_wrist is not None:
                frame_wrist_bgr = cam_wrist.get_frame()
                if frame_wrist_bgr is not None:
                    frame_wrist_rgb = _resize_frame_to_224(frame_wrist_bgr)

            leader_action_arr = _pack_dict_to_array(leader_action_raw)
            follower_obs_arr = _pack_dict_to_array(obs_raw if isinstance(obs_raw, dict) else {})

            if sent_actions_out is not None:
                sent_actions_out.append(_pack_dict_to_array(sent_action_raw))

            t = time.time() - start_time_ref[0]
            storage.append(
                {
                    "t": t,
                    "leader_action": leader_action_arr,
                    "follower_obs": follower_obs_arr,
                    "image": frame_main_rgb,
                    "image_wrist": frame_wrist_rgb,
                    "prompt": prompt_text,
                }
            )
            frame_count += 1

        elapsed = time.time() - tick_start
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(
                f"[WARN] Loop overran by {-remaining:.3f}s (actual fps < target {fps})",
                file=sys.stderr,
            )

    if collect and frame_count > 0:
        phase_duration = time.time() - loop_phase_start
        actual_fps = frame_count / phase_duration if phase_duration > 0 else 0.0
        print(f"Episode {start_time_ref[1]} recorded {frame_count} frames @ {actual_fps:.2f} fps (target {fps}).")


# ---- Core operations: record / playback / train / infer / group_rl ----

def record_episode(
    episode_idx: int,
    prompt_text: str,
    follower: SO101Follower,
    leader: SO101Leader,
    cam_main: Optional[CameraReader],
    cam_wrist: Optional[CameraReader],
    client: Client,
    target_fps: float = DEFAULT_TARGET_FPS,
):
    """Record a single episode, save to disk, and queue upload to server."""
    print(f"Recording episode {episode_idx} with prompt: {prompt_text!r}")

    # Setup phase (teleop, no recording)
    _run_phase(
        leader=leader,
        follower=follower,
        cam_main=cam_main,
        cam_wrist=cam_wrist,
        fps=target_fps,
        collect=False,
        storage=[],
        start_time_ref=[0.0, episode_idx],
        sent_actions_out=None,
        prompt_text=prompt_text,
    )

    # Record phase
    episode_buffer: List[dict] = []
    episode_sent_actions: List[np.ndarray] = []
    start_t = time.time()
    start_time_ref = [start_t, episode_idx]

    _run_phase(
        leader=leader,
        follower=follower,
        cam_main=cam_main,
        cam_wrist=cam_wrist,
        fps=target_fps,
        collect=True,
        storage=episode_buffer,
        start_time_ref=start_time_ref,
        sent_actions_out=episode_sent_actions,
        prompt_text=prompt_text,
    )

    # Add future action horizons
    add_actions(episode_buffer, episode_sent_actions, action_horizon=ACTION_HORIZON)

    # Save episode (in EPISODE_DIR)
    out_name = _episode_filename(episode_idx)
    with open(out_name, "wb") as f:
        pickle.dump(episode_buffer, f)
    print(f"Saved {out_name} ({len(episode_buffer)} frames).")

    # Queue upload to training server
    key = _episode_key(episode_idx)
    client.queue_upload_ht(key, episode_buffer)
    print("Saved and queued for upload.")


def playback_episode(
    episode_idx: int,
    follower: SO101Follower,
    fps: float = DEFAULT_TARGET_FPS,
):
    """Play back a saved episode on the follower."""
    filename = _episode_filename(episode_idx)
    if not os.path.exists(filename):
        print(f"[ERROR] Episode file '{filename}' does not exist.")
        return

    try:
        with open(filename, "rb") as f:
            episode = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load episode file '{filename}': {e}")
        return

    if not isinstance(episode, list) or len(episode) == 0:
        print("[ERROR] Episode file is empty or invalid.")
        return

    print(f"Playing back episode {episode_idx} from {filename}...")
    dt = 1.0 / fps

    try:
        for i, step in enumerate(episode):
            if "actions" in step:
                arr = np.asarray(step["actions"][0], dtype=float)
            elif "leader_action" in step:
                arr = np.asarray(step["leader_action"], dtype=float)
            else:
                print(f"[WARN] Step {i} has no usable action; skipping.")
                continue

            if arr.shape[0] != len(JOINT_KEYS):
                print(f"[WARN] Step {i} action has wrong shape {arr.shape}; skipping.")
                continue

            cmd = {k: float(arr[j]) for j, k in enumerate(JOINT_KEYS)}
            follower.send_action(cmd)
            time.sleep(dt)

        print("Playback complete.")
    except KeyboardInterrupt:
        print("\n[INFO] Playback interrupted by user.")


def _train_keys(client: Client, keys: List[str], epochs: int, lr: float):
    if not keys:
        print("[ERROR] No valid episodes selected for training.")
        return

    print(f"Training on episodes: {keys}")
    print("Waiting for upload to complete...")
    while client.upload_queue_empty_ht() is False:
        time.sleep(0.1)

    print("Training...")
    resp = client.send_message_ll(
        {
            "command": "train",
            "num_epochs": int(epochs),
            "pkl_data_keys": keys,
            "lr": float(lr),
        }
    )
    print("Done.")
    if resp is not None:
        print("Train response:", resp)


def train_episodes(
    episodes_spec: str,
    client: Client,
    epochs: int = 1,
    lr: float = 1e-6,
):
    """Trigger training on the server for selected episodes (supports ranges like 1,3,7-10)."""
    if episodes_spec == "all":
        idxs = _existing_episode_indices()
    else:
        idxs = _parse_int_list_with_ranges(episodes_spec)

    keys: List[str] = []
    for idx in idxs:
        filename = _episode_filename(idx)
        if os.path.exists(filename):
            keys.append(_episode_key(idx))
        else:
            print(f"[WARN] Episode file '{filename}' not found; skipping.")

    _train_keys(client=client, keys=keys, epochs=epochs, lr=lr)


def infer_loop(
    seconds: float,
    prompt_text: str,
    follower: SO101Follower,
    cam_main: Optional[CameraReader],
    cam_wrist: Optional[CameraReader],
    client: Client,
    fps: float = DEFAULT_TARGET_FPS,
):
    """
    Live inference loop using background camera readers.

    Each cycle:
      - Capture current follower observation.
      - Get main + wrist frames from CameraReader(s), resize to 224x224 RGB.
      - Send obs to server.
      - Execute returned actions.
    """
    print(f"Starting infer loop for {seconds} seconds with prompt: {prompt_text!r}")
    t_end = time.time() + seconds
    dt = 1.0 / fps

    try:
        while time.time() < t_end:
            obs_raw = follower.get_observation()
            leader_action_arr = _pack_dict_to_array(obs_raw if isinstance(obs_raw, dict) else {})

            image_main = None
            image_wrist = None

            if cam_main is not None:
                frame_main_bgr = cam_main.get_frame()
                if frame_main_bgr is not None:
                    image_main = _resize_frame_to_224(frame_main_bgr)

            if cam_wrist is not None:
                frame_wrist_bgr = cam_wrist.get_frame()
                if frame_wrist_bgr is not None:
                    image_wrist = _resize_frame_to_224(frame_wrist_bgr)

            if image_main is None or image_wrist is None:
                time.sleep(dt)
                continue

            obs_payload = {
                "leader_action": leader_action_arr,
                "image": image_main,
                "image_wrist": image_wrist,
                "prompt": prompt_text,
            }

            result = client.send_message_ll({"command": "infer", "obs": obs_payload})
            if not isinstance(result, dict) or "actions" not in result:
                print("[WARN] Invalid inference result:", result)
                continue

            actions = np.asarray(result["actions"], dtype=float)
            if actions.ndim != 2 or actions.shape[1] != len(JOINT_KEYS):
                print("[WARN] Inference actions shape unexpected:", actions.shape)
                continue

            for a in actions[:ACTION_HORIZON]:
                cmd = {k: float(a[i]) for i, k in enumerate(JOINT_KEYS)}
                follower.send_action(cmd)
                time.sleep(dt)

            time.sleep(0.1)

        print("Inference loop finished.")
    except KeyboardInterrupt:
        print("\n[INFO] Inference loop interrupted by user.")


def _group_rl_single_rollout(
    group_num: int,
    rollout_num: int,
    max_dur: float,
    prompt_text: str,
    follower: SO101Follower,
    leader: SO101Leader,
    cam_main: Optional[CameraReader],
    cam_wrist: Optional[CameraReader],
    client: Client,
    fps: float,
) -> Tuple[float, str]:
    """
    Run one Group-RL rollout:
      - user resets (teleop setup, not recorded)
      - policy runs up to max_dur (user presses Enter when done)
      - duration saved
      - rollout recorded like record, BUT we do not record "server inference cycles";
        we only record per executed action step.
    Returns (duration, key).
    """
    key = _group_rollout_key(group_num, rollout_num)
    filename = _group_rollout_filename(group_num, rollout_num)
    dt = 1.0 / fps

    print(f"\n[GROUP_RL] Rollout {rollout_num}/{key}: reset/setup phase (teleop).")
    _run_phase(
        leader=leader,
        follower=follower,
        cam_main=cam_main,
        cam_wrist=cam_wrist,
        fps=fps,
        collect=False,
        storage=[],
        start_time_ref=[0.0, key],
        sent_actions_out=None,
        prompt_text=prompt_text,
    )

    print(f"[GROUP_RL] Rollout {rollout_num}: running policy. Press Enter when task is complete. (Ctrl-C aborts rollout)")
    stop_flag = {"stop": False}
    waiter = threading.Thread(target=_enter_waiter, args=("Complete? Press Enter: ", stop_flag), daemon=True)
    waiter.start()

    start_wall = time.time()
    t_end = start_wall + max_dur

    episode_buffer: List[dict] = []
    episode_sent_actions: List[np.ndarray] = []

    duration = max_dur
    aborted = False

    try:
        while time.time() < t_end and not stop_flag["stop"]:
            # Prepare obs + images for inference call
            obs_raw = follower.get_observation()
            obs_arr = _pack_dict_to_array(obs_raw if isinstance(obs_raw, dict) else {})

            image_main = None
            image_wrist = None
            if cam_main is not None:
                bgr = cam_main.get_frame()
                if bgr is not None:
                    image_main = _resize_frame_to_224(bgr)
            if cam_wrist is not None:
                bgr = cam_wrist.get_frame()
                if bgr is not None:
                    image_wrist = _resize_frame_to_224(bgr)

            # If we can't see both cams, skip this cycle.
            if image_main is None or image_wrist is None:
                time.sleep(dt)
                continue

            obs_payload = {
                "leader_action": obs_arr,
                "image": image_main,
                "image_wrist": image_wrist,
                "prompt": prompt_text,
            }

            # Inference call (not recorded as steps)
            result = client.send_message_ll({"command": "infer", "obs": obs_payload})
            if not isinstance(result, dict) or "actions" not in result:
                print("[WARN] Invalid inference result:", result)
                time.sleep(0.05)
                continue

            actions = np.asarray(result["actions"], dtype=float)
            if actions.ndim != 2 or actions.shape[1] != len(JOINT_KEYS):
                print("[WARN] Inference actions shape unexpected:", actions.shape)
                time.sleep(0.05)
                continue

            # Execute actions; record per executed step
            for a in actions[:ACTION_HORIZON]:
                if stop_flag["stop"] or time.time() >= t_end:
                    break

                # Capture *current* obs/images for the recorded step.
                # (If you prefer "before action" vs "after action" semantics, adjust here.)
                step_obs_raw = follower.get_observation()
                step_obs_arr = _pack_dict_to_array(step_obs_raw if isinstance(step_obs_raw, dict) else {})

                step_image_main = None
                step_image_wrist = None
                if cam_main is not None:
                    bgr = cam_main.get_frame()
                    if bgr is not None:
                        step_image_main = _resize_frame_to_224(bgr)
                if cam_wrist is not None:
                    bgr = cam_wrist.get_frame()
                    if bgr is not None:
                        step_image_wrist = _resize_frame_to_224(bgr)

                # Execute the policy action
                cmd = {k: float(a[i]) for i, k in enumerate(JOINT_KEYS)}
                follower.send_action(cmd)

                # Record the executed step
                t_rel = time.time() - start_wall
                a_arr = np.asarray(a, dtype=np.float32)
                episode_sent_actions.append(a_arr.astype(np.float32))
                episode_buffer.append(
                    {
                        "t": float(t_rel),
                        # Keep field name "leader_action" for compatibility;
                        # here it is the policy action that was executed.
                        "leader_action": a_arr.astype(np.float32),
                        "follower_obs": step_obs_arr,
                        "image": step_image_main,
                        "image_wrist": step_image_wrist,
                        "prompt": prompt_text,
                    }
                )

                time.sleep(dt)

            # small gap between horizons
            time.sleep(0.05)

        if stop_flag["stop"]:
            duration = max(0.0, time.time() - start_wall)
        else:
            duration = max_dur

    except KeyboardInterrupt:
        aborted = True
        duration = max_dur
        print("\n[GROUP_RL] Rollout aborted with Ctrl-C; marking duration = max_dur.")
    finally:
        # Add action horizons like record
        add_actions(episode_buffer, episode_sent_actions, action_horizon=ACTION_HORIZON)

        # Save rollout
        with open(filename, "wb") as f:
            pickle.dump(episode_buffer, f)
        print(f"[GROUP_RL] Saved rollout: {filename} ({len(episode_buffer)} frames).")

        # Queue upload
        client.queue_upload_ht(key, episode_buffer)
        print(f"[GROUP_RL] Queued upload for key='{key}'.")

        if aborted:
            print("[GROUP_RL] (Tip) If you want partial rollouts excluded from training, just don't select them.")

    return float(duration), key


def group_rl(
    group_num: int,
    n: int,
    max_dur: float,
    lr: float,
    epochs: int,
    follower: SO101Follower,
    leader: SO101Leader,
    cam_main: Optional[CameraReader],
    cam_wrist: Optional[CameraReader],
    client: Client,
    fps: float = DEFAULT_TARGET_FPS,
    prompt_text: str = DEFAULT_INFER_PROMPT,
):
    """
    Group-RL flow:
      1) Generate n rollouts (policy inference + execution); record each rollout.
      2) Save durations as group_<num>.json
      3) Present rollouts sorted by duration; user selects which to train.
      4) Train using selected rollouts with lr/epochs.
      5) Save checkpoint named group_<num>.
    """
    print(f"\n[GROUP_RL] Starting group_rl num={group_num} n={n} max_dur={max_dur} lr={lr} epochs={epochs}")

    durations: List[Dict[str, Any]] = []
    for i in range(1, n + 1):
        dur, key = _group_rl_single_rollout(
            group_num=group_num,
            rollout_num=i,
            max_dur=max_dur,
            prompt_text=prompt_text,
            follower=follower,
            leader=leader,
            cam_main=cam_main,
            cam_wrist=cam_wrist,
            client=client,
            fps=fps,
        )
        durations.append({"rollout": i, "duration": float(dur), "key": key})

    # Save durations json
    json_name = _group_json_filename(group_num)
    with open(json_name, "w") as f:
        json.dump(
            {
                "group": group_num,
                "n": n,
                "max_dur": float(max_dur),
                "lr": float(lr),
                "epochs": int(epochs),
                "rollouts": durations,
            },
            f,
            indent=2,
        )
    print(f"\n[GROUP_RL] Saved durations: {json_name}")

    # Present rollouts sorted by duration
    sorted_rollouts = sorted(durations, key=lambda x: x.get("duration", float("inf")))
    print("\n[GROUP_RL] Rollouts sorted by duration (lower is better):")
    for rank, r in enumerate(sorted_rollouts, start=1):
        print(f"  {rank:02d}) rollout={r['rollout']:d}  duration={r['duration']:.3f}s  key={r['key']}")

    # User selects rollouts to train
    try:
        sel = input("\nSelect rollouts to train (e.g., 1,4,5,8 or 1-3): ").strip()
    except EOFError:
        sel = ""

    chosen = _parse_int_list_with_ranges(sel)
    if not chosen:
        print("[GROUP_RL] No rollouts selected; skipping training and save.")
        return

    chosen_keys: List[str] = []
    for rnum in chosen:
        if 1 <= rnum <= n:
            chosen_keys.append(_group_rollout_key(group_num, rnum))
        else:
            print(f"[WARN] Rollout {rnum} out of range 1..{n}; skipping.")

    if not chosen_keys:
        print("[GROUP_RL] No valid chosen rollouts; skipping.")
        return

    # Train
    _train_keys(client=client, keys=chosen_keys, epochs=epochs, lr=lr)

    # # Save checkpoint
    # ckpt_name = f"group_{group_num}"
    # client.send_message_ll({"command": "save", "name": ckpt_name})
    # print(f"[GROUP_RL] Saved checkpoint '{ckpt_name}'.")


# ---- CLI / main ----

def parse_command(line: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a line like:
      record num=1 prompt="move ..."
    into:
      ("record", {"num": "1", "prompt": "move ..."})
    """
    parts = shlex.split(line)
    if not parts:
        return "", {}
    cmd = parts[0]
    args: Dict[str, str] = {}
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            args[k] = v
    return cmd, args


def main():
    parser = argparse.ArgumentParser(description="SO101 teleop, training, and inference CLI.")
    parser.add_argument("--ip", type=str, default="10.246.224.208", help="pickleio server IP")
    parser.add_argument("--debug", action="store_true", help="Enable pickleio Client debug mode.")
    args = parser.parse_args()

    client = Client(args.ip, debug=args.debug)

    # Ensure episode directory exists
    _ensure_episode_dir()
    # Optional: uncomment if you want auto-sync on startup
    # _reupload_existing_episodes(client)

    follower = None
    leader = None
    cam_main: Optional[CameraReader] = None
    cam_wrist: Optional[CameraReader] = None

    exit_event = threading.Event()

    print("Initializing robots and cameras...")
    try:
        follower_cfg = SO101FollowerConfig(
            port="/dev/tty.usbmodem5A7A0158161",
            id="main_follower",
            use_degrees=False,
        )
        leader_cfg = SO101LeaderConfig(
            port="/dev/tty.usbmodem5A7A0158891",
            id="main_leader",
            use_degrees=False,
        )
        follower = SO101Follower(follower_cfg)
        leader = SO101Leader(leader_cfg)
        follower.connect(calibrate=False)
        leader.connect(calibrate=False)
        print("  - Robots connected.")
    except Exception as e:
        print(f"[WARN] Failed to connect robots: {e}")
        follower = None
        leader = None

    # Start background camera readers (fixed indices)
    cam_main = CameraReader(MAIN_CAM_INDEX, name="main", width=CAM_WIDTH, height=CAM_HEIGHT)
    cam_wrist = CameraReader(WRIST_CAM_INDEX, name="wrist", width=CAM_WIDTH, height=CAM_HEIGHT)
    cam_main.start()
    cam_wrist.start()

    # Always-on camera display thread
    cam_display = CameraDisplay(cam_main=cam_main, cam_wrist=cam_wrist, exit_event=exit_event, fps=30.0)
    cam_display.start()

    print("\npickleio server IP:", args.ip)
    print("Robot connection:", "OK" if (follower is not None and leader is not None) else "FAILED")
    print(f"Cameras: main index={MAIN_CAM_INDEX}, wrist index={WRIST_CAM_INDEX}")
    print("\nCommands:")
    print('  record num=1 prompt="move the red cube on the green cube"')
    print("      # if num is omitted it counts up automatically from 1")
    print("  teleop")
    print("      # teleoperation only (setup phase), no recording or saving")
    print("  playback num=1")
    print("      # plays back episode one")
    print("  train episodes=all epochs=1 lr=1e-6")
    print("      # episodes supports ranges, e.g. episodes=1,3,5,7-10")
    print("  infer seconds=5 prompt=\"move the red cube on the green cube\"")
    print("      # live inference and execution loop")
    print("  sync")
    print("      # reupload all locally-saved episodes to server")
    print("  load checkpoint=NAME")
    print("      # loads a server checkpoint named NAME")
    print("  save checkpoint=NAME")
    print("      # saves the current server policy as checkpoint NAME")
    print("  store checkpoint=NAME")
    print("      # alias for save")
    print("  group_rl lr=1e-5 epochs=1 num=1 n=10 max_dur=60")
    print("      # generates n rollouts (policy inference), saves durations group_<num>.json,")
    print("      # lets you choose rollouts to train, then saves checkpoint group_<num>")
    print("  quit / exit")
    print()

    next_idx = _next_episode_index()
    next_group = _next_group_num()

    try:
        while not exit_event.is_set():
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            cmd, cmd_args = parse_command(line)

            if cmd in ("quit", "exit"):
                break

            if cmd == "record":
                if follower is None or leader is None:
                    print("[ERROR] Robots must be connected for recording.")
                    continue

                num_str = cmd_args.get("num")
                if num_str is not None:
                    try:
                        episode_idx = int(num_str)
                    except ValueError:
                        print(f"[ERROR] Invalid num value: {num_str}")
                        continue
                    next_idx = max(next_idx, episode_idx + 1)
                else:
                    episode_idx = next_idx
                    next_idx += 1

                prompt_text = cmd_args.get("prompt", DEFAULT_RECORD_PROMPT)
                record_episode(
                    episode_idx=episode_idx,
                    prompt_text=prompt_text,
                    follower=follower,
                    leader=leader,
                    cam_main=cam_main,
                    cam_wrist=cam_wrist,
                    client=client,
                    target_fps=DEFAULT_TARGET_FPS,
                )

            elif cmd == "teleop":
                if follower is None or leader is None:
                    print("[ERROR] Robots must be connected for teleop.")
                    continue

                print("Starting teleop (no recording). Press Enter in the terminal to stop.")
                _run_phase(
                    leader=leader,
                    follower=follower,
                    cam_main=cam_main,
                    cam_wrist=cam_wrist,
                    fps=DEFAULT_TARGET_FPS,
                    collect=False,
                    storage=[],
                    start_time_ref=[0.0, 0],
                    sent_actions_out=None,
                    prompt_text=DEFAULT_RECORD_PROMPT,
                )
                print("Teleop finished.")

            elif cmd == "playback":
                if follower is None:
                    print("[ERROR] Follower must be connected for playback.")
                    continue
                num_str = cmd_args.get("num")
                if num_str is None:
                    print("[ERROR] playback requires num=IDX")
                    continue
                try:
                    episode_idx = int(num_str)
                except ValueError:
                    print(f"[ERROR] Invalid num value: {num_str}")
                    continue
                playback_episode(episode_idx=episode_idx, follower=follower, fps=DEFAULT_TARGET_FPS)

            elif cmd == "train":
                episodes_spec = cmd_args.get("episodes", "all")
                epochs_str = cmd_args.get("epochs", "1")
                lr_str = cmd_args.get("lr", "1e-6")
                try:
                    epochs = int(epochs_str)
                except ValueError:
                    print(f"[ERROR] Invalid epochs value: {epochs_str}")
                    continue
                try:
                    lr = float(lr_str)
                except ValueError:
                    print(f"[ERROR] Invalid lr value: {lr_str}")
                    continue

                train_episodes(
                    episodes_spec=episodes_spec,
                    client=client,
                    epochs=epochs,
                    lr=lr,
                )

            elif cmd == "infer":
                if follower is None:
                    print("[ERROR] Follower must be connected for inference.")
                    continue
                seconds_str = cmd_args.get("seconds", "5")
                prompt_text = cmd_args.get("prompt", DEFAULT_INFER_PROMPT)
                try:
                    seconds = float(seconds_str)
                except ValueError:
                    print(f"[ERROR] Invalid seconds value: {seconds_str}")
                    continue

                infer_loop(
                    seconds=seconds,
                    prompt_text=prompt_text,
                    follower=follower,
                    cam_main=cam_main,
                    cam_wrist=cam_wrist,
                    client=client,
                    fps=DEFAULT_TARGET_FPS,
                )

            elif cmd == "sync":
                _reupload_existing_episodes(client)

            elif cmd == "load":
                checkpoint = cmd_args.get("checkpoint")
                if checkpoint is None:
                    print("[ERROR] load requires checkpoint=NAME")
                    continue
                client.send_message_ll({"command": "load", "name": checkpoint})
                print(f"Loaded checkpoint '{checkpoint}'.")

            elif cmd in ("save", "store"):
                checkpoint = cmd_args.get("checkpoint")
                if checkpoint is None:
                    print(f"[ERROR] {cmd} requires checkpoint=NAME")
                    continue
                client.send_message_ll({"command": "save", "name": checkpoint})
                print(f"Saved checkpoint '{checkpoint}'.")

            elif cmd == "group_rl":
                if follower is None or leader is None:
                    print("[ERROR] Robots must be connected for group_rl.")
                    continue

                lr_str = cmd_args.get("lr", "1e-5")
                epochs_str = cmd_args.get("epochs", "1")
                num_str = cmd_args.get("num")
                n_str = cmd_args.get("n", "10")
                max_dur_str = cmd_args.get("max_dur", "60")
                prompt_text = cmd_args.get("prompt", DEFAULT_INFER_PROMPT)

                try:
                    lr = float(lr_str)
                except ValueError:
                    print(f"[ERROR] Invalid lr value: {lr_str}")
                    continue
                try:
                    epochs = int(epochs_str)
                except ValueError:
                    print(f"[ERROR] Invalid epochs value: {epochs_str}")
                    continue
                try:
                    n = int(n_str)
                except ValueError:
                    print(f"[ERROR] Invalid n value: {n_str}")
                    continue
                try:
                    max_dur = float(max_dur_str)
                except ValueError:
                    print(f"[ERROR] Invalid max_dur value: {max_dur_str}")
                    continue

                if num_str is not None:
                    try:
                        group_num = int(num_str)
                    except ValueError:
                        print(f"[ERROR] Invalid num value: {num_str}")
                        continue
                    next_group = max(next_group, group_num + 1)
                else:
                    group_num = next_group
                    next_group += 1

                group_rl(
                    group_num=group_num,
                    n=n,
                    max_dur=max_dur,
                    lr=lr,
                    epochs=epochs,
                    follower=follower,
                    leader=leader,
                    cam_main=cam_main,
                    cam_wrist=cam_wrist,
                    client=client,
                    fps=DEFAULT_TARGET_FPS,
                    prompt_text=prompt_text,
                )

            else:
                print(f"[ERROR] Unknown command: {cmd}")

    finally:
        exit_event.set()

        if cam_display is not None:
            cam_display.stop()

        if cam_main is not None:
            cam_main.stop()
        if cam_wrist is not None:
            cam_wrist.stop()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if follower is not None:
            try:
                follower.disconnect()
            except Exception:
                pass
        if leader is not None:
            try:
                leader.disconnect()
            except Exception:
                pass
        print("Exiting.")


if __name__ == "__main__":
    main()
