import dataclasses
import functools
import logging
import platform
import time
from typing import Any
import orbax.checkpoint as ocp

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.policies.so101_policy as so101_policy
import openpi.transforms as _transforms

from pickleio import Server

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    # lr applied manually
    tx = _optimizer.create_optimizer(config.optimizer, _optimizer.ConstantSchedule(1), weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding

import optax
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    lr: at.Float[at.Array, ""]
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params) 
    scaled_updates = jax.tree.map(lambda g: lr * g, updates)
    new_params = optax.apply_updates(params, scaled_updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

def make_infer_fn(
    config: _config.TrainConfig,
    *,
    use_ema: bool = True,
):
    """Return a Python-level infer_fn that:
      - applies input_transforms (outside jit),
      - calls a jitted core using current state params,
      - applies output_transforms (outside jit).
    """

    data_config = config.data.create(config.assets_dirs, config.model)

    input_transforms = _transforms.compose([            
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    output_transforms = _transforms.compose([
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])

    @jax.jit
    def _infer_jitted(
        state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        inputs_batched: dict,
        **sample_kwargs,
    ):

        # Choose which params to use (EMA or raw)
        params = (
            state.ema_params
            if (use_ema and state.ema_params is not None)
            else state.params
        )

        # Rebuild model with *current* params and set eval mode
        model = nnx.merge(state.model_def, params)
        model.eval()

        # Wrap dict into model's Observation type
        observation = _model.Observation.from_dict(inputs_batched)

        # Split rng for sampling
        rng, sample_rng = jax.random.split(rng)

        # Run model's sampling in eval mode
        actions = model.sample_actions(sample_rng, observation, **sample_kwargs)

        # Return actions + updated rng; no transforms/unbatching here
        return actions, rng

    def infer_fn(
        state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        obs: dict,
        **sample_kwargs,
    ) -> tuple[at.KeyArrayLike, dict]:
        # -------- pre: outside jit --------
        # Copy structure so transforms can't mutate caller's obs containers
        inputs = jax.tree.map(lambda x: x, obs)

        # Apply your composed input transforms (Python-level)
        # (these can normalize, reshape, etc.)
        inputs = input_transforms(inputs)

        # Convert leaves to jnp and add batch dim for the model
        inputs_batched = jax.tree.map(
            lambda x: jnp.asarray(x)[jnp.newaxis, ...],
            inputs,
        )

        # -------- core: inside jit --------
        actions_batched, new_rng = _infer_jitted(
            state,
            rng,
            inputs_batched,
            **sample_kwargs,
        )

        # -------- post: outside jit --------
        # Build raw outputs PyTree (still batched, JAX types)
        raw_outputs_batched = {
            "state": inputs_batched.get("state", inputs_batched),
            "actions": actions_batched,
        }

        # Remove batch dim + convert to numpy
        raw_outputs = jax.tree.map(
            lambda x: np.array(x[0, ...]),
            raw_outputs_batched,
        )

        # Apply your composed output transforms (Python-level)
        outputs = output_transforms(raw_outputs)

        return new_rng, outputs

    return infer_fn


from typing import Sequence
def create_dataloader(config: _config.TrainConfig, num_epochs: int, pkl_data_keys: Sequence[str]) -> Any:
    data_config = config.data.create(config.assets_dirs, config.model)
    data_loader = _data_loader.create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=None,
        shuffle=True,
        num_batches=-num_epochs,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=False,
        framework="jax",
        pkl_data_keys=pkl_data_keys,
    )
    return data_loader

def main(config: _config.TrainConfig):
    init_logging()




    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng, infer_rng = jax.random.split(rng, 3)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
    #     config.checkpoint_dir,
    #     keep_period=config.keep_period,
    #     overwrite=config.overwrite,
    #     resume=config.resume,
    # )
    
    init_wandb(config, resuming=False, enabled=config.wandb_enabled)

    # data_loader = _data_loader.create_data_loader(
    #     config,
    #     sharding=data_sharding,
    #     shuffle=True,
    # )
    # data_iter = iter(data_loader)
    # batch = next(data_iter)
    # logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # if resuming:
    #     train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, None),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    example_obs = so101_policy.make_so101_example()

    infer_fn = make_infer_fn(
        config,
        use_ema=config.ema_decay
    )


    server = Server(debug=True)
    # print private ip
    print("Starting Server")
    server.start()
    print("Private IP:", server.get_private_ip())

    infos = []
    while True:

        with server.receive_ll(timeout=0.1) as msg:
            if msg is None:
                continue

            if msg["command"] == "train":
                dataloader = create_dataloader(config, num_epochs=msg["num_epochs"], pkl_data_keys=msg["pkl_data_keys"])
                data_iter = iter(dataloader)

                print(f"Training for {msg['num_epochs']} epochs on {msg['pkl_data_keys']}")
                while True:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    
                    with sharding.set_mesh(mesh):
                        train_state, info = ptrain_step(train_rng, train_state, batch, msg["lr"])
                    infos.append(info)
                    stacked_infos = common_utils.stack_forest(infos)
                    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                    info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                    wandb.log(reduced_info, step=int(jax.device_get(train_state.step)))

                    print(info_str)
                    infos = []

                print("Finished training")

            elif msg["command"] == "infer":
                print("Running inference")
                infer_rng, outputs = infer_fn(train_state, infer_rng, msg["obs"])
                print("Done")
                server.send_response_ll(outputs)

            elif msg["command"] == "stop":
                print("Stopping server")
                break
                
            elif msg["command"] == "save":
                if not msg.get("name"):
                    print("Please provide a name for the checkpoint.")
                    continue
                print(f"Saving checkpoint with name {msg['name']}")
                # _checkpoints.save_state(checkpoint_manager, train_state, None, int(jax.device_get(train_state.step)))
                checkpoint_dir = config.checkpoint_dir / msg['name']
                with ocp.CheckpointManager(
                    checkpoint_dir,
                    item_handlers={
                        "assets": _checkpoints.CallbackHandler(),
                        "train_state": ocp.PyTreeCheckpointHandler(),
                        "params": ocp.PyTreeCheckpointHandler(),
                    },
                    options=ocp.CheckpointManagerOptions(
                        max_to_keep=1,
                        create=True,
                        enable_async_checkpointing=False,
                    ),
                ) as mngr:
                    _checkpoints.save_state(mngr, train_state, None, 0)
                print("Checkpoint saved")

            elif msg["command"] == "load":
                if not msg.get("name"):
                    print("Please provide a name for the checkpoint.")
                    continue
                print(f"Loading checkpoint from name {msg['name']}")
                checkpoint_dir = config.checkpoint_dir / msg['name']
                if not checkpoint_dir.exists():
                    print(f"Checkpoint directory {checkpoint_dir} does not exist.")
                    continue
                with ocp.CheckpointManager(
                    checkpoint_dir,
                    item_handlers={
                        "assets": _checkpoints.CallbackHandler(),
                        "train_state": ocp.PyTreeCheckpointHandler(),
                        "params": ocp.PyTreeCheckpointHandler(),
                    },
                    options=ocp.CheckpointManagerOptions(
                        max_to_keep=1,
                        create=False,
                        enable_async_checkpointing=False,
                    ),
                ) as mngr:
                    train_state = _checkpoints.restore_state(mngr, train_state, None)
                print("Checkpoint loaded")
    logging.info("Done training.")

if __name__ == "__main__":
    main(_config.cli())
