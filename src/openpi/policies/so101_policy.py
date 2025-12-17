import dataclasses
import pathlib

import einops
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model

def make_so101_example() -> dict:
    return {
        "leader_action": np.random.rand(6),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(_transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Use leader_action only as state.
        state = np.asarray(data["leader_action"])

        # Base image is required and mapped to base_0_rgb.
        base_image = _parse_image(data["image"])

        # Optional wrist image.
        wrist_raw = data.get("image_wrist", None)
        has_wrist = wrist_raw is not None
        if has_wrist:
            wrist_image = _parse_image(wrist_raw)
        else:
            wrist_image = np.zeros_like(base_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Three camera slots:
                # - base_0_rgb: always main image
                # - left_wrist_0_rgb: optional wrist image (masked if missing)
                # - right_wrist_0_rgb: unused, always masked
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (
                    base_image,
                    wrist_image if has_wrist else np.zeros_like(base_image),
                    np.zeros_like(base_image),
                )
                image_masks = (
                    np.True_,                # base_0_rgb always present
                    np.bool_(has_wrist),     # left_wrist_0_rgb only if wrist exists
                    np.False_,               # right_wrist_0_rgb unused
                )

            case _model.ModelType.PI0_FAST:
                # For FAST-style models, follow the Droid layout style:
                # - base_0_rgb: main image
                # - base_1_rgb: optional second view from wrist if present, else zeros (masked out if missing)
                # - wrist_0_rgb: unused, masked
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                base_1 = wrist_image if has_wrist else np.zeros_like(base_image)
                wrist_0 = np.zeros_like(base_image)
                images = (base_image, base_1, wrist_0)
                image_masks = (
                    np.True_,                # base_0_rgb always present
                    np.bool_(has_wrist),     # base_1_rgb only if wrist exists
                    np.False_,               # wrist_0_rgb unused
                )

            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Optional supervised actions (for training).
        if "actions" in data:
            inputs["actions"] = np.copy(data["actions"])

        # Optional prompt.
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # SO101 actions: first 6 dims are relevant.
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, :6]}

