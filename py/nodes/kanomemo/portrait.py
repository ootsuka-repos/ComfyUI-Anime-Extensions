"""Kanomemo portrait segmentation nodes backed by imgutils ISNetIS."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io, ui

from ...node_utils import mk_name


_SEGMENT_LOCK = threading.RLock()
_HF_HOME = Path(folder_paths.models_dir) / "huggingface"
_HF_HUB_CACHE = _HF_HOME / "hub"
_ISNETIS_SCALE = 1024


def _configure_huggingface_cache() -> None:
    """Keep imgutils' anime-seg snapshot inside the shared ComfyUI model tree."""

    _HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(_HF_HUB_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("ONNX_MODE", "cpu")
    try:
        import huggingface_hub.constants as hub_constants
    except ImportError as exc:
        raise RuntimeError("Kanomemo imgutils requires huggingface_hub") from exc
    hub_constants.HF_HOME = str(_HF_HOME)
    hub_constants.HF_HUB_CACHE = str(_HF_HUB_CACHE)


def _segment_one(image: torch.Tensor, *, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 3 or image.shape[-1] < 3:
        raise RuntimeError("Kanomemo imgutils expects an RGB IMAGE tensor")
    if scale != _ISNETIS_SCALE:
        raise RuntimeError(f"Kanomemo imgutils requires fixed scale {_ISNETIS_SCALE}")

    rgb = image[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    source = Image.fromarray((rgb.numpy() * 255.0).round().astype(np.uint8), "RGB")
    with _SEGMENT_LOCK:
        _configure_huggingface_cache()
        try:
            from imgutils.segment import segment_rgba_with_isnetis
        except ImportError as exc:
            raise RuntimeError(
                "Kanomemo imgutils is not installed in the shared ComfyUI environment"
            ) from exc
        _, rgba = segment_rgba_with_isnetis(source, scale=scale)

    rgba_array = np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255.0
    segmented = torch.from_numpy(np.ascontiguousarray(rgba_array[..., :3]))
    foreground_mask = torch.from_numpy(np.ascontiguousarray(rgba_array[..., 3]))
    return segmented, foreground_mask


class KanomemoImgutilsSegmentRGBA(io.ComfyNode):
    """Extract an anime character with imgutils.segment_rgba_with_isnetis."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name("Kanomemo", "SegmentRGBA"),
            display_name="Kanomemo Character Segment (imgutils)",
            category="Kanomemo/portrait",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "scale",
                    default=_ISNETIS_SCALE,
                    min=_ISNETIS_SCALE,
                    max=_ISNETIS_SCALE,
                    step=1,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="foreground_mask"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, scale: int):
        segmented: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for sample in image:
            rgb, mask = _segment_one(sample, scale=int(scale))
            segmented.append(rgb)
            masks.append(mask)
        return io.NodeOutput(torch.stack(segmented, dim=0), torch.stack(masks, dim=0))


class KanomemoSaveRGBA(io.ComfyNode):
    """Save an IMAGE plus foreground mask as a transparent PNG output."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name("Kanomemo", "SaveRGBA"),
            display_name="Kanomemo Save RGBA",
            category="Kanomemo/portrait",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("foreground_mask"),
                io.String.Input("filename_prefix", default="kanomemo/portrait"),
            ],
            outputs=[io.Image.Output(display_name="image")],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        foreground_mask: torch.Tensor,
        filename_prefix: str,
    ):
        if image.ndim != 4 or foreground_mask.ndim != 3:
            raise RuntimeError("Kanomemo Save RGBA expects batched IMAGE and MASK values")
        if image.shape[0] != foreground_mask.shape[0]:
            raise RuntimeError("Kanomemo Save RGBA image/mask batch sizes differ")
        output_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            str(filename_prefix),
            folder_paths.get_output_directory(),
            int(image.shape[2]),
            int(image.shape[1]),
        )
        results: list[ui.SavedResult] = []
        for index, (rgb_tensor, alpha_tensor) in enumerate(zip(image, foreground_mask, strict=True)):
            if tuple(rgb_tensor.shape[:2]) != tuple(alpha_tensor.shape):
                raise RuntimeError("Kanomemo Save RGBA image/mask dimensions differ")
            rgb = rgb_tensor[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
            alpha = alpha_tensor.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
            rgba = np.concatenate(
                (
                    (rgb.numpy() * 255.0).round().astype(np.uint8),
                    (alpha.numpy()[..., None] * 255.0).round().astype(np.uint8),
                ),
                axis=2,
            )
            file_name = f"{filename.replace('%batch_num%', str(index))}_{counter:05}_.png"
            Image.fromarray(rgba, "RGBA").save(Path(output_dir) / file_name, compress_level=4)
            results.append(ui.SavedResult(file_name, subfolder, io.FolderType.output))
            counter += 1
        return io.NodeOutput(image, ui=ui.SavedImages(results))
