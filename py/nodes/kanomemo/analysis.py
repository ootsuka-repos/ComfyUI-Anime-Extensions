"""Shared imgutils analysis nodes for every Doujin Forge product."""
from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io, ui

from ...node_utils import mk_name
from .portrait import _configure_huggingface_cache


_ANALYSIS_LOCK = threading.RLock()
_MOBILE_SAM_LOCK = threading.RLock()
_MOBILE_SAM: Any | None = None
_OPERATIONS = ("face", "head", "censor", "nudenet", "wd14", "ocr")


def _to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise RuntimeError("Kanomemo imgutils analysis expects exactly one RGB IMAGE")
    rgb = image[0, ..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    pixels = (rgb.contiguous().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _items(detections: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for box, label, score in detections:
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        output.append({
            "box": [int(round(float(value))) for value in box],
            "label": str(label),
            "score": float(score),
        })
    return output


def _analyse(
    image: Image.Image,
    *,
    operation: str,
    level: str,
    version: str,
    threshold: float,
) -> dict[str, Any]:
    _configure_huggingface_cache()
    selected = str(operation).strip().lower()
    if selected not in _OPERATIONS:
        raise ValueError(f"unsupported imgutils analysis operation: {operation!r}")
    score = min(1.0, max(0.0, float(threshold)))
    with _ANALYSIS_LOCK:
        if selected == "face":
            from imgutils.detect.face import detect_faces

            return {"operation": selected, "items": _items(detect_faces(
                image,
                level=(str(level).strip() or "s"),
                version=(str(version).strip() or "v1.4"),
                conf_threshold=score,
            ))}
        if selected == "head":
            from imgutils.detect.head import detect_heads

            selected_level = str(level).strip() or None
            return {"operation": selected, "items": _items(detect_heads(
                image, level=selected_level, conf_threshold=score,
            ))}
        if selected == "censor":
            from imgutils.detect.censor import detect_censors

            return {"operation": selected, "items": _items(detect_censors(
                image,
                level=(str(level).strip() or "s"),
                version=(str(version).strip() or "v1.0"),
                conf_threshold=score,
            ))}
        if selected == "nudenet":
            from imgutils.detect.nudenet import detect_with_nudenet

            return {"operation": selected, "items": _items(
                detect_with_nudenet(image, score_threshold=score)
            )}
        if selected == "wd14":
            from imgutils.tagging import get_wd14_tags

            rating, features, characters = get_wd14_tags(
                image, general_threshold=score, character_threshold=score,
            )
            return {
                "operation": selected,
                "rating": {str(key): float(value) for key, value in rating.items()},
                "features": {str(key): float(value) for key, value in features.items()},
                "characters": {str(key): float(value) for key, value in characters.items()},
            }
        from imgutils.ocr import detect_text_with_ocr

        return {
            "operation": selected,
            "items": _items(detect_text_with_ocr(image, box_threshold=score)),
        }


class KanomemoImgutilsAnalysis(io.ComfyNode):
    """Run non-generative imgutils checks in the shared ComfyUI process."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name("Kanomemo", "AnalyzeImgutils"),
            display_name="Kanomemo Image Analysis (imgutils)",
            category="Kanomemo/analysis",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("operation", options=list(_OPERATIONS), default="face"),
                io.String.Input("level", default="s"),
                io.String.Input("version", default=""),
                io.Float.Input("threshold", default=0.25, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.String.Output(display_name="json")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, operation, level, version, threshold):
        result = _analyse(
            _to_pil(image),
            operation=operation,
            level=level,
            version=version,
            threshold=threshold,
        )
        serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        return io.NodeOutput(serialized, ui=ui.PreviewText(serialized))


def _mobile_sam_model():
    global _MOBILE_SAM
    with _MOBILE_SAM_LOCK:
        if _MOBILE_SAM is None:
            _configure_huggingface_cache()
            from huggingface_hub import hf_hub_download
            from ultralytics import SAM

            checkpoint = hf_hub_download(
                repo_id="dhkim2810/MobileSAM", filename="mobile_sam.pt"
            )
            # ComfyUI executes nodes inside torch.inference_mode().  CUDA
            # BatchNorm rejects model parameters created as inference tensors
            # (CPU happens to accept them), so construct the cached model with
            # ordinary tensors.  Ultralytics still applies inference mode while
            # running predict().
            with torch.inference_mode(False):
                _MOBILE_SAM = SAM(checkpoint)
        return _MOBILE_SAM


class KanomemoMobileSAMMask(io.ComfyNode):
    """Segment one detection box with shared MobileSAM."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name("Kanomemo", "MobileSAMMask"),
            display_name="Kanomemo Object Mask (MobileSAM)",
            category="Kanomemo/analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("x0", default=0, min=0),
                io.Int.Input("y0", default=0, min=0),
                io.Int.Input("x1", default=512, min=0),
                io.Int.Input("y1", default=512, min=0),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, image, x0, y0, x1, y1):
        source = _to_pil(image)
        left = min(source.width, max(0, int(x0)))
        top = min(source.height, max(0, int(y0)))
        right = min(source.width, max(0, int(x1)))
        bottom = min(source.height, max(0, int(y1)))
        if right <= left or bottom <= top:
            return io.NodeOutput(torch.zeros((1, source.height, source.width), dtype=torch.float32))
        with _MOBILE_SAM_LOCK:
            model = _mobile_sam_model()
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            prediction = model.predict(
                np.asarray(source), bboxes=[[left, top, right, bottom]],
                device=device, verbose=False,
            )[0]
        masks = getattr(prediction, "masks", None)
        if masks is None or len(masks.data) == 0:
            return io.NodeOutput(torch.zeros((1, source.height, source.width), dtype=torch.float32))
        raw = masks.data[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        mask = Image.fromarray((raw > 0.5).astype(np.uint8) * 255, "L")
        if mask.size != source.size:
            mask = mask.resize(source.size, Image.Resampling.NEAREST)
        return io.NodeOutput(
            torch.from_numpy(np.asarray(mask, dtype=np.float32)[None, ...] / 255.0)
        )
