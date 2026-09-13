"""Shared WD14-ViT heatmap safety nodes used by the product repositories."""
from __future__ import annotations

import json
import threading
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from comfy_api.latest import io, ui

from ...node_utils import mk_name
from .portrait import _configure_huggingface_cache


MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
CENSOR_TAGS: tuple[str, ...] = (
    "pussy", "spread_pussy", "clitoris", "penis", "anus", "vaginal", "anal",
    "sex", "pussy_juice", "erection", "testicles",
)
PROB_THR = 0.30
HEAT_THR = 0.45
HEAT_THR_TAG = {
    "penis": 0.28,
    "erection": 0.28,
    "pussy": 0.35,
    "spread_pussy": 0.35,
    "clitoris": 0.35,
}
STROKE_FAMILIES: dict[str, tuple[str, ...]] = {
    "crotch": ("pussy", "spread_pussy", "clitoris", "pussy_juice", "anus"),
    "penis": ("penis", "erection", "testicles"),
}
_GENITAL_LABELS = {"pussy", "penis"}
_MODEL_LOCK = threading.RLock()
_MODEL: tuple | None = None


def _to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise RuntimeError("heatmap nodes expect exactly one RGB IMAGE")
    rgb = image[0, ..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    return Image.fromarray((rgb.contiguous().numpy() * 255.0).round().astype(np.uint8), "RGB")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_model() -> tuple:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        _configure_huggingface_cache()
        try:
            import pandas as pd
            import timm
            from huggingface_hub import hf_hub_download
            from timm.data import create_transform, resolve_data_config
        except ImportError as exc:
            raise RuntimeError(
                "heatmap nodes require pandas, timm, and huggingface_hub"
            ) from exc
        # ComfyUI invokes a node inside ``torch.inference_mode()``.  Models and
        # tensors constructed in that context become inference tensors, which
        # cannot participate in the Grad-CAM pass below.  Create the cached
        # model outside it even on the first node invocation.
        with torch.inference_mode(False):
            device = _device()
            model = timm.create_model(f"hf-hub:{MODEL_REPO}", pretrained=True)
            model.eval().to(device)
            csv_path = hf_hub_download(repo_id=MODEL_REPO, filename="selected_tags.csv")
            labels = tuple(str(value) for value in pd.read_csv(csv_path)["name"])
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        _MODEL = model, labels, transform, device
        return _MODEL


def _score_tensor(image: Image.Image) -> tuple[torch.Tensor, tuple[str, ...]]:
    model, labels, transform, device = _load_model()
    x = transform(image.convert("RGB")).unsqueeze(0)[:, [2, 1, 0]].to(device)
    with torch.inference_mode():
        scores = torch.sigmoid(model.forward_head(model.forward_features(x))).squeeze(0)
    return scores.detach().to(device="cpu"), labels


def vit_scores(image: Image.Image, wanted: Iterable[str] | None = None) -> dict[str, float]:
    with _MODEL_LOCK:
        scores, labels = _score_tensor(image)
    selected = None if wanted is None else {str(tag).strip() for tag in wanted if str(tag).strip()}
    return {
        name: float(scores[index])
        for index, name in enumerate(labels)
        if selected is None or name in selected
    }


def heatmap_tag_masks(
    image: Image.Image,
    *,
    tags: tuple[str, ...] = CENSOR_TAGS,
    prob_thr: float = PROB_THR,
    heat_thr: float = HEAT_THR,
) -> dict[str, Image.Image]:
    """Produce the same mirrored Grad-CAM tag masks as the former local backend."""

    with _MODEL_LOCK:
        model, labels, transform, device = _load_model()

        def tag_heatmaps(source: Image.Image):
            # ``enable_grad`` alone does not undo ComfyUI's surrounding
            # inference-mode context.  The complete forward/gradient pass must
            # run with ordinary tensors or ``retain_grad`` raises
            # "can't retain_grad on Tensor that has requires_grad=False".
            with torch.inference_mode(False), torch.enable_grad():
                x = transform(source.convert("RGB")).unsqueeze(0)[:, [2, 1, 0]].to(device)
                x.requires_grad_(False)
                features = model.forward_features(x)
                if not features.requires_grad:
                    # Some compatible classifiers expose feature maps detached
                    # from their parameter graph.  Grad-CAM only differentiates
                    # the head with respect to these maps, so make that explicit
                    # rather than silently emitting an empty censor mask.
                    features = features.detach().requires_grad_(True)
                features.retain_grad()
                probabilities = torch.sigmoid(model.forward_head(features)).squeeze(0)
                probabilities_cpu = probabilities.detach().to(device="cpu")
                picked = [
                    (index, name, float(probabilities_cpu[index]))
                    for index, name in enumerate(labels)
                    if name in tags and float(probabilities_cpu[index]) >= prob_thr
                ]
                if not picked:
                    return [], None
                indexes = torch.tensor([index for index, _name, _score in picked], device=device)
                gradients = torch.autograd.grad(
                    outputs=probabilities[indexes],
                    inputs=features,
                    grad_outputs=torch.eye(len(picked), device=device),
                    is_grads_batched=True,
                )[0]
                values = gradients.detach().mean(2, keepdim=True).mul(features.detach().unsqueeze(0)).mean(-1)
                token_count = values.shape[-1]
                side = int(token_count ** 0.5)
                if side * side != token_count:
                    values = values[..., -side * side:]
                values = torch.clamp(values.reshape(len(picked), side, side), min=0)
                maxima = values.reshape(len(picked), -1).max(-1)[0].clamp(min=1e-6)
                return picked, values / maxima[:, None, None]

        source = image.convert("RGB")
        picked, heatmaps = tag_heatmaps(source)
        if not picked or heatmaps is None:
            return {}
        mirrored, mirrored_heatmaps = tag_heatmaps(source.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
        if mirrored and mirrored_heatmaps is not None:
            by_name = {name: index for index, (_model_index, name, _score) in enumerate(mirrored)}
            for index, (_model_index, name, _score) in enumerate(picked):
                mirrored_index = by_name.get(name)
                if mirrored_index is not None:
                    heatmaps[index] = torch.maximum(heatmaps[index], mirrored_heatmaps[mirrored_index].flip(-1))
        big = torch.nn.functional.interpolate(
            heatmaps[:, None], size=(source.height, source.width), mode="bilinear", align_corners=False
        )[:, 0]
        output: dict[str, Image.Image] = {}
        for index, (_model_index, name, _score) in enumerate(picked):
            threshold = HEAT_THR_TAG.get(name, heat_thr)
            mask = (big[index] >= threshold).to(dtype=torch.uint8).to(device="cpu").numpy() * 255
            output[name] = Image.fromarray(mask, mode="L")
        return output


def _draw_capsules(
    draw: ImageDraw.ImageDraw,
    mask: Image.Image,
    *,
    min_frac: float,
    max_a: float,
    max_b: float,
    scale: float = 1.1,
    per_component: bool = True,
) -> int:
    try:
        import cv2
    except ImportError:
        return 0
    array = (np.asarray(mask) > 127).astype("uint8")
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(array, connectivity=8)
    min_area = int(array.size * (min_frac if per_component else 0.0002))
    keep = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return 0
    groups = [(labels == index) for index in keep] if per_component else [np.isin(labels, keep)]
    drawn = 0
    for group in groups:
        ys, xs = np.nonzero(group)
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        center = points.mean(0)
        covariance = np.cov((points - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, 1]
        a = min(max_a, max(12.0, 2.0 * float(np.sqrt(max(eigenvalues[1], 1.0))) * scale))
        b = min(max_b, max(12.0, 2.0 * float(np.sqrt(max(eigenvalues[0], 1.0))) * scale))
        b = min(b, a)
        p0 = (center[0] - major[0] * (a - b), center[1] - major[1] * (a - b))
        p1 = (center[0] + major[0] * (a - b), center[1] + major[1] * (a - b))
        draw.line([p0, p1], fill=255, width=int(2 * b))
        for px, py in (p0, p1):
            draw.ellipse([px - b, py - b, px + b, py + b], fill=255)
        drawn += 1
    return drawn


def _brush_mask_from_tags(tag_masks: dict[str, Image.Image], size: tuple[int, int]) -> Image.Image:
    short = float(min(size))
    output = Image.new("L", size, 0)
    draw = ImageDraw.Draw(output)
    drawn = 0
    for family, members in STROKE_FAMILIES.items():
        masks = [tag_masks[tag] for tag in members if tag in tag_masks]
        if not masks:
            continue
        if len(masks) >= 2:
            count = np.zeros((masks[0].height, masks[0].width), dtype=np.uint8)
            for mask in masks:
                count += (np.asarray(mask) > 127).astype(np.uint8)
            core = Image.fromarray(np.where(count >= 2, 255, 0).astype(np.uint8), mode="L")
            target = core if core.getbbox() else None
        else:
            target = None
        if target is None:
            target = masks[0]
            for mask in masks[1:]:
                target = Image.composite(mask, target, mask)
        drawn += _draw_capsules(
            draw,
            target,
            min_frac=0.002,
            scale=1.2,
            max_a=short * 0.35,
            max_b=short * 0.14,
            per_component=(family != "crotch"),
        )
    if "anus" in tag_masks:
        drawn += _draw_capsules(
            draw,
            tag_masks["anus"],
            min_frac=0.0008,
            scale=1.1,
            max_a=short * 0.12,
            max_b=short * 0.08,
        )
    if drawn == 0:
        for mask in tag_masks.values():
            drawn += _draw_capsules(
                draw, mask, min_frac=0.004, max_a=short * 0.25, max_b=short * 0.12
            )
    return output


def _anchor_boxes(image: Image.Image) -> list[tuple[tuple[int, int, int, int], str]]:
    """Use the shared imgutils genital detectors as conservative heatmap anchors."""

    from .analysis import _analyse

    found: list[tuple[tuple[int, int, int, int], str, float]] = []
    for level in ("s", "n"):
        try:
            result = _analyse(image, operation="censor", level=level, version="v1.0", threshold=0.30)
            found.extend(
                (tuple(int(value) for value in item["box"]), str(item["label"]).lower(), float(item["score"]))
                for item in result.get("items", [])
                if isinstance(item, dict) and isinstance(item.get("box"), list) and len(item["box"]) == 4
            )
        except Exception:
            continue
    try:
        result = _analyse(image, operation="nudenet", level="", version="", threshold=0.25)
        for item in result.get("items", []):
            if not isinstance(item, dict) or not isinstance(item.get("box"), list) or len(item["box"]) != 4:
                continue
            label = str(item.get("label", "")).lower()
            if "genitalia" not in label and "anus" not in label:
                continue
            normalized = "pussy" if "female" in label or "anus" in label else "penis"
            found.append((tuple(int(value) for value in item["box"]), normalized, float(item.get("score", 0.0))))
    except Exception:
        pass
    output: list[tuple[tuple[int, int, int, int], str]] = []
    for box, label, _score in found:
        if label not in _GENITAL_LABELS or box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(label == previous_label and _iou(box, previous_box) > 0.65 for previous_box, previous_label in output):
            continue
        output.append((box, label))
    return output


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if not intersection:
        return 0.0
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / max(1, union)


def _add_anchor_capsules(
    mask: Image.Image,
    anchors: list[tuple[tuple[int, int, int, int], str]],
    *,
    expand: float = 0.08,
) -> Image.Image:
    output = mask.copy()
    draw = ImageDraw.Draw(output)
    for (x0, y0, x1, y1), _label in anchors:
        ex, ey = int((x1 - x0) * expand), int((y1 - y0) * expand)
        left, top, right, bottom = x0 - ex, y0 - ey, x1 + ex, y1 + ey
        radius = max(8, min(right - left, bottom - top) // 3)
        draw.rounded_rectangle([left, top, right, bottom], radius=radius, fill=255)
    return output


def _keep_components_touching(mask: Image.Image, anchor: Image.Image) -> Image.Image:
    try:
        import cv2
    except ImportError:
        return mask
    array = (np.asarray(mask) > 127).astype("uint8")
    anchors = np.asarray(anchor) > 127
    count, labels = cv2.connectedComponents(array, connectivity=8)
    keep = np.zeros_like(array)
    for index in range(1, count):
        component = labels == index
        if (component & anchors).any():
            keep[component] = 255
    return Image.fromarray(keep, mode="L")


def _blur_per_component(image: Image.Image, mask: Image.Image, strength: float) -> Image.Image:
    try:
        import cv2
    except ImportError:
        radius = max(4, int(min(image.size) * 0.012 * strength))
        return Image.composite(image.filter(ImageFilter.GaussianBlur(radius)), image, mask)
    array = (np.asarray(mask) > 127).astype("uint8")
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(array, connectivity=8)
    output = image
    for index in range(1, count):
        x, y, width, height = (int(stats[index, column]) for column in (
            cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT
        ))
        radius = min(24, max(8, int(min(width, height) * 0.06 * strength)))
        pad = radius * 2
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(image.width, x + width + pad), min(image.height, y + height + pad)
        blurred = output.crop((left, top, right, bottom)).filter(ImageFilter.GaussianBlur(radius))
        component = Image.fromarray(np.where(labels == index, 255, 0).astype("uint8"), mode="L").crop(
            (left, top, right, bottom)
        )
        base = output.crop((left, top, right, bottom))
        output = output.copy() if output is image else output
        output.paste(Image.composite(blurred, base, component), (left, top))
    return output


def _refine_with_zoom(image: Image.Image, tag_masks: dict[str, Image.Image], margin: float = 0.35) -> None:
    for members in STROKE_FAMILIES.values():
        masks = [tag_masks[tag] for tag in members if tag in tag_masks]
        if not masks:
            continue
        union = masks[0]
        for mask in masks[1:]:
            union = Image.composite(mask, union, mask)
        box = union.getbbox()
        if box is None:
            continue
        x0, y0, x1, y1 = box
        mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
        left, top = max(0, x0 - mx), max(0, y0 - my)
        right, bottom = min(image.width, x1 + mx), min(image.height, y1 + my)
        if right - left < 64 or bottom - top < 64:
            continue
        for tag, sub_mask in heatmap_tag_masks(image.crop((left, top, right, bottom)), tags=members).items():
            full = Image.new("L", image.size, 0)
            full.paste(sub_mask, (left, top))
            base = tag_masks.get(tag)
            tag_masks[tag] = full if base is None else Image.composite(full, base, full)


def _tile_scan(image: Image.Image, tag_masks: dict[str, Image.Image], prob_thr: float = 0.45) -> list[Image.Image]:
    rescued: list[Image.Image] = []
    width, height = image.size
    tiles = (
        (0, 0, int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), 0, width, int(height * 0.6)),
        (0, int(height * 0.4), int(width * 0.6), height),
        (int(width * 0.4), int(height * 0.4), width, height),
    )
    for left, top, right, bottom in tiles:
        crop = image.crop((left, top, right, bottom))
        scores = vit_scores(crop, CENSOR_TAGS)
        needed = [
            tag for tag, score in scores.items()
            if score >= prob_thr and (tag not in tag_masks or tag_masks[tag].crop((left, top, right, bottom)).getbbox() is None)
        ]
        if not needed:
            continue
        for tag, sub_mask in heatmap_tag_masks(crop, tags=tuple(needed)).items():
            full = Image.new("L", image.size, 0)
            full.paste(sub_mask, (left, top))
            base = tag_masks.get(tag)
            tag_masks[tag] = full if base is None else Image.composite(full, base, full)
            rescued.append(full)
    return rescued


def _pixelate(image: Image.Image, mask: Image.Image) -> Image.Image:
    block = max(12, min(image.size) // 80)
    small = image.resize((max(1, image.width // block), max(1, image.height // block)), Image.Resampling.NEAREST)
    return Image.composite(small.resize(image.size, Image.Resampling.NEAREST), image, mask)


def heatmap_censor(image: Image.Image, *, mode: str, blur_strength: float, tags: tuple[str, ...]) -> tuple[Image.Image, list[str]]:
    with _MODEL_LOCK:
        tag_masks = heatmap_tag_masks(image, tags=tags)
        _refine_with_zoom(image, tag_masks)
        rescued = _tile_scan(image, tag_masks)
        anchors = _anchor_boxes(image)
        labels = [*tag_masks, *(f"det:{label}" for _box, label in anchors)]
        if not tag_masks and not anchors:
            return image.convert("RGB"), labels
        mask = _brush_mask_from_tags(tag_masks, image.size)
        if anchors:
            anchor_mask = _add_anchor_capsules(Image.new("L", image.size, 0), anchors)
            touch_base = anchor_mask
            for rescued_mask in rescued:
                touch_base = Image.composite(rescued_mask, touch_base, rescued_mask)
            mask = _keep_components_touching(mask, touch_base)
            mask = Image.composite(anchor_mask, mask, anchor_mask)
        if mask.getextrema()[1] <= 0:
            return image.convert("RGB"), labels
        if str(mode).lower() == "blur":
            return _blur_per_component(image.convert("RGB"), mask, float(blur_strength)), labels
        return _pixelate(image.convert("RGB"), mask), labels


def _parse_tags(serialized: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(serialized).split(",") if item.strip())
    return values or CENSOR_TAGS


class WD14ViTScores(io.ComfyNode):
    """Return calibrated WD14-ViT scores from the shared ComfyUI process."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name("Image", "WD14ViTScores"),
            display_name="WD14 ViT Scores",
            category="ComfyUIExtensions/Image/analysis",
            inputs=[io.Image.Input("image"), io.String.Input("tags", default="")],
            outputs=[io.String.Output(display_name="json")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, tags):
        wanted = _parse_tags(tags) if str(tags).strip() else None
        result = {"scores": vit_scores(_to_pil(image), wanted)}
        serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        return io.NodeOutput(serialized, ui=ui.PreviewText(serialized))


class HeatmapCensor(io.ComfyNode):
    """Apply the former local WD14-ViT heatmap censor in the shared ComfyUI process."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name("Image", "HeatmapCensor"),
            display_name="Heatmap Censor (WD14 ViT)",
            category="ComfyUIExtensions/Image/analysis",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=["blur", "pixelate"], default="blur"),
                io.Float.Input("blur_strength", default=0.5, min=0.1, max=5.0, step=0.1),
                io.String.Input("tags", default=",".join(CENSOR_TAGS)),
            ],
            outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="json")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, mode, blur_strength, tags):
        output, labels = heatmap_censor(
            _to_pil(image), mode=str(mode), blur_strength=float(blur_strength), tags=_parse_tags(tags)
        )
        serialized = json.dumps({"labels": labels}, ensure_ascii=False, separators=(",", ":"))
        return io.NodeOutput(_to_tensor(output), serialized, ui=ui.PreviewText(serialized))
