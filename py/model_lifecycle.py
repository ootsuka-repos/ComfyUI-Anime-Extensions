"""Release extension-owned models when a ComfyUI prompt finishes."""
from __future__ import annotations

import os
import sys
import logging

from comfy_api.latest import Caching
from server import PromptServer


_PACKAGE = __package__
_LOG = logging.getLogger(__name__)
_IMGUTILS_CACHES = {
    "imgutils.generic.yolo": ("_open_models_for_repo_id",),
    "imgutils.detect.nudenet": ("_open_nudenet_yolo",),
    "imgutils.tagging.wd14": ("_get_wd14_model", "_get_wd14_weights"),
    "imgutils.ocr.detect": ("_open_ocr_detection_model",),
    "imgutils.ocr.recognize": ("_open_ocr_recognition_model",),
    "imgutils.segment.isnetis": ("_get_model",),
}


def _clear_global(module, attribute, lock) -> None:
    with getattr(module, lock):
        setattr(module, attribute, None)


def clear_extension_models() -> None:
    # Only inspect already imported backends; cleanup must not load optional models.
    cleanups = []
    for name, attribute, lock in (
        ("nodes.kanomemo.analysis", "_MOBILE_SAM", "_MOBILE_SAM_LOCK"),
        ("nodes.kanomemo.heatmap", "_MODEL", "_MODEL_LOCK"),
    ):
        module = sys.modules.get(f"{_PACKAGE}.{name}")
        if module is not None:
            cleanups.append((name, _clear_global, (module, attribute, lock)))
    for name in ("irodori_tts", "irodori_character_voice"):
        module = sys.modules.get(f"{_PACKAGE}.modules.{name}.inference_runtime")
        if module is not None:
            cleanups.append((name, module.clear_cached_runtime, ()))
    for name, functions in _IMGUTILS_CACHES.items():
        module = sys.modules.get(name)
        if module is not None:
            for function in functions:
                cleanups.append((f"{name}.{function}", getattr(module, function).cache_clear, ()))
    for name, cleanup, arguments in cleanups:
        try:
            cleanup(*arguments)
        except Exception:
            # One backend must not prevent the remaining backends from releasing.
            _LOG.exception("Failed to unload %s", name)


class ModelLifecycle(Caching.CacheProvider):
    """Own model-cache cleanup without retaining external node outputs."""

    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        pass

    def should_cache(self, context, value=None):
        return False

    def on_prompt_end(self, prompt_id: str) -> None:
        if os.environ.get("COMFYUI_FORGE_AUTO_UNLOAD", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        try:
            clear_extension_models()
        finally:
            # The worker consumes this after saving history, outside execution.
            # It unloads managed models, resets node caches and runs GC.
            PromptServer.instance.prompt_queue.set_flag("free_memory", True)
            _LOG.info("Forge model cleanup finished; ComfyUI cache release queued")


model_lifecycle = ModelLifecycle()
