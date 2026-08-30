from __future__ import annotations

from pathlib import Path

import folder_paths


def extension_data_dir() -> Path:
    root = Path(folder_paths.models_dir) / "irodori"
    root.mkdir(parents=True, exist_ok=True)
    return root


def image_encoder_cache_root() -> Path:
    root = extension_data_dir() / "image_encoders"
    root.mkdir(parents=True, exist_ok=True)
    return root


def timm_model_cache_dir() -> Path:
    cache_dir = image_encoder_cache_root()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
