import json
from pathlib import Path

import folder_paths
import torch
from comfy_api.latest import io
from safetensors import safe_open

from ...modules.irodori_tts.inference_runtime import (
    list_available_runtime_devices,
    list_available_runtime_precisions,
)


IO_MODEL_CONFIG = io.Custom("IRODORI_MODEL_CONFIG")
IO_LORA_STACK = io.Custom("IRODORI_LORA_STACK")
IO_REF_CONFIG = io.Custom("IRODORI_REF_CONFIG")
IO_VOICE_DESIGN_CONFIG = io.Custom("IRODORI_VOICE_DESIGN_CONFIG")
IO_CFG_CONFIG = io.Custom("IRODORI_CFG_CONFIG")
IO_RESCALE_CONFIG = io.Custom("IRODORI_RESCALE_CONFIG")
IO_SCHEDULE_CONFIG = io.Custom("IRODORI_SCHEDULE_CONFIG")
IO_TRIM_TAIL_CONFIG = io.Custom("IRODORI_TRIM_TAIL_CONFIG")


def available_devices() -> list[str]:
    return list_available_runtime_devices()


def available_precisions(device: str = "cuda") -> list[str]:
    try:
        return list_available_runtime_precisions(device)
    except Exception:
        return ["fp32", "bf16"]


def none_if_non_positive(value):
    return None if value is None or value <= 0 else value


def resolve_checkpoint_path(model_name: str) -> str:
    checkpoint_path = folder_paths.get_full_path("checkpoints", model_name)
    return checkpoint_path if checkpoint_path else model_name


def peek_latent_dim_from_checkpoint(checkpoint_path: str) -> int:
    path = Path(checkpoint_path)
    if path.suffix.lower() == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        raw_config = metadata.get("config_json")
        if raw_config is None:
            raise ValueError(f"checkpoint metadata missing config_json: {path}")
        config = json.loads(raw_config)
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = payload.get("model_config") if isinstance(payload, dict) else None

    if not isinstance(config, dict) or "latent_dim" not in config:
        raise ValueError(f"checkpoint model_config missing latent_dim: {path}")
    return int(config["latent_dim"])


SIDECAR_CODEC_RELPATH = "codec/weights.pth"


def codec_repo_for_latent_dim(latent_dim: int) -> str:
    if int(latent_dim) == 32:
        return "Aratako/Semantic-DACVAE-Japanese-32dim"
    if int(latent_dim) == 128:
        return "facebook/dacvae-watermarked"
    raise ValueError(f"Unsupported checkpoint latent_dim={latent_dim}.")


def sidecar_codec_path(checkpoint_path: str | Path) -> Path | None:
    """Codec shipped next to the checkpoint, if the pair was fine-tuned together."""

    candidate = Path(checkpoint_path).parent / SIDECAR_CODEC_RELPATH
    return candidate if candidate.is_file() else None


def codec_source_for_checkpoint(checkpoint_path: str | Path, latent_dim: int) -> str:
    """Codec DACVAECodec.load should read for this checkpoint.

    A fine-tuned codec is not interchangeable with the public one even at the
    same latent_dim, so a sidecar next to the checkpoint wins over the
    latent_dim routing.
    """

    sidecar = sidecar_codec_path(checkpoint_path)
    if sidecar is not None:
        return str(sidecar)
    return codec_repo_for_latent_dim(int(latent_dim))


def resolve_lora_path(lora_name: str) -> str | None:
    if not lora_name or lora_name == "None":
        return None
    return folder_paths.get_full_path_or_raise("loras", lora_name)
