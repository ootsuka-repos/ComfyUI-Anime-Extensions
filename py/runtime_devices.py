"""Irodori device choices shared by schemas and inference without loading models."""

from __future__ import annotations

import torch


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None or not hasattr(backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def _is_xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except AttributeError:
        return False


def resolve_runtime_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
        return resolved
    if resolved.type == "mps":
        if resolved.index is not None:
            raise ValueError("MPS device index is not supported. Use 'mps'.")
        if not _is_mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if resolved.type == "xpu":
        if resolved.index is not None:
            raise ValueError("XPU device index is not supported. Use 'xpu'.")
        if not _is_xpu_available():
            raise ValueError("XPU device requested but torch.xpu.is_available() is False.")
        return torch.device("xpu")
    raise ValueError(
        f"Unsupported inference device={resolved!s}. Expected one of: cpu, cuda, mps, xpu."
    )


def list_available_runtime_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if _is_mps_available():
        devices.append("mps")
    if _is_xpu_available():
        devices.append("xpu")
    devices.append("cpu")
    return devices


def list_available_runtime_precisions(device: str | torch.device) -> list[str]:
    resolved = resolve_runtime_device(device)
    if resolved.type in ("cuda", "xpu"):
        return ["fp32", "bf16"]
    return ["fp32"]
