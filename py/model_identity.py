"""Read-only identities of registered local model files; never loads tensors."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import threading
from collections import OrderedDict
from pathlib import Path

from aiohttp import web
import folder_paths

_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_LOCK = threading.Lock()


class ModelChangedError(RuntimeError):
    pass


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_identity(path: Path, *, allow_empty: bool = False) -> dict:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode) or (before.st_size <= 0 and not allow_empty):
        raise ValueError("Model must be a non-empty regular file")
    key = (str(resolved), *_stamp(before))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            result = dict(_CACHE[key])
        else:
            with resolved.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                if _stamp(opened) != _stamp(before):
                    raise ModelChangedError("Model changed before fingerprinting; retry")
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if _stamp(os.fstat(handle.fileno())) != _stamp(before):
                    raise ModelChangedError("Model changed during fingerprinting; retry")
            result = {"sha256": digest, "size": before.st_size}
            _CACHE[key] = result
            if len(_CACHE) > 256:
                _CACHE.popitem(last=False)
        if path.resolve(strict=True) != resolved or _stamp(resolved.stat()) != _stamp(before):
            _CACHE.pop(key, None)
            raise ModelChangedError("Model changed during fingerprinting; retry")
        return dict(result)


def model_path_identity(path: str | Path) -> str:
    """Fingerprint a worker-selected weight file or model directory."""
    root = Path(path)
    if root.is_file():
        return _file_identity(root)["sha256"]
    if not root.is_dir():
        raise FileNotFoundError(f"Worker model is unavailable: {root}")
    suffixes = {".safetensors", ".bin", ".pt", ".pth", ".json", ".model", ".txt", ".tiktoken"}

    def snapshot():
        files = sorted(file for file in root.rglob("*")
                       if file.is_file() and file.suffix.lower() in suffixes
                       and not any(part.startswith(".") for part in file.relative_to(root).parts))
        records = tuple((file.relative_to(root).as_posix(), str(file.resolve(strict=True)), *_stamp(file.stat()))
                        for file in files)
        return files, records

    files, before = snapshot()
    if not files:
        raise FileNotFoundError(f"Worker model directory has no weights/configuration: {root}")
    records = [{"name": file.relative_to(root).as_posix(),
                **_file_identity(file, allow_empty=file.suffix.lower() == ".txt")} for file in files]
    if snapshot()[1] != before:
        raise ModelChangedError("Worker model directory changed during fingerprinting; retry")
    return _digest(records)


def _resolve_model(category: str, name: str) -> Path:
    if (not isinstance(category, str) or category not in folder_paths.folder_names_and_paths
            or not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Use a registered model category and relative model filename")
    if name not in folder_paths.get_filename_list(category):
        raise FileNotFoundError(f"Registered model is unavailable: {category}/{name}")
    path = folder_paths.get_full_path(category, name)
    if path is None:
        raise FileNotFoundError(f"Registered model is unavailable: {category}/{name}")
    return Path(path)


def _codec_repo(checkpoint: Path) -> str:
    # Match Irodori's config_json latent_dim routing, without importing torch.
    if checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("Irodori codec identity requires a safetensors checkpoint")
    with checkpoint.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid safetensors header")
        length = int.from_bytes(prefix, "little")
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors metadata length")
        header = json.loads(source.read(length))
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        raise ValueError("Invalid safetensors metadata object")
    config = json.loads(header["__metadata__"].get("config_json", "null"))
    latent_dim = config.get("latent_dim") if isinstance(config, dict) else None
    repositories = {32: "Aratako/Semantic-DACVAE-Japanese-32dim", 128: "facebook/dacvae-watermarked"}
    if latent_dim not in repositories:
        raise ValueError("Irodori checkpoint has unsupported or missing latent_dim metadata")
    return repositories[latent_dim]


def _cached_codec(repo: str) -> Path:
    # DACVAECodec.load without local_dir resolves this exact default file.
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(repo, "weights.pth")
    if not isinstance(path, str) or not Path(path).is_file():
        raise FileNotFoundError(f"Irodori codec is not cached locally: {repo}/weights.pth")
    return Path(path)


def irodori_runtime_identity(checkpoint: str | Path, codec_repo: str, *, allow_missing_codec: bool = False) -> tuple[str, str | None]:
    """Identity used by loaded Irodori runtimes and graph cache fingerprints."""
    checkpoint_digest = _file_identity(Path(checkpoint))["sha256"]
    try:
        codec_digest = _file_identity(_cached_codec(codec_repo))["sha256"]
    except FileNotFoundError:
        if not allow_missing_codec:
            raise
        codec_digest = None
    return checkpoint_digest, codec_digest


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_identity(payload: object) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"models", "include_irodori_codec"}:
        raise ValueError("Expected models and optional include_irodori_codec")
    models = payload.get("models")
    include_codec = payload.get("include_irodori_codec", False)
    if not isinstance(models, list) or not 1 <= len(models) <= 32 or not isinstance(include_codec, bool):
        raise ValueError("models must contain 1–32 entries; include_irodori_codec must be boolean")
    records = []
    checkpoints = []
    for item in models:
        if not isinstance(item, dict) or set(item) != {"category", "name"}:
            raise ValueError("Each model must specify category and name")
        path = _resolve_model(item["category"], item["name"])
        records.append({**item, **_file_identity(path)})
        if item["category"] == "checkpoints":
            checkpoints.append((path, records[-1]))
    result = {"version": 1, "models": sorted(records, key=lambda record: (record["category"], record["name"]))}
    if include_codec:
        if len(checkpoints) != 1:
            raise ValueError("Irodori codec identity requires exactly one checkpoint")
        checkpoint, record = checkpoints[0]
        repo = _codec_repo(checkpoint)
        if _file_identity(checkpoint) != {key: record[key] for key in ("sha256", "size")}:
            raise ModelChangedError("Irodori checkpoint changed while resolving codec; retry")
        files = [{"name": "weights.pth", **_file_identity(_cached_codec(repo))}]
        result["irodori_codec"] = {"repo": repo, "files": files, "sha256": _digest({"repo": repo, "files": files})}
    result["identity"] = _digest(result)
    return result


async def model_identity_route(request):
    try:
        payload = await request.json()
        result = await asyncio.to_thread(model_identity, payload)
        return web.json_response(result)
    except ModelChangedError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError:
        return web.json_response({"error": "Local model file could not be read"}, status=503)
