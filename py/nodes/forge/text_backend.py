"""Client for the locally hosted OpenAI-compatible storyboard model."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_MODEL = os.environ.get("DOUJIN_FORGE_OPENAI_MODEL", "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF")

from .text_types import (
    AgentResultError,
    AgentTextResult,
    raise_if_refusal_text,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8888/v1"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_structured_generation(log_tag: str) -> bool:
    """Return whether a call expects a bounded machine-readable result."""

    # Comic scenario and storyboard calls return schema-validated JSON.  Keep
    # the prefix form because retries add a suffix to the storyboard tag.
    return log_tag.startswith(("comic_story", "comic_dialogue_")) or log_tag in {
        "art-prompt",
        "heroine_names",
        "heroine_names:review",
        "lettering",
        "promo",
        "prompt_plan",
        "sales_agent_fill",
        "thumb",
        "vlm.gate",
        "video.appearance",
    }


def _thinking_enabled(log_tag: str) -> bool:
    if log_tag.startswith("scenario:proofread:"):
        return _env_bool("DOUJIN_FORGE_PROOFREAD_ENABLE_THINKING", False)
    # Local reasoning models can consume the completion budget as hidden
    # reasoning and return empty message.content for structured JSON requests.
    # Structured product calls therefore require an explicit opt-in; narrative
    # prose can continue to use the global default.
    if _is_structured_generation(log_tag):
        return _env_bool("DOUJIN_FORGE_STRUCTURED_ENABLE_THINKING", False)
    if log_tag.startswith("scenario:"):
        return _env_bool("DOUJIN_FORGE_SCENARIO_ENABLE_THINKING", False)
    return _env_bool("DOUJIN_FORGE_OPENAI_ENABLE_THINKING", True)


def _response_format(log_tag: str) -> dict[str, Any] | None:
    """Constrain comic planning to the required outer JSON contract.

    ``json_object`` only guarantees that a response parses.  In live comic
    planning, the local model could return a valid nested character or layout
    reference object (for example ``{"id": ..., "version": ...}``) as the
    entire answer.  The local llama-server supports OpenAI ``json_schema``
    response formats, so require the scenario or storyboard *root* object
    here.  Product validators still own all dynamic constraints such as exact
    page IDs, seeds, layouts, and dialogue limits; this grammar merely makes a
    structurally unrelated root response impossible.

    Keep the constraint limited to calls whose ``log_tag`` begins with
    ``comic_story`` until every other product schema has equivalent live
    compatibility coverage.
    """

    if log_tag.startswith("comic_storyboard"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "comic_storyboard_chunk_root",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "comic_chunk"},
                        "pages": {"type": "array", "items": {}},
                    },
                    "required": ["kind", "pages"],
                    "additionalProperties": False,
                },
            },
        }
    if log_tag.startswith("comic_dialogue_review"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "comic_dialogue_review_root",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "comic_dialogue_review"},
                        "pages": {"type": "array", "items": {}},
                    },
                    "required": ["kind", "pages"],
                    "additionalProperties": False,
                },
            },
        }
    if log_tag.startswith("comic_story"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "comic_scenario_root",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "schema_version": {"type": "integer"},
                        "kind": {"type": "string", "const": "comic_scenario"},
                        "title": {"type": "string"},
                        "synopsis": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "character": {"type": "object"},
                        "outfit_prompt": {"type": "string"},
                        "partner_present": {"type": "boolean"},
                        "cover": {"type": "object"},
                        "page_count": {"type": "integer"},
                        "pages": {"type": "array", "items": {}},
                    },
                    "required": [
                        "schema_version",
                        "kind",
                        "title",
                        "synopsis",
                        "tags",
                        "character",
                        "outfit_prompt",
                        "partner_present",
                        "cover",
                        "page_count",
                        "pages",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    return None


def _max_tokens(log_tag: str) -> int | None:
    """Return the completion limit for an LLM request.

    Comic planning returns a complete, schema-validated document.  A fixed
    completion ceiling corrupts a valid outer JSON object into a completed
    nested object when the server cuts generation off mid-document.  Unsloth's
    OpenAI endpoint defines ``max_tokens: null`` as "generate until EOS", so
    that is the safe default for comic scenario and storyboard requests.  The
    server's context window remains the actual safety bound.

    ``DOUJIN_FORGE_COMIC_MAX_TOKENS`` can be set to a positive integer for a
    deliberate operator cap, or to ``none``/``null``/``unlimited`` to retain
    the default EOS behaviour.
    """

    if log_tag.startswith(("comic_story", "comic_dialogue_")):
        raw = os.getenv("DOUJIN_FORGE_COMIC_MAX_TOKENS")
        if raw is None or raw.strip().lower() in {"", "none", "null", "unlimited", "auto"}:
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "DOUJIN_FORGE_COMIC_MAX_TOKENS must be positive or 'unlimited'"
            ) from exc
        if value <= 0:
            raise ValueError("DOUJIN_FORGE_COMIC_MAX_TOKENS must be positive")
        return value

    name = (
        "DOUJIN_FORGE_OPENAI_MAX_TOKENS_TITLE"
        if "title" in log_tag
        else "DOUJIN_FORGE_OPENAI_MAX_TOKENS"
    )
    value = int(os.getenv(name, "4096"))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _selected_model(model: str | None = None) -> str:
    return model or os.getenv("DOUJIN_FORGE_OPENAI_MODEL") or DEFAULT_MODEL


def _local_lifecycle_url(base_url: str, action: str) -> str | None:
    """Return one local Unsloth lifecycle endpoint.

    The completion surface is OpenAI-compatible at ``/v1`` while model
    lifecycle operations live at ``/api/inference``.  Keeping this conversion
    in one place prevents callers from accidentally treating an arbitrary
    OpenAI-compatible endpoint as a locally managed Unsloth runtime.
    """

    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    if action not in {"load", "unload"}:
        raise ValueError(f"unsupported local model lifecycle action: {action}")
    # The local runtime exposes lifecycle endpoints beside its OpenAI ``/v1``
    # surface.  Do not attempt to manage an arbitrary OpenAI-compatible host.
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        return None
    return f"{normalized[:-3]}/api/inference/{action}"


def _model_path_and_variant(model: str) -> tuple[str, str | None]:
    """Split the optional ``repository:GGUF_VARIANT`` model selector."""

    model_path, separator, variant = model.partition(":")
    if not model_path:
        raise ValueError("DOUJIN_FORGE_OPENAI_MODEL must not be empty")
    return model_path, variant if separator and variant else None


def _model_record(
    body: object,
    model_path: str,
) -> dict[str, object] | None:
    """Find ``model_path`` in an OpenAI ``/models`` response."""

    if not isinstance(body, dict):
        raise ValueError("local OpenAI models response must be an object")
    models = body.get("data")
    if not isinstance(models, list):
        raise ValueError("local OpenAI models response has no data list")
    for item in models:
        if not isinstance(item, dict):
            continue
        item_path, _ = _model_path_and_variant(str(item.get("id", "")))
        if item_path == model_path:
            return item
    return None


def _auto_load_timeout_seconds() -> float:
    raw = os.getenv("DOUJIN_FORGE_OPENAI_AUTO_LOAD_TIMEOUT", "600")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(
            "DOUJIN_FORGE_OPENAI_AUTO_LOAD_TIMEOUT must be a positive number"
        ) from exc
    if timeout <= 0:
        raise ValueError(
            "DOUJIN_FORGE_OPENAI_AUTO_LOAD_TIMEOUT must be a positive number"
        )
    return timeout


def _is_no_model_loaded_response(response: httpx.Response) -> bool:
    """Whether a local completion request failed solely because no model runs."""

    return response.status_code == 400 and "no model loaded" in response.text.lower()


def _local_load_options() -> dict[str, object]:
    """Retain explicit local runtime settings when reloading after GPU handoff."""
    options: dict[str, object] = {}
    for suffix, field, minimum, maximum in (
        ("CONTEXT_LENGTH", "max_seq_length", 0, 1_048_576),
        ("PARALLEL", "n_parallel", 1, 64),
        ("UBATCH", "n_ubatch", 1, 65_536),
    ):
        name = "DOUJIN_FORGE_OPENAI_" + suffix
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        options[field] = value
    for suffix, field, choices in (
        ("SPECULATIVE_TYPE", "speculative_type", {"auto", "off"}),
        ("FLASH_ATTENTION", "flash_attention", {"auto", "on", "off"}),
    ):
        name = "DOUJIN_FORGE_OPENAI_" + suffix
        value = os.getenv(name)
        if value is None:
            continue
        if value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")
        if field == "flash_attention":
            options["llama_extra_args"] = ["--flash-attn", value]
        else:
            options[field] = value
    return options


async def _load_local_model_after_missing_model(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    selected_model: str,
    headers: dict[str, str],
    log_tag: str,
) -> bool:
    """Load the selected local model once, returning ``False`` if unsupported.

    This is deliberately a recovery path for the exact local ``No model
    loaded`` error, rather than a preflight on every request.  It keeps normal
    calls fast, never manages remote endpoints, and respects an operator
    opt-out through ``DOUJIN_FORGE_OPENAI_AUTO_LOAD=false``.
    """

    if not _env_bool("DOUJIN_FORGE_OPENAI_AUTO_LOAD", True):
        return False
    load_url = _local_lifecycle_url(base_url, "load")
    if load_url is None:
        return False

    model_path, configured_variant = _model_path_and_variant(selected_model)
    try:
        models_response = await client.get(f"{base_url}/models", headers=headers)
        if models_response.status_code == 404:
            return False
        models_response.raise_for_status()
        record = _model_record(models_response.json(), model_path)
        # Do not ask the local runtime to download or otherwise guess a model
        # that was not explicitly registered for this installation.
        if record is None:
            return False
        if bool(record.get("loaded")):
            return True

        load_payload: dict[str, object] = {"model_path": model_path, **_local_load_options()}
        variant = configured_variant or record.get("quant")
        if isinstance(variant, str) and variant.strip():
            load_payload["gguf_variant"] = variant.strip()
        load_response = await client.post(
            load_url,
            json=load_payload,
            headers=headers,
        )
        if load_response.status_code == 404:
            return False
        load_response.raise_for_status()

        deadline = time.monotonic() + _auto_load_timeout_seconds()
        while True:
            models_response = await client.get(f"{base_url}/models", headers=headers)
            models_response.raise_for_status()
            record = _model_record(models_response.json(), model_path)
            if record is not None and bool(record.get("loaded")):
                return True
            if time.monotonic() >= deadline:
                raise AgentResultError(
                    f"{log_tag}: local model did not become ready within "
                    f"{_auto_load_timeout_seconds():g} seconds"
                )
            await asyncio.sleep(1.0)
    except httpx.HTTPStatusError as exc:
        raise _http_error(f"{log_tag}: local model auto-load", exc.response) from exc
    except httpx.RequestError:
        # The original completion response is still more useful if a local
        # compatibility server does not implement Unsloth's lifecycle API.
        return False
    except (TypeError, ValueError) as exc:
        raise AgentResultError(
            f"{log_tag}: local model auto-load response was invalid"
        ) from exc


def release_local_model_for_gpu(*, model: str | None = None) -> bool:
    """Unload the resident local LLM before a ComfyUI GPU stage.

    A local storyboard model can legitimately occupy both GPUs. Leaving it
    resident makes a later ComfyUI render fail with a misleading OOM. This is
    deliberately limited to the local Unsloth endpoint; an externally supplied
    compatible endpoint is never lifecycle-managed by this product.

    Returns ``True`` only when a model was actually unloaded.  A missing local
    runtime is harmless because there is no resident model competing for VRAM.
    """

    if not _env_bool("DOUJIN_FORGE_OPENAI_RELEASE_BEFORE_COMFY", True):
        return False
    base_url = os.getenv("DOUJIN_FORGE_OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    unload_url = _local_lifecycle_url(base_url, "unload")
    if unload_url is None:
        return False

    # The completion model permits a ``:quant`` suffix, while Unsloth's
    # lifecycle API expects the repository identifier.
    model_path = _selected_model(model).partition(":")[0]
    headers = {
        "Authorization": "Bearer "
        + os.getenv("DOUJIN_FORGE_OPENAI_API_KEY", "not-needed"),
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            models_response = client.get(f"{base_url}/models", headers=headers)
            if models_response.status_code == 404:
                return False
            models_response.raise_for_status()
            record = _model_record(models_response.json(), model_path)
            if record is None or not bool(record.get("loaded")):
                return False
            response = client.post(
                unload_url,
                json={"model_path": model_path},
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _http_error("comic_gpu_handoff", exc.response) from exc
    except httpx.RequestError:
        # If the local endpoint is absent there cannot be a model using this
        # process's VRAM.  Treat connection loss as an already-released state.
        return False
    except (TypeError, ValueError) as exc:
        raise AgentResultError(
            "comic_gpu_handoff: local OpenAI lifecycle response was invalid"
        ) from exc
    return True


def _coerce_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item.get("text", "")
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _http_error(log_tag: str, response: httpx.Response) -> AgentResultError:
    detail = " ".join(response.text.split())[:500]
    message = f"{log_tag}: local OpenAI API HTTP {response.status_code}"
    if detail:
        message += f": {detail}"
    return AgentResultError(message)


async def run_openai_agent(
    user_prompt: str,
    system_prompt: str,
    *,
    log_tag: str,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    on_text: Callable[[str], None] | None = None,
    response_format: dict[str, Any] | None = None,
    image_data_urls: list[str] | None = None,
    temperature: float | None = None,
    repetition_penalty: float | None = None,
    min_p: float | None = None,
    **_ignored: object,
) -> AgentTextResult:
    """Submit one completion, recovering once from a locally unloaded model."""

    started = time.perf_counter()
    base_url = os.getenv("DOUJIN_FORGE_OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    selected_model = _selected_model(model)
    user_content: str | list[dict[str, Any]] = user_prompt
    if image_data_urls:
        user_content = [{"type": "text", "text": user_prompt}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
        ]
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "max_tokens": _max_tokens(log_tag),
        "reasoning_effort": os.getenv("DOUJIN_FORGE_OPENAI_REASONING_EFFORT", "low"),
        "temperature": float(os.getenv("DOUJIN_FORGE_OPENAI_TEMPERATURE", "0.7")) if temperature is None else temperature,
        "top_p": float(os.getenv("DOUJIN_FORGE_OPENAI_TOP_P", "0.8")),
        "top_k": int(os.getenv("DOUJIN_FORGE_OPENAI_TOP_K", "20")),
        "min_p": float(os.getenv("DOUJIN_FORGE_OPENAI_MIN_P", "0.05")) if min_p is None else min_p,
        "repetition_penalty": float(
            os.getenv("DOUJIN_FORGE_OPENAI_REPETITION_PENALTY", "1.1")
        ) if repetition_penalty is None else repetition_penalty,
        "enable_thinking": _thinking_enabled(log_tag),
        "enable_tools": _env_bool("DOUJIN_FORGE_OPENAI_ENABLE_TOOLS", True),
        "enabled_tools": _env_list(
            "DOUJIN_FORGE_OPENAI_ENABLED_TOOLS",
            ("web_search", "python", "terminal"),
        ),
    }
    # An explicit tool list in the old interface meant a controlled agent
    # session. The local server has no legacy agent-tool mapping, so keep such
    # calls text-only rather than exposing unrelated interactive tools.
    if (
        allowed_tools is not None
        or log_tag.startswith("scenario:")
        or _is_structured_generation(log_tag)
    ):
        payload["enable_tools"] = False
        payload["enabled_tools"] = []
    effective_response_format = response_format or _response_format(log_tag)
    if log_tag.startswith("scenario:proofread:"):
        payload["chat_template_kwargs"] = {"enable_thinking": _thinking_enabled(log_tag)}
    if effective_response_format is not None:
        payload["response_format"] = effective_response_format

    if urlparse(base_url).hostname == "api.deepseek.com":
        for key in ("top_k", "min_p", "repetition_penalty", "enable_thinking",
                    "enable_tools", "enabled_tools", "chat_template_kwargs"):
            payload.pop(key, None)
        payload["thinking"] = {"type": "enabled" if _thinking_enabled(log_tag) else "disabled"}
        if image_data_urls:
            payload["model"] = os.getenv("DOUJIN_FORGE_VISION_MODEL", "deepseek-v4-flash-vision-exp")
        if effective_response_format and effective_response_format.get("type") == "json_schema":
            schema = effective_response_format["json_schema"]["schema"]
            payload["messages"][0]["content"] += "\nReturn only JSON matching this schema:\n" + json.dumps(schema)
            payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": "Bearer "
        + os.getenv("DOUJIN_FORGE_OPENAI_API_KEY", "not-needed"),
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0)
        ) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if _is_no_model_loaded_response(response) and await _load_local_model_after_missing_model(
                client,
                base_url=base_url,
                selected_model=selected_model,
                headers=headers,
                log_tag=log_tag,
            ):
                # The initial request never reached inference.  Resubmit the
                # exact same payload once after the selected local model has
                # proved ready; do not turn arbitrary generation failures into
                # an unbounded retry loop.
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _http_error(log_tag, exc.response) from exc
    except httpx.RequestError as exc:
        raise AgentResultError(
            f"{log_tag}: local OpenAI API request failed: {exc}"
        ) from exc

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AgentResultError(
            f"{log_tag}: local OpenAI API returned an invalid response"
        ) from exc
    text = _coerce_content(content)
    if not text.strip():
        raise AgentResultError(f"{log_tag}: local OpenAI API returned empty content")
    raise_if_refusal_text(text)
    if on_text is not None:
        on_text(text)
    return AgentTextResult(
        text=text,
        duration_ms=int((time.perf_counter() - started) * 1000),
        total_cost_usd=0.0,
        usage=body.get("usage") if isinstance(body, dict) else None,
    )
