"""Read-only runtime availability for plugin submission checks."""
import asyncio
import subprocess

from aiohttp import web
from server import PromptServer

from .sol_h3 import runtime_status, model_fingerprint


@PromptServer.instance.routes.get("/ComfyUIExtensions/Forge/SolH3/status")
async def sol_status(_request):
    return web.json_response(await asyncio.to_thread(runtime_status))


@PromptServer.instance.routes.post("/ComfyUIExtensions/Forge/sol-model-identity")
async def sol_model_identity(request):
    try:
        payload = await request.json()
        if (not isinstance(payload, dict) or set(payload) != {"task"}
                or payload["task"] not in ("t2va", "fl2va", "ref2va")):
            raise ValueError("Expected task: t2va, fl2va or ref2va")
        digest = await asyncio.to_thread(model_fingerprint, payload["task"])
        return web.json_response({"version": 1, "sha256": digest})
    except (ValueError, TypeError):
        return web.json_response({"error": "Invalid Sol model identity request or configuration"}, status=400)
    except FileNotFoundError:
        return web.json_response({"error": "Required local Sol model assets are unavailable"}, status=404)
    except (OSError, RuntimeError, KeyError, subprocess.SubprocessError):
        return web.json_response({"error": "Sol model identity could not be read; check server configuration and retry"}, status=503)
