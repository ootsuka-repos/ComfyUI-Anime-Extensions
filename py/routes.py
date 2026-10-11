"""HTTP endpoints registered by ComfyExtension.on_load, never during import."""
from __future__ import annotations

import asyncio
import subprocess

from aiohttp import web

from .model_identity import model_identity_route


async def yue2_status(_request):
    from .nodes.yue2 import runtime_status

    return web.json_response(await asyncio.to_thread(runtime_status))


async def sol_status(_request):
    from .nodes.media.sol_h3 import runtime_status

    return web.json_response(await asyncio.to_thread(runtime_status))


async def sol_model_identity(request):
    from .nodes.media.sol_h3 import model_fingerprint

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


_ROUTES = (
    web.route("GET", "/yue2/status", yue2_status),
    web.route("GET", "/ComfyUIExtensions/SolH3/status", sol_status),
    web.route("POST", "/ComfyUIExtensions/sol-model-identity", sol_model_identity),
    web.route("POST", "/ComfyUIExtensions/model-identity", model_identity_route),
)


def register_routes(routes: web.RouteTableDef) -> None:
    for route in _ROUTES:
        if route not in routes:
            routes.route(route.method, route.path, **route.kwargs)(route.handler)
