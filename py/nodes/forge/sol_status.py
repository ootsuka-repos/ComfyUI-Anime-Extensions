"""Read-only runtime availability for plugin submission checks."""
import asyncio

from aiohttp import web
from server import PromptServer

from .sol_h3 import runtime_status


@PromptServer.instance.routes.get("/ComfyUIExtensions/Forge/SolH3/status")
async def sol_status(_request):
    return web.json_response(await asyncio.to_thread(runtime_status))
