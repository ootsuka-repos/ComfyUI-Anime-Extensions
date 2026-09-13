from __future__ import annotations

from comfy_api.latest import ComfyExtension
from comfy_api.latest import io
from typing_extensions import override

from .py import NODES


class Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> Extension:
    from comfy_api.latest import ComfyAPI
    from aiohttp import web
    from server import PromptServer
    from .py.nodes.yue2 import runtime_status
    from .py.model_lifecycle import model_lifecycle

    await ComfyAPI().caching.register_provider(model_lifecycle)

    if hasattr(PromptServer, 'instance'):
        async def yue2_status(request):
            return web.json_response(runtime_status())
        PromptServer.instance.routes.get('/yue2/status')(yue2_status)
        from .py.model_identity import model_identity_route
        PromptServer.instance.routes.post("/ComfyUIExtensions/Forge/model-identity")(model_identity_route)
    return Extension()


WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
