from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class Extension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        from comfy_api.latest import ComfyAPI
        from server import PromptServer

        from .py.model_lifecycle import model_lifecycle
        from .py.routes import register_routes

        register_routes(PromptServer.instance.routes)
        await ComfyAPI().caching.register_provider(model_lifecycle)

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .py import get_node_list

        return get_node_list()


async def comfy_entrypoint() -> Extension:
    return Extension()


WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
