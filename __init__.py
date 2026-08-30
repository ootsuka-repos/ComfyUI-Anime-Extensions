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
    return Extension()


WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
