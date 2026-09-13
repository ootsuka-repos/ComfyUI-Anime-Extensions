from __future__ import annotations

from comfy_api.latest import io

from .nodes.media import nodes as media_nodes
from .nodes.media.model_loaders import nodes as model_loader_nodes
from .nodes.character_voice import nodes as character_voice_nodes
from .nodes.image_tools import nodes as image_tools_nodes
from .nodes.wrapper import nodes as wrapper_nodes
from .nodes.yue2 import nodes as yue2_nodes

NODES: list[type[io.ComfyNode]] = [
    *media_nodes,
    *model_loader_nodes,
    *wrapper_nodes,
    *character_voice_nodes,
    *image_tools_nodes,
    *yue2_nodes,
]

__all__ = ["NODES"]
