from __future__ import annotations

from comfy_api.latest import io

from .nodes.character_voice import nodes as character_voice_nodes
from .nodes.kanomemo import nodes as kanomemo_nodes
from .nodes.wrapper import nodes as wrapper_nodes
from .nodes.yue2 import nodes as yue2_nodes

NODES: list[type[io.ComfyNode]] = [
    *wrapper_nodes,
    *character_voice_nodes,
    *kanomemo_nodes,
    *yue2_nodes,
]

__all__ = ["NODES"]
