from __future__ import annotations

from comfy_api.latest import io


def get_node_list() -> list[type[io.ComfyNode]]:
    """Discover node classes without initializing inference backends or HTTP routes."""
    from .nodes.character_voice import nodes as character_voice_nodes
    from .nodes.image_tools import nodes as image_tools_nodes
    from .nodes.media import nodes as media_nodes
    from .nodes.media.model_loaders import nodes as model_loader_nodes
    from .nodes.wrapper import nodes as wrapper_nodes
    from .nodes.yue2 import nodes as yue2_nodes

    return [
        *media_nodes,
        *model_loader_nodes,
        *wrapper_nodes,
        *character_voice_nodes,
        *image_tools_nodes,
        *yue2_nodes,
    ]


__all__ = ["get_node_list"]
