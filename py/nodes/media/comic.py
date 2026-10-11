"""Arrange image batches into comic pages."""
from __future__ import annotations

import math

import torch
from comfy_api.latest import io


class ComicPage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.ComicPage",
            display_name="Comic Page",
            category="ComfyUIExtensions/Comic",
            inputs=[
                io.Image.Input("panels"),
                io.Int.Input("columns", default=2, min=1),
                io.Int.Input("gutter", default=16, min=0),
                io.Boolean.Input("right_to_left", default=True),
            ],
            outputs=[io.Image.Output("page")],
        )

    @classmethod
    def execute(cls, panels, columns, gutter, right_to_left):
        count, height, width, channels = panels.shape
        if not count or columns < 1 or gutter < 0:
            raise ValueError(
                "Provide panels, at least one column and a nonnegative gutter"
            )
        columns = min(columns, count)
        rows = math.ceil(count / columns)
        page = torch.ones(
            (
                1,
                rows * height + (rows + 1) * gutter,
                columns * width + (columns + 1) * gutter,
                channels,
            ),
            dtype=panels.dtype,
            device=panels.device,
        )
        for index, panel in enumerate(panels):
            row, column = divmod(index, columns)
            if right_to_left:
                column = columns - 1 - column
            y, x = gutter + row * (height + gutter), gutter + column * (width + gutter)
            page[0, y : y + height, x : x + width] = panel
        return io.NodeOutput(page)
