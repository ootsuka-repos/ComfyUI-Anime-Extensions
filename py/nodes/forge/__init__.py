"""Headless work assembly and VRM operations owned by the ComfyUI runtime."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import folder_paths
import torch
from comfy import model_management
from comfy_api.latest import io


def input_file(name: str) -> Path:
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Use an existing uploaded file inside ComfyUI/input")
    return path


def run_process(command: list[str], directory: Path) -> None:
    with (directory / "process.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(
                    f"Process failed ({process.returncode}); inspect {directory / 'process.log'}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def avatar(mode: str, name: str, source_video: str = "", avatar_file: str = ""):
    blender = shutil.which(os.environ.get("COMFYUI_FORGE_BLENDER", "blender"))
    if not blender:
        raise RuntimeError(
            "Install Blender with the VRM addon in the ComfyUI host, or set COMFYUI_FORGE_BLENDER"
        )
    source = input_file(source_video) if mode == "retarget" else None
    input_vrm = input_file(avatar_file) if mode == "retarget" else None
    ffmpeg = shutil.which("ffmpeg") if source is not None else None
    if source is not None and not ffmpeg:
        raise RuntimeError(
            "Install ffmpeg on the ComfyUI host for VRM video/audio output"
        )
    relative = "doujin-forge/avatar/" + uuid.uuid4().hex
    directory = Path(folder_paths.get_output_directory()) / relative
    directory.mkdir(parents=True)
    command = [
        blender,
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).with_name("blender_bridge.py")),
        "--",
        "--mode",
        mode,
        "--name",
        name,
        "--vrm-output",
        str(directory / "avatar.vrm"),
        "--poster-output",
        str(directory / "poster.png"),
        "--blend-output",
        str(directory / "scene.blend"),
    ]
    if source is not None:
        from .pose_2d import extract_pose_2d

        motion = directory / "motion.json"
        metadata = extract_pose_2d(
            source,
            motion,
            check_interrupt=model_management.throw_exception_if_processing_interrupted,
        )
        command += [
            "--input-vrm",
            str(input_vrm),
            "--motion-2d",
            str(motion),
            "--skip-vrma",
            "--video-output",
            str(directory / "dance.mp4"),
            "--fps",
            str(metadata["fps"]),
        ]
    run_process(command, directory)
    if source is not None:
        shutil.copyfile(input_vrm, directory / "avatar.vrm")
        silent = directory / "dance-silent.mp4"
        (directory / "dance.mp4").replace(silent)
        run_process(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-i",
                str(silent),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(directory / "dance.mp4"),
            ],
            directory,
        )
        silent.unlink()
    required = ["avatar.vrm", "poster.png", "scene.blend"] + (
        ["dance.mp4"] if source is not None else []
    )
    if any(
        not (directory / name).is_file() or not (directory / name).stat().st_size
        for name in required
    ):
        raise RuntimeError(
            f"Blender output is incomplete; inspect {directory / 'process.log'}"
        )
    files = [
        {"filename": p.name, "subfolder": relative, "type": "output"}
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix in {".vrm", ".png", ".blend", ".mp4", ".json"}
    ]
    if not files:
        raise RuntimeError(
            f"Blender returned no artifacts; inspect {directory / 'process.log'}"
        )
    manifest = json.dumps(
        {"mode": mode, "name": name, "files": files}, ensure_ascii=False
    )
    return io.NodeOutput(manifest, ui={"text": [manifest], "files": files})


class ForgeVRMStarter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.Forge.VRMStarter",
            display_name="Forge VRM Starter",
            category="Doujin Forge/Avatar",
            inputs=[io.String.Input("name", default="Avatar")],
            outputs=[io.String.Output("manifest")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, name):
        return avatar("starter", name)


class ForgeVRMDance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.Forge.VRMDance",
            display_name="Forge VRM Dance",
            category="Doujin Forge/Avatar",
            inputs=[
                io.String.Input("name", default="Avatar"),
                io.String.Input("source_video"),
                io.String.Input("avatar_file"),
            ],
            outputs=[io.String.Output("manifest")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, name, source_video, avatar_file):
        return avatar("retarget", name, source_video, avatar_file)


class ForgeComicPage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.Forge.ComicPage",
            display_name="Forge Comic Page",
            category="Doujin Forge/Comic",
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


from .text import ForgeTextCompletion, ForgeTextModelRelease

nodes = [ForgeVRMStarter, ForgeVRMDance, ForgeComicPage, ForgeTextCompletion, ForgeTextModelRelease]
