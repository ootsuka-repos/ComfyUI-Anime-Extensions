"""Native ComfyUI loaders whose execution fingerprints follow local weights."""

import asyncio
from pathlib import Path

import folder_paths
import nodes as native
from comfy_api.latest import io

from ...model_identity import _file_identity


def _digest(category, name):
    return _file_identity(Path(folder_paths.get_full_path_or_raise(category, name)))["sha256"]


def _schema(name, loader, outputs):
    inputs = []
    for group, specifications in loader.INPUT_TYPES().items():
        for key, specification in specifications.items():
            options = specification[0]
            settings = specification[1] if len(specification) > 1 else {}
            inputs.append(io.Combo.Input(key, options=options, optional=group == "optional", **settings))
    return io.Schema(node_id="ComfyUIExtensions_Forge_" + name, display_name="Forge " + name,
                     category="Doujin Forge/model loaders", inputs=inputs, outputs=outputs)


class ForgeCheckpointLoaderSimple(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("CheckpointLoaderSimple", native.CheckpointLoaderSimple,
                       [io.Model.Output(), io.Clip.Output(), io.Vae.Output()])

    @classmethod
    async def fingerprint_inputs(cls, ckpt_name):
        def calculate():
            return _digest("checkpoints", ckpt_name)

        return await asyncio.to_thread(calculate)

    @classmethod
    def execute(cls, ckpt_name):
        return io.NodeOutput(*native.CheckpointLoaderSimple().load_checkpoint(ckpt_name))


class ForgeUNETLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("UNETLoader", native.UNETLoader, [io.Model.Output()])

    @classmethod
    async def fingerprint_inputs(cls, unet_name, weight_dtype):
        def calculate():
            return _digest("diffusion_models", unet_name)

        return await asyncio.to_thread(calculate)

    @classmethod
    def execute(cls, unet_name, weight_dtype):
        return io.NodeOutput(*native.UNETLoader().load_unet(unet_name, weight_dtype))


class ForgeCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("CLIPLoader", native.CLIPLoader, [io.Clip.Output()])

    @classmethod
    async def fingerprint_inputs(cls, clip_name, type="stable_diffusion", device="default"):
        def calculate():
            return _digest("text_encoders", clip_name)

        return await asyncio.to_thread(calculate)

    @classmethod
    def execute(cls, clip_name, type="stable_diffusion", device="default"):
        return io.NodeOutput(*native.CLIPLoader().load_clip(clip_name, type, device))


class ForgeVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("VAELoader", native.VAELoader, [io.Vae.Output()])

    @classmethod
    async def fingerprint_inputs(cls, vae_name):
        def calculate():
            if vae_name == "pixel_space":
                return "pixel_space"
            if vae_name in native.VAELoader.image_taes:
                files = folder_paths.get_filename_list("vae_approx")
                return tuple(_digest("vae_approx", next(name for name in files if name.startswith(f"{vae_name}_{part}.")))
                             for part in ("encoder", "decoder"))
            category = "vae_approx" if Path(vae_name).stem in native.VAELoader.video_taes else "vae"
            return _digest(category, vae_name)

        return await asyncio.to_thread(calculate)

    @classmethod
    def execute(cls, vae_name):
        return io.NodeOutput(*native.VAELoader().load_vae(vae_name))


nodes = [ForgeCheckpointLoaderSimple, ForgeUNETLoader, ForgeCLIPLoader, ForgeVAELoader]
