from __future__ import annotations

import builtins
import importlib
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch  # ComfyUI supplies torch; initialize it before blocking optional imports.


class _IoType:
    def __init__(self, kind):
        self.kind = kind

    def Input(self, name, **kwargs):
        return SimpleNamespace(name=name, kind=self.kind, **kwargs)

    def Output(self, name=None, **kwargs):
        return SimpleNamespace(name=name, kind=self.kind, **kwargs)


@contextmanager
def _voice_nodes_without_backends():
    root = Path(__file__).resolve().parents[1]
    prefix = f"_voice_lazy_imports_{uuid.uuid4().hex}"
    original_import = builtins.__import__
    forbidden = {
        "dacvae", "torchaudio", "torchcodec", "transformers", "huggingface_hub",
        "safetensors", "sentencepiece", "soundfile", "imageio_ffmpeg", "peft",
        "timm", "silentcipher",
    }
    attempts = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in forbidden or "inference_runtime" in name:
            attempts.append(name)
            raise ModuleNotFoundError(f"Blocked optional backend: {name}", name=name)
        return original_import(name, globals, locals, fromlist, level)

    with patch.dict(sys.modules), TemporaryDirectory() as input_dir:
        sys.modules["torch"] = torch
        for suffix, directory in (("", root), (".py", root / "py"),
                                  (".py.nodes", root / "py" / "nodes")):
            module = ModuleType(prefix + suffix)
            module.__path__ = [str(directory)]
            sys.modules[module.__name__] = module

        folder_paths = ModuleType("folder_paths")
        folder_paths.base_path = input_dir
        folder_paths.get_filename_list = lambda _kind: ["model.safetensors"]
        folder_paths.get_input_directory = lambda: input_dir
        folder_paths.filter_files_content_types = lambda files, _types: files
        sys.modules["folder_paths"] = folder_paths

        comfy = ModuleType("comfy")
        comfy.utils = ModuleType("comfy.utils")
        comfy.model_management = ModuleType("comfy.model_management")
        sys.modules.update({
            "comfy": comfy,
            "comfy.utils": comfy.utils,
            "comfy.model_management": comfy.model_management,
        })

        io = SimpleNamespace(
            ComfyNode=object, Schema=lambda **kwargs: SimpleNamespace(**kwargs),
            Custom=_IoType, Hidden=SimpleNamespace(prompt="prompt", extra_pnginfo="extra_pnginfo"),
            UploadType=SimpleNamespace(audio="audio"),
        )
        for kind in ("String", "Int", "Float", "Combo", "Boolean", "Audio", "Image"):
            setattr(io, kind, _IoType(kind))
        latest = ModuleType("comfy_api.latest")
        latest.io = io
        latest.ui = SimpleNamespace()
        comfy_api = ModuleType("comfy_api")
        comfy_api.latest = latest
        sys.modules.update({"comfy_api": comfy_api, "comfy_api.latest": latest})

        with patch("builtins.__import__", side_effect=guarded_import):
            nodes = []
            for package in ("wrapper", "character_voice", "yue2"):
                module = importlib.import_module(f"{prefix}.py.nodes.{package}")
                nodes.extend(module.nodes)
            yield nodes, attempts


class VoiceLazyImportsTests(unittest.TestCase):
    def test_all_voice_schemas_are_available_without_optional_backends(self):
        with _voice_nodes_without_backends() as (nodes, attempts):
            schemas = {}
            for node in nodes:
                schema = node.define_schema()
                schemas[schema.node_id] = schema
            expected = {
                "EmojiPicker", "ModelLoader", "LoRAStack", "ReferenceAudio", "ReferenceAudioList",
                "VoiceDesignConfig", "CFGConfig", "RescaleConfig", "ScheduleConfig", "TrimTailConfig",
                "Sampler", "AudioSave", "CharacterVoiceSampler",
            }
            self.assertEqual(set(schemas), {
                *(f"ComfyUIExtensions.IrodoriTTS.{name}" for name in expected), "YuE2Generate",
            })
            self.assertEqual(attempts, [])

    def test_yue_requires_soundfile_only_after_consent_at_execution(self):
        with _voice_nodes_without_backends() as (nodes, attempts):
            node = next(node for node in nodes if node.__name__ == "YuE2Generate")
            kwargs = {"style": "test", "lyrics": "test", "seed": 1, "cot": "off", "abc": "", "cfg_scale": -1}
            with self.assertRaises(ValueError):
                node.execute(**kwargs, noncommercial=False)
            self.assertEqual(attempts, [])
            with self.assertRaises(ModuleNotFoundError) as raised:
                node.execute(**kwargs, noncommercial=True)
            self.assertEqual(raised.exception.name, "soundfile")


if __name__ == "__main__":
    unittest.main()
