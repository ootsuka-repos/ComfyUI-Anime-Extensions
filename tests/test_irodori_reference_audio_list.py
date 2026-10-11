from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_irodori_duration_scale import (  # noqa: E402
    _execute,
    _IoType,
    _load_sampler,
    _package,
)


def _load_reference_audio(input_dir: Path):
    root = Path(__file__).resolve().parents[1]
    prefix = f"_comfyui_extensions_irodori_reflist_{uuid.uuid4().hex}"
    _package(prefix, root)
    _package(f"{prefix}.py", root / "py")
    _package(f"{prefix}.py.nodes", root / "py" / "nodes")
    _package(f"{prefix}.py.nodes.wrapper", root / "py" / "nodes" / "wrapper")

    def annotated(name: str) -> str:
        if name.endswith(" [temp]"):
            return str(input_dir / "temp" / name[: -len(" [temp]")])
        return str(input_dir / name)

    folder_paths = ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(input_dir)
    folder_paths.filter_files_content_types = lambda files, _kinds: files
    folder_paths.get_annotated_filepath = annotated
    folder_paths.exists_annotated_filepath = lambda name: Path(annotated(name)).is_file()
    sys.modules["folder_paths"] = folder_paths

    latest = ModuleType("comfy_api.latest")
    latest.io = SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: SimpleNamespace(**kwargs),
        NodeOutput=lambda value: value,
        String=_IoType("string"),
        Float=_IoType("float"),
        Combo=_IoType("combo"),
        Boolean=_IoType("boolean"),
        Custom=_IoType,
        UploadType=SimpleNamespace(audio="audio"),
    )
    comfy_api = ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    node_utils = ModuleType(f"{prefix}.py.node_utils")
    node_utils.mk_name = lambda *parts: ".".join(parts)
    sys.modules[node_utils.__name__] = node_utils
    common = ModuleType(f"{prefix}.py.nodes.wrapper.common")
    common.CATEGORY = "Irodori"
    common.PACKAGE_NAME = "IrodoriTTS"
    sys.modules[common.__name__] = common
    irodori_common = ModuleType(f"{prefix}.py.nodes.wrapper.irodori_common")
    irodori_common.IO_REF_CONFIG = _IoType("IO_REF_CONFIG")
    sys.modules[irodori_common.__name__] = irodori_common

    module_name = f"{prefix}.py.nodes.wrapper.irodori_reference_audio"
    spec = importlib.util.spec_from_file_location(
        module_name, root / "py" / "nodes" / "wrapper" / "irodori_reference_audio.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("irodori_reference_audio.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class IrodoriReferenceAudioListTests(unittest.TestCase):
    def setUp(self) -> None:
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.input_dir = Path(self._tmp.name)
        (self.input_dir / "temp").mkdir()
        (self.input_dir / "a.wav").write_bytes(b"a")
        (self.input_dir / "temp" / "b.wav").write_bytes(b"b")
        self.module = _load_reference_audio(self.input_dir)
        self.node = self.module.IrodoriReferenceAudioList

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_resolves_every_clip_in_order(self) -> None:
        config = self.node.execute("b.wav [temp]\n\n  a.wav  \n", True, 60.0)

        self.assertEqual(
            config["ref_wavs"],
            [str(self.input_dir / "temp" / "b.wav"), str(self.input_dir / "a.wav")],
        )
        self.assertIsNone(config["ref_wav"])
        self.assertFalse(config["no_ref"])
        self.assertEqual(config["ref_normalize_db"], -16.0)
        self.assertTrue(config["ref_ensure_max"])
        self.assertEqual(config["max_ref_seconds"], 60.0)

    def test_rejects_empty_missing_and_video_entries(self) -> None:
        (self.input_dir / "clip.mp4").write_bytes(b"v")

        self.assertIsInstance(self.node.validate_inputs(audios=" \n"), str)
        self.assertIsInstance(self.node.validate_inputs(audios="a.wav\nmissing.wav"), str)
        self.assertIsInstance(self.node.validate_inputs(audios="clip.mp4"), str)
        self.assertIs(self.node.validate_inputs(audios="a.wav\nb.wav [temp]"), True)

    def test_fingerprint_follows_clip_contents(self) -> None:
        before = self.node.fingerprint_inputs(audios="a.wav\nb.wav [temp]")
        (self.input_dir / "a.wav").write_bytes(b"changed")

        self.assertNotEqual(before, self.node.fingerprint_inputs(audios="a.wav\nb.wav [temp]"))


class IrodoriSamplerReferenceListTests(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)

    def test_sampler_forwards_reference_lists(self) -> None:
        module, runtime = _load_sampler()

        _execute(module, ref_config={"ref_wav": None, "ref_wavs": ["x.wav", "y.wav"]})

        self.assertIsNone(runtime.request.ref_wav)
        self.assertEqual(runtime.request.ref_wavs, ["x.wav", "y.wav"])
        self.assertFalse(runtime.request.no_ref)

    def test_sampler_without_reference_stays_reference_free(self) -> None:
        module, runtime = _load_sampler()

        _execute(module)

        self.assertIsNone(runtime.request.ref_wavs)
        self.assertTrue(runtime.request.no_ref)


if __name__ == "__main__":
    unittest.main()
