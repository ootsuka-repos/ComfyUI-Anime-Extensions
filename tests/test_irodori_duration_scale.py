from __future__ import annotations

import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _package(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


class _KeywordObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ProgressBar:
    def __init__(self, total: int):
        self.total = total

    def update_absolute(self, _current: int, _total: int) -> None:
        return None


class _IoType:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def Input(self, name: str, **kwargs):
        return SimpleNamespace(name=name, kind=self.kind, **kwargs)

    def Output(self, **kwargs):
        return SimpleNamespace(kind=self.kind, **kwargs)


class _Runtime:
    def __init__(self) -> None:
        self.request: _KeywordObject | None = None

    def synthesize(self, request, **_kwargs):
        self.request = request
        audio = torch.ones((1, 8), dtype=torch.float32)
        return SimpleNamespace(audio=audio, audios=[audio], sample_rate=24_000)


def _load_sampler():
    root = Path(__file__).resolve().parents[1]
    prefix = f"_comfyui_extensions_irodori_duration_{uuid.uuid4().hex}"
    _package(prefix, root)
    _package(f"{prefix}.py", root / "py")
    _package(f"{prefix}.py.nodes", root / "py" / "nodes")
    _package(f"{prefix}.py.nodes.wrapper", root / "py" / "nodes" / "wrapper")
    _package(f"{prefix}.py.modules", root / "py" / "modules")
    _package(f"{prefix}.py.modules.irodori_tts", root / "py" / "modules" / "irodori_tts")

    comfy_utils = ModuleType("comfy.utils")
    comfy_utils.ProgressBar = _ProgressBar
    comfy = ModuleType("comfy")
    comfy.utils = comfy_utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils

    latest = ModuleType("comfy_api.latest")
    latest.io = SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: SimpleNamespace(**kwargs),
        NodeOutput=lambda value: value,
        String=_IoType("string"),
        Int=_IoType("int"),
        Float=_IoType("float"),
        Combo=_IoType("combo"),
        Boolean=_IoType("boolean"),
        Audio=_IoType("audio"),
    )
    comfy_api = ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    runtime = _Runtime()
    runtime_module = ModuleType(f"{prefix}.py.modules.irodori_tts.inference_runtime")
    runtime_module.RuntimeKey = _KeywordObject
    runtime_module.SamplingRequest = _KeywordObject
    runtime_module.get_cached_runtime = lambda _key: (runtime, False)
    runtime_module.clear_cached_runtime = lambda: None
    runtime_module.offload_cached_runtime = lambda: None
    sys.modules[runtime_module.__name__] = runtime_module

    node_utils = ModuleType(f"{prefix}.py.node_utils")
    node_utils.mk_name = lambda *parts: ".".join(parts)
    sys.modules[node_utils.__name__] = node_utils

    common = ModuleType(f"{prefix}.py.nodes.wrapper.common")
    common.CATEGORY = "Irodori"
    common.PACKAGE_NAME = "IrodoriTTS"
    sys.modules[common.__name__] = common

    irodori_common = ModuleType(f"{prefix}.py.nodes.wrapper.irodori_common")
    for name in (
        "IO_CFG_CONFIG",
        "IO_LORA_STACK",
        "IO_MODEL_CONFIG",
        "IO_REF_CONFIG",
        "IO_RESCALE_CONFIG",
        "IO_SCHEDULE_CONFIG",
        "IO_TRIM_TAIL_CONFIG",
        "IO_VOICE_DESIGN_CONFIG",
    ):
        setattr(irodori_common, name, _IoType(name))
    irodori_common.none_if_non_positive = lambda value: value if value > 0 else None
    sys.modules[irodori_common.__name__] = irodori_common

    module_name = f"{prefix}.py.nodes.wrapper.irodori_sampler"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "py" / "nodes" / "wrapper" / "irodori_sampler.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("irodori_sampler.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, runtime


def _execute(module, *, duration_scale: float | None = None, **legacy):
    kwargs = {
        "model_config": {"checkpoint": "model.safetensors", "codec_repo": "codec"},
        "text": "test",
        "seed": 1,
        "num_steps": 2,
        "batch_size": 1,
        "decode_mode": "sequential",
        "context_kv_cache": True,
        "max_text_len": 0,
        "trim_tail": True,
        **legacy,
    }
    if duration_scale is not None:
        kwargs["duration_scale"] = duration_scale
    return module.IrodoriTTSSampler.execute(**kwargs)


class IrodoriDurationScaleTests(unittest.TestCase):
    def test_schema_exposes_an_optional_bounded_scale(self) -> None:
        module, _runtime = _load_sampler()

        schema = module.IrodoriTTSSampler.define_schema()
        inputs = {input_spec.name: input_spec for input_spec in schema.inputs}
        scale = inputs["duration_scale"]

        self.assertTrue(scale.optional)
        self.assertEqual(scale.default, 1.0)
        self.assertEqual((scale.min, scale.max, scale.step), (0.1, 3.0, 0.01))

    def test_sampler_forwards_scale_to_the_automatic_duration_predictor(self) -> None:
        module, runtime = _load_sampler()

        output = _execute(module, duration_scale=0.75)

        self.assertEqual(runtime.request.duration_scale, 0.75)
        self.assertEqual(output["sample_rate"], 24_000)

    def test_sampler_defaults_to_one_and_ignores_legacy_manual_duration(self) -> None:
        module, runtime = _load_sampler()

        _execute(module, seconds=12.0, duration_config={"duration_scale": 0.5})

        self.assertEqual(runtime.request.seconds, None)
        self.assertEqual(runtime.request.duration_scale, 1.0)

    def test_sampler_rejects_unsafe_duration_scales(self) -> None:
        module, _runtime = _load_sampler()

        for value in (False, 0.0, 0.09, 3.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                _execute(module, duration_scale=value)


if __name__ == "__main__":
    unittest.main()
