from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def _package(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_analysis():
    root = Path(__file__).resolve().parents[1]
    prefix = "_comfyui_extensions_mobile_sam_test"
    _package(prefix, root)
    _package(f"{prefix}.py", root / "py")
    _package(f"{prefix}.py.nodes", root / "py" / "nodes")
    _package(f"{prefix}.py.nodes.kanomemo", root / "py" / "nodes" / "kanomemo")

    node_utils = ModuleType(f"{prefix}.py.node_utils")
    node_utils.mk_name = lambda *parts: ".".join(parts)
    sys.modules[node_utils.__name__] = node_utils

    portrait = ModuleType(f"{prefix}.py.nodes.kanomemo.portrait")
    portrait._configure_huggingface_cache = lambda: None
    sys.modules[portrait.__name__] = portrait

    latest = ModuleType("comfy_api.latest")
    latest.io = SimpleNamespace(ComfyNode=object)
    latest.ui = SimpleNamespace()
    comfy_api = ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    name = f"{prefix}.py.nodes.kanomemo.analysis"
    spec = importlib.util.spec_from_file_location(name, root / "py" / "nodes" / "kanomemo" / "analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("analysis.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MobileSamConstructionTests(unittest.TestCase):
    def test_model_is_not_created_as_an_inference_tensor(self):
        analysis = _load_analysis()

        class FakeSAM:
            def __init__(self, checkpoint):
                self.checkpoint = checkpoint
                self.inference_mode_enabled = torch.is_inference_mode_enabled()
                self.parameter = torch.ones(1)

        huggingface_hub = ModuleType("huggingface_hub")
        huggingface_hub.hf_hub_download = lambda **_kwargs: "mobile_sam.pt"
        ultralytics = ModuleType("ultralytics")
        ultralytics.SAM = FakeSAM

        with patch.dict(
            sys.modules,
            {"huggingface_hub": huggingface_hub, "ultralytics": ultralytics},
        ):
            with torch.inference_mode():
                model = analysis._mobile_sam_model()

        self.assertFalse(model.inference_mode_enabled)
        self.assertFalse(model.parameter.is_inference())
        self.assertEqual(model.checkpoint, "mobile_sam.pt")


if __name__ == "__main__":
    unittest.main()
