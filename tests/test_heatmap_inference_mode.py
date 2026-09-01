from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch


def _package(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_heatmap():
    root = Path(__file__).resolve().parents[1]
    prefix = "_comfyui_extensions_heatmap_test"
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

    analysis = ModuleType(f"{prefix}.py.nodes.kanomemo.analysis")
    analysis._analyse = lambda *_args, **_kwargs: []
    sys.modules[analysis.__name__] = analysis

    latest = ModuleType("comfy_api.latest")
    latest.io = SimpleNamespace(ComfyNode=object)
    latest.ui = SimpleNamespace()
    comfy_api = ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    name = f"{prefix}.py.nodes.kanomemo.heatmap"
    spec = importlib.util.spec_from_file_location(name, root / "py" / "nodes" / "kanomemo" / "heatmap.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("heatmap.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _TinyTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inference_modes: list[bool] = []

    def forward_features(self, image):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return image.mean(dim=1).reshape(1, 4, 1)

    @staticmethod
    def forward_head(features):
        return features.mean(dim=1)


class HeatmapInferenceModeTests(unittest.TestCase):
    def test_grad_cam_works_inside_comfy_inference_mode(self):
        heatmap = _load_heatmap()
        model = _TinyTagger()
        transform = lambda _image: torch.ones((3, 2, 2), dtype=torch.float32)

        with patch.object(heatmap, "_load_model", return_value=(model, ("pussy",), transform, "cpu")):
            with torch.inference_mode():
                masks = heatmap.heatmap_tag_masks(Image.new("RGB", (16, 16), "white"))

        self.assertEqual(set(masks), {"pussy"})
        self.assertFalse(any(model.inference_modes))
        self.assertGreater(masks["pussy"].getextrema()[1], 0)


if __name__ == "__main__":
    unittest.main()
