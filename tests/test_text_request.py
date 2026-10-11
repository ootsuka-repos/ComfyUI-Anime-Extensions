import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class TextRequestTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        package_name = "text_request_test_media"
        package = ModuleType(package_name)
        package.__path__ = [str(root / "py" / "nodes" / "media")]
        comfy = ModuleType("comfy")
        comfy.model_management = SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
        latest = ModuleType("comfy_api.latest")
        string_type = SimpleNamespace(
            Input=lambda name, **kwargs: SimpleNamespace(id=name, **kwargs),
            Output=lambda name, **kwargs: SimpleNamespace(id=name, **kwargs),
        )
        latest.io = SimpleNamespace(
            ComfyNode=object,
            Schema=lambda **kwargs: SimpleNamespace(**kwargs),
            String=string_type,
            NodeOutput=lambda *values, **kwargs: SimpleNamespace(result=values, **kwargs),
        )
        comfy_api = ModuleType("comfy_api")
        comfy_api.latest = latest
        self.backend_name = f"{package_name}.text_backend"
        modules = patch.dict(sys.modules, {
            package_name: package,
            "comfy": comfy,
            "comfy_api": comfy_api,
            "comfy_api.latest": latest,
            # Registration and request construction must work without the backend.
            self.backend_name: None,
        })
        modules.start()
        self.addCleanup(modules.stop)
        name = f"{package_name}.text"
        spec = importlib.util.spec_from_file_location(name, root / "py" / "nodes" / "media" / "text.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.module
        spec.loader.exec_module(self.module)

    def test_prompts_roundtrip_without_modification(self):
        user = '  「猫」の台詞: "こんにちは"\n次の行\\終わり  '
        system = '日本語で回答\n"引用符"と\tタブを保持'
        result = self.module.TextRequest.execute(user, system)
        self.assertEqual(json.loads(result.result[0]), {
            "user_prompt": user, "system_prompt": system, "log_tag": "text",
        })

    def test_blank_model_is_omitted(self):
        for model in ("", " \t\n"):
            with self.subTest(model=model):
                result = self.module.TextRequest.execute("質問", "", model)
                self.assertNotIn("model", json.loads(result.result[0]))

    def test_explicit_model_is_trimmed(self):
        result = self.module.TextRequest.execute("質問", "", "  local-model  ")
        self.assertEqual(json.loads(result.result[0])["model"], "local-model")


if __name__ == "__main__":
    unittest.main()
