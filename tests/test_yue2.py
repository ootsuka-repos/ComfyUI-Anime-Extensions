"""YuE2 node contract; no inference or network in unit tests."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class Yue2ContractTests(unittest.TestCase):
    def test_native_request_preserves_text_and_requires_consent(self):
        path = ROOT / 'py/nodes/yue2/worker.py'
        self.assertTrue(path.is_file(), 'YuE2 isolated node worker is missing')
        spec = importlib.util.spec_from_file_location('yue2_worker', path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        fields = dict(style='Pop\n', lyrics='[verse]\nHello　world\n', seed=42, cot='full', abc='', cfg_scale=2.0, noncommercial=True)
        native = worker.native_request(fields)
        self.assertEqual(native, {k:v for k,v in fields.items() if k not in ('noncommercial', 'abc')})
        with self.assertRaisesRegex(ValueError, 'CC-BY-NC'):
            worker.native_request({**fields, 'noncommercial': False})

    def test_default_cfg_uses_native_mode_default(self):
        spec = importlib.util.spec_from_file_location('yue2_worker', ROOT / 'py/nodes/yue2/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        fields = dict(style='Pop', lyrics='Hello', seed=42, cot='off', abc='', cfg_scale=-1.0, noncommercial=True)
        self.assertIsNone(worker.native_request(fields)['cfg_scale'])

    def test_node_is_registered_and_owns_audio_output(self):
        import sys
        sys.path.insert(0, str(ROOT.parents[1]))
        spec = importlib.util.spec_from_file_location('extensions_test', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        import asyncio
        extension = asyncio.run(module.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        found = [node for node in nodes if node.define_schema().node_id == 'YuE2Generate']
        self.assertEqual(len(found), 1, 'YuE2Generate must be registered as an actual ComfyUI node')
        schema = found[0].define_schema()
        self.assertTrue(schema.is_output_node)
        self.assertEqual(schema.outputs[0].io_type, 'AUDIO')

    def test_worker_saves_native_artifacts_and_truncation(self):
        import json
        import tempfile
        import types
        from unittest.mock import patch
        import numpy as np
        import soundfile as sf
        spec = importlib.util.spec_from_file_location('yue2_worker', ROOT / 'py/nodes/yue2/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        self.assertTrue(hasattr(worker, 'generate'), 'worker must execute the native pipeline, not HTTP')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            fields = dict(style='Pop', lyrics='Hello\n', seed=42, cot='full', abc='', cfg_scale=-1, noncommercial=True)
            seen = {}
            class Song:
                audio = np.ones((480, 2), dtype=np.float32) * 0.1
                latents = np.ones((4, 2), dtype=np.float32)
                truncated = {'abc': False, 'semantic': True}
                def save_artifacts(self, path):
                    path.mkdir()
                    sf.write(path / 'audio.flac', self.audio, 48000)
                    (path / 'request.json').write_text(json.dumps(seen['request']))
            class Pipeline:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    seen['load'] = kwargs
                    return cls()
                def __call__(self, **kwargs):
                    seen['request'] = kwargs
                    return Song()
                def close(self):
                    seen['closed'] = True
            with patch.dict('sys.modules', yue2=types.SimpleNamespace(YuE2Pipeline=Pipeline)):
                worker.generate(fields, directory, 'model', 'vae')
            self.assertEqual(seen['request']['lyrics'], fields['lyrics'])
            self.assertIsNone(seen['request']['cfg_scale'])
            self.assertTrue(seen['load']['local_files_only'])
            self.assertTrue(seen['closed'])
            self.assertTrue((directory / 'native/audio.flac').is_file())
            self.assertIs(json.loads((directory / 'completion.json').read_text())['truncated'], True)

    def test_node_executes_isolated_worker_and_returns_playable_audio(self):
        import json
        import sys
        import tempfile
        from unittest.mock import patch
        import numpy as np
        import soundfile as sf
        self.test_node_is_registered_and_owns_audio_output()
        node_module = sys.modules['extensions_test.py.nodes.yue2']
        self.assertIn('execute', node_module.YuE2Generate.__dict__, 'node must implement execution')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def run_worker(fields, directory):
                (directory / 'native').mkdir()
                sf.write(directory / 'native/audio.flac', np.ones((480, 2)) * 0.1, 48000)
                (directory / 'completion.json').write_text(json.dumps({'truncated': False}))
            with patch.object(node_module.folder_paths, 'get_output_directory', return_value=str(root)), patch.object(node_module, 'run_worker', side_effect=run_worker):
                result = node_module.YuE2Generate.execute(style='Pop', lyrics='Hello', seed=42, cot='full', abc='', cfg_scale=-1, noncommercial=True)
            self.assertEqual(result.result[0]['waveform'].shape, (1, 2, 480))
            self.assertFalse(result.result[1])
            self.assertTrue(list(root.glob('yue2/*/native/audio.flac')))

    def test_worker_cli_rejects_unconsented_inference(self):
        import json
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'intent.json').write_text(json.dumps({'noncommercial': False}))
            result = subprocess.run([sys.executable, str(ROOT / 'py/nodes/yue2/worker.py'), temp, 'model', 'vae'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, 'worker CLI must actually execute and validate intent')
            self.assertIn('CC-BY-NC', result.stderr)

    def test_missing_runtime_reports_unready_without_loading_models(self):
        import sys
        from unittest.mock import patch
        self.test_node_is_registered_and_owns_audio_output()
        node_module=sys.modules['extensions_test.py.nodes.yue2']
        self.assertTrue(hasattr(node_module, 'runtime_status'), 'extension readiness must verify installed runtime')
        with patch.dict('os.environ', {'COMFYUI_YUE2_CONFIG':'/nonexistent/yue2.json'}):
            status=node_module.runtime_status()
        self.assertFalse(status['ready'])
        self.assertFalse(status['model_loaded'])
        self.assertEqual(status['license'], 'CC-BY-NC-4.0')

    def test_comfy_interrupt_terminates_isolated_worker(self):
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        self.test_node_is_registered_and_owns_audio_output()
        module=sys.modules['extensions_test.py.nodes.yue2']
        self.assertTrue(hasattr(module, 'model_management'), 'node must honor ComfyUI interrupt and memory ownership')
        process=MagicMock()
        process.poll.return_value=None
        with tempfile.TemporaryDirectory() as temp, patch.object(module.Path, 'read_text', return_value='{"python":"unused","model":"unused","vae":"unused"}'), patch.object(module.subprocess,'Popen',return_value=process), patch.object(module.model_management,'unload_all_models'), patch.object(module.model_management,'throw_exception_if_processing_interrupted',side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                module.run_worker({},Path(temp))
        process.terminate.assert_called_once()
        process.wait.assert_called()

if __name__ == '__main__':
    unittest.main()
