
import asyncio
import json
import os
from pathlib import Path
import subprocess
import uuid
import time

import comfy.model_management as model_management
import folder_paths
import soundfile as sf
import torch
from comfy_api.latest import io, ui

from .worker import native_request


def runtime_config():
    config_path = Path(os.environ.get('COMFYUI_YUE2_CONFIG', str(Path(folder_paths.base_path) / 'runtimes/YuE2/comfyui.json')))
    return json.loads(config_path.read_text())


def runtime_status():
    result = {'service': 'yue2', 'model': 'm-a-p/YuE2-3B', 'license': 'CC-BY-NC-4.0',
              'ready': False, 'model_loaded': False}
    try:
        config_path = Path(os.environ.get('COMFYUI_YUE2_CONFIG', str(Path(folder_paths.base_path) / 'runtimes/YuE2/comfyui.json')))
        config = json.loads(config_path.read_text())
        result['ready'] = (os.access(config['python'], os.X_OK)
                           and (Path(config['model']) / 'model.safetensors').stat().st_size == 7261441640
                           and (Path(config['vae']) / 'model.safetensors').stat().st_size == 530512720)
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return result


def run_worker(fields, directory):
    config = runtime_config()
    (directory / 'intent.json').write_text(json.dumps(fields, ensure_ascii=False))
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'OMP_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [config['python'], '-B', str(Path(__file__).with_name('worker.py')), str(directory), config['model'], config['vae']]
    model_management.unload_all_models()
    with (directory / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            while process.poll() is None:
                model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f'YuE2 worker failed; see {directory / "worker.log"}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()



class YuE2Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='YuE2Generate', display_name='YuE2 Generate Song (noncommercial)',
            category='ComfyUI-Extensions/YuE2', is_output_node=True,
            inputs=[
                io.String.Input('style', default='', multiline=True),
                io.String.Input('lyrics', default='', multiline=True),
                io.Int.Input('seed', default=42, min=0, max=2147483647),
                io.Combo.Input('cot', options=['full', 'melody', 'off'], default='full'),
                io.String.Input('abc', default='', multiline=True),
                io.Float.Input('cfg_scale', default=-1, min=-1, max=20, tooltip='-1 uses the native mode default'),
                io.Boolean.Input('noncommercial', default=False, tooltip='I accept CC-BY-NC-4.0; noncommercial use only'),
            ],
            outputs=[io.Audio.Output(), io.Boolean.Output('truncated')],
        )


    @classmethod
    async def fingerprint_inputs(cls, **kwargs):
        def calculate():
            from ...model_identity import model_path_identity

            config = runtime_config()
            return (model_path_identity(config['model']), model_path_identity(config['vae']))

        return await asyncio.to_thread(calculate)

    @classmethod
    def execute(cls, style, lyrics, seed, cot, abc, cfg_scale, noncommercial):
        fields = dict(style=style, lyrics=lyrics, seed=seed, cot=cot, abc=abc,
                      cfg_scale=cfg_scale, noncommercial=noncommercial)
        native_request(fields)
        subfolder = 'yue2/' + uuid.uuid4().hex
        directory = Path(folder_paths.get_output_directory()) / subfolder
        directory.mkdir(parents=True)
        run_worker(fields, directory)
        completion = json.loads((directory / 'completion.json').read_text())
        samples, rate = sf.read(directory / 'native/audio.flac', dtype='float32', always_2d=True)
        audio = {'waveform': torch.from_numpy(samples.T.copy()).unsqueeze(0), 'sample_rate': rate}
        saved = ui.SavedAudios([ui.SavedResult('audio.flac', subfolder + '/native', io.FolderType.output)])
        return io.NodeOutput(audio, completion['truncated'], ui=saved)


nodes = [YuE2Generate]
