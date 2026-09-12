"""Native YuE2 request boundary for the isolated inference process."""
import json

import numpy as np


def generate(fields, directory, model, vae):
    # YuE2 is deliberately imported only inside its isolated interpreter.
    from yue2 import YuE2Pipeline

    request = native_request(fields)
    pipe = YuE2Pipeline.from_pretrained(model, vae=vae, device='cuda', local_files_only=True,
                                       backend='torch', quantization='none', memory_budget_gib=24,
                                       offload_ar=False)
    try:
        song = pipe(**request)
        if not np.isfinite(song.audio).all() or not np.isfinite(song.latents).all():
            raise ValueError('Non-finite YuE2 output')
        if not song.audio.size or float(np.sqrt(np.mean(song.audio.astype(np.float64) ** 2))) <= 1e-5:
            raise ValueError('Empty or silent YuE2 output')
        song.save_artifacts(directory / 'native')
        (directory / 'completion.json').write_text(json.dumps({'truncated': any(song.truncated.values()),
                                                              'model': 'm-a-p/YuE2-3B'}))
    finally:
        pipe.close()



def native_request(fields):
    if fields.get('noncommercial') is not True:
        raise ValueError('Explicit CC-BY-NC-4.0 noncommercial consent is required')
    request = {key: fields[key] for key in ('style', 'lyrics', 'seed', 'cot', 'cfg_scale')}
    if request['cfg_scale'] == -1:
        request['cfg_scale'] = None
    if fields.get('abc'):
        request['abc'] = fields['abc']
    return request


if __name__ == '__main__':
    import sys
    from pathlib import Path

    directory = Path(sys.argv[1])
    fields = json.loads((directory / 'intent.json').read_text())
    native_request(fields)
    generate(fields, directory, sys.argv[2], sys.argv[3])
