# YuE2: local music generation in ComfyUI

`YuE2Generate` runs through the ComfyUI queue, returns `AUDIO` and `truncated`, and provides a preview of the original 24-bit FLAC. It does not use an external YuE2 HTTP service. To avoid Torch/Transformers dependency conflicts, YuE2 inference runs in a separate Python child process. Interrupting ComfyUI stops the child process, and models are released when it exits. Do not install YuE2 in ComfyUI's shared Python environment.

## Local requirements

- Upstream: [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
- Validated version: YuE2 inference 0.1.6, upstream revision `88da114a67df892af0329472073b96a5ef700b93`.
- Separate Python 3.12 environment with Torch 2.10.0, Transformers 4.57.6, and CUDA BF16 support.
- YuE2-3B revision: `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`.
- YuE2-Vae revision: `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`.
- Models must be downloaded in advance. The node neither downloads models nor sends requests externally; it uses `local_files_only=True` and an offline environment. Upstream code validates the weight manifest.

Create `ComfyUI/runtimes/YuE2/comfyui.json` locally. Override this location with `COMFYUI_YUE2_CONFIG` in the ComfyUI startup environment. This is host-admin configuration; executable and output paths are not accepted as node inputs.

```json
{
  "python": "/absolute/path/to/YuE2/.venv/bin/python",
  "model": "/absolute/path/to/YuE2-3B/snapshot",
  "vae": "/absolute/path/to/YuE2-Vae/snapshot"
}
```

Restart ComfyUI when no jobs are queued or running. Check registration with `GET /object_info/YuE2Generate`, and inspect interpreter and weight locations and sizes with `GET /yue2/status`. The status endpoint does not guarantee successful model loading or music quality. Models are loaded per job and do not remain resident, so `model_loaded` is false.

## Inputs and outputs

- `style` / `lyrics`: supplied separately and passed to upstream unchanged, including line breaks.
- `cot`: `full` (default), `melody`, or `off`. `abc` is optional upstream score conditioning.
- `cfg_scale`: `-1` selects the upstream mode-specific default (`None`), without overriding it with a custom CFG value.
- `seed`: defaults to 42. The node retains upstream's non-quantized, 32-step ODE defaults and generation limits.
- `noncommercial`: defaults to false. **Set true only after explicitly agreeing to noncommercial use under CC-BY-NC-4.0.**

Weights by Multimodal Art Projection / the YuE2 authors are licensed under **CC BY-NC 4.0**. This differs from the upstream code's Apache 2.0 license. Japanese singing quality has not been validated.

The node saves `intent.json`, `worker.log`, `completion.json`, and `native/` under `ComfyUI/output/yue2/<random-id>/`. Native artifacts include the original audio, ABC, tokens, latents, request/configuration data, and results containing weight identities and hashes. Non-finite, empty, or silent model outputs fail validation. `truncated` is the logical OR of ABC and semantic truncation. Playable audio with `truncated=true` should not be treated as a completed full song.

Use the standard ComfyUI API flow: `/prompt` → `/history/{prompt_id}` → `/view`. In-memory history may be lost on restart. Clients should persist completion results and avoid automatically regenerating when history is unavailable. Original audio and native artifacts are retained.

## Tests

Run `tests/test_yue2.py` with ComfyUI's Python interpreter. These unit tests replace inference and do not demonstrate generation with real models. Run test files in separate processes because the existing tests modify `sys.modules`.
