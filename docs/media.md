# Text, video, comics, VRM, and model loaders

Client plugins own workflow JSON and submit it directly to the ComfyUI API. This extension runs inference and dedicated Python processing; it does not start a separate Engine or Publisher API.

| Node ID | Inputs and outputs |
| --- | --- |
| `ComfyUIExtensions.ComicPage` | Arrange an IMAGE batch into a page with configurable columns, gutters, and right-to-left order |
| `ComfyUIExtensions.VRMStarter` | Create a VRM, poster, and Blender scene for technical validation from a name |
| `ComfyUIExtensions.VRMDance` | Extract RTMW 2D poses from a video and animate a VRM from `input`; export video and an editable Blender scene |

The Blender bridge and pose extraction code are included in this extension and do not import the previous application repository.

VRM processing requires Blender, the VRM Add-on, and ffmpeg on the ComfyUI host. Set `COMFYUI_EXTENSIONS_BLENDER` to select Blender; the default is `blender` on PATH. Install the VRM Add-on in that Blender installation's script search path. RTMW uses `rtmlib` and `onnxruntime` in the existing ComfyUI Python environment. No environment or models are created inside the client plugin.

Input files must be inside `ComfyUI/input`. Results and processing logs are saved to `output/avatar/<id>/`; the history entry's `files` field describes downloadable artifacts. Cancellation stops Blender and ffmpeg, and pose extraction checks for cancellation between frames. Source video audio, when present, is included in the output video.

The client production library or calling agent handles planning, splitting, joining, and delivery validation. Do not submit a workflow to the same ComfyUI queue from inside a node and wait for it to finish.

## Text and vision generation

For canvas workflows, add `ComfyUIExtensions.TextRequest` (**Text Request**) and connect its `request_json` output to `ComfyUIExtensions.TextCompletion` (**Text Completion**). Enter ordinary text in `user_prompt` and `system_prompt`; quotes, line breaks, and Japanese text are encoded automatically. The optional `model` field overrides the host model; empty or whitespace-only values use the host default. Connect Text Completion's `text` output to ComfyUI's **Preview as Text** to display it on the canvas.

Text Completion also continues to accept direct JSON from client plugins. Required fields are `user_prompt`, `system_prompt`, and `log_tag` strings. The backend supports structured output, image inputs, generation parameters, and local model reloading. Tool execution is disabled in the node. `ComfyUIExtensions.TextModelRelease` requests local model unloading from the ComfyUI host.

Set `COMFYUI_EXTENSIONS_OPENAI_BASE_URL`, `COMFYUI_EXTENSIONS_OPENAI_MODEL`, and, if required, `COMFYUI_EXTENSIONS_OPENAI_API_KEY` in the ComfyUI process environment. The default endpoint is `http://127.0.0.1:8888/v1`. Endpoints and credentials are not workflow inputs. The selected text/VLM service must already be running; arbitrary providers are not started automatically.

Install `requirements-text.txt` in the ComfyUI environment for text-only usage. The HTTP backend loads at execution time, so discovering other nodes does not require `httpx`.

## Extension initialization

The extension uses the ComfyUI V3 Extension API. `comfy_entrypoint()` constructs the extension; `on_load()` registers the cache provider and HTTP routes. Importing the package or enumerating schemas does not require a running `PromptServer` or load the optional inference backends.

Route registration is centralized in `py/routes.py`. Existing endpoints are unchanged: `GET /yue2/status`, `GET /ComfyUIExtensions/SolH3/status`, `POST /ComfyUIExtensions/sol-model-identity`, and `POST /ComfyUIExtensions/model-identity`. Repeated registration on the same route table does not duplicate routes. Runtime status and content hashing run outside the HTTP event loop.

Media registration is separate from implementation: `avatar.py` owns VRM operations, `comic.py` owns page layout, and `text.py` / `sol_h3.py` own their respective nodes. Irodori device choices live in `py/runtime_devices.py`, so the model-loader schema does not import a speech model.

## Detecting replaced models

`ComfyUIExtensions_CheckpointLoaderSimple`, `ComfyUIExtensions_UNETLoader`, `ComfyUIExtensions_CLIPLoader`, and `ComfyUIExtensions_VAELoader` use the standard loaders' inputs, outputs, and loading logic, with SHA-256 file-content fingerprints added to ComfyUI's cache decisions. Replaced weights trigger reloading even when names and sizes match. A timestamp-only change with identical content does not. Irodori ModelLoader and the standard TTS resident runtime also check checkpoint and codec contents.

`POST /ComfyUIExtensions/model-identity` accepts:

```json
{"models":[{"category":"checkpoints","name":"model.safetensors"}]}
```

It returns sizes and SHA-256 hashes for registered models. Add `"include_irodori_codec": true` to include the already downloaded codec associated with an Irodori checkpoint. The endpoint does not load or download models. Hashing runs outside the event loop, and results are reused until file metadata changes. Missing files or changes during inspection cause errors; the endpoint does not fall back to filename-only identification.

Sol-H3-Spark and YuE2 also include weights, codecs, tokenizers, and related files in cache decisions. Each node hashes files in a separate thread. Reading large video models can take time on the first check after startup or after files change. Unchanged files reuse cached hashes.

`POST /ComfyUIExtensions/sol-model-identity` accepts `{"task":"fl2va"}` and returns a model identity in the form `{"version":1,"sha256":"…"}`. Supported tasks are `t2va`, `fl2va`, and `ref2va`. This endpoint does not download models or run inference. Music-video clients save the identity at production start and pass it to the Sol node's optional `expected_model_identity` input. The node checks it immediately before execution and refuses to run with different weights if models were replaced while the job was queued.

VRM Dance also fingerprints the source video and VRM contents, so replacing a same-named file triggers execution. If the source audio is shorter than the generated animation, silence is appended to preserve the full video duration.
