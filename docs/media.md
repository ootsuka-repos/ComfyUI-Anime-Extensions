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

`ComfyUIExtensions.TextCompletion` calls the configured provider from this extension. Client plugins retain prompts and validation logic for scripts, structure, and proofreading, then exchange JSON requests and results with the node. The backend supports structured output, image inputs, generation parameters, and local model reloading. Tool execution is disabled in the node. `ComfyUIExtensions.TextModelRelease` requests local model unloading from the ComfyUI host.

Set `COMFYUI_EXTENSIONS_OPENAI_BASE_URL`, `COMFYUI_EXTENSIONS_OPENAI_MODEL`, and, if required, `COMFYUI_EXTENSIONS_OPENAI_API_KEY` in the ComfyUI process environment. The default endpoint is `http://127.0.0.1:8888/v1`. Endpoints and credentials are not workflow inputs. The selected text/VLM service must already be running; arbitrary providers are not started automatically.

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
