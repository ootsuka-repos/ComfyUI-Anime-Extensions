# ComfyUI-Anime-Extensions

Custom nodes for ComfyUI: Irodori-TTS speech synthesis, image-conditioned Character Voice, image analysis and segmentation, YuE2 music generation, text and video generation, comic page layout, and VRM processing.

Models and inference caches are automatically released when a workflow finishes, whether it succeeds or fails. Cleanup covers ComfyUI-managed models and node output caches, plus the extension's Irodori, image processing, and imgutils caches. History and saved files are retained; subsequent workflows reload models as needed.

Set `COMFYUI_EXTENSIONS_AUTO_UNLOAD=0` in the ComfyUI environment to disable automatic cleanup. External text services are outside this cleanup process. `Release Local Text Model` separately requests model unloading from a supported local service; set `COMFYUI_EXTENSIONS_OPENAI_RELEASE_BEFORE_COMFY=0` to disable that request. With automatic cleanup enabled, `keep_gpu` does not keep models resident across workflows.

## Nodes

### YuE2

`YuE2Generate` runs music generation from lyrics and style through the ComfyUI queue. It returns `AUDIO`, a truncation flag, and the original FLAC. Inference runs in a separate Python environment without an external YuE2 HTTP service. Model weights are **CC-BY-NC-4.0 (noncommercial use only)**. See [setup and inputs](docs/yue2.md).

### Irodori-TTS

Category: `ComfyUIExtensions/IrodoriTTS`. Character Voice Sampler is in its `Character Voice` subcategory.

| Node | Function |
| --- | --- |
| IrodoriTTS Model Loader | Select a checkpoint and configure model/codec devices, precision, and caching |
| IrodoriTTS Sampler | Generate speech from text, with optional reference audio, VoiceDesign, and sampling settings |
| Irodori Character Voice Sampler | Generate speech with a Character Voice checkpoint and an optional character image |
| IrodoriTTS Reference Audio | Use an audio or video file as a speaker reference; configure normalization and maximum reference duration |
| IrodoriTTS VoiceDesign Config | Set a voice description (`caption`) for compatible models |
| IrodoriTTS CFG Config | Configure guidance for text, speaker, caption, and character conditioning |
| IrodoriTTS Rescale Config | Configure rescaling, truncation, and speaker K/V correction |
| IrodoriTTS Schedule Config | Select the standard TTS sampling schedule (`linear` / `sway`) |
| IrodoriTTS Trim Tail Config | Configure trimming of trailing silence and flat audio sections |
| IrodoriTTS LoRA Stack | Build LoRA settings; see current limitations below |
| IrodoriTTS Save Audio | Save `AUDIO` as WAV, MP3, or FLAC |
| IrodoriTTS Emoji Picker | Copy performance-direction emoji to the clipboard; no input or output sockets |

### Image Tools

Categories: `ComfyUIExtensions/Image/analysis` and `ComfyUIExtensions/Image/portrait`.

| Node | Function and outputs |
| --- | --- |
| Image Analysis (imgutils) | Return `face`, `head`, `censor`, `nudenet`, `wd14`, or `ocr` analysis as a JSON string |
| WD14 ViT Scores | Return WD14 ViT tag scores as JSON; `tags` accepts a comma-separated list, or leave it blank for all tags |
| Heatmap Censor (WD14 ViT) | Apply blur or pixelation using tag heatmaps; return the processed `IMAGE` and labels as JSON |
| Object Mask (MobileSAM) | Generate a `MASK` for an object specified by a rectangle (`x0, y0, x1, y1`) |
| Character Segment (imgutils) | Extract a character using ISNetIS; return `IMAGE` and `foreground_mask`, with `scale` fixed at 1024 |
| Save RGBA | Save `IMAGE` and `foreground_mask` as a transparent PNG |

Analysis, WD14 scores, heatmaps, and MobileSAM accept one RGB image at a time. Character Segment and Save RGBA support image batches.

### Text, video, comics, VRM, and model loaders

Categories under `ComfyUIExtensions`: `Text`, `Video`, `Comic`, `Avatar`, and `model loaders`.

| Node | Function and outputs |
| --- | --- |
| Text Completion | Send a JSON request to an OpenAI-compatible service for text or image-conditioned generation; return a string |
| Release Local Text Model | Request model unloading from a supported local text service; return whether a model was released |
| Sol-H3-Spark | Generate video from text, first/last frames, or reference media; return an artifact manifest |
| Comic Page | Arrange an IMAGE batch into a page using column count, gutters, and reading direction |
| VRM Starter | Create a starter VRM, poster, and Blender scene for technical validation |
| VRM Dance | Extract 2D poses from an input video and animate an input VRM; export video and an editable Blender scene |
| CheckpointLoaderSimple / UNETLoader / CLIPLoader / VAELoader | Use standard ComfyUI loading with SHA-256 content fingerprints for cache invalidation |

The model loaders detect replaced weights even when filenames and sizes stay the same. Irodori, YuE2, and Sol-H3-Spark also use model content in cache decisions. Initial hashing of large weights may take time.

See [nodes, configuration, and model identity APIs](docs/media.md) and [Sol-H3-Spark setup](docs/sol-h3-spark.md).

## Requirements

- ComfyUI with the `comfy_api.latest` Extension API and cache-provider prompt lifecycle support
- A Python environment compatible with ComfyUI and [requirements.txt](requirements.txt)
- PyTorch 2.10 or later for Irodori-TTS v4.1-Small, as specified in `requirements.txt`
- Git and internet access to obtain dependencies and models

PyTorch and torchaudio come from the ComfyUI environment and are deliberately omitted from `requirements.txt`. Dependencies include torchcodec for audio, transformers and DACVAE for model inference, and dghs-imgutils, timm, and ultralytics for image processing.

## Installation

### Linux / virtual environment

Activate the Python environment used by ComfyUI, then run the following commands with the appropriate path:

```bash
cd /path/to/ComfyUI
git clone https://github.com/ootsuka-repos/ComfyUI-Anime-Extensions.git custom_nodes/ComfyUI-Anime-Extensions
python -m pip install -r custom_nodes/ComfyUI-Anime-Extensions/requirements.txt
```

### Windows portable (PowerShell)

Run from the portable installation root containing both `ComfyUI` and `python_embeded`:

```powershell
Set-Location -LiteralPath "C:\path\to\ComfyUI_windows_portable"
git clone https://github.com/ootsuka-repos/ComfyUI-Anime-Extensions.git .\ComfyUI\custom_nodes\ComfyUI-Anime-Extensions
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-Anime-Extensions\requirements.txt
```

Restart ComfyUI after installation. To update, run `git pull` in this repository, reinstall dependencies in the same Python environment, and restart ComfyUI.

## Additional setup by feature

Model weights and dedicated runtimes are not bundled. In addition to installing `requirements.txt`, prepare the components needed for your chosen features.

| Feature | Additional setup |
| --- | --- |
| Text Completion | A running OpenAI-compatible text/VLM service. Set `COMFYUI_EXTENSIONS_OPENAI_BASE_URL` (default: `http://127.0.0.1:8888/v1`), `COMFYUI_EXTENSIONS_OPENAI_MODEL`, and, if required, `COMFYUI_EXTENSIONS_OPENAI_API_KEY` in the ComfyUI environment |
| YuE2 | A separate Python environment, downloaded model/VAE weights, and `ComfyUI/runtimes/YuE2/comfyui.json`. Override the configuration path with `COMFYUI_YUE2_CONFIG`. [Details](docs/yue2.md) |
| Sol-H3-Spark | Dedicated runtimes, models, a container, and ffmpeg / ffprobe. Configuration defaults to `ComfyUI/runtimes/sol-h3-spark/config.json`; override it with `COMFYUI_EXTENSIONS_SOL_CONFIG`. [Details](docs/sol-h3-spark.md) |
| VRM | Blender with the VRM Add-on installed in its environment. Video processing also requires ffmpeg. Select Blender through `COMFYUI_EXTENSIONS_BLENDER` or PATH |

Configure service endpoints, credentials, and runtime paths on the ComfyUI host.

## Model locations and downloads

The `Sol-H3-Spark` node handles video and music-video segments, from H3 drafts through LTX-2.5 finishing. Clients invoke it through workflow JSON. See [Sol-H3-Spark](docs/sol-h3-spark.md) for runtime setup, weights, and LTX-2.5 access approval.

Place Irodori-TTS and Character Voice checkpoints under ComfyUI's `models/checkpoints`. Model weights are not included in this repository.

```text
ComfyUI/
└─ models/
   ├─ checkpoints/
   │  └─ irodori_tts/
   │     ├─ model.safetensors
   │     └─ codec/
   │        └─ weights.pth    # optional: codec fine-tuned with this checkpoint
   ├─ irodori/                 # Character Voice downloads
   │  ├─ codecs/
   │  ├─ tokenizers/
   │  └─ image_encoders/
   └─ huggingface/
      └─ hub/                 # Image processing model downloads
```

- Model Loader selects the codec from the checkpoint's `latent_dim`: 32 uses `Aratako/Semantic-DACVAE-Japanese-32dim`, and 128 uses `facebook/dacvae-watermarked`. Other values raise an error.
- A `codec/weights.pth` sitting next to the checkpoint overrides that routing and is loaded instead. A codec fine-tuned alongside a checkpoint produces a latent space that the public codec cannot decode even at the same `latent_dim`, so ship the pair together.
- Standard TTS downloads its tokenizer and codec as needed. It follows each library's cache settings rather than explicitly using `models/irodori`.
- Character Voice downloads tokenizers, codecs, and image encoders under `models/irodori`.
- Image processing sets the Hugging Face cache to `models/huggingface/hub` at runtime. WD14 ViT uses `SmilingWolf/wd-vit-tagger-v3`; MobileSAM uses `dhkim2810/MobileSAM`. Additional imgutils models are downloaded as needed. This cache configuration is shared within the ComfyUI process.

## Basic usage

### Standard TTS and VoiceDesign

1. Select a compatible checkpoint in `IrodoriTTS Model Loader` and connect `irodori_model_config` to `IrodoriTTS Sampler`.
2. Enter the speech text in the sampler's `text` input. Optionally connect Reference Audio to `ref_config` and VoiceDesign Config to `voice_design_config`. VoiceDesign requires a checkpoint that supports caption conditioning.
3. Connect `audio` to an audio preview node or `IrodoriTTS Save Audio`, then run the workflow. The default save prefix is `output/audio/IrodoriTTS`.

Standard TTS estimates duration when the checkpoint has a duration predictor. Without one, it falls back to 30 seconds and ignores `duration_scale`. The scale defaults to 1.0, accepts 0.1–3.0, and multiplies the predicted duration; generation is limited to 0.5–30 seconds. There is no manual duration input. `trim_tail` is enabled by default, so the final audio may be shorter after trimming.

Reference Audio accepts audio/video files in ComfyUI's `input` directory and supports audio uploads. `max_ref_seconds` accepts 1–120 seconds and defaults to 120. Audio extraction from video uses imageio-ffmpeg or ffmpeg.

### Character Voice

Select a Character Voice checkpoint in the same Model Loader and connect it to `Irodori Character Voice Sampler`. `character_image` is optional; for image batches, only the first image is used.

Set duration with `seconds` (default: 30; range: 1–120). CFG, Rescale, and Trim Tail settings are supported. There are no reference audio, VoiceDesign, LoRA, or Schedule inputs. Speaker/caption CFG settings and speaker K/V correction are not used.

### Character segmentation and transparent PNGs

Connect `image` and `foreground_mask` from `Character Segment (imgutils)` to the matching inputs of `Save RGBA`. A mask value of 1 is opaque; 0 is transparent. The default save prefix is `output/image_tools/portrait`.

For MobileSAM, specify the rectangle in pixel coordinates of the original image. Invalid rectangles or missing masks produce an all-zero mask.

### Text, video, comics, and VRM

- **Text:** Set `Text Completion` → `request_json` to `{"user_prompt":"Write a short line of dialogue.","system_prompt":"Reply in English.","log_tag":"example"}`. Use `image_data_urls` for image conditioning and `response_format` for structured output. Tool execution is disabled in this node.
- **Comics:** Feed an IMAGE batch of equally sized panels into `Comic Page`, then connect `page` to an image-saving node. This node arranges panels; drawing, speech bubbles, and lettering are separate steps.
- **Video:** Select a `task` in `Sol-H3-Spark`. `t2va` accepts text only; `fl2va` uses a first or last frame; `ref2va` requires references containing an image or video. Place input files inside ComfyUI's `input` directory.
- **VRM:** `VRM Starter` creates a model for technical validation. For `VRM Dance`, specify `avatar_file` and `source_video` from `input`. Results are saved to `output/avatar/<id>/`. Source audio, when present, is included in the output video.

Display names, node IDs, categories, environment variables, and API paths use the extension's current naming. There are no aliases for legacy identifiers. Re-add affected nodes in existing workflows and update host settings and API clients to the names documented here.

## Current limitations

- YuE2 does not download models automatically. Use requires explicit agreement to noncommercial use through `noncommercial=true`. A `truncated=true` output indicates that generation was cut short.
- Sol-H3-Spark generates 121 frames at 24 fps per run, with a selectable duration of 4–approximately 5.04 seconds. Clients must split and join longer works.
- VRM Dance uses 2D pose-based animation. It saves an editable scene and video, but does not export VRMA.
- **The LoRA UI and standard TTS runtime expect different formats.** LoRA Stack passes paths selected from files under `models/loras`, while the runtime requires an adapter directory containing `adapter_config.json` and weights. Selecting a LoRA file alone is therefore insufficient.
- Standard TTS accepts at most one LoRA. LoRA Stack's `strength` is not passed to the inference request, so strength changes have no effect. Dynamic runtime LoRA loading cannot be combined with `compile_model=True`.
- `runtime_cache_policy=offload_after_use` differs by sampler: standard TTS releases the cached runtime, while Character Voice moves it to CPU. `keep_gpu` retains it and `unload_after_use` discards it, subject to workflow-end cleanup described above.

## Troubleshooting

- **Missing nodes:** Check ComfyUI's startup log for import errors and install dependencies into the Python environment ComfyUI actually uses.
- **Missing checkpoints:** Place them under `models/checkpoints`, then refresh the model list or restart ComfyUI.
- **`config_json` or `latent_dim` errors:** Use a compatible checkpoint containing Irodori configuration metadata. Model Loader also lists checkpoints intended for other models.
- **Slow first run:** Initial downloads and model initialization take time. Check console progress and network access.
- **Insufficient VRAM for speech:** Use `codec_device=cpu` and `codec_precision=fp32` in Model Loader, plus `batch_size=1` and `decode_mode=sequential` in the sampler. Select `runtime_cache_policy=offload_after_use` or `unload_after_use` to release resources after generation.
- **Image analysis batch errors:** Pass one image at a time to analysis, heatmap, and MobileSAM nodes.

## License

Code is provided under the [MIT License](LICENSE). Downloaded models and dependencies retain their respective licenses.
