# Sol-H3-Spark for Doujin Forge

`ComfyUIExtensions.Forge.SolH3` runs the official Sol-H3-Spark pipeline as owned
child processes. Model execution belongs here; the Codex/Claude plugin contains
editable workflow JSON and the existing production planning/composition library.
Legacy MiniMax pruned/fused/VDN/SLA graphs are not used as fallbacks.

Upstream: [Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/),
Sana `sol-engine` revision `8e0db4fa562d727ea28b8d63015c196db7d97cae`.
Use the whole Sana checkout: the package imports shared Sol attention code.
The official recipe targets Linux aarch64 and one NVIDIA GB10 / DGX Spark.

## Prepare the ComfyUI host

Keep generated dependencies outside this extension checkout and the plugin.
The default layout is:

```text
ComfyUI/runtimes/
  Sana/                         # whole pinned upstream checkout
  sol-h3-spark/
    sources/                    # pinned FastVideo, FA4, ComfyUI, LTX-2, upscaler
    stage1-env/                 # separate AI dependencies
    stage2-env/
    dependencies/fa4/
    dependencies/offline-context/
    fixed-prompt.pt             # prescribed generic Gemma/connector cache
    paths-t2va.json
    paths-fl2va.json
    paths-ref2va.json
    config.json                 # extension host configuration
    jobs/                       # staged media, worker logs and receipts
```

1. Clone Sana and check out the exact revision above. From
   `models/minimax_h3/Sol-H3-Spark`, read `docs/setup.md`, `docs/validation.md`,
   `configs/dependencies.json` and `configs/checkpoints.json`.
2. Obtain the listed weights using `download_checkpoints.py` with `--task t2va`,
   `--task fl2va` and `--task ref2va`, sharing one `--output-dir`. Include the offline
   weights for building the generic prompt cache. Downloads are pinned; do not fetch
   the entire unfiltered MiniMax-H3 repository. The three-task native H3 subset is
   about 134 GiB, excluding the other models. Existing HF cache paths can instead be
   supplied through the flat `--paths` overrides accepted by `prepare.py`.
3. LTX-2.5 requires approved [Hugging Face access](https://huggingface.co/Lightricks/LTX-2.5)
   and local `hf auth login`. Credentials belong to the host; never put them in
   workflow JSON or send them in chat. A 403 means setup cannot proceed to inference.
4. Fetch sources using upstream `prepare.py --fetch-sources`. It validates source
   pins and applies the exact declared FastVideo Spark loader patch. Install
   FastVideo/fastvideo-kernel into Stage1, and LTX core/pipelines/kernels into Stage2.
   Compile kernels for this GPU; Stage2 requires `all2all_cpp` even with one GPU.
   Follow upstream's separate FA4 and offline-context dependency targets.
5. Build `Dockerfile.qwen` as `sol-h3-spark-qwen`; it retains the pinned NGC Torch
   stack and ComfyUI v0.30.0. This source is separate from the live ComfyUI server.
   The upstream `qwen_python.sh` wrapper runs the container with no network and
   the current user's UID/GID. Its package/runtime/weight/source mounts preserve
   absolute paths. The weight mount should contain checkpoint files, not credentials.
6. Run `prepare.py` for each task with the corresponding interpreters, sources and
   flat weight overrides. Write `paths-t2va.json`, `paths-fl2va.json` and
   `paths-ref2va.json` under the runtime directory. For a first setup use
   `--cache-pending`, then build `fixed-prompt.pt` with upstream `runtime.cache_builder`
   using the exact INT8 Gemma and INT8 dev-connector checkpoints. Arbitrary or BF16
   text embeddings cannot substitute for this cache.
7. Connect the manifests to the extension:

   ```bash
   python3 scripts/configure_sol_h3.py \
     --comfy-root /absolute/ComfyUI \
     --sana-root /absolute/ComfyUI/runtimes/Sana \
     --runtime-root /absolute/ComfyUI/runtimes/sol-h3-spark \
     --weights-root /absolute/huggingface/hub
   ```

   To record a setup still downloading, add `--allow-pending`; this explicitly does
   not make inference ready. `COMFYUI_FORGE_SOL_CONFIG` can select another config path.
   Restart ComfyUI after installing/updating the extension, once its queue is idle.

Upstream `*-observed.txt` files are version inventories, not resolved install locks.
Do not install these conflicting stacks into live ComfyUI's Python or the plugin.
Model files and the generic cache are never downloaded during a node invocation.

## Workflow inputs and outputs

The node has `task`, `prompt`, `seed`, `duration`, `output_prefix`, optional
`first_frame` / `last_frame` filenames and `reference_images`, `reference_videos`,
`reference_audios` as JSON strings containing uploaded filename lists.
Comfy input files and annotated `name [temp]` files are accepted, copied into the
dedicated runtime job so that the Qwen container can see them.

- T2VA accepts text only.
- FL2VA uses first frame, last frame, or both.
- Ref2VA requires at least one image/video, with at most 9 images, 3 video files,
  3 audio files and 12 total references. The upstream media preparer also validates
  soundtrack counts and prompt numbering. The node orders images, videos and audio.

All tasks use the fixed native 1344×768 / 121-frame / 24-fps audio/video recipe.
`duration` selects a 4–121/24 second delivery after generation; both generated
endpoints are retained. Longer MVs use the plugin's timeline renderer, splitting
intervals at 120 delivery frames and chaining continuation frames.
Output history `files` contains `video.mp4` and `sol-report.json`.

`GET /ComfyUIExtensions/Forge/SolH3/status` checks the frozen recipe and task path
manifests without importing GPU libraries. The plugin checks it before recording
a submit intent. Missing prerequisites therefore do not queue work or leave an
ambiguous submission receipt. `prepared` means filesystem preparation, not a
successful GPU inference test.

Every node invocation includes loading and a full upstream warmup, then one formal
request. `total_node_seconds` includes this cost; `startup_and_warmup_s` and the
formal request's `e2e_s` are recorded separately. The published hot benchmark is
not the per-node latency of this integration. Workers close after each invocation,
also on cancellation; there is no hidden permanent GPU service.

Comfy's own resident models are unloaded before launching Sol. The extension does
not stop independent services. A large external LLM may need to be paused by the
host operator after verifying that it is idle; restore it when GPU testing ends.
Failed jobs retain request JSON, `process.log` and worker logs under `jobs/`.
Successful jobs remove staged media and large transient latent tensors while
retaining receipts and logs. The model cache is shared and preserved.

## Verification status

The node registration, workflow bindings and production call paths must be checked
against the live ComfyUI schema. These checks do not prove model inference.
A Sol pass requires the real H3 and LTX workers, full warmup, a successful formal
request and decoded 121-frame native video with audio. Retain failures separately
from subsequent successful retries. In particular, previous native H3 inference
receipts are not Sol-H3-Spark validation.

Code is MIT under this repository's license. Upstream code and downloaded weights
retain their own terms; see the pinned package's `THIRD_PARTY_NOTICES.md`.
