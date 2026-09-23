# Sol-H3-Spark for ComfyUI-Anime-Extensions

`ComfyUIExtensions.SolH3` runs the official Sol-H3-Spark pipeline as owned
child processes. Model execution belongs here; the Doujin Forge plugin (an omp
extension package) contains editable workflow JSON and the existing production
planning/composition library. Legacy MiniMax pruned/fused/VDN/SLA graphs are not
used as fallbacks.

Upstream: [Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/),
Sana `sol-engine` revision `8e0db4fa562d727ea28b8d63015c196db7d97cae`.
Use the whole Sana checkout: the package imports shared Sol attention code.
The official recipe targets Linux aarch64 and one NVIDIA GB10 / DGX Spark.

Sol's Qwen3-VL-32B NVFP4 AWQ model encodes prompts and reference media into
H3-specific token features (`[1, tokens, 5120]`) and MiniMax modality tags. It is
separate from the external LLM used for writing scenarios and prompts. The existing
`qwen38-flashnext-serving` Qwen3.8 chat/vision API remains that writing service;
its generated text or general embeddings cannot replace the native H3 features.
The Sol container exposes no additional chat API and exits with its owning job.

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
   Preserve the official Gemma and connector filenames: the offline cache builder
   checks their names after resolving symlinks. HF snapshot symlinks resolve to
   hash-named blobs; use the upstream local download layout, or hard-link verified
   blobs to their official filenames under the runtime's checkpoint directory.
   Set `gemma_tokenizer` to the same named file as `offline_gemma`.
   Keep native H3 `vae/config.json` even for T2VA: FastVideo initializes video
   geometry from it before lazy loading. T2VA does not need native video VAE
   weights; FL2VA and Ref2VA require every VAE shard for reference encoding.
3. LTX-2.5 requires approved [Hugging Face access](https://huggingface.co/Lightricks/LTX-2.5)
   and local `hf auth login`. Credentials belong to the host; never put them in
   workflow JSON or send them in chat. A 403 means setup cannot proceed to inference.
4. Fetch sources using upstream `prepare.py --fetch-sources`. It validates source
   pins and applies the exact declared FastVideo Spark loader patch. Install
   FastVideo/fastvideo-kernel into Stage1, and LTX core/pipelines/kernels into Stage2.
   Compile kernels for this GPU; Stage2 requires `all2all_cpp` even with one GPU.
   Follow upstream's separate FA4 and offline-context dependency targets.
5. From this extension checkout, run
   `docker build -f docker/sol-h3-qwen.Dockerfile -t sol-h3-spark-qwen .`.
   It retains the pinned NGC Torch stack and ComfyUI v0.30.0, and supplies the
   missing native audio dependency described below. This source is separate from
   the live ComfyUI server.
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
   not make inference ready. `COMFYUI_EXTENSIONS_SOL_CONFIG` can select another config path.
   Restart ComfyUI after installing/updating the extension, once its queue is idle.

Upstream `*-observed.txt` files are version inventories, not resolved install locks.
Do not install these conflicting stacks into live ComfyUI's Python or the plugin.
Model files and the generic cache are never downloaded during a node invocation.

On this GB10 host, the fresh build needed dependency adjustments. The recorded
FA4 `apache-tvm-ffi==0.1.13rc1` is absent from PyPI and upstream tags; the published
`0.1.13.post3` satisfies Quack 0.5.3's supported range and passed real FA4 GPU
attention comparisons against FP32 reference math, with and without a causal mask.
Keep the FA4 CuTe version at `4.6.0.dev0` in its separate target.

Stage2's Torch CUDA 13.2 headers require a matching compiler. System CUDA 13.0
caused an incompatible-header build failure. The working build installed
`nvidia-cuda-nvcc==13.2.78`, `nvidia-nvvm==13.2.78`,
`nvidia-cuda-crt==13.2.78` and `nvidia-cuda-cccl==13.2.75` into Stage2 only,
then set `CUDA_HOME` to that environment's `site-packages/nvidia/cu13` directory
and used its `bin/nvcc`. The live ComfyUI environment and system CUDA remain
independent. Both environment import/GPU checks passed; these are preparation
checks, not a claim of completed Sol model inference.

Upstream's Qwen Dockerfile excludes TorchAudio while the NGC image does not
provide it. ComfyUI's `comfy.sd` import therefore fails. The 2.10 release wheel
and source also require a newer Torch C++ interface than NGC 25.11 contains.
The extension Dockerfile compiles TorchAudio 2.9.1 revision
`a224ab24a7f4797f6707051257265e223e12576f` against NGC's existing headers, with
optional native audio CUDA kernels disabled. This preserves the prescribed Torch
and Qwen GPU stack. The real Qwen wrapper passed `comfy.sd`/MiniMax imports,
audio resampling and a BF16 GPU matrix multiplication. Stage1 owns actual audio
sample encoding in its independent environment.

## Workflow inputs and outputs

The node has `task`, `prompt`, `seed`, `duration`, `output_prefix`, optional
`first_frame` / `last_frame` filenames and `reference_images`, `reference_videos`,
`reference_audios` as JSON strings containing uploaded filename lists.
Comfy input files and annotated `name [temp]` files are accepted, copied into the
dedicated runtime job so that the Qwen container can see them.

- T2VA accepts text only.
- FL2VA uses first frame, last frame, or both.
  Match the 1344×768 canvas when possible. Upstream stretches the first supplied
  keyframe; a portrait input can distort. Ref2VA preserves the aspect ratio of
  image references and is preferable when an image only specifies appearance.
- Ref2VA requires at least one image/video, with at most 9 images, 3 video files,
  3 audio files and 12 total references. The upstream media preparer also validates
  soundtrack counts and prompt numbering. The node orders images, videos and audio.

All tasks use the fixed native 1344×768 / 121-frame / 24-fps audio/video recipe.
`duration` selects a 4–121/24 second delivery after generation; both generated
endpoints are retained. Longer MVs use the plugin's timeline renderer, splitting
intervals at 120 delivery frames and chaining continuation frames.
Output history `files` contains `video.mp4` and `sol-report.json`.

`GET /ComfyUIExtensions/SolH3/status` checks the frozen recipe, task paths,
H3 VAE metadata and task-specific H3 shards without importing GPU libraries.
The node repeats these checks before creating a job. The plugin checks it before recording
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
Comfy's allocator environment is removed from Sol's child environment: its
`cudaMallocAsync` backend cannot run the native H3 VAE's CUDA graph pool checks.
Sol then uses the default native allocator and its prescribed Stage2 expandable
segments. This does not change the allocator used by the live ComfyUI process.
Failed jobs retain request JSON, `process.log` and worker logs under `jobs/`.
Successful jobs remove staged media and large transient latent tensors while
retaining receipts and logs. The model cache is shared and preserved.

After verifying the replacement, legacy Comfy-format H3 pruned/fused checkpoints,
their old VAE/encoder/LoRA and VDN weights can be removed when no other workflow uses
them. Preserve every checkpoint in the Sol manifests, including native H3 partitions.
Do not delete the whole `Comfy-Org/MiniMax-H3` HF cache: Sol's NVFP4 AWQ encoder
shares that repository with the obsolete Comfy-format weights. Removing a snapshot
symlink alone does not reclaim its blob's disk space.

## Verification status

The node registration, workflow bindings and production call paths must be checked
against the live ComfyUI schema. These checks do not prove model inference.
A Sol pass requires the real H3 and LTX workers, full warmup, a successful formal
request and decoded 121-frame native video with audio. Retain failures separately
from subsequent successful retries. In particular, previous native H3 inference
receipts are not Sol-H3-Spark validation.

On 2026-09-13–14 (JST), the plugin's T2VA, FL2VA, Ref2VA and MV production commands
all passed on this GB10 host. Native outputs decoded to 121 frames at 1344×768/24 fps
with finite, non-silent audio. Formal generation took 62.8–72.8 seconds; complete
commands including model loading, full warmup and delivery took 459.9–552.6 seconds.
The three video modes delivered 120 frames; the MV delivered 96 frames with its supplied
music. Frame review found subtitle-like text in T2VA and changes to reference details
in conditioned tasks. Audio was checked numerically, without subjective listening.
Detailed receipts are retained on the validation host and are not bundled here.

Code is MIT under this repository's license. Upstream code and downloaded weights
retain their own terms; see the pinned package's `THIRD_PARTY_NOTICES.md`.
