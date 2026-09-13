"""Sol-H3-Spark execution owned by ComfyUI, using the pinned upstream recipe."""
from __future__ import annotations

import asyncio

import json
import importlib.util
import math
import re
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import folder_paths
from comfy import model_management
from comfy_api.latest import io

SANA_REVISION = "8e0db4fa562d727ea28b8d63015c196db7d97cae"
NATIVE_FRAMES, FPS = 121, 24


def runtime_config() -> dict:
    default = Path(folder_paths.base_path) / "runtimes/sol-h3-spark/config.json"
    filename = Path(os.environ.get("COMFYUI_EXTENSIONS_SOL_CONFIG", str(default))).expanduser()
    if not filename.is_file():
        raise RuntimeError(f"Sol-H3-Spark runtime is not prepared: {filename}. See docs/sol-h3-spark.md.")
    config = json.loads(filename.read_text())
    if config.get("schema") != 1:
        raise ValueError("Unsupported Sol-H3-Spark host configuration")
    package = Path(config["package"]).expanduser().resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(package), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if revision != SANA_REVISION:
        raise ValueError(f"Sol-H3-Spark requires Sana revision {SANA_REVISION}; found {revision}")
    config["package"] = str(package)
    root = Path(config["runtime_root"]).expanduser().resolve(strict=True)
    if not root.is_dir() or root == Path("/"):
        raise ValueError("Use a dedicated Sol-H3-Spark runtime directory")
    config["runtime_root"] = str(root)
    return config


def uploaded_file(name: str) -> Path:
    """Accept Comfy input/temp annotations without exposing arbitrary host paths."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Reference filenames must be nonempty strings")
    path = Path(folder_paths.get_annotated_filepath(name)).resolve()
    roots = [Path(folder_paths.get_input_directory()).resolve(),
             Path(folder_paths.get_temp_directory()).resolve()]
    if not any(root in path.parents for root in roots) or not path.is_file():
        raise ValueError(f"Use an uploaded ComfyUI input/temp file: {name}")
    return path


def task_paths(config: dict, task: str) -> dict:
    filename = Path(config["package"]) / "runtime/config.py"
    spec = importlib.util.spec_from_file_location("extensions_sol_upstream_config", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_recipe(task)
    paths = module.load_paths(config["paths"][task], task=task)
    root = Path(paths["h3_model"])
    # FastVideo initializes video geometry from this metadata even when T2VA
    # skips native H3 VAE encode/decode and therefore needs no VAE weights.
    json.loads((root / "vae/config.json").read_text())
    components = ["transformer_ref" if task == "ref2va" else "transformer"]
    if task != "t2va":
        components.append("vae")
    for component in components:
        directory = root / component
        json.loads((directory / "config.json").read_text())
        index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            if not (directory / shard).is_file():
                raise FileNotFoundError(f"Missing Sol {task} checkpoint shard: {directory / shard}")
    return paths


def model_fingerprint(task: str, *, resolved_paths: dict | None = None) -> str:
    """Fingerprint only the selected Sol task's local inference assets."""
    from ...model_identity import model_path_identity, _digest

    paths = resolved_paths if resolved_paths is not None else task_paths(runtime_config(), task)
    h3 = Path(paths["h3_model"])
    assets = {
        "h3_transformer": h3 / ("transformer_ref" if task == "ref2va" else "transformer"),
        "h3_vae_config": h3 / "vae/config.json",
    }
    if task != "t2va":
        assets["h3_vae"] = h3 / "vae"
    for key in ("ref2va_lora" if task == "ref2va" else "vsa_lora", "transformer",
                "refiner_lora", "output_video_vae", "audio_vae", "adapter_dir",
                "h3_upscaler_checkpoint", "qwen_checkpoint", "prompt_cache"):
        assets[key] = Path(paths[key])
    return _digest({key: model_path_identity(path) for key, path in sorted(assets.items())})


def require_model_identity(task: str, expected: str, *, resolved_paths: dict | None = None) -> None:
    """Reject changed weights before admitting a pinned project to a worker."""
    if not isinstance(expected, str) or (expected and re.fullmatch(r"[0-9a-f]{64}", expected) is None):
        raise ValueError("expected_model_identity must be empty or a SHA-256 digest")
    if expected and model_fingerprint(task, resolved_paths=resolved_paths) != expected:
        raise ValueError("Sol model content changed from this project's pinned identity; restore the original weights or use a new project")


def runtime_status() -> dict:
    """Read the pinned recipe and task paths without loading GPU libraries."""
    result = {"implementation": "Sol-H3-Spark", "revision": SANA_REVISION, "tasks": {},
              "validation": "Filesystem and frozen recipe only; does not prove GPU inference"}
    try:
        config = runtime_config()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
        return result
    for task in ("t2va", "fl2va", "ref2va"):
        try:
            task_paths(config, task)
            result["tasks"][task] = {"prepared": True}
        except (OSError, ValueError, KeyError) as error:
            result["tasks"][task] = {"prepared": False, "error": str(error)}
    return result


def reference_list(raw: str, limit: int) -> list[str]:
    values = json.loads(raw)
    if (not isinstance(values, list) or len(values) > limit
            or any(not isinstance(value, str) or not value for value in values)):
        raise ValueError(f"References must be a JSON list of at most {limit} uploaded filenames")
    return values


def run_owned(command: list[str], *, directory: Path, environment: dict | None = None) -> None:
    """Upstream infer handles SIGTERM and closes its three owned worker groups."""
    with (directory / "process.log").open("a") as log:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"Sol-H3-Spark process failed ({process.returncode}); see {directory / 'process.log'}")
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                # Pipeline.close may spend 40 seconds per owned worker. Do not
                # kill the parent early and strand CUDA workers or Qwen Docker.
                process.wait()


def video_info(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    streams = json.loads(result.stdout)["streams"]
    video = next(item for item in streams if item.get("codec_type") == "video")
    if not any(item.get("codec_type") == "audio" for item in streams):
        raise ValueError("Sol-H3-Spark did not produce its required audio track")
    if (video["width"], video["height"], video["avg_frame_rate"]) != (1344, 768, "24/1"):
        raise ValueError("Sol-H3-Spark output does not match the frozen 1344x768/24fps recipe")
    return video


def deliver_video(source: Path, output: Path, frame_count: int, directory: Path) -> None:
    """Resample the entire native clip, retaining both endpoints and audio timing."""
    if not 96 <= frame_count <= NATIVE_FRAMES:
        raise ValueError("Sol delivery must contain 96–121 frames")
    if frame_count == NATIVE_FRAMES:
        shutil.copyfile(source, output)
    else:
        indices = [round(i * (NATIVE_FRAMES - 1) / (frame_count - 1)) for i in range(frame_count)]
        select = "+".join(f"eq(n\\,{index})" for index in indices)
        run_owned(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                   "-vf", f"select={select},setpts=N/({FPS}*TB)",
                   "-af", f"atempo={NATIVE_FRAMES / frame_count},apad,atrim=end={frame_count / FPS},asetpts=PTS-STARTPTS",
                   "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output)],
                  directory=directory)
    if int(video_info(output)["nb_read_frames"]) != frame_count:
        raise ValueError("Sol delivery frame count differs from the requested duration")


class SolH3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.SolH3",
            display_name="Sol-H3-Spark",
            category="ComfyUIExtensions/Video",
            inputs=[
                io.Combo.Input("task", options=["t2va", "fl2va", "ref2va"], default="t2va"),
                io.String.Input("prompt", multiline=True),
                io.Int.Input("seed", default=42, min=0, max=2**63 - 1),
                io.Float.Input("duration", default=NATIVE_FRAMES / FPS, min=4, max=NATIVE_FRAMES / FPS),
                io.String.Input("output_prefix", default="sol-h3-spark"),
                io.String.Input("first_frame", default="", optional=True),
                io.String.Input("last_frame", default="", optional=True),
                io.String.Input("reference_images", default="[]", optional=True),
                io.String.Input("reference_videos", default="[]", optional=True),
                io.String.Input("reference_audios", default="[]", optional=True),
                io.String.Input("expected_model_identity", default="", optional=True),
            ],
            outputs=[io.String.Output("manifest")],
            is_output_node=True,
        )

    @classmethod
    async def fingerprint_inputs(cls, **kwargs):
        def calculate():
            from ...model_identity import _file_identity

            references = []
            for field in ("first_frame", "last_frame"):
                if kwargs.get(field):
                    references.append(kwargs[field])
            for field, limit in (("reference_images", 9), ("reference_videos", 3), ("reference_audios", 3)):
                references.extend(reference_list(kwargs.get(field, "[]"), limit))
            return (model_fingerprint(kwargs["task"]),
                    tuple(_file_identity(uploaded_file(name))["sha256"] for name in references))

        return await asyncio.to_thread(calculate)

    @classmethod
    def execute(cls, task, prompt, seed, duration, output_prefix,
                first_frame="", last_frame="", reference_images="[]",
                reference_videos="[]", reference_audios="[]", expected_model_identity=""):
        if (task not in {"t2va", "fl2va", "ref2va"} or not prompt.strip()
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("Provide a Sol task, nonempty prompt and nonnegative 63-bit seed")
        if not math.isfinite(duration) or not 4 <= duration <= NATIVE_FRAMES / FPS:
            raise ValueError("Sol generates 121 frames at 24fps; clip duration must be 4–5.041666666666667 seconds. Use MV segments for longer works.")
        references = [(kind, name) for kind, raw, limit in (
            ("image", reference_images, 9), ("video", reference_videos, 3),
            ("audio", reference_audios, 3),
        ) for name in reference_list(raw, limit)]
        if len(references) > 12:
            raise ValueError("Sol Ref2VA allows at most 12 reference inputs")
        frames = {key: value for key, value in (("first_frame", first_frame), ("last_frame", last_frame)) if value}
        if task == "t2va" and (references or frames):
            raise ValueError("T2VA accepts only text; select FL2VA or Ref2VA for media inputs")
        if task == "fl2va" and (not frames or references):
            raise ValueError("FL2VA requires first/last frame inputs without reference lists")
        if task == "ref2va" and (frames or not any(kind in {"image", "video"} for kind, _ in references)):
            raise ValueError("Ref2VA requires an image or video reference; use FL2VA for first/last frames")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install ffmpeg and ffprobe on the ComfyUI host")
        config = runtime_config()
        resolved_paths = task_paths(config, task)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        prefix = (output_root / output_prefix).resolve()
        if prefix == output_root or output_root not in prefix.parents:
            raise ValueError("output_prefix must be a path below ComfyUI/output")
        job_id = uuid.uuid4().hex
        job = Path(config["runtime_root"]) / "jobs" / job_id
        job.mkdir(parents=True)
        # Launch from the same immutable path selection that admission verifies.
        paths = job / "runtime-paths.json"
        paths.write_text(json.dumps(resolved_paths, ensure_ascii=False) + "\n")
        inputs = job / "inputs"
        inputs.mkdir()
        case = {"case_id": "generation", "task": task, "prompt": prompt, "seed": seed}

        def stage(name):
            source = uploaded_file(name)
            target = inputs / (uuid.uuid4().hex + source.suffix.lower())
            shutil.copyfile(source, target)
            return str(target)

        case.update({key: stage(value) for key, value in frames.items()})
        if references:
            case["references"] = [{"type": kind, "path": stage(name)} for kind, name in references]
        request = job / "request.jsonl"
        request.write_text(json.dumps(case, ensure_ascii=False) + "\n")
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                   SOL_H3_SPARK_RUNTIME_ROOT=config["runtime_root"],
                   SOL_H3_SPARK_QWEN_IMAGE=config["qwen_image"],
                   SOL_H3_SPARK_QWEN_WEIGHTS_ROOT=config["weights_root"],
                   SOL_H3_SPARK_COMFY_ROOT=config["comfy_root"])
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "COMFYUI_EXTENSIONS_OPENAI_API_KEY", "OPENAI_API_KEY"):
            env.pop(key, None)
        # Comfy enables cudaMallocAsync, which cannot run H3 VAE CUDA graphs.
        # Let Sol own allocator settings, including Stage2's expandable segments.
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            env.pop(key, None)
        package = Path(config["package"])
        # Sol owns isolated CUDA workers; release Comfy's resident model weights
        # before starting them. No shared LLM service is stopped by this node.
        require_model_identity(task, expected_model_identity, resolved_paths=resolved_paths)
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        started = time.monotonic()
        run_owned([sys.executable, "-B", str(package / "infer.py"), "--paths", str(paths),
                   "--prompts", str(request), "--task", task, "--output-dir", str(job / "run")],
                  directory=job, environment=env)
        report = json.loads((job / "run/results.json").read_text())
        if report.get("status") != "PASS" or len(report.get("requests", [])) != 1:
            raise RuntimeError(f"Sol returned an incomplete report: {job / 'run/results.json'}")
        source = Path(report["requests"][0]["output"]).resolve(strict=True)
        if job not in source.parents or int(video_info(source)["nb_read_frames"]) != NATIVE_FRAMES:
            raise ValueError("Sol returned an invalid native output")
        directory = prefix.parent / (prefix.name + "-" + job_id)
        directory.mkdir(parents=True)
        output = directory / "video.mp4"
        frame_count = min(NATIVE_FRAMES, math.ceil(duration * FPS))
        deliver_video(source, output, frame_count, job)
        report.update(upstream_revision=SANA_REVISION, total_node_seconds=time.monotonic() - started,
                      delivered_frames=frame_count, delivered_duration=frame_count / FPS,
                      native_frames=NATIVE_FRAMES, worker_lifetime="one node invocation; full warmup included in node time")
        (directory / "sol-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        # Large latent captures and staged media are transient. Receipts, worker
        # logs and model caches remain under the runtime for diagnosis/reuse.
        shutil.rmtree(inputs)
        for capture in (job / "run").rglob("*.pt"):
            capture.unlink()
        relative = str(directory.relative_to(output_root))
        files = [{"filename": name, "subfolder": relative, "type": "output"}
                 for name in ("video.mp4", "sol-report.json")]
        manifest = json.dumps({"model": "Sol-H3-Spark", "task": task, "files": files,
                               "frames": frame_count, "fps": FPS}, ensure_ascii=False)
        return io.NodeOutput(manifest, ui={"text": [manifest], "files": files})
