import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import IO_REF_CONFIG


VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def _is_video_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _file_digest(path: str | os.PathLike[str]) -> str:
    m = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            m.update(chunk)
    return m.hexdigest()


def _find_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _extract_video_audio(video_path: str) -> str:
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "Video reference audio requires imageio-ffmpeg or ffmpeg. "
            "Install requirements.txt or use an audio file as reference."
        )

    digest = _file_digest(video_path)
    output_dir = Path(folder_paths.get_temp_directory()) / "irodori_tts" / "reference_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{digest}.wav"
    if output_path.exists():
        return str(output_path)

    temp_path = output_path.with_suffix(".tmp.wav")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Failed to extract audio from video: {stderr or video_path}")

    temp_path.replace(output_path)
    print(f"[IrodoriTTS] extracted reference audio: {video_path} -> {output_path}")
    return str(output_path)


class IrodoriReferenceAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])

        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "ReferenceAudio"),
            display_name="IrodoriTTS Reference Audio",
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "audio",
                    options=sorted(files),
                    upload=io.UploadType.audio,
                    tooltip="話者参照に使う音声ファイルをinputフォルダから選択します。",
                ),
                io.Boolean.Input(
                    "normalize_ref_audio",
                    default=False,
                    tooltip="参照音声を-16dB基準で正規化します。音量差が大きい参照音声で有効です。",
                ),
                io.Float.Input(
                    "max_ref_seconds",
                    default=30.0,
                    min=1.0,
                    max=120.0,
                    step=1.0,
                    tooltip="参照音声として使用する最大秒数です。長い音声は先頭からこの秒数に切り詰めます。",
                ),
                io.Custom("AUDIO_UI").Input("audioUI", optional=True),
                io.Custom("AUDIOUPLOAD").Input("upload", optional=True),
            ],
            outputs=[
                IO_REF_CONFIG.Output(display_name="irodori_ref_config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: str,
        normalize_ref_audio: bool,
        max_ref_seconds: float,
        audioUI=None,
        upload=None,
    ):
        audio_path = folder_paths.get_annotated_filepath(audio)
        if _is_video_file(audio_path):
            audio_path = _extract_video_audio(audio_path)
        config = {
            "ref_wav": audio_path,
            "ref_latent": None,
            "no_ref": False,
            "ref_normalize_db": -16.0 if normalize_ref_audio else None,
            "ref_ensure_max": bool(normalize_ref_audio),
            "max_ref_seconds": float(max_ref_seconds),
        }
        return io.NodeOutput(config)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        audio = kwargs.get("audio")
        audio_path = folder_paths.get_annotated_filepath(audio)
        return _file_digest(audio_path)

    @classmethod
    def validate_inputs(cls, **kwargs):
        audio = kwargs.get("audio")
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        audio_path = folder_paths.get_annotated_filepath(audio)
        if _is_video_file(audio_path) and _find_ffmpeg() is None:
            return "Video reference audio requires imageio-ffmpeg or ffmpeg. Install requirements.txt or select an audio file."
        return True
