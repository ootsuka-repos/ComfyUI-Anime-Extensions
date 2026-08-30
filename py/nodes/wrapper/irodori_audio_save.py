from __future__ import annotations

import wave
from pathlib import Path

import folder_paths
import torch
from comfy_api.latest import io, ui

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME


class IrodoriTTSAudioSave(io.ComfyNode):
    """Save an AUDIO value as WAV, MP3, or FLAC in ComfyUI's output directory."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "AudioSave"),
            display_name="IrodoriTTS Save Audio",
            category=CATEGORY,
            search_aliases=["save audio", "export wav", "export mp3", "export flac"],
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input(
                    "filename_prefix",
                    default="audio/IrodoriTTS",
                    tooltip="ComfyUI output directory below which to save the audio.",
                ),
                io.Combo.Input(
                    "format",
                    options=["wav", "mp3", "flac"],
                    default="wav",
                    tooltip="WAV is lossless PCM. MP3 is compressed; FLAC is lossless compressed.",
                ),
                io.Combo.Input(
                    "mp3_quality",
                    options=["V0", "128k", "320k"],
                    default="320k",
                    tooltip="MP3 quality setting. It is ignored for WAV and FLAC.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[io.Audio.Output("audio")],
        )

    @staticmethod
    def _save_wav(audio: dict, filename_prefix: str):
        waveform_batch = audio.get("waveform")
        if not isinstance(waveform_batch, torch.Tensor) or waveform_batch.ndim != 3:
            raise ValueError("Expected AUDIO waveform with shape [batch, channels, samples].")

        sample_rate = int(audio.get("sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError(f"Invalid AUDIO sample rate: {sample_rate}")

        output_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
        )
        results = []
        for batch_number, waveform in enumerate(waveform_batch.detach().to(device="cpu", dtype=torch.float32)):
            if waveform.ndim != 2 or waveform.shape[0] < 1:
                raise ValueError(
                    "Expected each AUDIO batch item to have shape [channels, samples], "
                    f"got {tuple(waveform.shape)}."
                )

            file_stem = filename.replace("%batch_num%", str(batch_number))
            file_name = f"{file_stem}_{counter:05}.wav"
            output_path = Path(output_dir) / file_name

            pcm16 = (
                waveform.clamp(-1.0, 1.0)
                .transpose(0, 1)
                .contiguous()
                .mul(32767.0)
                .round()
                .to(torch.int16)
                .numpy()
            )
            with wave.open(str(output_path), "wb") as handle:
                handle.setnchannels(int(waveform.shape[0]))
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm16.tobytes())

            results.append(ui.SavedResult(file_name, subfolder, io.FolderType.output))
            counter += 1

        return ui.SavedAudios(results)

    @classmethod
    def execute(
        cls,
        audio: dict,
        filename_prefix: str,
        format: str,
        mp3_quality: str,
    ):
        if audio is None:
            raise ValueError("IrodoriTTS Save Audio: input audio is None.")

        selected_format = str(format).strip().lower()
        if selected_format == "wav":
            saved = cls._save_wav(audio, filename_prefix)
        elif selected_format in {"mp3", "flac"}:
            saved = ui.AudioSaveHelper.get_save_audio_ui(
                audio,
                filename_prefix=filename_prefix,
                cls=cls,
                format=selected_format,
                quality=str(mp3_quality),
            )
        else:
            raise ValueError(f"Unsupported audio format: {format!r}")

        return io.NodeOutput(audio, ui=saved)
