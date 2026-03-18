"""Base class and shared helpers for delivery-chain operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from lib.proc import run_command


def ensure_mono_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float32)
    return np.mean(audio, axis=1, dtype=np.float32)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    return ensure_mono_audio(audio), int(sr)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)


def peak_normalize(audio: np.ndarray, limit: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak > limit:
        audio = audio * (limit / peak)
    return audio.astype(np.float32)


def soft_clip(audio: np.ndarray, limit: float = 0.98, drive: float = 2.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-8)
    if peak <= limit:
        return audio.astype(np.float32)
    normalized = audio / peak
    clipped = np.tanh(drive * normalized) / np.tanh(drive)
    return (clipped * limit).astype(np.float32)


def ffmpeg_filter_to_wav(
    input_path: Path,
    output_path: Path,
    filter_chain: str,
    sample_rate: int | None = None,
) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        "-af", f"{filter_chain},aresample=resampler=soxr:precision=28",
        "-ac", "1",
    ]
    if sample_rate is not None:
        command.extend(["-ar", str(sample_rate)])
    command.extend(["-c:a", "pcm_s16le", str(output_path)])
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg filter failed: {filter_chain}")


def standardize_final_output(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Output to final format (mono, sample rate, codec) per config."""
    final_cfg = config["final_output"]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        "-ac", str(final_cfg["channels"]),
        "-af", "aresample=resampler=soxr:precision=28",
        "-ar", str(final_cfg["sample_rate"]),
        "-c:a", str(final_cfg["codec_name"]),
        str(output_path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg final standardization failed")


class DeliveryOperator(ABC):
    """Abstract base for delivery-chain operators."""

    @property
    @abstractmethod
    def op_name(self) -> str:
        """Operator key used in chain grammar (e.g. 'resample', 'codec')."""
        ...

    def apply(
        self,
        input_path: Path,
        output_path: Path,
        params: dict[str, Any],
        config: dict[str, Any],
        seed: int,
        op_index: int,
    ) -> dict[str, Any]:
        """
        Apply this operator. Read from input_path, write to output_path.
        Return metadata dict (op, mode, codec, etc.) for tracing.
        """
        metadata = {"op": self.op_name}
        self._apply_impl(input_path, output_path, params, config, seed, op_index, metadata)
        return metadata

    @abstractmethod
    def _apply_impl(
        self,
        input_path: Path,
        output_path: Path,
        params: dict[str, Any],
        config: dict[str, Any],
        seed: int,
        op_index: int,
        metadata: dict[str, Any],
    ) -> None:
        """Implement the transformation; update metadata as needed."""
        ...
