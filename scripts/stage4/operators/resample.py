"""O1. Resample operator: 16k→8k, 16k→24k, 16k→32k→16k (paper §8.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.proc import run_command

from .base import DeliveryOperator


class ResampleOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "resample"

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
        mode = params["mode"]
        metadata["mode"] = mode
        if mode == "16k_to_8k":
            self._resample_once(input_path, output_path, 8000)
        elif mode == "16k_to_24k":
            self._resample_once(input_path, output_path, 24000)
        elif mode == "16k_to_32k":
            self._resample_once(input_path, output_path, 32000)
        elif mode == "8k_to_16k":
            self._resample_once(input_path, output_path, 16000)
        elif mode == "24k_to_16k":
            self._resample_once(input_path, output_path, 16000)
        elif mode == "32k_to_16k":
            self._resample_once(input_path, output_path, 16000)
        elif mode == "16k_to_8k_to_16k":
            self._resample_roundtrip(input_path, output_path, 8000, 16000)
        elif mode == "16k_to_24k_to_16k":
            self._resample_roundtrip(input_path, output_path, 24000, 16000)
        elif mode == "16k_to_32k_to_16k":
            self._resample_roundtrip(input_path, output_path, 32000, 16000)
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    @staticmethod
    def _resample_once(input_path: Path, output_path: Path, sample_rate: int) -> None:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-af", "aresample=resampler=soxr:precision=28",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        result = run_command(command)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg resample step failed")

    @classmethod
    def _resample_roundtrip(
        cls,
        input_path: Path,
        output_path: Path,
        mid_sample_rate: int,
        final_sample_rate: int,
    ) -> None:
        mid_path = output_path.with_name(f"{output_path.stem}__mid_{mid_sample_rate}.wav")
        try:
            cls._resample_once(input_path, mid_path, mid_sample_rate)
            cls._resample_once(mid_path, output_path, final_sample_rate)
        finally:
            try:
                mid_path.unlink()
            except FileNotFoundError:
                pass
