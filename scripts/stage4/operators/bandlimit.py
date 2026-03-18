"""O2. BandLimit operator: NB (300–3400 Hz), WB (50–7000 Hz) (paper §8.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DeliveryOperator, ffmpeg_filter_to_wav


class BandLimitOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "bandlimit"

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
        if mode == "NB":
            filter_chain = (
                "highpass=f=250:poles=2,"
                "lowpass=f=3400:poles=2,"
                "compand=attacks=0.01:decays=0.15:points=-80/-80|-18/-18|0/-3"
            )
        elif mode == "WB":
            filter_chain = "highpass=f=50:poles=2,lowpass=f=7000:poles=2"
        else:
            raise ValueError(f"Unsupported bandlimit mode: {mode}")
        ffmpeg_filter_to_wav(input_path, output_path, filter_chain, sample_rate=None)
