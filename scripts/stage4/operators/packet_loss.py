"""O4. PacketLoss operator (paper §8.1, §13.3)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from .base import DeliveryOperator, load_audio, write_audio


class PacketLossOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "packet_loss"

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
        loss_rate_pct = float(params["loss_rate_pct"])
        frame_ms = int(config["packet_loss"]["frame_ms"])
        avg_burst_frames = float(params.get("avg_burst_frames", config["packet_loss"].get("avg_burst_frames", 2.5)))
        concealment = str(params.get("concealment", config["packet_loss"].get("concealment", "repeat_fade")))
        audio, sr = load_audio(input_path)
        dropped = self._apply_packet_loss_numpy(
            audio,
            sr,
            loss_rate_pct,
            frame_ms,
            avg_burst_frames,
            concealment,
            seed + op_index,
        )
        write_audio(output_path, dropped, sr)
        metadata["loss_rate_pct"] = loss_rate_pct
        metadata["frame_ms"] = frame_ms
        metadata["avg_burst_frames"] = avg_burst_frames
        metadata["concealment"] = concealment

    @staticmethod
    def _apply_packet_loss_numpy(
        audio: np.ndarray,
        sample_rate: int,
        loss_rate_pct: float,
        frame_ms: int,
        avg_burst_frames: float,
        concealment: str,
        seed: int,
    ) -> np.ndarray:
        frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
        dropped = max(0.0, min(0.95, loss_rate_pct / 100.0))
        avg_burst_frames = max(1.0, avg_burst_frames)
        bad_to_good = min(1.0, 1.0 / avg_burst_frames)
        good_to_bad = min(1.0, dropped * bad_to_good / max(1e-6, 1.0 - dropped))
        rng = random.Random(seed)
        frames = []
        previous = np.zeros(frame_len, dtype=np.float32)
        previous_previous = previous.copy()
        in_bad_state = False
        for start in range(0, len(audio), frame_len):
            frame = audio[start : start + frame_len]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            transition = bad_to_good if in_bad_state else good_to_bad
            if rng.random() < transition:
                in_bad_state = not in_bad_state
            if in_bad_state:
                if concealment == "repeat_fade":
                    concealed = previous.copy() * 0.85
                elif concealment == "interpolate" and len(frames) > 0:
                    concealed = 0.5 * (previous + previous_previous)
                elif concealment == "noise_fill":
                    frame_seed = rng.randrange(0, 2**32)
                    concealed = np.random.default_rng(frame_seed).uniform(
                        -0.02, 0.02, size=frame_len
                    ).astype(np.float32)
                else:
                    concealed = previous.copy()
                frames.append(concealed.astype(np.float32))
            else:
                frames.append(frame.copy())
                previous_previous = previous.copy()
                previous = frame.copy()
        merged = np.concatenate(frames)[: len(audio)]
        return merged.astype(np.float32)
