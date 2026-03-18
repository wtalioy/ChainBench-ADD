"""Composite telephony-session operator with codec, burst loss, jitter, and AGC."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .base import DeliveryOperator, ffmpeg_filter_to_wav, load_audio, write_audio
from .codec import _roundtrip_codec
from .packet_loss import PacketLossOperator


def _telephony_band_filter(profile: str) -> str:
    if profile == "NB":
        return (
            "highpass=f=250:poles=2,"
            "lowpass=f=3400:poles=2,"
            "compand=attacks=0.01:decays=0.15:points=-80/-80|-20/-20|0/-4"
        )
    if profile == "WB":
        return "highpass=f=50:poles=2,lowpass=f=7000:poles=2"
    raise ValueError(f"Unsupported telephony profile: {profile}")


def _agc_filter_chain(profile: str) -> str:
    if profile == "none":
        return "anull"
    if profile == "mild":
        return "compand=attacks=0.01:decays=0.2:points=-80/-80|-24/-20|-8/-7|0/-2,alimiter=limit=0.97"
    if profile == "telephony":
        return "compand=attacks=0.003:decays=0.2:points=-80/-80|-30/-22|-12/-9|0/-3,alimiter=limit=0.95"
    raise ValueError(f"Unsupported AGC profile: {profile}")


def _apply_jitter_buffer_model(
    audio: np.ndarray,
    sample_rate: int,
    jitter_ms: float,
    seed: int,
    frame_ms: int,
) -> np.ndarray:
    if jitter_ms <= 0:
        return audio.astype(np.float32)
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    max_offset = max(1, int(sample_rate * jitter_ms / 1000.0))
    rng = np.random.default_rng(seed)
    accum = np.zeros(len(audio) + max_offset * 2 + frame_len, dtype=np.float32)
    counts = np.zeros_like(accum)
    base_offset = max_offset
    for start in range(0, len(audio), frame_len):
        frame = audio[start : start + frame_len]
        if len(frame) == 0:
            continue
        target_start = base_offset + start + int(rng.integers(-max_offset, max_offset + 1))
        target_start = max(0, target_start)
        target_end = target_start + len(frame)
        accum[target_start:target_end] += frame
        counts[target_start:target_end] += 1.0
    valid = counts > 0
    accum[valid] /= counts[valid]
    if not np.any(valid):
        return audio.astype(np.float32)
    first = int(np.argmax(valid))
    trimmed = accum[first : first + len(audio)]
    if len(trimmed) < len(audio):
        trimmed = np.pad(trimmed, (0, len(audio) - len(trimmed)))
    return trimmed[: len(audio)].astype(np.float32)


class TelephonySessionOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "telephony_session"

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
        profile = str(params.get("profile", "NB")).upper()
        codec = str(params["codec"])
        bitrate = params.get("bitrate")
        session_sr = int(params.get("encode_sample_rate", 8000 if profile == "NB" else 16000))
        decode_sr = int(params.get("decode_sample_rate", session_sr))
        loss_rate_pct = float(params.get("loss_rate_pct", 0.0))
        avg_burst_frames = float(params.get("avg_burst_frames", config["packet_loss"].get("avg_burst_frames", 3.0)))
        concealment = str(params.get("concealment", config["packet_loss"].get("concealment", "repeat_fade")))
        jitter_ms = float(params.get("jitter_ms", 0.0))
        agc_profile = str(params.get("agc_profile", "telephony")).lower()
        frame_ms = int(params.get("frame_ms", config["packet_loss"]["frame_ms"]))

        metadata.update(
            {
                "profile": profile,
                "codec": codec,
                "bitrate": bitrate or "",
                "encode_sample_rate": session_sr,
                "decode_sample_rate": decode_sr,
                "loss_rate_pct": loss_rate_pct,
                "avg_burst_frames": avg_burst_frames,
                "concealment": concealment,
                "jitter_ms": jitter_ms,
                "agc_profile": agc_profile,
                "frame_ms": frame_ms,
            }
        )

        preprocessed_path = output_path.with_name(f"{output_path.stem}__telephony_pre.wav")
        decoded_path = output_path.with_name(f"{output_path.stem}__telephony_decoded.wav")
        packeted_path = output_path.with_name(f"{output_path.stem}__telephony_packeted.wav")

        try:
            ffmpeg_filter_to_wav(
                input_path,
                preprocessed_path,
                _telephony_band_filter(profile),
                sample_rate=session_sr,
            )
            _roundtrip_codec(
                preprocessed_path,
                decoded_path,
                codec=codec,
                bitrate=bitrate,
                encode_sample_rate=session_sr,
                decode_sample_rate=decode_sr,
            )

            audio, sr = load_audio(decoded_path)
            packeted = PacketLossOperator._apply_packet_loss_numpy(
                audio,
                sr,
                loss_rate_pct,
                frame_ms,
                avg_burst_frames,
                concealment,
                seed + op_index,
            )
            packeted = _apply_jitter_buffer_model(packeted, sr, jitter_ms, seed + op_index + 1, frame_ms)
            write_audio(packeted_path, packeted, sr)

            agc_filter = _agc_filter_chain(agc_profile)
            if agc_filter == "anull":
                shutil.copy2(packeted_path, output_path)
            else:
                ffmpeg_filter_to_wav(packeted_path, output_path, agc_filter, sample_rate=None)
        finally:
            for path in (preprocessed_path, decoded_path, packeted_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
