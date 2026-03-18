"""O5. RIR operator: room impulse response (paper §8.1, §13.2)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from lib.logging import get_logger

from .base import DeliveryOperator, load_audio, peak_normalize, write_audio

try:
    import pyroomacoustics as pra
except ImportError:
    pra = None


LOGGER = get_logger("stage4")
RIR_WARNING_CACHE: set[str] = set()
FFT_CONVOLUTION_THRESHOLD = 2_000_000


def _log_rir_warning_once(key: str, message: str) -> None:
    if key in RIR_WARNING_CACHE:
        return
    LOGGER.warning(message)
    RIR_WARNING_CACHE.add(key)


def _convolve_audio(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    output_len = len(audio) + len(rir) - 1
    if len(audio) * len(rir) <= FFT_CONVOLUTION_THRESHOLD:
        return np.convolve(audio, rir, mode="full").astype(np.float32)
    fft_size = 1 << (output_len - 1).bit_length()
    audio_fft = np.fft.rfft(audio, n=fft_size)
    rir_fft = np.fft.rfft(rir, n=fft_size)
    convolved = np.fft.irfft(audio_fft * rir_fft, n=fft_size)[:output_len]
    return np.asarray(convolved, dtype=np.float32)


@lru_cache(maxsize=128)
def _inverse_sabine_cached(rt60: float, room_dim_key: tuple[float, ...]) -> tuple[float, int]:
    if pra is None:
        raise RuntimeError("pyroomacoustics is not installed")
    return pra.inverse_sabine(rt60, list(room_dim_key))


def _synthesize_rir(
    sample_rate: int,
    room_dim: list[float],
    distance: float,
    rt60: float,
    seed: int,
    rir_cfg: dict[str, Any],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sound_speed = float(rir_cfg["sound_speed_mps"])
    max_tail_sec = min(float(rir_cfg["max_tail_sec"]), max(0.3, rt60 * 1.2))
    num_samples = max(64, int(sample_rate * max_tail_sec))
    rir = np.zeros(num_samples, dtype=np.float32)
    direct_delay = int((distance / sound_speed) * sample_rate)
    direct_delay = min(max(0, direct_delay), num_samples - 1)
    direct_gain = 1.0 / max(distance, 0.5)
    rir[direct_delay] += direct_gain
    room_scale = sum(room_dim) / len(room_dim)
    min_reflections = int(rir_cfg["min_reflections"])
    max_reflections = int(rir_cfg["max_reflections"])
    reflection_count = min_reflections + int((room_scale / 10.0) * (max_reflections - min_reflections))
    reflection_count = max(min_reflections, min(max_reflections, reflection_count))
    time_axis = np.arange(num_samples, dtype=np.float32) / sample_rate
    envelope = np.exp(-6.91 * time_axis / max(rt60, 1e-3))
    for _ in range(reflection_count):
        delay_sec = rng.uniform(distance / sound_speed, min(max_tail_sec * 0.95, 0.25 + rt60 * 0.5))
        delay_idx = min(num_samples - 1, max(1, int(delay_sec * sample_rate)))
        amp = rng.uniform(0.02, 0.18) / max(distance, 0.5)
        amp *= envelope[delay_idx]
        rir[delay_idx] += amp * rng.uniform(0.6, 1.0)
    noise_floor = rng.standard_normal(num_samples).astype(np.float32) * envelope * 0.003
    rir += noise_floor
    norm = float(np.max(np.abs(rir)) + 1e-8)
    rir = rir / norm
    rir[direct_delay] = max(rir[direct_delay], 0.8)
    return rir.astype(np.float32)


def _apply_rir_synthetic(
    audio: np.ndarray,
    sample_rate: int,
    room: dict[str, Any],
    distance: float,
    rt60: float,
    seed: int,
    rir_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    rir = _synthesize_rir(sample_rate, room["room_dim"], distance, rt60, seed, rir_cfg)
    convolved = _convolve_audio(audio, rir)
    return peak_normalize(convolved), {"backend": "synthetic", "rir_num_samples": int(len(rir))}


def _sample_rir_positions(
    room_dim: list[float],
    distance: float,
    seed: int,
    rir_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    dims = np.asarray(room_dim, dtype=np.float64)
    if dims.ndim != 1 or dims.size not in {2, 3}:
        raise ValueError(f"room_dim must have length 2 or 3, got {room_dim}")
    rng = np.random.default_rng(seed)
    margin = float(rir_cfg.get("wall_margin_m", 0.4))
    span_x = max(0.0, float(dims[0]) - 2.0 * margin)
    actual_distance_x = min(float(distance), span_x)
    center_x = float(dims[0]) / 2.0
    positions = np.zeros((2, dims.size), dtype=np.float64)
    positions[0, 0] = center_x + actual_distance_x / 2.0
    positions[1, 0] = center_x - actual_distance_x / 2.0
    if dims.size >= 2:
        lateral_span = max(0.0, float(dims[1]) - 2.0 * margin)
        lateral_offset = 0.0 if lateral_span == 0.0 else float(rng.uniform(-0.2 * lateral_span, 0.2 * lateral_span))
        center_y = float(dims[1]) / 2.0
        positions[0, 1] = center_y + lateral_offset
        positions[1, 1] = center_y - lateral_offset * 0.5
    if dims.size == 3:
        source_height = float(rir_cfg.get("source_height_m", 1.5))
        mic_height = float(rir_cfg.get("mic_height_m", 1.5))
        positions[0, 2] = min(max(source_height, margin), float(dims[2]) - margin)
        positions[1, 2] = min(max(mic_height, margin), float(dims[2]) - margin)
    source_pos = positions[0]
    mic_pos = positions[1]
    actual_distance = float(np.linalg.norm(source_pos - mic_pos))
    return source_pos, mic_pos, actual_distance


def _apply_rir_pyroomacoustics(
    audio: np.ndarray,
    sample_rate: int,
    room: dict[str, Any],
    distance: float,
    rt60: float,
    seed: int,
    rir_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if pra is None:
        raise RuntimeError("pyroomacoustics is not installed")
    room_dim = [float(v) for v in room["room_dim"]]
    source_pos, mic_pos, actual_distance = _sample_rir_positions(room_dim, distance, seed, rir_cfg)
    absorption, max_order = _inverse_sabine_cached(float(rt60), tuple(room_dim))
    absorption = float(np.clip(absorption, 0.05, 0.99))
    max_order = int(max(1, min(max_order, int(rir_cfg.get("pyroomacoustics_max_order_cap", 24)))))
    shoebox = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
        air_absorption=bool(rir_cfg.get("pyroomacoustics_air_absorption", True)),
    )
    shoebox.add_source(source_pos)
    shoebox.add_microphone_array(np.asarray(mic_pos, dtype=np.float64).reshape(-1, 1))
    shoebox.compute_rir()
    rir = np.asarray(shoebox.rir[0][0], dtype=np.float32)
    if rir.size == 0:
        raise RuntimeError("pyroomacoustics returned an empty RIR")
    max_tail_sec = min(float(rir_cfg["max_tail_sec"]), max(0.3, rt60 * 1.2))
    rir = rir[: max(64, int(sample_rate * max_tail_sec))]
    convolved = _convolve_audio(audio, rir)
    return peak_normalize(convolved), {
        "backend": "pyroomacoustics",
        "rir_num_samples": int(len(rir)),
        "absorption": absorption,
        "max_order": max_order,
        "source_position": [round(float(v), 4) for v in source_pos],
        "mic_position": [round(float(v), 4) for v in mic_pos],
        "effective_distance": round(actual_distance, 4),
    }


def _apply_rir(
    audio: np.ndarray,
    sample_rate: int,
    room: dict[str, Any],
    distance: float,
    rt60: float,
    seed: int,
    rir_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    backend = str(rir_cfg.get("backend", "synthetic")).lower()
    fallback_backend = str(rir_cfg.get("fallback_backend", "synthetic")).lower()
    if backend == "pyroomacoustics":
        try:
            return _apply_rir_pyroomacoustics(audio, sample_rate, room, distance, rt60, seed, rir_cfg)
        except Exception as exc:
            if fallback_backend and fallback_backend != "pyroomacoustics":
                _log_rir_warning_once(
                    "pyroomacoustics_fallback",
                    f"pyroomacoustics RIR failed; falling back to {fallback_backend}: {exc}",
                )
                reverbed, meta = _apply_rir_synthetic(audio, sample_rate, room, distance, rt60, seed, rir_cfg)
                meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"
                return reverbed, meta
            raise
    if backend == "synthetic":
        return _apply_rir_synthetic(audio, sample_rate, room, distance, rt60, seed, rir_cfg)
    raise ValueError(f"Unsupported RIR backend: {backend}")


class RIROperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "rir"

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
        room = params["room"]
        distance = float(params["distance"])
        rt60 = float(params["rt60"])
        metadata["requested_backend"] = str(config["rir"].get("backend", "synthetic")).lower()
        metadata["fallback_backend"] = str(config["rir"].get("fallback_backend", "synthetic")).lower()
        metadata["rt60"] = rt60
        metadata["distance"] = distance
        metadata["room_name"] = room["name"]
        metadata["room_dim"] = room["room_dim"]
        audio, sr = load_audio(input_path)
        reverbed, rir_meta = _apply_rir(
            audio, sr, room, distance, rt60, seed + op_index, config["rir"]
        )
        metadata.update(rir_meta)
        write_audio(output_path, reverbed, sr)
