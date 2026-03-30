"""Canonical task-key builders derived from metadata rows."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

OPERATOR_MULTISET_KEY_CACHE_FIELD = "__operator_multiset_key"


@lru_cache(maxsize=16384)
def _parse_json_list_from_string(value: str) -> tuple[Any, ...]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return ()
    return tuple(parsed) if isinstance(parsed, list) else ()


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return list(_parse_json_list_from_string(value))
    parsed = value
    return parsed if isinstance(parsed, list) else []


def parse_operator_seq(operator_seq_value: Any) -> list[str]:
    return [str(item) for item in _parse_json_list(operator_seq_value)]


def parse_operator_params(operator_params_value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _parse_json_list(operator_params_value) if isinstance(item, dict)]


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _normalize_compact_number(value: Any) -> str | None:
    parsed = _parse_float(value)
    if parsed is None:
        return _normalize_text(value)
    rounded = round(parsed, 3)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _bucket_value(value: Any, *, bounds: list[tuple[float, str]], fallback: str) -> str | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    for upper_bound, label in bounds:
        if parsed <= upper_bound:
            return label
    return fallback


def _bucket_bitrate(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    return _bucket_value(
        text[:-1] if text.lower().endswith("k") else text,
        bounds=[(16, "very_low"), (24, "low"), (32, "medium")],
        fallback="high",
    )


def _bucket_packet_loss(value: Any) -> str | None:
    return _bucket_value(value, bounds=[(2, "light"), (5, "moderate")], fallback="severe")


def _bucket_rt60(value: Any) -> str | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    if parsed < 0.35:
        return "short"
    if parsed < 0.65:
        return "medium"
    return "long"


def _bucket_distance(value: Any) -> str | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    if parsed < 1.0:
        return "near"
    if parsed < 2.5:
        return "mid"
    return "far"


def _bucket_snr(value: Any) -> str | None:
    return _bucket_value(value, bounds=[(12, "low"), (24, "medium")], fallback="high")


def _bucket_sample_rate(value: Any) -> str | None:
    return _bucket_value(
        value,
        bounds=[(8000, "8k"), (16000, "16k"), (24000, "24k")],
        fallback="32k_plus",
    )


def _append_token(tokens: list[str], prefix: str, value: str | None) -> None:
    if value:
        tokens.append(f"{prefix}={value}")


def _append_text_token(tokens: list[str], prefix: str, value: Any) -> None:
    _append_token(tokens, prefix, _normalize_text(value))


def _codec_signature_tokens(operator: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    _append_text_token(tokens, "codec", operator.get("codec"))
    _append_token(tokens, "bitrate", _bucket_bitrate(operator.get("bitrate")))
    _append_token(tokens, "encode_sr", _bucket_sample_rate(operator.get("encode_sample_rate")))
    return tokens


def _packet_loss_signature_tokens(operator: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    _append_token(tokens, "plr", _bucket_packet_loss(operator.get("loss_rate_pct")))
    _append_text_token(tokens, "concealment", operator.get("concealment"))
    return tokens


def _rir_signature_tokens(operator: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    _append_token(tokens, "rt60", _bucket_rt60(operator.get("rt60")))
    _append_token(tokens, "distance", _bucket_distance(operator.get("distance")))
    room = operator.get("room")
    if isinstance(room, dict):
        _append_text_token(tokens, "room", room.get("name"))
    return tokens


def _call_path_signature_tokens(operator: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    _append_text_token(tokens, "profile", operator.get("profile"))
    _append_text_token(tokens, "codec", operator.get("codec"))
    _append_token(tokens, "plr", _bucket_packet_loss(operator.get("loss_rate_pct")))
    _append_token(tokens, "encode_sr", _bucket_sample_rate(operator.get("encode_sample_rate")))
    _append_text_token(tokens, "concealment", operator.get("concealment"))
    _append_text_token(tokens, "agc", operator.get("agc_profile"))
    _append_token(tokens, "jitter", _normalize_compact_number(operator.get("jitter_ms")))
    return tokens


def _operator_signature_tokens(operator: dict[str, Any]) -> list[str]:
    op_name = _normalize_text(operator.get("op"))
    if not op_name:
        return []
    if op_name in {"codec", "reencode"}:
        return _codec_signature_tokens(operator)
    if op_name == "packet_loss":
        return _packet_loss_signature_tokens(operator)
    if op_name == "bandlimit":
        return [f"bandwidth={value}" for value in [_normalize_text(operator.get("mode"))] if value]
    if op_name == "noise":
        tokens: list[str] = []
        _append_token(tokens, "snr", _bucket_snr(operator.get("snr_db")))
        _append_text_token(tokens, "noise", operator.get("noise_type"))
        return tokens
    if op_name == "rir":
        return _rir_signature_tokens(operator)
    if op_name == "resample":
        return [f"resample={value}" for value in [_normalize_text(operator.get("mode"))] if value]
    if op_name == "call_path":
        return _call_path_signature_tokens(operator)
    return []


def operator_signature_sequence(value: Any) -> list[str]:
    if not isinstance(value, dict):
        seq = parse_operator_seq(value)
        return [str(item) for item in seq]

    params = parse_operator_params(value.get("operator_params", "[]"))
    seq = parse_operator_seq(value.get("operator_seq", "[]"))
    if not params:
        return seq

    signatures: list[str] = []
    for index, operator in enumerate(params):
        op_name = _normalize_text(operator.get("op")) or (seq[index] if index < len(seq) else "")
        if not op_name:
            continue
        op_tokens = _operator_signature_tokens(operator)
        signatures.append(f"{op_name}[{','.join(sorted(op_tokens))}]" if op_tokens else op_name)
    return signatures or seq


def operator_multiset_key(value: Any) -> str:
    if isinstance(value, dict):
        cached = value.get(OPERATOR_MULTISET_KEY_CACHE_FIELD)
        if isinstance(cached, str):
            return cached
        signatures = operator_signature_sequence(value)
        computed = json.dumps(sorted(signatures), ensure_ascii=False, separators=(",", ":")) if signatures else ""
        value[OPERATOR_MULTISET_KEY_CACHE_FIELD] = computed
        return computed
    signatures = operator_signature_sequence(value)
    return json.dumps(sorted(signatures), ensure_ascii=False, separators=(",", ":")) if signatures else ""


def path_endpoint_key(value: Any) -> str:
    signatures = operator_signature_sequence(value)
    return signatures[-1] if signatures else ""
