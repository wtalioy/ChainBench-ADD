"""O3. Codec and O7. ReEncode operators (paper §8.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.proc import run_command

from .base import DeliveryOperator


def _roundtrip_codec(
    input_path: Path,
    output_path: Path,
    codec: str,
    bitrate: str | None = None,
    encode_sample_rate: int | None = None,
    decode_sample_rate: int | None = None,
) -> None:
    codec = codec.lower()
    encoded_path: Path
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path), "-ac", "1",
    ]
    if encode_sample_rate is not None:
        command.extend(["-ar", str(encode_sample_rate)])
    if codec == "aac":
        encoded_path = output_path.with_name(f"{output_path.stem}__encoded.m4a")
        command.extend(["-c:a", "aac", "-movflags", "+faststart"])
        if bitrate:
            command.extend(["-b:a", bitrate])
    elif codec == "opus":
        encoded_path = output_path.with_name(f"{output_path.stem}__encoded.ogg")
        command.extend(["-c:a", "libopus", "-application", "voip", "-vbr", "on"])
        if bitrate:
            command.extend(["-b:a", bitrate])
    elif codec == "pcm_mulaw":
        encoded_path = output_path.with_name(f"{output_path.stem}__encoded_mulaw.wav")
        command.extend(["-c:a", "pcm_mulaw"])
    elif codec == "pcm_alaw":
        encoded_path = output_path.with_name(f"{output_path.stem}__encoded_alaw.wav")
        command.extend(["-c:a", "pcm_alaw"])
    elif codec == "gsm":
        encoded_path = output_path.with_name(f"{output_path.stem}__encoded.gsm")
        command.extend(["-c:a", "gsm"])
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    command.append(str(encoded_path))
    try:
        r_enc = run_command(command)
        if r_enc.returncode != 0:
            raise RuntimeError(r_enc.stderr.strip() or f"ffmpeg encode failed for {codec}")
        cmd_dec = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(encoded_path),
            "-ac", "1",
        ]
        if decode_sample_rate is not None:
            cmd_dec.extend(["-ar", str(decode_sample_rate)])
        cmd_dec.extend(["-c:a", "pcm_s16le", str(output_path)])
        r_dec = run_command(cmd_dec)
        if r_dec.returncode != 0:
            raise RuntimeError(r_dec.stderr.strip() or f"ffmpeg decode failed for {codec}")
    finally:
        try:
            encoded_path.unlink()
        except FileNotFoundError:
            pass


class CodecOperator(DeliveryOperator):
    @property
    def op_name(self) -> str:
        return "codec"

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
        codec = params["codec"]
        bitrate = params.get("bitrate")
        encode_sample_rate = params.get("encode_sample_rate")
        decode_sample_rate = params.get("decode_sample_rate", encode_sample_rate)
        metadata["codec"] = codec
        if bitrate:
            metadata["bitrate"] = bitrate
        if encode_sample_rate:
            metadata["encode_sample_rate"] = encode_sample_rate
        if decode_sample_rate:
            metadata["decode_sample_rate"] = decode_sample_rate
        _roundtrip_codec(input_path, output_path, codec, bitrate, encode_sample_rate, decode_sample_rate)


class ReEncodeOperator(DeliveryOperator):
    """O7. ReEncode: same or cross codec re-encode (paper §8.1)."""

    @property
    def op_name(self) -> str:
        return "reencode"

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
        codec = params.get("codec")
        if codec is None:
            codec = params.get("default_codec", "aac")
        bitrate = params.get("bitrate")
        encode_sample_rate = params.get("encode_sample_rate")
        decode_sample_rate = params.get("decode_sample_rate", encode_sample_rate)
        metadata["codec"] = codec
        if bitrate:
            metadata["bitrate"] = bitrate
        if encode_sample_rate:
            metadata["encode_sample_rate"] = encode_sample_rate
        if decode_sample_rate:
            metadata["decode_sample_rate"] = decode_sample_rate
        _roundtrip_codec(input_path, output_path, codec, bitrate, encode_sample_rate, decode_sample_rate)
