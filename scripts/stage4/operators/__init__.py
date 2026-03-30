"""Delivery-chain operators as individual classes (paper §8.1). Registry and dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DeliveryOperator, standardize_final_output
from .bandlimit import BandLimitOperator
from .codec import CodecOperator, ReEncodeOperator
from .noise import NoiseOperator
from .packet_loss import PacketLossOperator
from .resample import ResampleOperator
from .call_path import CallPathOperator
from .rir import RIROperator

OPERATOR_REGISTRY: dict[str, type[DeliveryOperator]] = {
    "resample": ResampleOperator,
    "bandlimit": BandLimitOperator,
    "codec": CodecOperator,
    "reencode": ReEncodeOperator,
    "packet_loss": PacketLossOperator,
    "noise": NoiseOperator,
    "call_path": CallPathOperator,
    "rir": RIROperator,
}
OPERATOR_INSTANCES: dict[str, DeliveryOperator] = {
    op_name: cls() for op_name, cls in OPERATOR_REGISTRY.items()
}


def apply_operator(
    current_path: Path,
    operator: dict[str, Any],
    temp_dir: Path,
    op_index: int,
    config: dict[str, Any],
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    """Dispatch to the appropriate operator class and return (output_path, metadata)."""
    op_name = operator["op"]
    instance = OPERATOR_INSTANCES.get(op_name)
    if instance is None:
        raise ValueError(f"Unsupported operator: {op_name}")
    output_path = temp_dir / f"step_{op_index:02d}_{op_name}.wav"
    metadata = instance.apply(
        current_path,
        output_path,
        operator,
        config,
        seed,
        op_index,
    )
    return output_path, metadata


__all__ = [
    "DeliveryOperator",
    "OPERATOR_REGISTRY",
    "OPERATOR_INSTANCES",
    "apply_operator",
    "standardize_final_output",
    "ResampleOperator",
    "BandLimitOperator",
    "CodecOperator",
    "ReEncodeOperator",
    "PacketLossOperator",
    "NoiseOperator",
    "CallPathOperator",
    "RIROperator",
]
