"""Single-job rendering and manifest building for stage4."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lib.audio import ffprobe_audio
from lib.config import relative_to_workspace

from .operators import apply_operator, standardize_final_output


def derive_render_seed(job: dict[str, Any], config: dict[str, Any]) -> int:
    row = job["source_row"]
    base_seed = int(row.get("seed") or config["seed"])
    stable_offset = int(hashlib.sha1(job["sample_id"].encode("utf-8")).hexdigest()[:8], 16) % 100000
    return base_seed + stable_offset


def validate_final_output(probe: dict[str, Any], config: dict[str, Any]) -> str | None:
    validation = config["validation"]
    if probe["duration"] < float(validation["min_duration_sec"]):
        return f"duration too short: {probe['duration']:.3f}s"
    if probe["duration"] > float(validation["max_duration_sec"]):
        return f"duration too long: {probe['duration']:.3f}s"
    if probe["sample_rate"] != int(validation["sample_rate"]):
        return f"unexpected sample rate: {probe['sample_rate']}"
    if probe["channels"] != int(validation["channels"]):
        return f"unexpected channels: {probe['channels']}"
    if probe["codec_name"] != str(validation["codec_name"]):
        return f"unexpected codec name: {probe['codec_name']}"
    if probe.get("size", 0) <= 0:
        return "empty output file"
    return None


def should_probe_intermediate_outputs(config: dict[str, Any]) -> bool:
    trace_cfg = config.get("trace", {})
    return bool(trace_cfg.get("probe_intermediate_outputs", False))


def should_store_intermediate_paths(config: dict[str, Any]) -> bool:
    trace_cfg = config.get("trace", {})
    return bool(trace_cfg.get("store_intermediate_paths", True))


def should_write_trace_json(config: dict[str, Any]) -> bool:
    trace_cfg = config.get("trace", {})
    return bool(trace_cfg.get("write_json", True))


def should_copy_compatible_outputs(config: dict[str, Any]) -> bool:
    perf_cfg = config.get("performance", {})
    return bool(perf_cfg.get("copy_compatible_outputs", True))


def is_probe_compatible_with_final_output(probe: dict[str, Any], config: dict[str, Any]) -> bool:
    final_cfg = config["final_output"]
    return (
        probe["sample_rate"] == int(final_cfg["sample_rate"])
        and probe["channels"] == int(final_cfg["channels"])
        and probe["codec_name"] == str(final_cfg["codec_name"])
        and probe.get("size", 0) > 0
    )


def append_trace_step(
    trace_steps: list[dict[str, Any]],
    current_path: Path,
    next_path: Path,
    op_index: int,
    op_meta: dict[str, Any],
    config: dict[str, Any],
) -> None:
    step: dict[str, Any] = {
        "step_index": op_index,
        "operator": op_meta,
    }
    if should_store_intermediate_paths(config):
        step["input_path"] = str(current_path)
        step["output_path"] = str(next_path)
    if should_probe_intermediate_outputs(config):
        step["ffprobe"] = ffprobe_audio(next_path)
    trace_steps.append(step)


def write_trace_if_enabled(trace: dict[str, Any], trace_path: Path, config: dict[str, Any]) -> None:
    if not should_write_trace_json(config):
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def render_single_job(
    job: dict[str, Any],
    config: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    row = job["source_row"]
    family_name = job["family_name"]
    source_audio_path = Path(job["source_audio_path_abs"])
    if not source_audio_path.exists():
        return {
            "status": "failed",
            "job": job,
            "error": f"missing source audio: {source_audio_path}",
        }

    family_dir_name = family_name
    final_cfg = config["final_output"]
    extension = final_cfg["extension"]
    output_audio_path = (
        output_root / "audio" / family_dir_name / row["language"] / row["split"] / row["speaker_id"]
        / f"{job['sample_id']}{extension}"
    )
    trace_path = (
        output_root / "traces" / family_dir_name / row["language"] / row["split"] / row["speaker_id"]
        / f"{job['sample_id']}.json"
    )

    if output_audio_path.exists() and not config.get("overwrite", False):
        probe = ffprobe_audio(output_audio_path)
        derived_seed = derive_render_seed(job, config)
        if probe is not None and validate_final_output(probe, config) is None:
            trace_payload: dict[str, Any] | list[Any] = []
            if should_write_trace_json(config) and trace_path.exists():
                try:
                    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    trace_payload = []
            return {
                "status": "skipped_existing",
                "job": job,
                "output_audio_path": output_audio_path,
                "trace_path": trace_path if should_write_trace_json(config) else None,
                "probe": probe,
                "trace": trace_payload,
                "seed": derived_seed,
            }

    trace_steps: list[dict[str, Any]] = []
    seed = derive_render_seed(job, config)
    try:
        source_probe: dict[str, Any] | None = None
        if not job["operators"] and should_copy_compatible_outputs(config):
            source_probe = ffprobe_audio(source_audio_path)
            if source_probe is not None and validate_final_output(source_probe, config) is None:
                output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_audio_path, output_audio_path)
                trace = {
                    "job_id": job["job_id"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                    "family_name": family_name,
                    "template_id": job["template_id"],
                    "source_audio_path": relative_to_workspace(source_audio_path, workspace_root),
                    "output_audio_path": relative_to_workspace(output_audio_path, workspace_root),
                    "operators": job["operators"],
                    "variant_index": int(job.get("variant_index", 0)),
                    "seed": seed,
                    "steps": trace_steps,
                    "final_probe": source_probe,
                }
                write_trace_if_enabled(trace, trace_path, config)
                return {
                    "status": "ok",
                    "job": job,
                    "output_audio_path": output_audio_path,
                    "trace_path": trace_path if should_write_trace_json(config) else None,
                    "probe": source_probe,
                    "trace": trace,
                    "seed": seed,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            current_path = source_audio_path

            for op_index, operator in enumerate(job["operators"], start=1):
                next_path, op_meta = apply_operator(
                    current_path, operator, temp_dir, op_index, config, seed
                )
                append_trace_step(trace_steps, current_path, next_path, op_index, op_meta, config)
                current_path = next_path

            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            final_probe = ffprobe_audio(current_path)
            if (
                final_probe is not None
                and should_copy_compatible_outputs(config)
                and is_probe_compatible_with_final_output(final_probe, config)
            ):
                shutil.copy2(current_path, output_audio_path)
            else:
                standardize_final_output(current_path, output_audio_path, config)
                final_probe = ffprobe_audio(output_audio_path)
                if final_probe is None:
                    return {
                        "status": "failed",
                        "job": job,
                        "error": f"ffprobe failed for final output: {output_audio_path}",
                    }
            validation_error = validate_final_output(final_probe, config)
            if validation_error is not None:
                return {"status": "failed", "job": job, "error": validation_error}

            trace = {
                "job_id": job["job_id"],
                "sample_id": job["sample_id"],
                "parent_id": job["parent_id"],
                "family_name": family_name,
                "template_id": job["template_id"],
                "source_audio_path": relative_to_workspace(source_audio_path, workspace_root),
                "output_audio_path": relative_to_workspace(output_audio_path, workspace_root),
                "operators": job["operators"],
                "variant_index": int(job.get("variant_index", 0)),
                "seed": seed,
                "steps": trace_steps,
                "final_probe": final_probe,
            }
            write_trace_if_enabled(trace, trace_path, config)
            return {
                "status": "ok",
                "job": job,
                "output_audio_path": output_audio_path,
                "trace_path": trace_path if should_write_trace_json(config) else None,
                "probe": final_probe,
                "trace": trace,
                "seed": seed,
            }
    except Exception as exc:
        return {
            "status": "failed",
            "job": job,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_manifest_row(
    job: dict[str, Any],
    render_result: dict[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    row = job["source_row"]
    probe = render_result["probe"]
    operator_seq = [op["op"] for op in job["operators"]]
    trace = render_result.get("trace")

    codec = ""
    bitrate = ""
    packet_loss = ""
    bandwidth_mode = ""
    snr = ""
    rt60 = ""
    rir_backend = ""
    room_dim = ""
    distance = ""
    for op in job["operators"]:
        if op["op"] in {"codec", "reencode"}:
            codec = op["codec"]
            bitrate = op.get("bitrate", bitrate)
        elif op["op"] == "packet_loss":
            packet_loss = str(op["loss_rate_pct"])
        elif op["op"] == "bandlimit":
            bandwidth_mode = op["mode"]
        elif op["op"] == "noise":
            snr = str(op["snr_db"])
        elif op["op"] == "rir":
            rt60 = str(op["rt60"])
            room_dim = json.dumps(op["room"]["room_dim"])
            distance = str(op["distance"])
    if isinstance(trace, dict):
        for step in trace.get("steps", []):
            operator = step.get("operator", {})
            if operator.get("op") == "rir":
                rir_backend = str(operator.get("backend", ""))
                break

    return {
        "sample_id": job["sample_id"],
        "parent_id": job["parent_id"],
        "split": row["split"],
        "language": row["language"],
        "source_corpus": row["source_corpus"],
        "speaker_id": row["speaker_id"],
        "source_speaker_id": row["source_speaker_id"],
        "utterance_id": row["utterance_id"],
        "transcript": row["transcript"],
        "raw_transcript": row.get("raw_transcript", ""),
        "label": row["label"],
        "generator_family": row["generator_family"],
        "generator_name": row["generator_name"],
        "chain_family": job["family_name"],
        "chain_template_id": job["template_id"],
        "chain_variant_index": str(int(job.get("variant_index", 0))),
        "operator_seq": json.dumps(operator_seq),
        "operator_params": json.dumps(job["operators"], ensure_ascii=False, sort_keys=True),
        "seed": str(render_result.get("seed", "")),
        "duration_sec": f"{probe['duration']:.3f}",
        "sample_rate": str(probe["sample_rate"]),
        "channels": str(probe["channels"]),
        "codec_name": probe["codec_name"],
        "sample_fmt": probe.get("sample_fmt", ""),
        "snr": snr,
        "rt60": rt60,
        "rir_backend": rir_backend,
        "room_dim": room_dim,
        "distance": distance,
        "codec": codec,
        "bitrate": bitrate,
        "packet_loss": packet_loss,
        "bandwidth_mode": bandwidth_mode,
        "clean_parent_path": row["clean_parent_path"],
        "audio_path": relative_to_workspace(render_result["output_audio_path"], workspace_root),
        "trace_path": (
            relative_to_workspace(render_result["trace_path"], workspace_root)
            if render_result.get("trace_path") is not None
            else ""
        ),
        "license_tag": row.get("license_tag", ""),
    }


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_language: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_family[row["chain_family"]][row["language"]] += 1
        by_language[row["language"]][row["split"]] += 1
    return {
        "families": {family: dict(counter) for family, counter in by_family.items()},
        "languages": {
            language: {
                "selected_samples": sum(counter.values()),
                "split_sample_counts": dict(counter),
            }
            for language, counter in by_language.items()
        },
    }
