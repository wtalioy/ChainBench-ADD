#!/usr/bin/env python3
"""Stage 2 clean master preparation for ChainBench-ADD.

Reads the Stage-1 curated manifest and renders standardized clean-parent WAVs:
- mono
- 16 kHz
- 16-bit PCM
- loudness normalization
- leading/trailing silence trim

It then validates the rendered audio and writes Stage-2 manifests and summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common_logging import get_logger, setup_logging
from tqdm.auto import tqdm


LOGGER = get_logger("stage2")


@dataclass
class RenderResult:
    ok: bool
    status: str
    input_row: dict[str, str]
    output_relpath: str | None = None
    output_duration_sec: float | None = None
    output_sample_rate: int | None = None
    output_channels: int | None = None
    output_codec_name: str | None = None
    output_sample_fmt: str | None = None
    output_size_bytes: int | None = None
    error: str | None = None
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage2_clean_master_preparation.json",
        help="Path to the Stage-2 JSON config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit a progress update every N files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally process only the first N rows after filtering.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("zh", "en"),
        help="Restrict processing to one or more languages.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command: list[str], timeout_sec: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=stdout,
            stderr=(stderr + f"\nTIMEOUT after {timeout_sec}s").strip(),
        )


def relative_to_workspace(path: Path, workspace_root: Path) -> str:
    return str(path.resolve().relative_to(workspace_root.resolve())).replace(os.sep, "/")


def resolve_path(path_str: str, workspace_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return workspace_root / path


def load_stage1_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_filter_chain(config: dict[str, Any]) -> str:
    filters: list[str] = []
    trim_cfg = config["trim"]
    if trim_cfg.get("enabled", True):
        threshold_db = float(trim_cfg["threshold_db"])
        start_duration = float(trim_cfg["start_duration_sec"])
        stop_duration = float(trim_cfg["stop_duration_sec"])
        filters.append(
            "silenceremove="
            f"start_periods=1:start_duration={start_duration}:start_threshold={threshold_db}dB:"
            f"stop_periods=-1:stop_duration={stop_duration}:stop_threshold={threshold_db}dB"
        )

    loudnorm_cfg = config["loudnorm"]
    if loudnorm_cfg.get("enabled", True):
        filters.append(
            "loudnorm="
            f"I={float(loudnorm_cfg['integrated_lufs'])}:"
            f"LRA={float(loudnorm_cfg['lra'])}:"
            f"TP={float(loudnorm_cfg['true_peak_db'])}"
        )
    return ",".join(filters)


def ffprobe_audio(path: Path, timeout_sec: int) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name,sample_fmt",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command, timeout_sec=timeout_sec)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        fmt = payload["format"]
        return {
            "duration": float(fmt["duration"]),
            "size": int(fmt["size"]),
            "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
            "codec_name": str(stream.get("codec_name", "")),
            "sample_fmt": str(stream.get("sample_fmt", "")),
        }
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None


def validate_output(probe: dict[str, Any], config: dict[str, Any]) -> str | None:
    audio_cfg = config["audio_output"]
    validation_cfg = config["validation"]

    if probe["sample_rate"] != int(audio_cfg["sample_rate"]):
        return f"unexpected sample_rate={probe['sample_rate']}"
    if probe["channels"] != int(audio_cfg["channels"]):
        return f"unexpected channels={probe['channels']}"
    if probe["codec_name"] != str(audio_cfg["codec_name"]):
        return f"unexpected codec_name={probe['codec_name']}"
    sample_fmt = str(probe["sample_fmt"])
    if not sample_fmt.startswith(str(audio_cfg["sample_fmt"])):
        return f"unexpected sample_fmt={sample_fmt}"
    if probe["duration"] < float(validation_cfg["min_duration_sec"]):
        return f"duration too short: {probe['duration']:.3f}s"
    if probe["duration"] > float(validation_cfg["max_duration_sec"]):
        return f"duration too long: {probe['duration']:.3f}s"
    if probe["size"] <= 0:
        return "empty output file"
    return None


def render_single_row(
    row: dict[str, str],
    config: dict[str, Any],
    workspace_root: Path,
    output_audio_root: Path,
    filter_chain: str,
) -> RenderResult:
    timeout_cfg = config.get("timeouts", {})
    ffmpeg_timeout_sec = int(timeout_cfg.get("ffmpeg_sec", 120))
    ffprobe_timeout_sec = int(timeout_cfg.get("ffprobe_sec", 30))
    input_path = resolve_path(row["stage1_audio_path"], workspace_root)
    if not input_path.exists():
        return RenderResult(
            ok=False,
            status="missing_input",
            input_row=row,
            error=f"missing input file: {input_path}",
        )

    extension = str(config["audio_output"]["extension"])
    output_path = output_audio_root / row["language"] / row["split"] / row["speaker_id"] / f"{row['sample_id']}{extension}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overwrite = bool(config.get("overwrite", False))
    if output_path.exists() and not overwrite:
        probe = ffprobe_audio(output_path, timeout_sec=ffprobe_timeout_sec)
        if probe is not None:
            validation_error = validate_output(probe, config)
            if validation_error is None:
                return RenderResult(
                    ok=True,
                    status="skipped_existing",
                    input_row=row,
                    output_relpath=relative_to_workspace(output_path, workspace_root),
                    output_duration_sec=probe["duration"],
                    output_sample_rate=probe["sample_rate"],
                    output_channels=probe["channels"],
                    output_codec_name=probe["codec_name"],
                    output_sample_fmt=probe["sample_fmt"],
                    output_size_bytes=probe["size"],
                    skipped=True,
                )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(config["audio_output"]["channels"]),
        "-ar",
        str(config["audio_output"]["sample_rate"]),
    ]
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.extend(
        [
            "-c:a",
            str(config["audio_output"]["codec_name"]),
            str(output_path),
        ]
    )

    result = run_command(command, timeout_sec=ffmpeg_timeout_sec)
    if result.returncode != 0:
        status = "ffmpeg_timeout" if result.returncode == 124 else "ffmpeg_failed"
        return RenderResult(
            ok=False,
            status=status,
            input_row=row,
            error=result.stderr.strip() or "ffmpeg failed with unknown error",
        )

    probe = ffprobe_audio(output_path, timeout_sec=ffprobe_timeout_sec)
    if probe is None:
        return RenderResult(
            ok=False,
            status="ffprobe_failed",
            input_row=row,
            error=f"ffprobe failed for output: {output_path}",
        )

    validation_error = validate_output(probe, config)
    if validation_error is not None:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        return RenderResult(
            ok=False,
            status="validation_failed",
            input_row=row,
            error=validation_error,
        )

    return RenderResult(
        ok=True,
        status="rendered",
        input_row=row,
        output_relpath=relative_to_workspace(output_path, workspace_root),
        output_duration_sec=probe["duration"],
        output_sample_rate=probe["sample_rate"],
        output_channels=probe["channels"],
        output_codec_name=probe["codec_name"],
        output_sample_fmt=probe["sample_fmt"],
        output_size_bytes=probe["size"],
    )


def make_stage2_row(result: RenderResult, preprocess_desc: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.input_row)
    parent_id = row["sample_id"]
    row["parent_id"] = parent_id
    row["clean_parent_path"] = result.output_relpath or ""
    row["audio_path"] = result.output_relpath or ""
    row["preprocess_stage"] = "stage2_clean_master"
    row["preprocess_steps"] = json.dumps(preprocess_desc["steps"])
    row["preprocess_params"] = json.dumps(preprocess_desc["params"], sort_keys=True)
    row["source_duration_sec"] = row["duration_sec"]
    row["source_sample_rate"] = row["sample_rate"]
    row["source_channels"] = row["channels"]
    row["source_codec_name"] = row["codec_name"]
    row["duration_sec"] = f"{float(result.output_duration_sec or 0.0):.3f}"
    row["sample_rate"] = str(result.output_sample_rate or "")
    row["channels"] = str(result.output_channels or "")
    row["codec_name"] = result.output_codec_name or ""
    row["sample_fmt"] = result.output_sample_fmt or ""
    row["output_size_bytes"] = str(result.output_size_bytes or "")
    row["chain_family"] = "clean_parent"
    row["operator_seq"] = "[]"
    row["operator_params"] = "{}"
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, Any] = {}
    split_counts_by_lang: dict[str, Counter] = defaultdict(Counter)
    speaker_counts_by_lang: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        language = row["language"]
        split = row["split"]
        split_counts_by_lang[language][split] += 1
        speaker_counts_by_lang[language].add(row["speaker_id"])

    for language, split_counter in split_counts_by_lang.items():
        by_language[language] = {
            "selected_samples": sum(split_counter.values()),
            "selected_speakers": len(speaker_counts_by_lang[language]),
            "split_sample_counts": dict(split_counter),
        }
    return by_language


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = Path.cwd()
    config_path = resolve_path(args.config, workspace_root)
    config = load_json(config_path)
    stage1_manifest_path = resolve_path(config["stage1_manifest"], workspace_root)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_audio_root = output_root / "audio"
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    output_audio_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading Stage-1 manifest from %s", stage1_manifest_path)
    rows = load_stage1_rows(stage1_manifest_path)
    LOGGER.info("loaded %d Stage-1 rows", len(rows))

    if args.language:
        rows = [row for row in rows if row["language"] in set(args.language)]
        LOGGER.info("after language filter: %d rows", len(rows))
    if args.limit > 0:
        rows = rows[: args.limit]
        LOGGER.info("after --limit: %d rows", len(rows))
    if not rows:
        raise RuntimeError("No Stage-1 rows selected for Stage-2 processing")

    filter_chain = build_filter_chain(config)
    LOGGER.info("using ffmpeg filter chain: %s", filter_chain if filter_chain else "<none>")
    LOGGER.info(
        "timeouts: ffmpeg=%ss, ffprobe=%ss",
        int(config.get("timeouts", {}).get("ffmpeg_sec", 120)),
        int(config.get("timeouts", {}).get("ffprobe_sec", 30)),
    )

    preprocess_desc = {
        "steps": [
            "trim_silence",
            "downmix_to_mono",
            "resample_16khz",
            "loudness_normalization",
            "encode_pcm_s16le",
        ],
        "params": {
            "trim": config["trim"],
            "loudnorm": config["loudnorm"],
            "audio_output": config["audio_output"],
            "validation": config["validation"],
        },
    }

    success_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counters = Counter()
    workers = int(config["workers"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                render_single_row,
                row,
                config,
                workspace_root,
                output_audio_root,
                filter_chain,
            ): row
            for row in rows
        }

        total = len(future_map)
        with tqdm(total=total, desc="stage2 render", unit="file", dynamic_ncols=True) as progress:
            for idx, future in enumerate(as_completed(future_map), start=1):
                result = future.result()
                counters[result.status] += 1
                if result.ok:
                    success_rows.append(make_stage2_row(result, preprocess_desc))
                else:
                    failures.append(
                        {
                            "sample_id": result.input_row["sample_id"],
                            "language": result.input_row["language"],
                            "split": result.input_row["split"],
                            "speaker_id": result.input_row["speaker_id"],
                            "status": result.status,
                            "error": result.error or "",
                        }
                    )

                progress.update(1)
                if idx <= 5 or idx % args.log_every == 0 or idx == total:
                    progress.set_postfix(
                        rendered=counters["rendered"],
                        skipped=counters["skipped_existing"],
                        failed=idx - (counters["rendered"] + counters["skipped_existing"]),
                    )

    LOGGER.info("Sorting success rows ...")
    success_rows.sort(key=lambda row: (row["language"], row["split"], row["speaker_id"], row["utterance_id"]))
    if not success_rows:
        raise RuntimeError("Stage-2 produced zero valid clean masters")

    write_csv(manifest_root / "clean_parent_manifest.csv", success_rows)
    for language in sorted({row["language"] for row in success_rows}):
        subset = [row for row in success_rows if row["language"] == language]
        write_csv(manifest_root / f"clean_parent_manifest_{language}.csv", subset)

    with (manifest_root / "stage2_failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "stage1_manifest": relative_to_workspace(stage1_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "total_input_rows": len(rows),
        "successful_rows": len(success_rows),
        "failed_rows": len(failures),
        "status_counts": dict(counters),
        "languages": summarize_rows(success_rows),
    }
    with (manifest_root / "stage2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-2 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.info(
        "Stage-2 finished: success=%d, failed=%d, manifests written to %s",
        len(success_rows),
        len(failures),
        manifest_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
