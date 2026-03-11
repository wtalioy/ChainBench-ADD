#!/usr/bin/env python3
"""Stage 3 spoof clean-parent generation for ChainBench-ADD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common_logging import clean_stream_line, format_elapsed, get_logger, setup_logging
from tqdm.auto import tqdm


LOGGER = get_logger("stage3-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage3_spoof_generation.json",
        help="Path to the Stage-3 config JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--runner-log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level passed to generator batch runners.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Emit Stage-3 progress every N validated spoof samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally process only the first N real clean parents.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write job plans/manifests but do not launch generators.",
    )
    parser.add_argument(
        "--only-generator",
        action="append",
        help="Restrict processing to one or more configured generator keys.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("zh", "en"),
        help="Restrict processing to one or more languages.",
    )
    parser.add_argument(
        "--generators-per-parent",
        type=int,
        default=0,
        help="Override the configured number of assigned generators per parent.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_str: str, workspace_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else workspace_root / path


def relative_to_workspace(path: Path, workspace_root: Path) -> str:
    return str(path.resolve().relative_to(workspace_root.resolve())).replace(os.sep, "/")


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_command_streaming(
    command: list[str],
    cwd: Path,
    log_path: Path,
    log_prefix: str,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n" + " ".join(command) + "\n\nOUTPUT:\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            log_handle.write(raw_line)
            log_handle.flush()
            line = raw_line.rstrip()
            if line:
                LOGGER.info("%s > %s", log_prefix, clean_stream_line(line))

        return process.wait()


def get_conda_env_names() -> set[str]:
    result = run_command(["conda", "env", "list", "--json"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query conda envs: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    envs = set()
    for env_path in payload.get("envs", []):
        envs.add(Path(env_path).name)
    return envs


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    normalized_rows = []
    for row in rows:
        normalized_rows.append({field: row.get(field, "") for field in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_traceback_or_tail(log_path: Path, max_lines: int = 40) -> str:
    if not log_path.exists():
        return "<log file missing>"
    with log_path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if not lines:
        return "<log file empty>"
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith("Traceback (most recent call last):"):
            return "\n".join(lines[idx : idx + max_lines])
    return "\n".join(lines[-max_lines:])


def ffprobe_audio(path: Path) -> dict[str, Any] | None:
    result = run_command(
        [
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
    )
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
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


def build_postprocess_filter_chain(config: dict[str, Any]) -> str:
    post_cfg = config.get("postprocess", {})
    filters: list[str] = []
    trim_cfg = post_cfg.get("trim", {})
    if trim_cfg.get("enabled", False):
        filters.append(
            "silenceremove="
            f"start_periods=1:start_duration={float(trim_cfg['start_duration_sec'])}:start_threshold={float(trim_cfg['threshold_db'])}dB:"
            f"stop_periods=-1:stop_duration={float(trim_cfg['stop_duration_sec'])}:stop_threshold={float(trim_cfg['threshold_db'])}dB"
        )
    loudnorm_cfg = post_cfg.get("loudnorm", {})
    if loudnorm_cfg.get("enabled", False):
        filters.append(
            "loudnorm="
            f"I={float(loudnorm_cfg['integrated_lufs'])}:"
            f"LRA={float(loudnorm_cfg['lra'])}:"
            f"TP={float(loudnorm_cfg['true_peak_db'])}"
        )
    return ",".join(filters)


def postprocess_audio(raw_path: Path, final_path: Path, config: dict[str, Any]) -> str | None:
    post_cfg = config.get("postprocess", {})
    if not post_cfg.get("enabled", False):
        if raw_path.resolve() != final_path.resolve():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(raw_path.read_bytes())
        return None

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp.wav")
    audio_cfg = post_cfg["audio_output"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(audio_cfg["channels"]),
        "-ar",
        str(audio_cfg["sample_rate"]),
    ]
    filter_chain = build_postprocess_filter_chain(config)
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.extend(["-c:a", str(audio_cfg["codec_name"]), str(tmp_path)])
    result = run_command(command)
    if result.returncode != 0:
        return result.stderr.strip() or "ffmpeg postprocess failed"
    tmp_path.replace(final_path)
    return None


def validate_spoof_output(
    probe: dict[str, Any],
    source_duration_sec: float,
    config: dict[str, Any],
) -> str | None:
    validation = config["validation"]
    if probe["size"] <= 0:
        return "empty output file"
    if probe["duration"] < float(validation["min_duration_sec"]):
        return f"duration too short: {probe['duration']:.3f}s"
    if probe["duration"] > float(validation["max_duration_sec"]):
        return f"duration too long: {probe['duration']:.3f}s"
    duration_ratio = probe["duration"] / max(source_duration_sec, 1e-6)
    if duration_ratio < float(validation["min_duration_ratio"]):
        return f"duration ratio too small: {duration_ratio:.3f}"
    if duration_ratio > float(validation["max_duration_ratio"]):
        return f"duration ratio too large: {duration_ratio:.3f}"

    post_cfg = config.get("postprocess", {})
    if post_cfg.get("enabled", False):
        audio_cfg = post_cfg["audio_output"]
        if probe["sample_rate"] != int(audio_cfg["sample_rate"]):
            return f"unexpected sample_rate={probe['sample_rate']}"
        if probe["channels"] != int(audio_cfg["channels"]):
            return f"unexpected channels={probe['channels']}"
        if probe["codec_name"] != str(audio_cfg["codec_name"]):
            return f"unexpected codec_name={probe['codec_name']}"
        if not str(probe["sample_fmt"]).startswith(str(audio_cfg["sample_fmt"])):
            return f"unexpected sample_fmt={probe['sample_fmt']}"
    return None


def choose_prompt_reference(
    row: dict[str, str],
    speaker_rows: list[dict[str, str]],
    generator_key: str,
    seed: int,
) -> dict[str, str]:
    alternatives = [candidate for candidate in speaker_rows if candidate["parent_id"] != row["parent_id"]]
    if not alternatives:
        return row
    alternatives.sort(key=lambda item: item["parent_id"])
    idx = random.Random(f"{seed}:{generator_key}:{row['parent_id']}").randrange(len(alternatives))
    return alternatives[idx]


def get_active_generators(config: dict[str, Any], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    selected = {}
    only = set(args.only_generator or [])
    for key, value in config["generators"].items():
        if not value.get("enabled", False):
            continue
        if only and key not in only:
            continue
        selected[key] = value
    if not selected:
        raise RuntimeError("No active generators selected")
    return selected


def preflight_generators(
    generator_cfgs: dict[str, dict[str, Any]],
    workspace_root: Path,
    plan_only: bool,
) -> None:
    missing_envs = []
    missing_repos = []
    for key, generator in generator_cfgs.items():
        repo_path = workspace_root / generator["repo_path"]
        if not repo_path.exists():
            missing_repos.append(f"{key}:{repo_path}")

    if missing_repos:
        raise RuntimeError("Missing generator repo paths: " + ", ".join(missing_repos))
    if plan_only:
        return

    env_names = get_conda_env_names()
    for key, generator in generator_cfgs.items():
        env_name = generator["conda_env"]
        if env_name not in env_names:
            missing_envs.append(f"{key}:{env_name}")
    if missing_envs:
        raise RuntimeError("Missing generator conda envs: " + ", ".join(missing_envs))


def assign_generators(
    rows: list[dict[str, str]],
    generator_cfgs: dict[str, dict[str, Any]],
    generators_per_parent: int,
    seed: int,
) -> list[dict[str, Any]]:
    counts = Counter()
    tiebreak = {key: idx for idx, key in enumerate(sorted(generator_cfgs))}
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    jobs: list[dict[str, Any]] = []

    rows_by_speaker: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_speaker[(row["language"], row["speaker_id"])].append(row)

    for row in shuffled_rows:
        supported = [
            key
            for key, gen_cfg in generator_cfgs.items()
            if row["language"] in gen_cfg.get("supported_languages", [])
        ]
        if len(supported) < generators_per_parent:
            raise RuntimeError(
                f"Parent {row['parent_id']} language={row['language']} has only {len(supported)} compatible generators"
            )
        supported.sort(key=lambda key: (counts[key], tiebreak[key], key))
        chosen = supported[:generators_per_parent]
        for assignment_idx, generator_key in enumerate(chosen, start=1):
            counts[generator_key] += 1
            prompt_row = choose_prompt_reference(
                row,
                rows_by_speaker[(row["language"], row["speaker_id"])],
                generator_key,
                seed,
            )
            spoof_parent_id = f"{row['parent_id']}__{generator_key}"
            jobs.append(
                {
                    "source_row": row,
                    "generator_key": generator_key,
                    "assignment_idx": assignment_idx,
                    "prompt_row": prompt_row,
                    "spoof_parent_id": spoof_parent_id,
                }
            )
    return jobs


def enrich_jobs(
    assignments: list[dict[str, Any]],
    generator_cfgs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    jobs_by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_root = output_root / "audio_raw"
    final_root = output_root / "audio"
    seed = int(config["seed"])

    for item in assignments:
        row = item["source_row"]
        generator_key = item["generator_key"]
        prompt_row = item["prompt_row"]
        generator = generator_cfgs[generator_key]
        job_id = item["spoof_parent_id"]
        raw_path = raw_root / generator_key / row["language"] / row["split"] / row["speaker_id"] / f"{job_id}.wav"
        final_path = final_root / row["language"] / row["split"] / row["speaker_id"] / f"{job_id}.wav"
        jobs_by_generator[generator_key].append(
            {
                "source_row": row,
                "prompt_row": prompt_row,
                "job_id": job_id,
                "sample_id": job_id,
                "parent_id": job_id,
                "source_parent_id": row["parent_id"],
                "prompt_parent_id": prompt_row["parent_id"],
                "assignment_idx": item["assignment_idx"],
                "generator_key": generator_key,
                "generator_name": generator["generator_name"],
                "generator_family": generator["generator_family"],
                "language": row["language"],
                "split": row["split"],
                "source_corpus": row["source_corpus"],
                "speaker_id": row["speaker_id"],
                "source_speaker_id": row["source_speaker_id"],
                "utterance_id": row["utterance_id"],
                "text": row["transcript"],
                "transcript": row["transcript"],
                "raw_transcript": row.get("raw_transcript", ""),
                "prompt_text": prompt_row["transcript"],
                "prompt_audio_path": str(resolve_path(prompt_row["clean_parent_path"], workspace_root)),
                "source_audio_path": str(resolve_path(row["clean_parent_path"], workspace_root)),
                "source_duration_sec": float(row["duration_sec"]),
                "source_sample_rate": row["sample_rate"],
                "source_channels": row.get("channels", ""),
                "source_codec_name": row.get("codec_name", ""),
                "raw_output_path": str(raw_path.resolve()),
                "output_path": str(raw_path.resolve()),
                "final_output_path": str(final_path.resolve()),
                "raw_output_relpath": relative_to_workspace(raw_path, workspace_root),
                "final_output_relpath": relative_to_workspace(final_path, workspace_root),
                "source_clean_parent_path": row.get("clean_parent_path", ""),
                "license_tag": row.get("license_tag", ""),
                "speaker_gender": row.get("speaker_gender", ""),
                "speaker_age": row.get("speaker_age", ""),
                "speaker_accent": row.get("speaker_accent", ""),
                "speaker_variant": row.get("speaker_variant", ""),
                "sentence_id": row.get("sentence_id", ""),
                "locale": row.get("locale", ""),
                "sentence_domain": row.get("sentence_domain", ""),
                "seed": seed,
            }
        )
    return jobs_by_generator


def run_generator_batch(
    generator_key: str,
    generator: dict[str, Any],
    jobs: list[dict[str, Any]],
    output_root: Path,
    workspace_root: Path,
    runner_log_level: str,
) -> dict[str, Any]:
    jobs_path = output_root / "jobs" / f"{generator_key}.jsonl"
    results_path = output_root / "results" / f"{generator_key}.jsonl"
    adapter_cfg_path = output_root / "jobs" / f"{generator_key}.adapter_config.json"
    log_path = output_root / "logs" / f"{generator_key}.log"
    write_jsonl(jobs_path, jobs)
    adapter_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_cfg_path.write_text(json.dumps(generator["adapter_config"], ensure_ascii=False, indent=2), encoding="utf-8")

    runner_script = workspace_root / "scripts" / "stage3_generator_batch_runner.py"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        generator["conda_env"],
        "python",
        "-u",
        str(runner_script),
        "--adapter",
        generator["adapter"],
        "--repo-path",
        str((workspace_root / generator["repo_path"]).resolve()),
        "--config-path",
        str(adapter_cfg_path.resolve()),
        "--jobs-path",
        str(jobs_path.resolve()),
        "--results-path",
        str(results_path.resolve()),
        "--log-level",
        runner_log_level,
    ]
    LOGGER.info("%s > launch %d jobs in env=%s", generator_key, len(jobs), generator["conda_env"])
    returncode = run_command_streaming(command, cwd=workspace_root, log_path=log_path, log_prefix=generator_key)
    return {
        "generator_key": generator_key,
        "returncode": returncode,
        "results_path": results_path,
        "log_path": log_path,
    }


def materialize_generator_jobs(
    jobs_by_generator: dict[str, list[dict[str, Any]]],
    generator_cfgs: dict[str, dict[str, Any]],
    output_root: Path,
) -> None:
    for generator_key, jobs in jobs_by_generator.items():
        jobs_path = output_root / "jobs" / f"{generator_key}.jsonl"
        adapter_cfg_path = output_root / "jobs" / f"{generator_key}.adapter_config.json"
        write_jsonl(jobs_path, jobs)
        adapter_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_cfg_path.write_text(
            json.dumps(generator_cfgs[generator_key]["adapter_config"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def collect_spoof_rows(
    jobs_by_generator: dict[str, list[dict[str, Any]]],
    generator_cfgs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_root: Path,
    workspace_root: Path,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Counter]]:
    spoof_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stats_by_generator: dict[str, Counter] = defaultdict(Counter)
    processed = 0
    total_results = sum(len(jobs) for jobs in jobs_by_generator.values())
    started_at = time.monotonic()

    with tqdm(total=total_results, desc="stage3 validate", unit="job", dynamic_ncols=True) as validate_progress:
        for generator_key, jobs in jobs_by_generator.items():
            job_map = {job["job_id"]: job for job in jobs}
            results_path = output_root / "results" / f"{generator_key}.jsonl"
            results = load_jsonl(results_path)
            stats = stats_by_generator[generator_key]
            for result in results:
                processed += 1
                job = job_map[result["job_id"]]
                stats[result["status"]] += 1
                if result["status"] == "failed":
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "generation",
                            "error": result.get("error", ""),
                            "source_parent_id": job["source_parent_id"],
                            "prompt_parent_id": job["prompt_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                raw_path = Path(job["raw_output_path"])
                final_path = Path(job["final_output_path"])
                post_error = postprocess_audio(raw_path, final_path, config)
                if post_error is not None:
                    stats["postprocess_failed"] += 1
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "postprocess",
                            "error": post_error,
                            "source_parent_id": job["source_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                probe = ffprobe_audio(final_path)
                if probe is None:
                    stats["ffprobe_failed"] += 1
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "ffprobe",
                            "error": f"ffprobe failed for {final_path}",
                            "source_parent_id": job["source_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                validation_error = validate_spoof_output(probe, float(job["source_duration_sec"]), config)
                if validation_error is not None:
                    stats["validation_failed"] += 1
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "validation",
                            "error": validation_error,
                            "source_parent_id": job["source_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                generator = generator_cfgs[generator_key]
                spoof_rows.append(
                    {
                        "sample_id": job["sample_id"],
                        "parent_id": job["parent_id"],
                        "source_parent_id": job["source_parent_id"],
                        "prompt_parent_id": job["prompt_parent_id"],
                        "split": job["split"],
                        "language": job["language"],
                        "source_corpus": job["source_corpus"],
                        "speaker_id": job["speaker_id"],
                        "source_speaker_id": job["source_speaker_id"],
                        "utterance_id": job["utterance_id"],
                        "transcript": job["transcript"],
                        "raw_transcript": job.get("raw_transcript", ""),
                        "label": "spoof",
                        "generator_family": generator["generator_family"],
                        "generator_name": generator["generator_name"],
                        "chain_family": "clean_parent",
                        "operator_seq": "[]",
                        "operator_params": "{}",
                        "seed": str(job["seed"]),
                        "duration_sec": f"{probe['duration']:.3f}",
                        "sample_rate": str(probe["sample_rate"]),
                        "channels": str(probe["channels"]),
                        "codec_name": probe["codec_name"],
                        "sample_fmt": probe["sample_fmt"],
                        "source_duration_sec": str(job["source_duration_sec"]),
                        "source_sample_rate": job["source_sample_rate"],
                        "source_channels": job.get("source_channels", ""),
                        "source_codec_name": job.get("source_codec_name", ""),
                        "clean_parent_path": job["final_output_relpath"],
                        "audio_path": job["final_output_relpath"],
                        "source_clean_parent_path": job.get("source_clean_parent_path", ""),
                        "raw_generator_output_path": job["raw_output_relpath"],
                        "prompt_audio_path": relative_to_workspace(Path(job["prompt_audio_path"]), workspace_root),
                        "output_size_bytes": str(probe["size"]),
                        "license_tag": job.get("license_tag", ""),
                        "speaker_gender": job.get("speaker_gender", ""),
                        "speaker_age": job.get("speaker_age", ""),
                        "speaker_accent": job.get("speaker_accent", ""),
                        "speaker_variant": job.get("speaker_variant", ""),
                        "sentence_id": job.get("sentence_id", ""),
                        "locale": job.get("locale", ""),
                        "sentence_domain": job.get("sentence_domain", ""),
                        "assignment_idx": str(job["assignment_idx"]),
                        "generator_key": generator_key,
                        "stage3_status": result["status"],
                    }
                )
                stats["validated_ok"] += 1
                validate_progress.update(1)
                if processed <= 5 or processed % log_every == 0 or processed == total_results:
                    validate_progress.set_postfix(
                        generator=generator_key,
                        ok=len(spoof_rows),
                        fail=len(failures),
                        elapsed=format_elapsed(time.monotonic() - started_at),
                    )

            missing = len(jobs) - len(results)
            if missing > 0:
                stats["missing_results"] += missing
                for job in jobs:
                    if job["job_id"] not in {result["job_id"] for result in results}:
                        failures.append(
                            {
                                "job_id": job["job_id"],
                                "generator_key": generator_key,
                                "stage": "results",
                                "error": "missing result entry",
                                "source_parent_id": job["source_parent_id"],
                            }
                        )

    return spoof_rows, failures, stats_by_generator


def summarize_spoof_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, Counter] = defaultdict(Counter)
    by_generator: dict[str, Counter] = defaultdict(Counter)
    speakers_by_language: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_language[row["language"]][row["split"]] += 1
        by_generator[row["generator_key"]][row["language"]] += 1
        speakers_by_language[row["language"]].add(row["speaker_id"])
    return {
        "languages": {
            language: {
                "selected_samples": sum(counter.values()),
                "selected_speakers": len(speakers_by_language[language]),
                "split_sample_counts": dict(counter),
            }
            for language, counter in by_language.items()
        },
        "generators": {key: dict(counter) for key, counter in by_generator.items()},
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    stage3_started_at = time.monotonic()

    workspace_root = Path.cwd()
    config_path = resolve_path(args.config, workspace_root)
    config = load_json(config_path)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage2_manifest_path = resolve_path(config["stage2_manifest"], workspace_root)
    rows = load_csv_rows(stage2_manifest_path)
    LOGGER.info("loaded %d clean parents from %s", len(rows), stage2_manifest_path)
    if args.language:
        allowed_languages = set(args.language)
        rows = [row for row in rows if row["language"] in allowed_languages]
        LOGGER.info("language filter -> %d rows", len(rows))
    if args.limit > 0:
        rows = rows[: args.limit]
        LOGGER.info("limit -> %d rows", len(rows))
    if not rows:
        raise RuntimeError("No Stage-2 rows selected for Stage-3 processing")

    generator_cfgs = get_active_generators(config, args)
    LOGGER.info("generators=%s", ", ".join(sorted(generator_cfgs)))
    preflight_generators(generator_cfgs, workspace_root, args.plan_only)
    if args.plan_only:
        LOGGER.success("preflight ok | checked repos")
    else:
        LOGGER.success("preflight ok | checked repos + envs")

    generators_per_parent = (
        int(args.generators_per_parent)
        if args.generators_per_parent > 0
        else int(config["generators_per_parent"])
    )
    if generators_per_parent > len(generator_cfgs):
        raise RuntimeError(
            f"Requested generators_per_parent={generators_per_parent}, but only {len(generator_cfgs)} active generators selected"
        )

    assignments = assign_generators(
        rows,
        generator_cfgs,
        generators_per_parent,
        int(config["seed"]),
    )
    LOGGER.info(
        "planned jobs=%d from parents=%d with generators_per_parent=%d",
        len(assignments),
        len(rows),
        generators_per_parent,
    )
    jobs_by_generator = enrich_jobs(assignments, generator_cfgs, config, workspace_root, output_root)

    assignment_summary = Counter(item["generator_key"] for item in assignments)
    (output_root / "jobs").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    materialize_generator_jobs(jobs_by_generator, generator_cfgs, output_root)
    plan_path = output_root / "jobs" / "stage3_job_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": relative_to_workspace(config_path, workspace_root),
                "stage2_manifest": relative_to_workspace(stage2_manifest_path, workspace_root),
                "input_clean_parents": len(rows),
                "jobs_total": len(assignments),
                "generators_per_parent": generators_per_parent,
                "jobs_per_generator": dict(assignment_summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.plan_only:
        LOGGER.success("plan-only | wrote job plan to %s", plan_path)
        return 0

    futures = []
    results_meta = []
    max_workers = max(1, int(config.get("workers", 1)))
    generation_started_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for generator_key, jobs in jobs_by_generator.items():
            LOGGER.info(
                "%s > queued jobs=%d",
                generator_key,
                len(jobs),
            )
            futures.append(
                executor.submit(
                    run_generator_batch,
                    generator_key,
                    generator_cfgs[generator_key],
                    jobs,
                    output_root,
                    workspace_root,
                    args.runner_log_level,
                )
            )
        with tqdm(total=len(futures), desc="stage3 generate", unit="gen", dynamic_ncols=True) as generate_progress:
            for future in as_completed(futures):
                result = future.result()
                results_meta.append(result)
                generate_progress.update(1)
                generate_progress.set_postfix(
                    finished=f"{len(results_meta)}/{len(futures)}",
                    elapsed=format_elapsed(time.monotonic() - generation_started_at),
                )
                LOGGER.success(
                    "%s > runner done | rc=%d | finished=%d/%d | elapsed=%s | log=%s",
                    result["generator_key"],
                    result["returncode"],
                    len(results_meta),
                    len(futures),
                    format_elapsed(time.monotonic() - generation_started_at),
                    result["log_path"],
                )

    failed_runners = [item for item in results_meta if item["returncode"] != 0]
    if failed_runners:
        for item in failed_runners:
            LOGGER.error(
                "%s > runner failed | rc=%d | log=%s\n%s",
                item["generator_key"],
                item["returncode"],
                item["log_path"],
                extract_traceback_or_tail(item["log_path"]),
            )
        raise RuntimeError(
            f"Stage-3 generation failed for {len(failed_runners)} runner(s); see logged traceback snippets above"
        )

    spoof_rows, failures, stats_by_generator = collect_spoof_rows(
        jobs_by_generator,
        generator_cfgs,
        config,
        output_root,
        workspace_root,
        args.log_every,
    )
    spoof_rows.sort(key=lambda row: (row["language"], row["split"], row["speaker_id"], row["sample_id"]))

    if not spoof_rows:
        raise RuntimeError("Stage-3 produced zero valid spoof clean parents")

    manifest_root = output_root / "manifests"
    write_csv(manifest_root / "spoof_clean_manifest.csv", spoof_rows)
    for language in sorted({row["language"] for row in spoof_rows}):
        subset = [row for row in spoof_rows if row["language"] == language]
        write_csv(manifest_root / f"spoof_clean_manifest_{language}.csv", subset)

    real_rows = rows
    all_rows = []
    all_fieldnames = []
    fieldname_set = set()
    for collection in (real_rows, spoof_rows):
        for row in collection:
            for key in row.keys():
                if key not in fieldname_set:
                    fieldname_set.add(key)
                    all_fieldnames.append(key)
    all_rows.extend(real_rows)
    all_rows.extend(spoof_rows)
    write_csv(manifest_root / "clean_parent_manifest_all.csv", all_rows, fieldnames=all_fieldnames)

    failures_path = manifest_root / "stage3_failures.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "stage2_manifest": relative_to_workspace(stage2_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "input_clean_parents": len(rows),
        "jobs_total": len(assignments),
        "generators_per_parent": generators_per_parent,
        "valid_spoof_clean_parents": len(spoof_rows),
        "failed_jobs": len(failures),
        "jobs_per_generator": dict(assignment_summary),
        "runner_status": {
            item["generator_key"]: {
                "returncode": item["returncode"],
                "log_path": relative_to_workspace(item["log_path"], workspace_root),
            }
            for item in results_meta
        },
        "generator_result_counts": {
            generator_key: dict(counter) for generator_key, counter in stats_by_generator.items()
        },
        "spoof_stats": summarize_spoof_rows(spoof_rows),
    }
    summary_path = manifest_root / "stage3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-3 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.success(
        "finished | elapsed=%s | valid=%d | fail=%d | manifests=%s",
        format_elapsed(time.monotonic() - stage3_started_at),
        len(spoof_rows),
        len(failures),
        manifest_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
