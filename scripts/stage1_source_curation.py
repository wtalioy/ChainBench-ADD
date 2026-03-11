#!/usr/bin/env python3
"""Stage 1 source curation for ChainBench-ADD.

This script:
1. Scans AISHELL-3 and Common Voice metadata.
2. Applies transcript and audio-quality filters.
3. Samples target speakers and utterances deterministically.
4. Assigns speaker-disjoint train/dev/test splits.
5. Organizes selected raw audio via symlinks and writes manifests/stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
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


LOGGER = get_logger("stage1")
EN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
MEAN_VOL_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
MAX_VOL_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
SILENCE_DUR_RE = re.compile(r"silence_duration:\s*(\d+(?:\.\d+)?)")


@dataclass
class Candidate:
    source_speaker_id: str
    utterance_id: str
    transcript: str
    raw_transcript: str
    source_audio_path: str
    source_split: str
    source_corpus: str
    language: str
    license_tag: str
    speaker_meta: dict[str, str]
    extra_meta: dict[str, str]


@dataclass
class ProbedCandidate:
    candidate: Candidate
    duration: float
    sample_rate: int
    channels: int
    codec_name: str


@dataclass
class AcceptedSample:
    candidate: Candidate
    duration: float
    sample_rate: int
    channels: int
    codec_name: str
    mean_volume_db: float
    max_volume_db: float
    silence_duration_sec: float
    speech_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage1_curation.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-every-speakers",
        type=int,
        default=10,
        help="Emit a progress log every N speakers during curation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def counter_summary(counter: Counter, top_k: int = 5) -> str:
    if not counter:
        return "none"
    parts = [f"{key}={value}" for key, value in counter.most_common(top_k)]
    return ", ".join(parts)


def normalize_english_transcript(text: str) -> str | None:
    tokens = EN_TOKEN_RE.findall(text.strip())
    if len(tokens) < 4:
        return None
    if tokens and all(token.isdigit() for token in tokens):
        return None
    return " ".join(tokens)


def normalize_aishell_transcript(text: str) -> str | None:
    tokens = text.strip().split()
    chars = "".join(tokens[::2]).strip()
    if len(ZH_CHAR_RE.findall(chars)) < 4:
        return None
    if chars.isdigit():
        return None
    return chars


def load_aishell_speaker_meta(dataset_root: Path) -> dict[str, dict[str, str]]:
    speaker_meta: dict[str, dict[str, str]] = {}
    meta_path = dataset_root / "spk-info.txt"
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            speaker_meta[parts[0]] = {
                "age_group": parts[1],
                "gender": parts[2],
                "accent": parts[3],
            }
    return speaker_meta


def load_aishell_candidates(
    lang_cfg: dict[str, Any],
    counters: dict[str, Counter],
) -> dict[str, list[Candidate]]:
    dataset_root = Path(lang_cfg["dataset_root"])
    speaker_meta = load_aishell_speaker_meta(dataset_root)
    speaker_to_candidates: dict[str, list[Candidate]] = defaultdict(list)
    lines_seen = 0

    for source_split in ("train", "test"):
        transcript_path = dataset_root / source_split / "content.txt"
        wav_root = dataset_root / source_split / "wav"
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                lines_seen += 1
                line = line.rstrip("\n")
                if not line:
                    continue
                utterance_id, raw_transcript = line.split("\t", 1)
                transcript = normalize_aishell_transcript(raw_transcript)
                if transcript is None:
                    counters["text"]["rejected_short_or_numeric"] += 1
                    continue
                source_speaker_id = utterance_id[:7]
                audio_path = wav_root / source_speaker_id / utterance_id
                if not audio_path.exists():
                    counters["text"]["missing_audio_file"] += 1
                    continue
                speaker_to_candidates[source_speaker_id].append(
                    Candidate(
                        source_speaker_id=source_speaker_id,
                        utterance_id=Path(utterance_id).stem,
                        transcript=transcript,
                        raw_transcript=raw_transcript,
                        source_audio_path=str(audio_path),
                        source_split=source_split,
                        source_corpus=lang_cfg["source_corpus"],
                        language="zh",
                        license_tag=lang_cfg["license_tag"],
                        speaker_meta=speaker_meta.get(source_speaker_id, {}),
                        extra_meta={},
                    )
                )
                counters["text"]["accepted"] += 1
                if lines_seen % 20000 == 0:
                    LOGGER.info(
                        "[zh] scanned %d transcript rows, valid rows=%d, speakers=%d",
                        lines_seen,
                        counters["text"]["accepted"],
                        len(speaker_to_candidates),
                    )
    return speaker_to_candidates


def load_common_voice_candidates(
    lang_cfg: dict[str, Any],
    counters: dict[str, Counter],
) -> dict[str, list[Candidate]]:
    dataset_root = Path(lang_cfg["dataset_root"])
    tsv_path = dataset_root / "validated.tsv"
    clips_root = dataset_root / "clips"
    speaker_to_candidates: dict[str, list[Candidate]] = defaultdict(list)
    rows_seen = 0

    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows_seen += 1
            transcript = normalize_english_transcript(row.get("sentence", ""))
            if transcript is None:
                counters["text"]["rejected_short_or_numeric"] += 1
                continue
            source_speaker_id = row["client_id"]
            audio_path = clips_root / row["path"]
            if not audio_path.exists():
                counters["text"]["missing_audio_file"] += 1
                continue
            speaker_to_candidates[source_speaker_id].append(
                Candidate(
                    source_speaker_id=source_speaker_id,
                    utterance_id=Path(row["path"]).stem,
                    transcript=transcript,
                    raw_transcript=row.get("sentence", ""),
                    source_audio_path=str(audio_path),
                    source_split="validated",
                    source_corpus=lang_cfg["source_corpus"],
                    language="en",
                    license_tag=lang_cfg["license_tag"],
                    speaker_meta={
                        "age": row.get("age", "") or "",
                        "gender": row.get("gender", "") or "",
                        "accents": row.get("accents", "") or "",
                        "variant": row.get("variant", "") or "",
                    },
                    extra_meta={
                        "sentence_id": row.get("sentence_id", "") or "",
                        "locale": row.get("locale", "") or "",
                        "sentence_domain": row.get("sentence_domain", "") or "",
                    },
                )
            )
            counters["text"]["accepted"] += 1
            if rows_seen % 100000 == 0:
                LOGGER.info(
                    "[en] scanned %d validated rows, valid rows=%d, speakers=%d",
                    rows_seen,
                    counters["text"]["accepted"],
                    len(speaker_to_candidates),
                )
    return speaker_to_candidates


def ffprobe_audio(path: str) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    result = run_command(command)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        fmt = payload["format"]
        return {
            "duration": float(fmt["duration"]),
            "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
            "codec_name": stream.get("codec_name", ""),
        }
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None


def analyze_audio_quality(path: str, silence_noise_db: int, silence_min_duration: float) -> dict[str, float] | None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        path,
        "-af",
        f"silencedetect=noise={silence_noise_db}dB:d={silence_min_duration},volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = run_command(command)
    if result.returncode != 0:
        return None

    stderr = result.stderr
    mean_match = MEAN_VOL_RE.search(stderr)
    max_match = MAX_VOL_RE.search(stderr)
    if mean_match is None or max_match is None:
        return None

    silence_total = sum(float(match.group(1)) for match in SILENCE_DUR_RE.finditer(stderr))
    return {
        "mean_volume_db": float(mean_match.group(1)),
        "max_volume_db": float(max_match.group(1)),
        "silence_duration_sec": silence_total,
    }


def parallel_map(items: list[Any], worker, max_workers: int) -> list[tuple[Any, Any]]:
    if not items:
        return []
    results: list[tuple[Any, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, item): item for item in items}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                results.append((item, future.result()))
            except Exception:
                results.append((item, None))
    return results


def duration_rank(duration: float, preferred_min: float, preferred_max: float) -> tuple[int, float]:
    in_preferred = preferred_min <= duration <= preferred_max
    if in_preferred:
        return (0, abs(duration - ((preferred_min + preferred_max) / 2.0)))
    if duration < preferred_min:
        return (1, preferred_min - duration)
    return (1, duration - preferred_max)


def resolve_max_audio_checks(num_candidates: int, lang_cfg: dict[str, Any]) -> int:
    configured = int(lang_cfg.get("max_audio_checks_per_speaker", 0) or 0)
    if configured <= 0:
        return num_candidates
    return min(num_candidates, configured)


def curate_single_speaker(
    candidates: list[Candidate],
    lang_cfg: dict[str, Any],
    filters: dict[str, Any],
    workers: int,
    rng: random.Random,
    language: str,
    source_speaker_id: str,
) -> tuple[list[AcceptedSample] | None, dict[str, int]]:
    stats = Counter()
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    target_utts = lang_cfg["target_utterances_per_speaker"]
    min_utts = lang_cfg["min_utterances_per_speaker"]
    max_checks = resolve_max_audio_checks(len(shuffled), lang_cfg)
    batch_size = min(12, max(4, workers * 2))

    accepted: list[AcceptedSample] = []
    checked = 0
    cursor = 0

    while cursor < len(shuffled) and checked < max_checks and len(accepted) < target_utts:
        remaining_budget = max_checks - checked
        batch = shuffled[cursor : cursor + min(batch_size, remaining_budget)]
        cursor += len(batch)

        probed_batch: list[ProbedCandidate] = []
        for candidate, probe in parallel_map(batch, lambda item: ffprobe_audio(item.source_audio_path), workers):
            checked += 1
            if probe is None:
                stats["audio_probe_failed"] += 1
                continue
            duration = float(probe["duration"])
            if duration < filters["min_duration_sec"] or duration > filters["max_duration_sec"]:
                stats["duration_rejected"] += 1
                continue
            probed_batch.append(
                ProbedCandidate(
                    candidate=candidate,
                    duration=duration,
                    sample_rate=int(probe["sample_rate"]),
                    channels=int(probe["channels"]),
                    codec_name=str(probe["codec_name"]),
                )
            )

        analyzed = parallel_map(
            probed_batch,
            lambda item: analyze_audio_quality(
                item.candidate.source_audio_path,
                filters["silence_noise_threshold_db"],
                filters["silence_min_duration_sec"],
            ),
            workers,
        )

        for probed, quality in analyzed:
            if quality is None:
                stats["audio_quality_failed"] += 1
                continue

            mean_volume_db = float(quality["mean_volume_db"])
            max_volume_db = float(quality["max_volume_db"])
            silence_duration_sec = float(quality["silence_duration_sec"])
            speech_ratio = max(0.0, 1.0 - (silence_duration_sec / probed.duration))

            if mean_volume_db < filters["min_mean_volume_db"]:
                stats["low_volume_rejected"] += 1
                continue
            if max_volume_db > filters["max_peak_volume_db"]:
                stats["clipping_rejected"] += 1
                continue
            if speech_ratio < filters["min_speech_ratio"]:
                stats["low_speech_ratio_rejected"] += 1
                continue

            accepted.append(
                AcceptedSample(
                    candidate=probed.candidate,
                    duration=probed.duration,
                    sample_rate=probed.sample_rate,
                    channels=probed.channels,
                    codec_name=probed.codec_name,
                    mean_volume_db=mean_volume_db,
                    max_volume_db=max_volume_db,
                    silence_duration_sec=silence_duration_sec,
                    speech_ratio=speech_ratio,
                )
            )

        LOGGER.info(
            "[%s][speaker=%s] checked %d/%d candidates in latest batch, accepted=%d, rejected=%s",
            language,
            source_speaker_id,
            checked,
            max_checks,
            len(accepted),
            counter_summary(stats),
        )

    if len(accepted) < min_utts:
        stats["speaker_rejected_insufficient_valid_utterances"] += 1
        LOGGER.info(
            "[%s][speaker=%s] rejected after checking %d/%d candidates: accepted=%d (< %d), stats=%s",
            language,
            source_speaker_id,
            checked,
            max_checks,
            len(accepted),
            min_utts,
            counter_summary(stats, top_k=8),
        )
        return None, dict(stats)

    accepted.sort(
        key=lambda item: (
            duration_rank(
                item.duration,
                filters["preferred_min_duration_sec"],
                filters["preferred_max_duration_sec"],
            ),
            -item.speech_ratio,
            abs(item.mean_volume_db + 20.0),
            item.candidate.utterance_id,
        )
    )
    LOGGER.info(
        "[%s][speaker=%s] accepted with %d valid utterances after %d checks",
        language,
        source_speaker_id,
        len(accepted),
        checked,
    )
    return accepted[:target_utts], dict(stats)


def compute_split_counts(total: int, split_cfg: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in split_cfg.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    priorities = sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for name, _ in priorities[:remainder]:
        counts[name] += 1
    return counts


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def relative_to_workspace(path: Path, workspace_root: Path) -> str:
    return str(path.resolve().relative_to(workspace_root.resolve())).replace(os.sep, "/")


def assign_splits(
    selected_speakers: list[dict[str, Any]],
    split_cfg: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    split_counts = compute_split_counts(len(selected_speakers), split_cfg)
    shuffled = list(selected_speakers)
    rng.shuffle(shuffled)
    split_map: dict[str, str] = {}
    cursor = 0
    for split_name in ("train", "dev", "test"):
        count = split_counts.get(split_name, 0)
        for speaker_bundle in shuffled[cursor : cursor + count]:
            split_map[speaker_bundle["speaker_id"]] = split_name
        cursor += count
    return split_map


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    workspace_root = Path.cwd()
    config_path = workspace_root / args.config
    config = load_json(config_path)
    output_root = workspace_root / config["output_root"]
    manifest_root = output_root / "manifests"
    raw_root = output_root / "raw"
    manifest_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    workers = int(config["workers"])
    filters = config["audio_filters"]

    all_manifest_rows: list[dict[str, Any]] = []
    selected_speaker_summary: dict[str, list[dict[str, Any]]] = {}
    overall_summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "seed": seed,
        "workers": workers,
        "languages": {},
    }

    for language in ("zh", "en"):
        lang_cfg = config["languages"][language]
        counters: dict[str, Counter] = {
            "text": Counter(),
            "selection": Counter(),
        }
        LOGGER.info("[%s] loading metadata from %s", language, lang_cfg["dataset_root"])
        if language == "zh":
            speaker_to_candidates = load_aishell_candidates(lang_cfg, counters)
        else:
            speaker_to_candidates = load_common_voice_candidates(lang_cfg, counters)

        min_text_candidates = lang_cfg["min_utterances_per_speaker"]
        candidate_speakers = [
            speaker_id
            for speaker_id, items in speaker_to_candidates.items()
            if len(items) >= min_text_candidates
        ]
        counters["selection"]["candidate_speakers_after_text_filter"] = len(candidate_speakers)
        LOGGER.info(
            "[%s] metadata scan complete: %d candidate speakers, text counters=%s",
            language,
            len(candidate_speakers),
            counter_summary(counters["text"], top_k=8),
        )
        if len(candidate_speakers) < lang_cfg["target_speakers"]:
            raise RuntimeError(
                f"{language}: only {len(candidate_speakers)} candidate speakers available; "
                f"target is {lang_cfg['target_speakers']}"
            )

        rng = random.Random(seed + (11 if language == "zh" else 17))
        rng.shuffle(candidate_speakers)

        speaker_bundles: list[dict[str, Any]] = []
        speaker_logs: list[dict[str, Any]] = []
        LOGGER.info(
            "[%s] curating speakers: target=%d, min_utts=%d, target_utts=%d",
            language,
            lang_cfg["target_speakers"],
            lang_cfg["min_utterances_per_speaker"],
            lang_cfg["target_utterances_per_speaker"],
        )
        with tqdm(
            total=len(candidate_speakers),
            desc=f"stage1 {language} speakers",
            unit="spk",
            dynamic_ncols=True,
        ) as progress:
            for source_speaker_id in candidate_speakers:
                if len(speaker_bundles) >= lang_cfg["target_speakers"]:
                    break

                accepted_samples, speaker_stats = curate_single_speaker(
                    speaker_to_candidates[source_speaker_id],
                    lang_cfg,
                    filters,
                    workers,
                    random.Random(f"{seed}:{language}:{source_speaker_id}"),
                    language,
                    source_speaker_id,
                )
                log_entry = {
                    "source_speaker_id": source_speaker_id,
                    "candidate_utterances_after_text_filter": len(speaker_to_candidates[source_speaker_id]),
                    "status": "accepted" if accepted_samples is not None else "rejected",
                    "stats": speaker_stats,
                }
                speaker_logs.append(log_entry)
                progress.update(1)

                if accepted_samples is None:
                    counters["selection"]["rejected_speakers_audio_filters"] += 1
                else:
                    speaker_bundles.append(
                        {
                            "source_speaker_id": source_speaker_id,
                            "samples": accepted_samples,
                            "speaker_meta": accepted_samples[0].candidate.speaker_meta,
                        }
                    )
                    counters["selection"]["accepted_speakers"] += 1
                    counters["selection"]["accepted_samples"] += len(accepted_samples)

                if progress.n <= 3 or progress.n % args.log_every_speakers == 0:
                    progress.set_postfix(
                        accepted=len(speaker_bundles),
                        target=lang_cfg["target_speakers"],
                        rejected=counters["selection"]["rejected_speakers_audio_filters"],
                        samples=counters["selection"]["accepted_samples"],
                    )

        target_shortfall = lang_cfg["target_speakers"] - len(speaker_bundles)
        if target_shortfall > 0:
            if config.get("allow_partial_target", False):
                LOGGER.warning(
                    "[%s] target shortfall: accepted %d speakers out of requested %d; proceeding with partial dataset",
                    language,
                    len(speaker_bundles),
                    lang_cfg["target_speakers"],
                )
            else:
                raise RuntimeError(
                    f"{language}: accepted {len(speaker_bundles)} speakers, "
                    f"but target is {lang_cfg['target_speakers']}"
                )

        speaker_bundles.sort(key=lambda bundle: bundle["source_speaker_id"])
        for idx, bundle in enumerate(speaker_bundles, start=1):
            bundle["speaker_id"] = f"{lang_cfg['speaker_id_prefix']}{idx:04d}"

        split_map = assign_splits(
            speaker_bundles,
            config["splits"],
            random.Random(seed + (101 if language == "zh" else 103)),
        )

        selected_speakers_payload: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        split_speaker_counts = Counter(split_map.values())
        split_sample_counts = Counter()
        LOGGER.info("[%s] materializing curated layout and manifests", language)

        for bundle in speaker_bundles:
            speaker_id = bundle["speaker_id"]
            split = split_map[speaker_id]
            selected_speakers_payload.append(
                {
                    "speaker_id": speaker_id,
                    "source_speaker_id": bundle["source_speaker_id"],
                    "split": split,
                    "language": language,
                    "source_corpus": lang_cfg["source_corpus"],
                    "speaker_meta": bundle["speaker_meta"],
                    "selected_utterances": [sample.candidate.utterance_id for sample in bundle["samples"]],
                }
            )
            for sample in bundle["samples"]:
                src = Path(sample.candidate.source_audio_path)
                dst = raw_root / language / split / speaker_id / f"{sample.candidate.utterance_id}{src.suffix}"
                ensure_symlink(src, dst)
                sample_id = f"{speaker_id}_{sample.candidate.utterance_id}"
                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "language": language,
                    "source_corpus": sample.candidate.source_corpus,
                    "speaker_id": speaker_id,
                    "source_speaker_id": sample.candidate.source_speaker_id,
                    "utterance_id": sample.candidate.utterance_id,
                    "transcript": sample.candidate.transcript,
                    "raw_transcript": sample.candidate.raw_transcript,
                    "label": "bona_fide",
                    "generator_family": "none",
                    "generator_name": "none",
                    "chain_family": "source_clean",
                    "operator_seq": "[]",
                    "operator_params": "{}",
                    "seed": seed,
                    "duration_sec": f"{sample.duration:.3f}",
                    "sample_rate": sample.sample_rate,
                    "channels": sample.channels,
                    "codec_name": sample.codec_name,
                    "mean_volume_db": f"{sample.mean_volume_db:.2f}",
                    "max_volume_db": f"{sample.max_volume_db:.2f}",
                    "silence_duration_sec": f"{sample.silence_duration_sec:.3f}",
                    "effective_speech_ratio": f"{sample.speech_ratio:.3f}",
                    "source_split": sample.candidate.source_split,
                    "source_audio_path": relative_to_workspace(src, workspace_root),
                    "stage1_audio_path": relative_to_workspace(dst, workspace_root),
                    "license_tag": sample.candidate.license_tag,
                    "speaker_gender": sample.candidate.speaker_meta.get("gender", ""),
                    "speaker_age": sample.candidate.speaker_meta.get("age", sample.candidate.speaker_meta.get("age_group", "")),
                    "speaker_accent": sample.candidate.speaker_meta.get("accent", sample.candidate.speaker_meta.get("accents", "")),
                    "speaker_variant": sample.candidate.speaker_meta.get("variant", ""),
                    "sentence_id": sample.candidate.extra_meta.get("sentence_id", ""),
                    "locale": sample.candidate.extra_meta.get("locale", ""),
                    "sentence_domain": sample.candidate.extra_meta.get("sentence_domain", ""),
                }
                manifest_rows.append(row)
                split_sample_counts[split] += 1

        manifest_rows.sort(key=lambda row: (row["language"], row["split"], row["speaker_id"], row["utterance_id"]))
        all_manifest_rows.extend(manifest_rows)
        selected_speaker_summary[language] = selected_speakers_payload

        language_summary = {
            "source_corpus": lang_cfg["source_corpus"],
            "target_speakers": lang_cfg["target_speakers"],
            "selected_speakers": len(speaker_bundles),
            "target_shortfall": max(0, lang_cfg["target_speakers"] - len(speaker_bundles)),
            "target_utterances_per_speaker": lang_cfg["target_utterances_per_speaker"],
            "min_utterances_per_speaker": lang_cfg["min_utterances_per_speaker"],
            "selected_samples": len(manifest_rows),
            "candidate_speakers_after_text_filter": len(candidate_speakers),
            "split_speaker_counts": dict(split_speaker_counts),
            "split_sample_counts": dict(split_sample_counts),
            "text_filter_counters": dict(counters["text"]),
            "selection_counters": dict(counters["selection"]),
            "speaker_logs_path": f"manifests/{language}_speaker_logs.json",
            "speaker_map_path": f"manifests/{language}_selected_speakers.json",
        }
        overall_summary["languages"][language] = language_summary

        with (manifest_root / f"{language}_speaker_logs.json").open("w", encoding="utf-8") as handle:
            json.dump(speaker_logs, handle, ensure_ascii=False, indent=2)
        with (manifest_root / f"{language}_selected_speakers.json").open("w", encoding="utf-8") as handle:
            json.dump(selected_speakers_payload, handle, ensure_ascii=False, indent=2)
        write_csv(manifest_root / f"clean_real_manifest_{language}.csv", manifest_rows)
        LOGGER.info(
            "[%s] finished: speakers=%d, samples=%d, split_speakers=%s, split_samples=%s",
            language,
            len(speaker_bundles),
            len(manifest_rows),
            dict(split_speaker_counts),
            dict(split_sample_counts),
        )

    all_manifest_rows.sort(key=lambda row: (row["language"], row["split"], row["speaker_id"], row["utterance_id"]))
    write_csv(manifest_root / "clean_real_manifest.csv", all_manifest_rows)
    with (manifest_root / "stage1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info("all languages finished: total samples=%d", len(all_manifest_rows))
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
