"""Stage 5 CLI: validation plus lean dataset metadata export."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm.auto import tqdm

from lib.config import load_json, relative_to_workspace, resolve_path
from lib.io import load_csv_rows, write_csv, write_json
from lib.logging import get_logger, setup_logging

from .export import export_single_audio
from .protocols import (
    UNSEEN_CHAIN_CONFIG_SPLIT_FIELD,
    UNSEEN_COMPOSITION_SPLIT_FIELD,
    UNSEEN_ORDER_SPLIT_FIELD,
    annotate_rows,
    check_speaker_disjoint,
    summarize_parent_coverage,
    summarize_task_splits,
)
from .validate import summarize_validation_rows, validate_single_row


LOGGER = get_logger("stage5")
DEFAULT_COVERAGE_CHAIN_FAMILIES = ["direct", "platform_like", "telephony", "simreplay", "hybrid"]
STANDARD_SPLIT_EXPORT_FIELD = "split_standard"
RELEASE_METADATA_FIELD_ORDER = [
    # Identity and paths
    "sample_id",
    "parent_id",
    "audio_path",
    "clean_parent_path",
    "trace_path",
    # Labels and evaluation splits
    "label",
    STANDARD_SPLIT_EXPORT_FIELD,
    UNSEEN_COMPOSITION_SPLIT_FIELD,
    UNSEEN_ORDER_SPLIT_FIELD,
    UNSEEN_CHAIN_CONFIG_SPLIT_FIELD,
    # Source/content metadata
    "language",
    "speaker_id",
    "source_speaker_id",
    "utterance_id",
    "transcript",
    "raw_transcript",
    "source_corpus",
    "license_tag",
    # Generator metadata
    "generator_family",
    "generator_name",
    # Delivery-chain metadata
    "chain_family",
    "chain_template_id",
    "chain_variant_index",
    "operator_seq",
    "chain_config",
    "operator_params",
    "codec",
    "bitrate",
    "packet_loss",
    "bandwidth_mode",
    "snr",
    "rt60",
    "rir_backend",
    "room_dim",
    "distance",
    "seed",
    # Audio stats
    "duration_sec",
    "sample_rate",
    "channels",
    "codec_name",
    "sample_fmt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage5_validation_packaging.json",
        help="Path to the Stage-5 JSON config file.",
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
        default=100,
        help="Update the progress display every N validated files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally validate/package only the first N rows after filtering.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("zh", "en"),
        help="Restrict processing to one or more languages.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Override worker count from config.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Stage-5 validation and directly construct the final dataset package.",
    )
    return parser.parse_args()


def write_stats_tables(
    rows: list[dict[str, Any]],
    stats_root: Path,
    workspace_root: Path,
) -> dict[str, str]:
    label_language_split_rows: list[dict[str, Any]] = []
    chain_family_rows: list[dict[str, Any]] = []
    generator_family_rows: list[dict[str, Any]] = []

    counts_by_label_lang_split: dict[tuple[str, str, str], int] = Counter()
    counts_by_chain_family_label: dict[tuple[str, str], int] = Counter()
    counts_by_generator_family_label: dict[tuple[str, str], int] = Counter()

    for row in rows:
        counts_by_label_lang_split[(row["label"], row["language"], row["split"])] += 1
        counts_by_chain_family_label[(row["chain_family"], row["label"])] += 1
        counts_by_generator_family_label[(row["generator_family"], row["label"])] += 1

    for (label, language, split), count in sorted(counts_by_label_lang_split.items()):
        label_language_split_rows.append(
            {
                "label": label,
                "language": language,
                "split": split,
                "samples": str(count),
            }
        )
    for (chain_family, label), count in sorted(counts_by_chain_family_label.items()):
        chain_family_rows.append(
            {
                "chain_family": chain_family,
                "label": label,
                "samples": str(count),
            }
        )
    for (generator_family, label), count in sorted(counts_by_generator_family_label.items()):
        generator_family_rows.append(
            {
                "generator_family": generator_family,
                "label": label,
                "samples": str(count),
            }
        )

    write_csv(stats_root / "stats_label_language_split.csv", label_language_split_rows)
    write_csv(stats_root / "stats_chain_family_label.csv", chain_family_rows)
    write_csv(stats_root / "stats_generator_family_label.csv", generator_family_rows)
    return {
        "label_language_split": relative_to_workspace(
            stats_root / "stats_label_language_split.csv",
            workspace_root,
        ),
        "chain_family_label": relative_to_workspace(
            stats_root / "stats_chain_family_label.csv",
            workspace_root,
        ),
        "generator_family_label": relative_to_workspace(
            stats_root / "stats_generator_family_label.csv",
            workspace_root,
        ),
    }


def summarize_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_id_counts = Counter(row["sample_id"] for row in rows)
    audio_path_counts = Counter(row["audio_path"] for row in rows)
    duplicate_sample_ids = {key: count for key, count in sample_id_counts.items() if count > 1}
    duplicate_audio_paths = {key: count for key, count in audio_path_counts.items() if count > 1}
    return {
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_audio_path_count": len(duplicate_audio_paths),
    }


def make_failure_record(row: dict[str, Any], status: str, error: str | None) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "parent_id": row["parent_id"],
        "split": row["split"],
        "language": row["language"],
        "speaker_id": row["speaker_id"],
        "status": status,
        "error": error or "",
        "audio_path": row["audio_path"],
    }


def build_release_metadata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_rows: list[dict[str, Any]] = []
    for row in rows:
        metadata_row = {
            "sample_id": row.get("sample_id", ""),
            "parent_id": row.get("parent_id", ""),
            "audio_path": row.get("audio_path", ""),
            "clean_parent_path": row.get("clean_parent_path", ""),
            "trace_path": row.get("trace_path", ""),
            "label": row.get("label", ""),
            STANDARD_SPLIT_EXPORT_FIELD: row.get("split", ""),
            UNSEEN_COMPOSITION_SPLIT_FIELD: row.get(UNSEEN_COMPOSITION_SPLIT_FIELD, ""),
            UNSEEN_ORDER_SPLIT_FIELD: row.get(UNSEEN_ORDER_SPLIT_FIELD, ""),
            UNSEEN_CHAIN_CONFIG_SPLIT_FIELD: row.get(UNSEEN_CHAIN_CONFIG_SPLIT_FIELD, ""),
            "language": row.get("language", ""),
            "speaker_id": row.get("speaker_id", ""),
            "source_speaker_id": row.get("source_speaker_id", ""),
            "utterance_id": row.get("utterance_id", ""),
            "transcript": row.get("transcript", ""),
            "raw_transcript": row.get("raw_transcript", ""),
            "source_corpus": row.get("source_corpus", ""),
            "license_tag": row.get("license_tag", ""),
            "generator_family": row.get("generator_family", ""),
            "generator_name": row.get("generator_name", ""),
            "chain_family": row.get("chain_family", ""),
            "chain_template_id": row.get("chain_template_id", ""),
            "chain_variant_index": row.get("chain_variant_index", ""),
            "operator_seq": row.get("operator_seq", ""),
            "chain_config": row.get("chain_config", "[]"),
            "operator_params": row.get("operator_params", ""),
            "codec": row.get("codec", ""),
            "bitrate": row.get("bitrate", ""),
            "packet_loss": row.get("packet_loss", ""),
            "bandwidth_mode": row.get("bandwidth_mode", ""),
            "snr": row.get("snr", ""),
            "rt60": row.get("rt60", ""),
            "rir_backend": row.get("rir_backend", ""),
            "room_dim": row.get("room_dim", ""),
            "distance": row.get("distance", ""),
            "seed": row.get("seed", ""),
            "duration_sec": row.get("duration_sec", ""),
            "sample_rate": row.get("sample_rate", ""),
            "channels": row.get("channels", ""),
            "codec_name": row.get("codec_name", ""),
            "sample_fmt": row.get("sample_fmt", ""),
        }
        metadata_rows.append(
            {
                field: metadata_row.get(field, "")
                for field in RELEASE_METADATA_FIELD_ORDER
            }
        )
    return metadata_rows


def load_selected_rows(
    input_manifest_path: Path,
    languages: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    LOGGER.info("loading Stage-4 manifest from %s", input_manifest_path)
    rows = load_csv_rows(input_manifest_path)
    LOGGER.info("loaded %d Stage-4 rows", len(rows))
    if languages:
        allowed_languages = set(languages)
        rows = [row for row in rows if row["language"] in allowed_languages]
        LOGGER.info("after language filter: %d rows", len(rows))
    if limit > 0:
        rows = rows[:limit]
        LOGGER.info("after --limit: %d rows", len(rows))
    if not rows:
        raise RuntimeError("No Stage-4 rows selected for Stage-5 processing")
    return rows


def _consume_results(
    pending: set[Future],
    *,
    counts: Counter,
    progress: Any,
    completed: int,
    log_every: int,
    on_result: Callable[[Any], None],
    progress_postfix: Callable[[int], dict[str, Any]],
) -> int:
    done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
    pending.clear()
    pending.update(still_pending)
    for future in done:
        completed += 1
        result = future.result()
        counts[result.status] += 1
        on_result(result)
        progress.update(1)
        if completed <= 5 or completed % log_every == 0:
            progress.set_postfix(**progress_postfix(completed))
    return completed


def _run_bounded_tasks(
    items: Iterable[dict[str, Any]],
    total: int,
    *,
    workers: int,
    desc: str,
    unit: str,
    submit_fn: Callable[[ThreadPoolExecutor, dict[str, Any]], Future],
    on_result: Callable[[Any], None],
    counts: Counter,
    log_every: int,
    progress_postfix: Callable[[int], dict[str, Any]],
) -> None:
    in_flight_limit = max(1, workers * 2)
    item_iter = iter(items)
    pending: set[Future] = set()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in range(min(in_flight_limit, total)):
            item = next(item_iter, None)
            if item is None:
                break
            pending.add(submit_fn(executor, item))

        with tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True) as progress:
            completed = 0
            while pending:
                completed = _consume_results(
                    pending,
                    counts=counts,
                    progress=progress,
                    completed=completed,
                    log_every=log_every,
                    on_result=on_result,
                    progress_postfix=progress_postfix,
                )
                while len(pending) < in_flight_limit:
                    item = next(item_iter, None)
                    if item is None:
                        break
                    pending.add(submit_fn(executor, item))
            if total > 0 and (completed <= 5 or completed % log_every != 0):
                progress.set_postfix(**progress_postfix(completed))


def validate_dataset_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    workspace_root: Path,
    workers: int,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    prepared_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts = Counter()
    _run_bounded_tasks(
        rows,
        len(rows),
        workers=workers,
        desc="stage5 validate",
        unit="file",
        submit_fn=lambda executor, row: executor.submit(validate_single_row, row, config, workspace_root),
        on_result=lambda result: (
            prepared_rows.append(dict(result.input_row))
            if result.ok
            else failures.append(make_failure_record(result.input_row, result.status, result.error))
        ),
        counts=counts,
        log_every=log_every,
        progress_postfix=lambda completed: {
            "kept": len(prepared_rows),
            "failed": completed - len(prepared_rows),
        },
    )
    return prepared_rows, failures, counts


def sort_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (row["language"], row["split"], row["speaker_id"], row["sample_id"]))
    return rows


def resolve_coverage_families(config: dict[str, Any]) -> list[str]:
    return list(
        config.get(
            "coverage_chain_families",
            DEFAULT_COVERAGE_CHAIN_FAMILIES,
        )
    )


def export_dataset_audio(
    rows: list[dict[str, Any]],
    workspace_root: Path,
    dataset_root: Path,
    workers: int,
    overwrite: bool,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    exported_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts = Counter()
    _run_bounded_tasks(
        rows,
        len(rows),
        workers=workers,
        desc="stage5 export",
        unit="audio",
        submit_fn=lambda executor, row: executor.submit(
            export_single_audio, row, workspace_root, dataset_root, overwrite
        ),
        on_result=lambda result: (
            exported_rows.append(result.output_row)
            if result.ok and result.output_row is not None
            else failures.append(make_failure_record(result.input_row, result.status, result.error))
        ),
        counts=counts,
        log_every=log_every,
        progress_postfix=lambda completed: {
            "copied": counts["exported_audio"],
            "skipped": counts["skipped_existing_audio"],
            "failed": completed - (counts["exported_audio"] + counts["skipped_existing_audio"]),
        },
    )
    return exported_rows, failures, counts


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = Path.cwd()
    config_path = resolve_path(args.config, workspace_root)
    config = load_json(config_path)
    input_manifest_path = resolve_path(config["input_manifest"], workspace_root)
    output_root = resolve_path(config["output_root"], workspace_root)
    manifest_root = output_root / "manifest"
    stats_root = manifest_root
    manifest_root.mkdir(parents=True, exist_ok=True)
    rows = load_selected_rows(input_manifest_path, args.language, args.limit)

    workers = args.workers if args.workers > 0 else int(config["workers"])
    if args.skip_validation:
        LOGGER.info("skip-validation mode: constructing dataset directly from input manifest")
        prepared_rows = [dict(row) for row in rows]
        failures: list[dict[str, Any]] = []
        counts = Counter({"packaged_without_validation": len(prepared_rows)})
    else:
        prepared_rows, failures, counts = validate_dataset_rows(
            rows,
            config,
            workspace_root,
            workers,
            args.log_every,
        )

    if not prepared_rows:
        raise RuntimeError("Stage-5 produced zero rows for dataset construction")

    LOGGER.info("copying final audio into dataset structure ...")
    exported_rows, export_failures, export_counts = export_dataset_audio(
        prepared_rows,
        workspace_root,
        output_root,
        workers,
        overwrite=bool(config.get("overwrite_audio", False)),
        log_every=args.log_every,
    )
    failures.extend(export_failures)
    counts.update(export_counts)
    if not exported_rows:
        raise RuntimeError("Stage-5 produced zero exported dataset rows")

    annotated_rows = sort_dataset_rows(annotate_rows(exported_rows))

    LOGGER.info("writing lean dataset metadata ...")
    metadata_rows = build_release_metadata_rows(annotated_rows)
    metadata_path = manifest_root / "metadata.csv"
    write_csv(metadata_path, metadata_rows, fieldnames=RELEASE_METADATA_FIELD_ORDER)

    failures_path = manifest_root / "stage5_failures.json"
    write_json(failures_path, failures)

    LOGGER.info("writing statistics tables ...")
    stats_tables = write_stats_tables(annotated_rows, stats_root, workspace_root)

    coverage_families = resolve_coverage_families(config)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": str(config.get("dataset_name", "ChainBench")),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "input_manifest": relative_to_workspace(input_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "validation_enabled": not args.skip_validation,
        "input_rows": len(rows),
        "dataset_rows": len(annotated_rows),
        "failed_rows": len(failures),
        "metadata_path": relative_to_workspace(metadata_path, workspace_root),
        "failures_path": relative_to_workspace(failures_path, workspace_root),
        "status_counts": dict(counts),
        "speaker_disjoint_check": check_speaker_disjoint(annotated_rows),
        "counterfactual_parent_coverage": summarize_parent_coverage(annotated_rows, coverage_families),
        "task_specific_splits": summarize_task_splits(annotated_rows),
        "duplicate_checks": summarize_duplicates(annotated_rows),
        "stats": summarize_validation_rows(annotated_rows),
        "stats_tables": stats_tables,
    }
    summary_path = manifest_root / "dataset_summary.json"
    write_json(summary_path, summary)

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-5 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.info(
        "Stage-5 finished: dataset_rows=%d, failed=%d, manifest_root=%s",
        len(annotated_rows),
        len(failures),
        manifest_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
