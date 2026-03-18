"""Simplified ChainBench evaluation CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.config import relative_to_workspace, resolve_path
from lib.io import load_csv_rows, write_json
from lib.logging import get_logger, setup_logging
from .config import BASELINE_IDS, TASK_IDS, load_eval_config
from .pipeline import run_all_baselines
from .tasks import TaskPack, build_task_packs


LOGGER = get_logger("eval")


def _apply_smoke_mode(rows: list[dict[str, Any]], config: dict[str, Any], smoke_limit: int = 500) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split_standard", "")).strip() or "train"
        by_split.setdefault(split, []).append(row)

    per_split = max(1, smoke_limit // 3)
    smoke_rows = [row for split in ("train", "dev", "test") for row in by_split.get(split, [])[:per_split]]
    for baseline_cfg in config["baselines"].values():
        baseline_cfg["train"]["epochs"] = min(baseline_cfg["train"]["epochs"], 2)
    config["smoke_limits"] = (100, 50, 50)  # max train, dev, test per pack
    return smoke_rows


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_args(args: argparse.Namespace) -> str | None:
    if args.eval_only and args.train_only:
        return "--eval-only and --train-only cannot be used together"
    if args.eval_only and args.force_retrain:
        return "--force-retrain cannot be used with --eval-only"
    if args.sample_ratio is not None and not 0.0 < args.sample_ratio <= 1.0:
        return "--sample-ratio must be in the interval (0, 1]"
    return None


def _load_config_from_args(args: argparse.Namespace, workspace_root: Path) -> tuple[Path, dict[str, Any] | None]:
    config_path = resolve_path(args.config, workspace_root)
    if not config_path.exists():
        LOGGER.error("Config not found: %s", config_path)
        return config_path, None

    try:
        config = load_eval_config(
            config_path,
            workspace_root,
            tasks_override=args.tasks,
            baselines_override=args.baselines,
        )
    except ValueError as exc:
        LOGGER.error(str(exc))
        return config_path, None

    if args.sample_ratio is not None:
        config["sample_ratio"] = args.sample_ratio
    return config_path, config


def _build_pack_config(config: dict[str, Any]) -> dict[str, Any]:
    pack_config: dict[str, Any] = {}
    if config.get("sample_ratio") is not None:
        pack_config["sample_ratio"] = config["sample_ratio"]
    if "smoke_limits" in config:
        pack_config["smoke_limits"] = config["smoke_limits"]
    return pack_config


def _build_summary(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    metadata_path: Path,
    output_root: Path,
    workspace_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": _utc_now_iso(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "metadata_path": relative_to_workspace(metadata_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "eval_only": args.eval_only,
        "train_only": args.train_only,
        "smoke": args.smoke,
        "sample_ratio": config.get("sample_ratio"),
        "force_retrain": args.force_retrain,
        "tasks": config["tasks"],
        "baselines": list(config["baselines"].keys()),
        "metadata_rows": len(rows),
        "baseline_runs": [],
        "baseline_metrics": [],
    }


def _write_summary_snapshot(path: Path, summary: dict[str, Any]) -> None:
    summary["generated_at_utc"] = _utc_now_iso()
    write_json(path, summary)


def _build_task_packs_with_logging(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[TaskPack]:
    packs = build_task_packs(rows, config["tasks"], _build_pack_config(config))
    LOGGER.info("effective sample_ratio=%s", config.get("sample_ratio"))
    for pack in packs:
        LOGGER.info(
            "task pack %s/%s: train=%d dev=%d test=%d",
            pack.task_id,
            pack.variant,
            len(pack.train_rows),
            len(pack.dev_rows),
            len(pack.test_rows),
        )
    return packs


def _finalize_task_pack_summary(summary: dict[str, Any], packs: list[TaskPack]) -> None:
    summary["task_packs_built"] = len(packs)
    if packs:
        summary["status"] = "ready"
        return
    summary["status"] = "no_packs"
    summary["message"] = "No task packs produced (e.g. no data for selected tasks)."


def _persist_final_summary(output_root: Path, summary: dict[str, Any], start_time: float) -> Path:
    summary_path = output_root / f"eval_summary_{start_time}.json"
    write_json(summary_path, summary)
    return summary_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench evaluation pipeline.")
    parser.add_argument(
        "--config",
        default="config/eval_baselines.json",
        help="Path to evaluation JSON config.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_IDS,
        help="Tasks to run (default: from config).",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINE_IDS,
        help="Baselines to run (default: from config).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training only and skip evaluation.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: limit metadata rows and per-split sizes for fast validation.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        help="Deterministically subsample each task pack split by this ratio (0, 1].",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached checkpoints and retrain selected baselines from scratch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    start_time = time.time()
    workspace_root = Path.cwd()
    error = _validate_args(args)
    if error is not None:
        LOGGER.error(error)
        return 1

    config_path, config = _load_config_from_args(args, workspace_root)
    if config is None:
        return 1

    metadata_path = Path(config["metadata_path"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_latest_path = output_root / "eval_summary_latest.json"

    if not metadata_path.exists():
        LOGGER.error("Metadata not found: %s", metadata_path)
        return 1

    rows = load_csv_rows(metadata_path)
    LOGGER.info("loaded %d rows from %s", len(rows), metadata_path)

    if args.smoke:
        rows = _apply_smoke_mode(rows, config)
        LOGGER.info("smoke mode: sampled %d rows across splits (train/dev/test)", len(rows))

    dataset_root = Path(config["dataset_root"])
    baselines = list(config["baselines"].keys())
    summary = _build_summary(
        args=args,
        config=config,
        config_path=config_path,
        metadata_path=metadata_path,
        output_root=output_root,
        workspace_root=workspace_root,
        rows=rows,
    )
    packs = _build_task_packs_with_logging(config, rows)
    _finalize_task_pack_summary(summary, packs)
    _write_summary_snapshot(summary_latest_path, summary)

    if summary["baselines"] and packs:
        def persist_baseline_snapshot(
            baseline_results: list[dict[str, Any]],
            baseline_metrics: list[dict[str, Any]],
        ) -> None:
            summary["baseline_runs"] = baseline_results
            summary["baseline_metrics"] = baseline_metrics
            _write_summary_snapshot(summary_latest_path, summary)

        baseline_results, baseline_metrics, baseline_error = run_all_baselines(
            output_root=output_root,
            dataset_root=dataset_root,
            packs=packs,
            baseline_names=baselines,
            baseline_configs=config["baselines"],
            eval_only=args.eval_only,
            train_only=args.train_only,
            force_retrain=args.force_retrain,
            on_snapshot=persist_baseline_snapshot,
        )
        if baseline_error:
            summary["baseline_error"] = baseline_error
        summary["baseline_runs"] = baseline_results
        summary["baseline_metrics"] = baseline_metrics
    else:
        summary["baseline_runs"] = []

    _write_summary_snapshot(summary_latest_path, summary)
    summary_path = _persist_final_summary(output_root, summary, start_time)
    LOGGER.info("updated live summary %s", summary_latest_path)
    LOGGER.info("wrote %s", summary_path)
    LOGGER.info("summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
    LOGGER.info("time taken: %s", time.time() - start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
