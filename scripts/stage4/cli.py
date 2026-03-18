"""Stage 4 CLI: delivery-chain rendering."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lib.logging import get_logger, setup_logging
from tqdm.auto import tqdm

from lib.config import load_json, relative_to_workspace, resolve_path
from lib.io import load_csv_rows, write_csv

from .chains import count_jobs, iter_sample_jobs
from .render import build_manifest_row, render_single_job, summarize_manifest


LOGGER = get_logger("stage4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage4_delivery_chain_rendering.json",
        help="Path to the Stage-4 config JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("direct", "platform_like", "telephony", "simreplay", "hybrid"),
        help="Restrict rendering to selected chain families.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("zh", "en"),
        help="Restrict processing to one or more languages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally process only the first N clean parents.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Override worker count from config.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Sample/render plan but do not actually create delivered children.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Update progress display every N finished jobs.",
    )
    return parser.parse_args()


def resolve_worker_count(args_workers: int, config: dict[str, Any]) -> int:
    if args_workers > 0:
        return args_workers
    config_workers = int(config.get("workers", 0))
    if config_workers > 0:
        return config_workers
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count))


def _consume_render_result(
    result: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    counts: Counter,
    workspace_root: Path,
) -> None:
    job = result["job"]
    counts[result["status"]] += 1
    if result["status"] in {"ok", "skipped_existing"}:
        manifest_rows.append(build_manifest_row(job, result, workspace_root))
        return
    failures.append(
        {
            "job_id": job["job_id"],
            "sample_id": job["sample_id"],
            "parent_id": job["parent_id"],
            "chain_family": job["family_name"],
            "chain_template_id": job["template_id"],
            "error": result["error"],
        }
    )


def _render_jobs(
    jobs: Iterable[dict[str, Any]],
    total: int,
    config: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
    workers: int,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts: Counter = Counter()
    in_flight_limit = max(1, workers * 2)
    job_iter = iter(jobs)
    pending: set[Future] = set()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in range(min(in_flight_limit, total)):
            job = next(job_iter, None)
            if job is None:
                break
            pending.add(executor.submit(render_single_job, job, config, workspace_root, output_root))

        with tqdm(total=total, desc="stage4 render", unit="job", dynamic_ncols=True) as progress:
            completed = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    _consume_render_result(
                        future.result(),
                        manifest_rows,
                        failures,
                        counts,
                        workspace_root,
                    )
                    progress.update(1)
                    if completed <= 5 or completed % log_every == 0 or completed == total:
                        progress.set_postfix(
                            ok=counts["ok"],
                            skipped=counts["skipped_existing"],
                            failed=counts["failed"],
                        )
                while len(pending) < in_flight_limit:
                    job = next(job_iter, None)
                    if job is None:
                        break
                    pending.add(
                        executor.submit(render_single_job, job, config, workspace_root, output_root)
                    )

    return manifest_rows, failures, counts


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = Path.cwd()
    config_path = resolve_path(args.config, workspace_root)
    config = load_json(config_path)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_manifest_path = resolve_path(config["input_manifest"], workspace_root)
    rows = load_csv_rows(input_manifest_path)
    LOGGER.info("loaded %d clean parents from %s", len(rows), input_manifest_path)
    if args.language:
        rows = [r for r in rows if r["language"] in set(args.language)]
        LOGGER.info("after language filter: %d rows", len(rows))
    if args.limit > 0:
        rows = rows[: args.limit]
        LOGGER.info("after --limit: %d rows", len(rows))
    if not rows:
        raise RuntimeError("No clean-parent rows selected for Stage-4 processing")

    selected_families = args.families or [
        name for name, family_cfg in config["families"].items() if family_cfg.get("enabled", False)
    ]
    LOGGER.info("selected chain families: %s", ", ".join(selected_families))

    jobs_per_family = count_jobs(rows, config, selected_families)
    jobs_total = sum(jobs_per_family.values())
    plan = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "input_manifest": relative_to_workspace(input_manifest_path, workspace_root),
        "input_clean_parents": len(rows),
        "selected_families": selected_families,
        "jobs_total": jobs_total,
        "jobs_per_family": dict(jobs_per_family),
    }
    jobs_dir = output_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "stage4_job_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.plan_only:
        LOGGER.info("plan-only mode: wrote job plan to %s", jobs_dir / "stage4_job_plan.json")
        return 0

    jobs = iter_sample_jobs(rows, config, selected_families, workspace_root)

    workers = resolve_worker_count(args.workers, config)
    LOGGER.info("using %d worker(s)", workers)
    manifest_rows, failures, counts = _render_jobs(
        jobs,
        jobs_total,
        config,
        workspace_root,
        output_root,
        workers,
        args.log_every,
    )

    if not manifest_rows:
        raise RuntimeError("Stage-4 produced zero delivered samples")

    manifest_rows.sort(
        key=lambda r: (r["chain_family"], r["language"], r["split"], r["speaker_id"], r["sample_id"])
    )
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_csv(manifest_root / "delivered_manifest.csv", manifest_rows)
    for family_name in selected_families:
        subset = [r for r in manifest_rows if r["chain_family"] == family_name]
        if subset:
            write_csv(manifest_root / f"delivered_manifest_{family_name}.csv", subset)
    for language in sorted({r["language"] for r in manifest_rows}):
        subset = [r for r in manifest_rows if r["language"] == language]
        write_csv(manifest_root / f"delivered_manifest_{language}.csv", subset)

    failures_path = manifest_root / "stage4_failures.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "input_manifest": relative_to_workspace(input_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "input_clean_parents": len(rows),
        "jobs_total": jobs_total,
        "delivered_samples": len(manifest_rows),
        "failed_jobs": len(failures),
        "status_counts": dict(counts),
        "stats": summarize_manifest(manifest_rows),
    }
    summary_path = manifest_root / "stage4_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-4 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.info(
        "Stage-4 finished: delivered_samples=%d, failed_jobs=%d, manifests=%s",
        len(manifest_rows),
        len(failures),
        manifest_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
