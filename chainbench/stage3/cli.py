"""Stage 3 CLI: spoof clean-parent generation."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

from chainbench.lib.cli import (
    LOG_LEVEL_CHOICES,
    add_language_filter_argument,
    add_limit_argument,
    add_log_level_argument,
    load_rows_with_filters,
    resolve_config_argument,
)
from chainbench.lib.logging import get_logger, setup_logging, format_elapsed
from chainbench.lib.config import default_workspace_root, load_json, relative_to_workspace, resolve_path
from chainbench.lib.io import write_csv, write_json
from chainbench.lib.summary import print_json, utc_now_iso

from .collect import collect_spoof_rows, summarize_spoof_rows
from .execution import materialize_generator_jobs, run_generator_batches
from .jobs import assign_generators, enrich_jobs, get_active_generators, preflight_generators
from .worker import INTERNAL_WORKER_FLAG, main as run_internal_worker


LOGGER = get_logger("stage3-main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage3.json",
        help="Path to the Stage-3 config JSON.",
    )
    add_log_level_argument(parser)
    parser.add_argument(
        "--runner-log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level passed to generator batch runners.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Emit Stage-3 progress every N validated spoof samples.",
    )
    add_limit_argument(parser, help_text="Optionally process only the first N real clean parents.")
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
    add_language_filter_argument(parser, help_text="Restrict processing to one or more languages.")
    parser.add_argument(
        "--generators-per-parent",
        type=int,
        default=0,
        help="Override the configured number of assigned generators per parent.",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    stage3_started_at = time.monotonic()

    workspace_root = default_workspace_root()
    config_path = resolve_config_argument(args.config, workspace_root)
    config = load_json(config_path)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage2_manifest_path = resolve_path(config["stage2_manifest"], workspace_root)
    rows = load_rows_with_filters(
        stage2_manifest_path,
        logger=LOGGER,
        row_label="clean parents",
        empty_error="No Stage-2 rows selected for Stage-3 processing",
        languages=args.language,
        limit=args.limit,
    )

    generator_cfgs = get_active_generators(config, args.only_generator)
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

    assignments = assign_generators(rows, generator_cfgs, generators_per_parent, int(config["seed"]))
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
    batch_paths_by_generator = materialize_generator_jobs(jobs_by_generator, generator_cfgs, output_root)
    plan_path = output_root / "jobs" / "stage3_job_plan.json"
    write_json(
        plan_path,
        {
            "generated_at_utc": utc_now_iso(),
            "config_path": relative_to_workspace(config_path, workspace_root),
            "stage2_manifest": relative_to_workspace(stage2_manifest_path, workspace_root),
            "input_clean_parents": len(rows),
            "jobs_total": len(assignments),
            "generators_per_parent": generators_per_parent,
            "jobs_per_generator": dict(assignment_summary),
        },
    )

    if args.plan_only:
        LOGGER.success("plan-only | wrote job plan to %s", plan_path)
        return 0

    results_meta = run_generator_batches(
        jobs_by_generator,
        generator_cfgs,
        batch_paths_by_generator,
        workspace_root=workspace_root,
        runner_log_level=args.runner_log_level,
        workers=max(1, int(config.get("workers", 1))),
        logger=LOGGER,
    )

    spoof_rows, failures, stats_by_generator = collect_spoof_rows(
        jobs_by_generator,
        generator_cfgs,
        config,
        output_root,
        workspace_root,
        args.log_every,
    )
    spoof_rows.sort(key=lambda r: (r["language"], r["split"], r["speaker_id"], r["sample_id"]))

    if not spoof_rows:
        raise RuntimeError("Stage-3 produced zero valid spoof clean parents")

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_csv(manifest_root / "spoof_clean_manifest.csv", spoof_rows)
    for language in sorted({r["language"] for r in spoof_rows}):
        subset = [r for r in spoof_rows if r["language"] == language]
        write_csv(manifest_root / f"spoof_clean_manifest_{language}.csv", subset)

    all_rows = []
    all_fieldnames = []
    fieldname_set = set()
    for collection in (rows, spoof_rows):
        for row in collection:
            for key in row.keys():
                if key not in fieldname_set:
                    fieldname_set.add(key)
                    all_fieldnames.append(key)
    all_rows.extend(rows)
    all_rows.extend(spoof_rows)
    write_csv(manifest_root / "clean_parent_manifest_all.csv", all_rows, fieldnames=all_fieldnames)

    failures_path = manifest_root / "stage3_failures.json"
    write_json(failures_path, failures)

    summary = {
        "generated_at_utc": utc_now_iso(),
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
            gk: dict(counter) for gk, counter in stats_by_generator.items()
        },
        "spoof_stats": summarize_spoof_rows(spoof_rows),
    }
    summary_path = manifest_root / "stage3_summary.json"
    write_json(summary_path, summary)

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-3 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.success(
        "finished | elapsed=%s | valid=%d | fail=%d | manifests=%s",
        format_elapsed(time.monotonic() - stage3_started_at),
        len(spoof_rows),
        len(failures),
        manifest_root,
    )
    print_json(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == INTERNAL_WORKER_FLAG:
        return run_internal_worker(raw_argv[1:])
    return run_pipeline(parse_args(raw_argv))


if __name__ == "__main__":
    sys.exit(main())
