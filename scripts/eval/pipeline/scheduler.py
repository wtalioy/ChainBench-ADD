"""Job assignment and parallel scheduling helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from lib.logging import get_logger

from ..tasks import TaskPack
from .checkpoints import shared_training_key
from .models import AssignedJob, RunRecord
from .runner import run_baseline
from .state import PipelineState

LOGGER = get_logger("eval.pipeline")


def phase_device_pool(phase_cfg: dict[str, Any]) -> list[str]:
    return [str(device) for device in phase_cfg["devices"]]


def execution_device_pool(
    baseline_cfg: dict[str, Any],
    *,
    eval_only: bool,
    train_only: bool,
) -> list[str]:
    if baseline_cfg["train"]["enabled"] and not eval_only:
        return phase_device_pool(baseline_cfg["train"])
    if baseline_cfg["eval"]["enabled"] and not train_only:
        return phase_device_pool(baseline_cfg["eval"])
    return phase_device_pool(baseline_cfg["train"])


def assign_jobs_to_devices(
    packs: list[TaskPack],
    baseline_names: list[str],
    baseline_configs: dict[str, dict[str, Any]],
    *,
    eval_only: bool,
    train_only: bool,
) -> list[AssignedJob]:
    pool_counters: dict[tuple[str, ...], int] = {}
    shared_training_devices: dict[tuple[tuple[str, ...], str], str] = {}
    assigned_jobs: list[AssignedJob] = []
    for index, (pack, baseline_name) in enumerate((pack, baseline_name) for pack in packs for baseline_name in baseline_names):
        device_pool = execution_device_pool(
            baseline_configs[baseline_name],
            eval_only=eval_only,
            train_only=train_only,
        )
        pool_key = tuple(device_pool)
        shared_key = None
        if baseline_configs[baseline_name]["train"]["enabled"] and not eval_only:
            shared_key = shared_training_key(pack, baseline_name, baseline_configs[baseline_name])
        if shared_key is not None and (pool_key, shared_key) in shared_training_devices:
            execution_device = shared_training_devices[(pool_key, shared_key)]
        else:
            slot_index = pool_counters.get(pool_key, 0)
            execution_device = device_pool[slot_index % len(device_pool)]
            pool_counters[pool_key] = slot_index + 1
            if shared_key is not None:
                shared_training_devices[(pool_key, shared_key)] = execution_device
        assigned_jobs.append(
            AssignedJob(
                index=index,
                pack=pack,
                baseline_name=baseline_name,
                execution_device=execution_device,
            )
        )
    return assigned_jobs


def group_jobs_by_execution_device(jobs: list[AssignedJob]) -> dict[str, list[AssignedJob]]:
    jobs_by_device: dict[str, list[AssignedJob]] = {}
    for job in jobs:
        jobs_by_device.setdefault(job.execution_device, []).append(job)
    return jobs_by_device


def run_job_queue(
    jobs: list[AssignedJob],
    *,
    baseline_configs: dict[str, dict[str, Any]],
    pipeline_state: PipelineState,
) -> list[tuple[int, RunRecord]]:
    results: list[tuple[int, RunRecord]] = []
    for job in jobs:
        results.append(
            (
                job.index,
                run_baseline(
                    job.pack,
                    job.baseline_name,
                    baseline_configs[job.baseline_name],
                    execution_device=job.execution_device,
                    pipeline_state=pipeline_state,
                    job=job,
                ),
            )
        )
    return results


def execute_assigned_jobs(
    assigned_jobs: list[AssignedJob],
    *,
    baseline_configs: dict[str, dict[str, Any]],
    pipeline_state: PipelineState,
) -> list[tuple[int, RunRecord]]:
    jobs_by_device = group_jobs_by_execution_device(assigned_jobs)
    if pipeline_state.monitor is not None:
        LOGGER.info(
            "scheduled %d baseline runs across %d execution device(s); live status: %s",
            len(assigned_jobs),
            len(jobs_by_device),
            pipeline_state.monitor.status_path,
        )

    if len(jobs_by_device) <= 1:
        return run_job_queue(
            assigned_jobs,
            baseline_configs=baseline_configs,
            pipeline_state=pipeline_state,
        )

    indexed_records: list[tuple[int, RunRecord]] = []
    with ThreadPoolExecutor(max_workers=len(jobs_by_device)) as executor:
        futures = [
            executor.submit(
                run_job_queue,
                device_jobs,
                baseline_configs=baseline_configs,
                pipeline_state=pipeline_state,
            )
            for device_jobs in jobs_by_device.values()
        ]
        for future in as_completed(futures):
            indexed_records.extend(future.result())
    return indexed_records
