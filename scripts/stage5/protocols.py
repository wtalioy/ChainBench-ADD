"""Lightweight metadata helpers for stage5."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any

from eval.task_splits import (
    UNSEEN_CHAIN_CONFIG_SPLIT_FIELD,
    UNSEEN_COMPOSITION_SPLIT_FIELD,
    UNSEEN_ORDER_SPLIT_FIELD,
    annotate_task_split_columns,
)
from metadata.task_keys import (
    CHAIN_CONFIG_KEY_CACHE_FIELD,
    CHAIN_CONFIG_VALUE_CACHE_FIELD,
    COMPOSITION_KEY_CACHE_FIELD,
    ORDER_KEY_CACHE_FIELD,
    build_chain_config_tokens,
    chain_config_key,
    composition_key,
    order_key,
)


TASK_KEY_CACHE_FIELDS = {
    UNSEEN_COMPOSITION_SPLIT_FIELD: COMPOSITION_KEY_CACHE_FIELD,
    UNSEEN_ORDER_SPLIT_FIELD: ORDER_KEY_CACHE_FIELD,
    UNSEEN_CHAIN_CONFIG_SPLIT_FIELD: CHAIN_CONFIG_KEY_CACHE_FIELD,
}


def build_chain_config(row: dict[str, Any]) -> str:
    cached = row.get(CHAIN_CONFIG_VALUE_CACHE_FIELD)
    if isinstance(cached, str):
        return cached
    chain_config = json.dumps(build_chain_config_tokens(row), ensure_ascii=False, separators=(",", ":"))
    row[CHAIN_CONFIG_VALUE_CACHE_FIELD] = chain_config
    return chain_config


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = annotate_task_split_columns(rows)
    for row in annotated:
        row["chain_config"] = build_chain_config(row)
    return annotated


def check_speaker_disjoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speakers_by_split[str(row["split"])].add(str(row["speaker_id"]))
    overlaps: dict[str, int] = {}
    split_names = sorted(speakers_by_split)
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            overlap = speakers_by_split[left] & speakers_by_split[right]
            overlaps[f"{left}__{right}"] = len(overlap)
    return {
        "splits": {split: len(speakers) for split, speakers in speakers_by_split.items()},
        "pairwise_overlap_counts": overlaps,
        "speaker_disjoint": all(count == 0 for count in overlaps.values()),
    }


def summarize_parent_coverage(rows: list[dict[str, Any]], required_families: list[str]) -> dict[str, Any]:
    families_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families_by_parent[str(row["parent_id"])].add(str(row["chain_family"]))
    coverage_counts = Counter()
    required = set(required_families)
    for families in families_by_parent.values():
        missing = required - families
        coverage_counts["parents_total"] += 1
        if not missing:
            coverage_counts["parents_with_required_families"] += 1
        else:
            coverage_counts["parents_missing_required_families"] += 1
    return dict(coverage_counts)


def _task_key(row: dict[str, Any], key_fn, cache_field: str) -> str:
    cached = row.get(cache_field)
    if isinstance(cached, str):
        return cached
    key = key_fn(row)
    row[cache_field] = key
    return key


def _nonempty_row_task_key(row: dict[str, Any], key_fn, cache_field: str) -> str:
    key = _task_key(row, key_fn, cache_field)
    return "" if key in {"", "[]"} else key


def summarize_task_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split_field, key_fn in (
        (UNSEEN_COMPOSITION_SPLIT_FIELD, composition_key),
        (UNSEEN_ORDER_SPLIT_FIELD, order_key),
        (UNSEEN_CHAIN_CONFIG_SPLIT_FIELD, chain_config_key),
    ):
        cache_field = TASK_KEY_CACHE_FIELDS[split_field]
        split_counts = Counter(str(row.get(split_field, "")).strip() or "excluded" for row in rows)
        holdout_rows = [row for row in rows if str(row.get(split_field, "")).strip() == "test"]
        holdout_keys = sorted(
            {
                key
                for row in holdout_rows
                if (key := _nonempty_row_task_key(row, key_fn, cache_field))
            }
        )
        holdout_families = sorted(
            {str(row.get("chain_family", "")).strip() for row in holdout_rows if str(row.get("chain_family", "")).strip()}
        )
        seen_control_rows = [
            row
            for row in rows
            if str(row.get("split", "")).strip() == "test"
            and str(row.get(split_field, "")).strip() == ""
            and _nonempty_row_task_key(row, key_fn, cache_field)
        ]
        family_matched_seen_rows = [
            row for row in seen_control_rows if not holdout_families or str(row.get("chain_family", "")).strip() in holdout_families
        ]
        summaries[split_field] = {
            "split_counts": dict(split_counts),
            "holdout_key_count": len(holdout_keys),
            "holdout_keys_preview": holdout_keys[:10],
            "holdout_test_rows": len(holdout_rows),
            "holdout_test_family_counts": dict(Counter(str(row.get("chain_family", "")).strip() for row in holdout_rows)),
            "holdout_test_label_counts": dict(Counter(str(row.get("label", "")).strip() for row in holdout_rows)),
            "seen_control_test_rows": len(seen_control_rows),
            "seen_control_test_family_counts": dict(
                Counter(str(row.get("chain_family", "")).strip() for row in seen_control_rows)
            ),
            "family_matched_seen_control_test_rows": len(family_matched_seen_rows),
            "family_matched_seen_control_family_counts": dict(
                Counter(str(row.get("chain_family", "")).strip() for row in family_matched_seen_rows)
            ),
            "family_matched_seen_control_label_counts": dict(
                Counter(str(row.get("label", "")).strip() for row in family_matched_seen_rows)
            ),
        }
    return summaries
