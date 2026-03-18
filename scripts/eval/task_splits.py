"""Deterministic task-specific split assignment for evaluation protocols."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any

from metadata.task_keys import (
    CHAIN_CONFIG_KEY_CACHE_FIELD,
    COMPOSITION_KEY_CACHE_FIELD,
    ORDER_KEY_CACHE_FIELD,
    chain_config_key,
    composition_key,
    order_key,
)


STANDARD_SPLIT_FIELD = "split"
UNSEEN_COMPOSITION_SPLIT_FIELD = "split_unseen_composition"
UNSEEN_ORDER_SPLIT_FIELD = "split_unseen_order"
UNSEEN_CHAIN_CONFIG_SPLIT_FIELD = "split_unseen_chain_config"
SUPPORTED_SPLIT_VALUES = {"train", "dev", "test"}
UNSEEN_HOLDOUT_RATIO = 0.25

TASK_SPLIT_SPECS = (
    (UNSEEN_COMPOSITION_SPLIT_FIELD, composition_key, UNSEEN_COMPOSITION_SPLIT_FIELD, COMPOSITION_KEY_CACHE_FIELD),
    (UNSEEN_ORDER_SPLIT_FIELD, order_key, None, ORDER_KEY_CACHE_FIELD),
    (UNSEEN_CHAIN_CONFIG_SPLIT_FIELD, chain_config_key, UNSEEN_CHAIN_CONFIG_SPLIT_FIELD, CHAIN_CONFIG_KEY_CACHE_FIELD),
)


def _task_key(row: dict[str, Any], key_fn, cache_field: str) -> str:
    cached = row.get(cache_field)
    if isinstance(cached, str):
        return cached
    key = key_fn(row)
    row[cache_field] = key
    return key


def has_task_specific_split(rows: list[dict[str, Any]], split_field: str) -> bool:
    return any(str(row.get(split_field, "")).strip() in SUPPORTED_SPLIT_VALUES for row in rows)


def select_holdout_keys(
    rows: list[dict[str, Any]],
    *,
    key_fn,
    salt: str,
    cache_field: str,
    split_field: str = STANDARD_SPLIT_FIELD,
    holdout_ratio: float = UNSEEN_HOLDOUT_RATIO,
) -> set[str]:
    test_keys_set: set[str] = set()
    for row in rows:
        if str(row.get(split_field, "")).strip() != "test":
            continue
        key = _task_key(row, key_fn, cache_field)
        if key not in {"", "[]"}:
            test_keys_set.add(key)
    test_keys = sorted(
        test_keys_set,
        key=lambda key: hashlib.sha1(f"{salt}\0{key}".encode("utf-8")).hexdigest(),
    )
    if len(test_keys) < 2:
        return set()

    target_count = min(
        len(test_keys) - 1,
        max(1, int(round(len(test_keys) * float(holdout_ratio)))),
    )
    return set(test_keys[:target_count])


def select_unseen_order_holdout_keys(
    rows: list[dict[str, Any]],
    *,
    split_field: str = STANDARD_SPLIT_FIELD,
) -> set[str]:
    """Select strict unseen-order holdouts.

    A held-out order is only valid if the same parameter-aware composition
    appears in train/dev under at least one different order, so test measures
    order generalization rather than composition or severity generalization.
    """

    orders_by_composition: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        base_split = str(row.get(split_field, "")).strip()
        if base_split not in SUPPORTED_SPLIT_VALUES:
            continue
        comp_key = _task_key(row, composition_key, COMPOSITION_KEY_CACHE_FIELD)
        ord_key = _task_key(row, order_key, ORDER_KEY_CACHE_FIELD)
        if not comp_key or not ord_key:
            continue
        orders_by_composition[comp_key][ord_key][base_split] += 1

    holdout_keys: set[str] = set()
    for composition, order_counts in orders_by_composition.items():
        candidate_orders = [
            ord_key
            for ord_key, counts in order_counts.items()
            if counts["test"] > 0
            and any(
                other_key != ord_key and (other_counts["train"] > 0 or other_counts["dev"] > 0)
                for other_key, other_counts in order_counts.items()
            )
        ]
        if not candidate_orders:
            continue
        holdout_keys.add(
            min(
                candidate_orders,
                key=lambda ord_key: hashlib.sha1(
                    f"{UNSEEN_ORDER_SPLIT_FIELD}\0{composition}\0{ord_key}".encode("utf-8")
                ).hexdigest(),
            )
        )
    return holdout_keys


def assign_task_split(
    row: dict[str, Any],
    *,
    holdout_keys: set[str],
    key_fn,
    cache_field: str,
    split_field: str = STANDARD_SPLIT_FIELD,
) -> str:
    base_split = str(row.get(split_field, "")).strip()
    key = _task_key(row, key_fn, cache_field)
    if key in {"", "[]"}:
        return ""
    if base_split == "test":
        return "test" if key in holdout_keys else ""
    if base_split in {"train", "dev"} and key not in holdout_keys:
        return base_split
    return ""


def annotate_task_split_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    holdout_keys_by_field = {
        split_field: (
            select_unseen_order_holdout_keys(rows)
            if salt is None
            else select_holdout_keys(rows, key_fn=key_fn, salt=salt, cache_field=cache_field)
        )
        for split_field, key_fn, salt, cache_field in TASK_SPLIT_SPECS
    }

    annotated: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        for split_field, key_fn, _, cache_field in TASK_SPLIT_SPECS:
            enriched[split_field] = assign_task_split(
                row,
                holdout_keys=holdout_keys_by_field[split_field],
                key_fn=key_fn,
                cache_field=cache_field,
            )
        annotated.append(enriched)
    return annotated
