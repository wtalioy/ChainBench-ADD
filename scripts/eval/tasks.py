"""Task derivation for supported evaluation tasks from metadata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from metadata.task_keys import (
    chain_config_key,
    composition_key,
    order_key,
)
from .task_splits import (
    UNSEEN_CHAIN_CONFIG_SPLIT_FIELD,
    UNSEEN_COMPOSITION_SPLIT_FIELD,
    UNSEEN_ORDER_SPLIT_FIELD,
    has_task_specific_split,
)

CHAIN_FAMILIES = ("direct", "platform_like", "telephony", "simreplay", "hybrid")
CURATED_CROSS_CHAIN_TRANSFERS = (
    {
        "train_family": "direct",
        "test_family": "platform_like",
        "motivation": "Canonical clean-to-platform transfer for distribution-channel robustness.",
    },
    {
        "train_family": "direct",
        "test_family": "telephony",
        "motivation": "Canonical clean-to-telephony transfer for communication-channel robustness.",
    },
    {
        "train_family": "direct",
        "test_family": "simreplay",
        "motivation": "Canonical clean-to-replay transfer for physical-channel robustness.",
    },
    {
        "train_family": "direct",
        "test_family": "hybrid",
        "motivation": "Canonical clean-to-hybrid transfer for the most deployment-realistic composed shift.",
    },
    {
        "train_family": "telephony",
        "test_family": "simreplay",
        "motivation": "Representative cross-mechanism transfer from communication artifacts to replay artifacts.",
    },
)
IN_CHAIN_TASK = "in_chain"
CROSS_CHAIN_TASK = "cross_chain"
COMPOSITION_GENERALIZATION_TASK = "composition_generalization"
ORDER_GENERALIZATION_TASK = "order_generalization"
HELD_OUT_CHAIN_CONFIGURATION_TASK = "held_out_chain_configuration"
SEEN_CONTROL_VARIANT = "seen_control"
UNSEEN_HOLDOUT_VARIANT = "unseen_holdout"
COUNTERFACTUAL_TASK = "counterfactual_consistency"
REQUIRED_COUNTERFACTUAL_FAMILIES = ("direct", "platform_like", "telephony", "simreplay")
SHARED_TRAINING_TASK_IDS = {
    COMPOSITION_GENERALIZATION_TASK,
    ORDER_GENERALIZATION_TASK,
    HELD_OUT_CHAIN_CONFIGURATION_TASK,
}
TASK_IDS = (
    IN_CHAIN_TASK,
    CROSS_CHAIN_TASK,
    COMPOSITION_GENERALIZATION_TASK,
    ORDER_GENERALIZATION_TASK,
    HELD_OUT_CHAIN_CONFIGURATION_TASK,
    COUNTERFACTUAL_TASK,
)


@dataclass
class TaskPack:
    """One evaluation task variant with train/dev/test rows."""

    task_id: str
    variant: str
    description: str
    train_rows: list[dict[str, Any]] = field(default_factory=list)
    dev_rows: list[dict[str, Any]] = field(default_factory=list)
    test_rows: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _base_split(row: dict[str, Any]) -> str:
    return str(row.get("split_standard", "")).strip()


def _stable_row_token(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _target_sample_size(size: int, ratio: float) -> int:
    if size <= 0 or ratio >= 1.0:
        return size
    return min(size, max(1, int(round(size * ratio))))


def _sample_rows(rows: list[dict[str, Any]], ratio: float, *, salt: str) -> list[dict[str, Any]]:
    if not rows or ratio >= 1.0:
        return rows
    target_size = _target_sample_size(len(rows), ratio)
    ranked = sorted(
        (
            hashlib.sha1(f"{salt}\0{_stable_row_token(row)}".encode("utf-8")).hexdigest(),
            index,
        )
        for index, row in enumerate(rows)
    )
    keep_indices = {index for _, index in ranked[:target_size]}
    return [row for index, row in enumerate(rows) if index in keep_indices]


def _sample_exact_rows(rows: list[dict[str, Any]], target_size: int, *, salt: str) -> list[dict[str, Any]]:
    if not rows or target_size >= len(rows):
        return rows
    if target_size <= 0:
        return []
    ranked = sorted(
        (
            hashlib.sha1(f"{salt}\0{_stable_row_token(row)}".encode("utf-8")).hexdigest(),
            index,
        )
        for index, row in enumerate(rows)
    )
    keep_indices = {index for _, index in ranked[:target_size]}
    return [row for index, row in enumerate(rows) if index in keep_indices]


def _sample_grouped_rows(
    rows: list[dict[str, Any]],
    ratio: float,
    *,
    salt: str,
    group_key_fn,
) -> list[dict[str, Any]]:
    if not rows or ratio >= 1.0:
        return rows
    group_keys = list(dict.fromkeys(group_key_fn(row) for row in rows))
    target_size = _target_sample_size(len(group_keys), ratio)
    ranked = sorted(
        (
            hashlib.sha1(f"{salt}\0{group_key}".encode("utf-8")).hexdigest(),
            group_key,
        )
        for group_key in group_keys
    )
    keep_group_keys = {group_key for _, group_key in ranked[:target_size]}
    return [row for row in rows if group_key_fn(row) in keep_group_keys]


def _sample_rows_by_group(
    rows: list[dict[str, Any]],
    ratio: float,
    *,
    salt: str,
    group_key_fn,
) -> list[dict[str, Any]]:
    if not rows or ratio >= 1.0:
        return rows
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group[str(group_key_fn(row))].append(row)
    sampled_rows: list[dict[str, Any]] = []
    for group_key in sorted(rows_by_group):
        group_rows = rows_by_group[group_key]
        target_size = _target_sample_size(len(group_rows), ratio)
        sampled_rows.extend(
            _sample_exact_rows(
                group_rows,
                target_size,
                salt=f"{salt}:{group_key}",
            )
        )
    return sampled_rows


def _counterfactual_meta(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id:
            by_parent[parent_id].append(row)

    family_coverage_histogram: Counter[int] = Counter()
    paired_parent_count = 0
    for parent_rows in by_parent.values():
        families = {_chain_family(row) for row in parent_rows}
        if not set(REQUIRED_COUNTERFACTUAL_FAMILIES).issubset(families):
            continue
        paired_parent_count += 1
        family_coverage_histogram[len(families)] += 1

    return {
        "paired_test_parent_count": paired_parent_count,
        "required_chain_families": list(REQUIRED_COUNTERFACTUAL_FAMILIES),
        "paired_test_family_coverage": {
            str(k): v for k, v in sorted(family_coverage_histogram.items())
        },
    }


def _rows_for_split(rows: list[dict[str, Any]], split: str, split_field: str = "split") -> list[dict[str, Any]]:
    if split_field == "split":
        return [row for row in rows if _base_split(row) == split]
    return [row for row in rows if str(row.get(split_field, "")).strip() == split]


def _chain_family(row: dict[str, Any]) -> str:
    return str(row.get("chain_family", "")).strip()


def _supported_chain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _chain_family(row) in CHAIN_FAMILIES]


def _nonempty_operator_key(row: dict[str, Any], key_fn) -> str:
    key = key_fn(row)
    return "" if key in {"", "[]"} else key


def _derive_generalization_partitions(
    rows: list[dict[str, Any]],
    *,
    key_fn,
    split_field: str,
    holdout_meta_key: str,
    seen_meta_key: str,
) -> dict[str, Any] | None:
    if split_field != "split" and has_task_specific_split(rows, split_field):
        usable_rows = [
            row
            for row in rows
            if str(row.get(split_field, "")).strip() in {"train", "dev", "test"}
        ]
        train_rows = _rows_for_split(usable_rows, "train", split_field)
        dev_rows = _rows_for_split(usable_rows, "dev", split_field)
        unseen_test_rows = _rows_for_split(usable_rows, "test", split_field)
        seen_candidate_rows = [
            row
            for row in _rows_for_split(rows, "test")
            if str(row.get(split_field, "")).strip() == "" and _nonempty_operator_key(row, key_fn)
        ]
        holdout_keys = sorted(
            {_nonempty_operator_key(row, key_fn) for row in unseen_test_rows if _nonempty_operator_key(row, key_fn)}
        )
        seen_keys = sorted(
            {_nonempty_operator_key(row, key_fn) for row in seen_candidate_rows if _nonempty_operator_key(row, key_fn)}
        )
        meta = {"split_field": split_field}
    else:
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = _nonempty_operator_key(row, key_fn)
            if key:
                by_key[key].append(row)

        seen_keys_set: set[str] = set()
        unseen_keys_set: set[str] = set()
        for key, grouped_rows in by_key.items():
            train_count = len(_rows_for_split(grouped_rows, "train"))
            test_count = len(_rows_for_split(grouped_rows, "test"))
            if test_count > 0 and train_count < 10:
                unseen_keys_set.add(key)
            elif train_count > 0:
                seen_keys_set.add(key)

        if not seen_keys_set or not unseen_keys_set:
            return None

        train_rows = [row for row in _rows_for_split(rows, "train") if _nonempty_operator_key(row, key_fn) in seen_keys_set]
        dev_rows = [row for row in _rows_for_split(rows, "dev") if _nonempty_operator_key(row, key_fn) in seen_keys_set]
        unseen_test_rows = [
            row for row in _rows_for_split(rows, "test") if _nonempty_operator_key(row, key_fn) in unseen_keys_set
        ]
        seen_candidate_rows = [
            row for row in _rows_for_split(rows, "test") if _nonempty_operator_key(row, key_fn) in seen_keys_set
        ]
        holdout_keys = sorted(unseen_keys_set)
        seen_keys = sorted(seen_keys_set)
        meta = {}

    if not train_rows or not unseen_test_rows or not seen_candidate_rows:
        return None
    return {
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "unseen_test_rows": unseen_test_rows,
        "seen_candidate_rows": seen_candidate_rows,
        "meta": meta,
        holdout_meta_key: holdout_keys,
        seen_meta_key: seen_keys,
    }


def _family_balanced_seen_control_rows(
    unseen_rows: list[dict[str, Any]],
    seen_candidate_rows: list[dict[str, Any]],
    *,
    salt_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unseen_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unseen_rows:
        family = _chain_family(row)
        if family:
            unseen_by_family[family].append(row)
    for row in seen_candidate_rows:
        family = _chain_family(row)
        if family:
            seen_by_family[family].append(row)

    comparison_families = sorted(set(unseen_by_family) & set(seen_by_family))
    if not comparison_families:
        return [], [], {
            "comparison_chain_families": [],
            "family_balance_reference_counts": {},
            "family_balance_target_counts": {},
            "family_balance_available_counts": {},
        }

    filtered_unseen_rows = [
        row for family in comparison_families for row in unseen_by_family[family]
    ]
    reference_counts = {family: len(unseen_by_family[family]) for family in comparison_families}
    available_counts = {family: len(seen_by_family[family]) for family in comparison_families}
    scale = min(1.0, min(available_counts[family] / reference_counts[family] for family in comparison_families))
    target_counts = {
        family: min(
            available_counts[family],
            max(1, int(reference_counts[family] * scale)),
        )
        for family in comparison_families
    }
    balanced_seen_rows: list[dict[str, Any]] = []
    for family in comparison_families:
        balanced_seen_rows.extend(
            _sample_exact_rows(
                seen_by_family[family],
                target_counts[family],
                salt=f"{salt_prefix}:{family}",
            )
        )
    return filtered_unseen_rows, balanced_seen_rows, {
        "comparison_chain_families": comparison_families,
        "family_balance_reference_counts": {family: reference_counts[family] for family in comparison_families},
        "family_balance_target_counts": {family: target_counts[family] for family in comparison_families},
        "family_balance_available_counts": {family: available_counts[family] for family in comparison_families},
    }


def build_in_chain_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """One pack per chain family with train/dev/test from the same family."""
    packs: list[TaskPack] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("chain_family", ""))].append(row)

    for family in CHAIN_FAMILIES:
        fam_rows = by_family.get(family)
        if not fam_rows:
            continue
        train = _rows_for_split(fam_rows, "train")
        dev = _rows_for_split(fam_rows, "dev")
        test = _rows_for_split(fam_rows, "test")
        if not test and not dev:
            continue
        packs.append(
            TaskPack(
                task_id=IN_CHAIN_TASK,
                variant=family,
                description=f"In-chain detection: {family}",
                train_rows=train,
                dev_rows=dev,
                test_rows=test,
                meta={"chain_family": family},
            )
        )
    return packs


def build_cross_chain_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Build a paper-oriented subset of representative cross-chain transfers."""
    packs: list[TaskPack] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("chain_family", ""))].append(row)

    # The paper only needs a compact set of interpretable transfer settings.
    # We keep direct -> non-direct targets as the main clean-to-degraded axis,
    # plus one communication-to-replay transfer explicitly called out in the plan.
    for transfer in CURATED_CROSS_CHAIN_TRANSFERS:
        train_family = transfer["train_family"]
        test_family = transfer["test_family"]
        if not by_family.get(train_family) or not by_family.get(test_family):
            continue
        train_rows = _rows_for_split(by_family[train_family], "train")
        dev_rows = _rows_for_split(by_family[train_family], "dev")
        test_rows = _rows_for_split(by_family[test_family], "test")
        if not train_rows or not test_rows:
            continue
        variant = f"{train_family}_to_{test_family}"
        packs.append(
            TaskPack(
                task_id=CROSS_CHAIN_TASK,
                variant=variant,
                description=f"Cross-chain: train {train_family} → test {test_family}",
                train_rows=train_rows,
                dev_rows=dev_rows,
                test_rows=test_rows,
                meta={
                    "train_chain_family": train_family,
                    "test_chain_family": test_family,
                    "selection_motivation": transfer["motivation"],
                },
            )
        )
    return packs


def build_composition_generalization_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Seen vs held-out operator composition generalization."""
    partitions = _derive_generalization_partitions(
        rows,
        key_fn=composition_key,
        split_field=UNSEEN_COMPOSITION_SPLIT_FIELD,
        holdout_meta_key="holdout_keys",
        seen_meta_key="seen_control_keys",
    )
    if partitions is None:
        return []
    unseen_test_rows, seen_control_rows, balance_meta = _family_balanced_seen_control_rows(
        partitions["unseen_test_rows"],
        partitions["seen_candidate_rows"],
        salt_prefix=COMPOSITION_GENERALIZATION_TASK,
    )
    if not unseen_test_rows or not seen_control_rows:
        return []
    return [
        TaskPack(
            task_id=COMPOSITION_GENERALIZATION_TASK,
            variant=UNSEEN_HOLDOUT_VARIANT,
            description="Composition generalization: unseen holdout",
            train_rows=partitions["train_rows"],
            dev_rows=partitions["dev_rows"],
            test_rows=unseen_test_rows,
            meta={
                **partitions["meta"],
                "comparison_variant": UNSEEN_HOLDOUT_VARIANT,
                "shared_training_group": COMPOSITION_GENERALIZATION_TASK,
                "holdout_keys": partitions["holdout_keys"],
                **balance_meta,
            },
        ),
        TaskPack(
            task_id=COMPOSITION_GENERALIZATION_TASK,
            variant=SEEN_CONTROL_VARIANT,
            description="Composition generalization: seen control",
            train_rows=partitions["train_rows"],
            dev_rows=partitions["dev_rows"],
            test_rows=seen_control_rows,
            meta={
                **partitions["meta"],
                "comparison_variant": SEEN_CONTROL_VARIANT,
                "paired_variant": UNSEEN_HOLDOUT_VARIANT,
                "family_matched_control": True,
                "family_balanced_control": True,
                "shared_training_group": COMPOSITION_GENERALIZATION_TASK,
                "seen_control_keys": partitions["seen_control_keys"],
                **balance_meta,
            },
        ),
    ]


def build_order_generalization_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Seen vs held-out operator order generalization."""
    partitions = _derive_generalization_partitions(
        rows,
        key_fn=order_key,
        split_field=UNSEEN_ORDER_SPLIT_FIELD,
        holdout_meta_key="holdout_keys",
        seen_meta_key="seen_control_keys",
    )
    if partitions is None:
        return []
    unseen_test_rows, seen_control_rows, balance_meta = _family_balanced_seen_control_rows(
        partitions["unseen_test_rows"],
        partitions["seen_candidate_rows"],
        salt_prefix=ORDER_GENERALIZATION_TASK,
    )
    if not unseen_test_rows or not seen_control_rows:
        return []
    return [
        TaskPack(
            task_id=ORDER_GENERALIZATION_TASK,
            variant=UNSEEN_HOLDOUT_VARIANT,
            description="Order generalization: unseen holdout",
            train_rows=partitions["train_rows"],
            dev_rows=partitions["dev_rows"],
            test_rows=unseen_test_rows,
            meta={
                **partitions["meta"],
                "comparison_variant": UNSEEN_HOLDOUT_VARIANT,
                "shared_training_group": ORDER_GENERALIZATION_TASK,
                "holdout_keys": partitions["holdout_keys"],
                **balance_meta,
            },
        ),
        TaskPack(
            task_id=ORDER_GENERALIZATION_TASK,
            variant=SEEN_CONTROL_VARIANT,
            description="Order generalization: seen control",
            train_rows=partitions["train_rows"],
            dev_rows=partitions["dev_rows"],
            test_rows=seen_control_rows,
            meta={
                **partitions["meta"],
                "comparison_variant": SEEN_CONTROL_VARIANT,
                "paired_variant": UNSEEN_HOLDOUT_VARIANT,
                "family_matched_control": True,
                "family_balanced_control": True,
                "shared_training_group": ORDER_GENERALIZATION_TASK,
                "seen_control_keys": partitions["seen_control_keys"],
                **balance_meta,
            },
        ),
    ]


def build_chain_configuration_generalization_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Seen vs held-out chain-configuration generalization."""
    partitions = _derive_generalization_partitions(
        rows,
        key_fn=chain_config_key,
        split_field=UNSEEN_CHAIN_CONFIG_SPLIT_FIELD,
        holdout_meta_key="holdout_keys",
        seen_meta_key="seen_control_keys",
    )
    if partitions is None:
        return []
    unseen_test_rows, seen_control_rows, balance_meta = _family_balanced_seen_control_rows(
        partitions["unseen_test_rows"],
        partitions["seen_candidate_rows"],
        salt_prefix=HELD_OUT_CHAIN_CONFIGURATION_TASK,
    )
    if not unseen_test_rows or not seen_control_rows:
        return []
    return [
        TaskPack(
            task_id=HELD_OUT_CHAIN_CONFIGURATION_TASK,
            variant=UNSEEN_HOLDOUT_VARIANT,
            description="Held-out chain configuration generalization: unseen holdout",
            train_rows=partitions["train_rows"],
            dev_rows=partitions["dev_rows"],
            test_rows=unseen_test_rows,
            meta={
                **partitions["meta"],
                "comparison_variant": UNSEEN_HOLDOUT_VARIANT,
                "shared_training_group": HELD_OUT_CHAIN_CONFIGURATION_TASK,
                "holdout_keys": partitions["holdout_keys"],
                **balance_meta,
            },
        ),
        TaskPack(
            task_id=HELD_OUT_CHAIN_CONFIGURATION_TASK,
            variant=SEEN_CONTROL_VARIANT,
            description="Held-out chain configuration generalization: seen control",
            train_rows=partitions["train_rows"],
            dev_rows=partitions["dev_rows"],
            test_rows=seen_control_rows,
            meta={
                **partitions["meta"],
                "comparison_variant": SEEN_CONTROL_VARIANT,
                "paired_variant": UNSEEN_HOLDOUT_VARIANT,
                "family_matched_control": True,
                "family_balanced_control": True,
                "shared_training_group": HELD_OUT_CHAIN_CONFIGURATION_TASK,
                "seen_control_keys": partitions["seen_control_keys"],
                **balance_meta,
            },
        ),
    ]


def build_counterfactual_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Build one pack for matched-parent counterfactual evaluation."""
    usable_rows = _supported_chain_rows(rows)
    train_rows = _rows_for_split(usable_rows, "train")
    dev_rows = _rows_for_split(usable_rows, "dev")
    test_candidates = _rows_for_split(usable_rows, "test")

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_candidates:
        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id:
            by_parent[parent_id].append(row)

    test_rows: list[dict[str, Any]] = []
    family_coverage_histogram: Counter[int] = Counter()
    for parent_rows in by_parent.values():
        families = {_chain_family(row) for row in parent_rows}
        if not set(REQUIRED_COUNTERFACTUAL_FAMILIES).issubset(families):
            continue
        family_coverage_histogram[len(families)] += 1
        test_rows.extend(sorted(parent_rows, key=lambda row: (_chain_family(row), str(row.get("sample_id", "")))))

    if not train_rows or not test_rows:
        return []

    return [
        TaskPack(
            task_id=COUNTERFACTUAL_TASK,
            variant="matched_parents",
            description="Counterfactual consistency across matched chain families",
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=test_rows,
            meta={
                "reference_chain_family": "direct",
                "chain_families": list(CHAIN_FAMILIES),
                "required_chain_families": list(REQUIRED_COUNTERFACTUAL_FAMILIES),
                "paired_test_parent_count": sum(family_coverage_histogram.values()),
                "paired_test_family_coverage": {
                    str(k): v for k, v in sorted(family_coverage_histogram.items())
                },
            },
        )
    ]


TASK_BUILDERS = {
    IN_CHAIN_TASK: build_in_chain_packs,
    CROSS_CHAIN_TASK: build_cross_chain_packs,
    COMPOSITION_GENERALIZATION_TASK: build_composition_generalization_packs,
    ORDER_GENERALIZATION_TASK: build_order_generalization_packs,
    HELD_OUT_CHAIN_CONFIGURATION_TASK: build_chain_configuration_generalization_packs,
    COUNTERFACTUAL_TASK: build_counterfactual_packs,
}


def _truncate_pack(
    pack: TaskPack,
    max_train: int,
    max_dev: int,
    max_test: int,
) -> TaskPack:
    """Truncate pack splits for smoke testing."""
    return TaskPack(
        task_id=pack.task_id,
        variant=pack.variant,
        description=pack.description,
        train_rows=pack.train_rows[:max_train],
        dev_rows=pack.dev_rows[:max_dev],
        test_rows=pack.test_rows[:max_test],
        meta={**pack.meta, "smoke_truncated": True},
    )


def _sample_pack(pack: TaskPack, sample_ratio: float) -> TaskPack:
    train_dev_scope = (
        str(pack.meta.get("shared_training_group", "")).strip() or pack.task_id
        if pack.task_id in SHARED_TRAINING_TASK_IDS
        else f"{pack.task_id}:{pack.variant}"
    )
    train_rows = _sample_rows(
        pack.train_rows,
        sample_ratio,
        salt=f"{train_dev_scope}:train",
    )
    dev_rows = _sample_rows(
        pack.dev_rows,
        sample_ratio,
        salt=f"{train_dev_scope}:dev",
    )
    if pack.task_id == COUNTERFACTUAL_TASK:
        test_rows = _sample_grouped_rows(
            pack.test_rows,
            sample_ratio,
            salt=f"{pack.task_id}:{pack.variant}:test",
            group_key_fn=lambda row: str(row.get("parent_id", "")).strip() or _stable_row_token(row),
        )
        meta = {**pack.meta, **_counterfactual_meta(test_rows), "sample_ratio": sample_ratio}
    elif pack.task_id in SHARED_TRAINING_TASK_IDS:
        test_rows = _sample_rows_by_group(
            pack.test_rows,
            sample_ratio,
            salt=f"{pack.task_id}:{pack.variant}:test",
            group_key_fn=_chain_family,
        )
        meta = {**pack.meta, "sample_ratio": sample_ratio}
    else:
        test_rows = _sample_rows(
            pack.test_rows,
            sample_ratio,
            salt=f"{pack.task_id}:{pack.variant}:test",
        )
        meta = {**pack.meta, "sample_ratio": sample_ratio}
    return TaskPack(
        task_id=pack.task_id,
        variant=pack.variant,
        description=pack.description,
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
        meta=meta,
    )


def build_task_packs(
    rows: list[dict[str, Any]],
    task_ids: list[str],
    config: dict[str, Any] | None = None,
) -> list[TaskPack]:
    """Build all requested task packs from metadata rows."""
    config = config or {}
    sample_ratio = config.get("sample_ratio")
    smoke_limits = config.get("smoke_limits")  # (max_train, max_dev, max_test) or None
    packs = [pack for task_id in task_ids for pack in TASK_BUILDERS[task_id](rows)]

    if sample_ratio is not None:
        sample_ratio = float(sample_ratio)
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in the interval (0, 1]")
        packs = [_sample_pack(pack, sample_ratio) for pack in packs]

    if smoke_limits:
        max_train, max_dev, max_test = smoke_limits
        packs = [_truncate_pack(p, max_train, max_dev, max_test) for p in packs]

    return packs
