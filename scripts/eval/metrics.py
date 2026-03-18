"""Basic metrics for normalized baseline score files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

from lib.config import load_json
from .tasks import COUNTERFACTUAL_TASK, TaskPack

BONA_LABELS = {"bona_fide", "bonafide", "1"}
SPOOF_LABELS = {"spoof", "0"}


def build_label_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        (row.get("sample_id") or "").strip(): (row.get("label") or "").strip()
        for row in rows
        if (row.get("sample_id") or "").strip()
    }


def load_scores_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _label_to_binary(value: str) -> int | None:
    normalized = (value or "").strip().lower()
    if normalized in BONA_LABELS:
        return 1
    if normalized in SPOOF_LABELS:
        return 0
    return None


def _binary_scores(scores: list[dict[str, Any]], score_key: str, label_key: str) -> list[tuple[float, int]]:
    result: list[tuple[float, int]] = []
    for row in scores:
        label = _label_to_binary(str(row.get(label_key, "")))
        if label is not None:
            result.append((_parse_score(row.get(score_key)), label))
    return result


def _apply_label_map(scores: list[dict[str, Any]], label_map: dict[str, str] | None) -> list[dict[str, Any]]:
    if not label_map:
        return scores
    enriched: list[dict[str, Any]] = []
    for row in scores:
        sample_id = (row.get("sample_id") or "").strip()
        if sample_id and not (row.get("label") or "").strip() and sample_id in label_map:
            enriched.append({**row, "label": label_map[sample_id]})
        else:
            enriched.append(row)
    return enriched


def _enrich_scores(
    scores: list[dict[str, Any]],
    *,
    label_map: dict[str, str] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    enriched = _apply_label_map(scores, label_map)
    if not metadata_rows:
        return enriched

    metadata_by_sample = {
        (row.get("sample_id") or "").strip(): row
        for row in metadata_rows
        if (row.get("sample_id") or "").strip()
    }
    merged_rows: list[dict[str, Any]] = []
    for row in enriched:
        sample_id = (row.get("sample_id") or "").strip()
        metadata = metadata_by_sample.get(sample_id)
        if not metadata:
            merged_rows.append(row)
            continue
        merged = {**metadata, **row}
        if not (merged.get("label") or "").strip():
            merged["label"] = metadata.get("label", "")
        merged_rows.append(merged)
    return merged_rows


def _validate_score_coverage(scores: list[dict[str, Any]], task_pack: TaskPack | None) -> None:
    if task_pack is None:
        return
    expected_ids = [
        (row.get("sample_id") or "").strip()
        for row in task_pack.test_rows
        if (row.get("sample_id") or "").strip()
    ]
    if not expected_ids:
        return
    seen_counts: dict[str, int] = defaultdict(int)
    unexpected_ids: set[str] = set()
    for row in scores:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        seen_counts[sample_id] += 1
        if sample_id not in expected_ids:
            unexpected_ids.add(sample_id)
    duplicate_ids = sorted(sample_id for sample_id, count in seen_counts.items() if count > 1)
    missing_ids = sorted(sample_id for sample_id in expected_ids if seen_counts.get(sample_id, 0) == 0)
    if duplicate_ids or missing_ids or unexpected_ids:
        raise ValueError(
            "scores.csv coverage mismatch: "
            f"missing={len(missing_ids)} duplicate={len(duplicate_ids)} unexpected={len(unexpected_ids)}"
        )


def compute_eer_from_labels(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """Compute EER and its threshold from binary labels and positive-class scores."""
    if labels.size == 0 or predictions.size == 0 or labels.size != predictions.size:
        return 1.0, 0.0
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    fnr = 1.0 - tpr
    best_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[best_index] + fnr[best_index]) / 2.0)
    threshold = float(thresholds[best_index])
    return eer, threshold


def _compute_eer_from_binary_scores(binary_scores: list[tuple[float, int]]) -> float:
    if not binary_scores:
        return 0.0
    labels = np.array([label for _, label in binary_scores], dtype=np.int32)
    predictions = np.array([score for score, _ in binary_scores], dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    eer, _ = compute_eer_from_labels(labels, predictions)
    return eer


def compute_eer(scores: list[dict[str, Any]], score_key: str = "score", label_key: str = "label") -> float:
    """Equal-error-rate estimate. Assumes higher score = bona fide."""
    return _compute_eer_from_binary_scores(_binary_scores(scores, score_key, label_key))


def _compute_auc_from_binary_scores(binary_scores: list[tuple[float, int]]) -> float:
    if not binary_scores:
        return 0.0
    ordered_scores = sorted(binary_scores, key=lambda item: item[0])
    n_pos = sum(1 for _, label in ordered_scores if label == 1)
    n_neg = len(ordered_scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = 0
    for i, (_, label) in enumerate(ordered_scores):
        if label == 1:
            rank_sum += i + 1
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(0.0, min(1.0, auc))


def compute_auc_simple(scores: list[dict[str, Any]], score_key: str = "score", label_key: str = "label") -> float:
    """AUC via trapezoidal rule on sorted scores (positive = bona fide)."""
    return _compute_auc_from_binary_scores(_binary_scores(scores, score_key, label_key))


def _compute_accuracy_from_binary_scores(binary_scores: list[tuple[float, int]], threshold: float) -> float:
    if not binary_scores:
        return 0.0
    correct = 0
    for score, label in binary_scores:
        if (1 if score >= threshold else 0) == label:
            correct += 1
    return correct / len(binary_scores)


def compute_accuracy(scores: list[dict[str, Any]], score_key: str = "score", label_key: str = "label", threshold: float | None = None) -> float:
    binary_scores = _binary_scores(scores, score_key, label_key)
    threshold = 0.5 if threshold is None else threshold
    return _compute_accuracy_from_binary_scores(binary_scores, threshold)


def _compute_f1_from_binary_scores(binary_scores: list[tuple[float, int]], threshold: float) -> float:
    if not binary_scores:
        return 0.0
    tp = fp = fn = 0
    for score, label in binary_scores:
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_f1(scores: list[dict[str, Any]], score_key: str = "score", label_key: str = "label", threshold: float = 0.5) -> float:
    return _compute_f1_from_binary_scores(_binary_scores(scores, score_key, label_key), threshold)


def _score_to_risk(score: float, label: int) -> float:
    normalized_score = max(0.0, min(1.0, score))
    return 1.0 - normalized_score if label == 1 else normalized_score


def _compute_counterfactual_metrics(
    scores: list[dict[str, Any]],
    *,
    reference_chain_family: str = "direct",
    threshold: float = 0.5,
) -> dict[str, float | int | str]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        parent_id = (row.get("parent_id") or "").strip()
        chain_family = (row.get("chain_family") or "").strip()
        if not parent_id or not chain_family:
            continue
        label = _label_to_binary(str(row.get("label", "")))
        if label is None:
            continue
        by_parent[parent_id].append(
            {
                "chain_family": chain_family,
                "score": _parse_score(row.get("score")),
                "label": label,
            }
        )

    paired_parent_count = 0
    paired_chain_count = 0
    parent_flip_count = 0
    flipped_chain_count = 0
    score_drifts: list[float] = []
    worst_chain_risks: list[float] = []

    for parent_rows in by_parent.values():
        by_family: dict[str, dict[str, Any]] = {}
        for row in parent_rows:
            by_family.setdefault(str(row["chain_family"]), row)
        reference_row = by_family.get(reference_chain_family)
        if reference_row is None or len(by_family) < 2:
            continue

        paired_parent_count += 1
        reference_pred = 1 if float(reference_row["score"]) >= threshold else 0
        parent_has_flip = False
        parent_worst_risk = _score_to_risk(float(reference_row["score"]), int(reference_row["label"]))

        for family, row in by_family.items():
            score = float(row["score"])
            parent_worst_risk = max(parent_worst_risk, _score_to_risk(score, int(row["label"])))
            if family == reference_chain_family:
                continue
            paired_chain_count += 1
            score_drifts.append(abs(score - float(reference_row["score"])))
            if (1 if score >= threshold else 0) != reference_pred:
                flipped_chain_count += 1
                parent_has_flip = True

        if parent_has_flip:
            parent_flip_count += 1
        worst_chain_risks.append(parent_worst_risk)

    return {
        "reference_chain_family": reference_chain_family,
        "counterfactual_parent_count": paired_parent_count,
        "counterfactual_chain_pairs": paired_chain_count,
        "counterfactual_flip_rate": (
            flipped_chain_count / paired_chain_count if paired_chain_count else 0.0
        ),
        "counterfactual_parent_flip_rate": (
            parent_flip_count / paired_parent_count if paired_parent_count else 0.0
        ),
        "score_drift": sum(score_drifts) / len(score_drifts) if score_drifts else 0.0,
        "worst_chain_risk": (
            sum(worst_chain_risks) / len(worst_chain_risks) if worst_chain_risks else 0.0
        ),
    }


def compute_metrics_for_scores(
    scores_path: Path,
    label_map: dict[str, str] | None = None,
    task_pack: TaskPack | None = None,
) -> dict[str, float | int | str]:
    scores = _enrich_scores(
        load_scores_csv(scores_path),
        label_map=label_map,
        metadata_rows=task_pack.test_rows if task_pack is not None else None,
    )
    _validate_score_coverage(scores, task_pack)
    binary_scores = _binary_scores(scores, "score", "label")
    calibrated_threshold = 0.5
    if binary_scores:
        labels = np.array([label for _, label in binary_scores], dtype=np.int32)
        predictions = np.array([score for score, _ in binary_scores], dtype=np.float64)
        _, calibrated_threshold = compute_eer_from_labels(labels, predictions)
    metrics: dict[str, float | int | str] = {
        "eer": _compute_eer_from_binary_scores(binary_scores),
        "auc": _compute_auc_from_binary_scores(binary_scores),
        "accuracy": _compute_accuracy_from_binary_scores(binary_scores, calibrated_threshold),
        "f1": _compute_f1_from_binary_scores(binary_scores, calibrated_threshold),
        "calibrated_threshold": calibrated_threshold,
        "n_samples": len(scores),
    }
    if task_pack is not None and task_pack.task_id == COUNTERFACTUAL_TASK:
        metrics.update(
            _compute_counterfactual_metrics(
                scores,
                reference_chain_family=str(task_pack.meta.get("reference_chain_family", "direct")),
            )
        )
    return metrics


def aggregate_run_metrics(
    output_root: Path,
    baseline_results: list[dict[str, Any]],
    label_maps: dict[tuple[str, str], dict[str, str]] | None = None,
    task_packs: dict[tuple[str, str], TaskPack] | None = None,
) -> list[dict[str, Any]]:
    """For each baseline run that has scores_path, compute metrics and return list of run metrics."""
    out: list[dict[str, Any]] = []
    for run in baseline_results:
        sp = run.get("scores_path")
        if not sp:
            out.append({**run, "metrics": None})
            continue
        path = Path(sp)
        if not path.is_absolute():
            path = output_root / sp
        if not path.exists():
            out.append({**run, "metrics": None})
            continue
        metrics_path = output_root / run["task_id"] / run["variant"] / run["baseline"] / "metrics.json"
        if metrics_path.exists() and metrics_path.stat().st_mtime >= path.stat().st_mtime:
            out.append({**run, "metrics": load_json(metrics_path)})
            continue
        task_key = (run["task_id"], run["variant"])
        label_map = (label_maps or {}).get(task_key)
        task_pack = (task_packs or {}).get(task_key)
        out.append(
            {
                **run,
                "metrics": compute_metrics_for_scores(path, label_map=label_map, task_pack=task_pack),
            }
        )
    return out
