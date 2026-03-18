"""Shared task-view loading and baseline view materialization."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.io import write_csv
from lib.logging import get_logger
from .progress import progress_iter
from .tasks import TaskPack

LOGGER = get_logger("eval.views")

ATTACK_TYPES = [f"A{idx:02d}" for idx in range(7, 20)]


@dataclass
class ASVspoofViewPaths:
    root: Path
    database_root: Path
    protocols_root: Path
    train_protocol: Path
    dev_protocol: Path
    eval_protocol_2019: Path
    eval_protocol_2021: Path
    eval_protocol_2021_la: Path
    eval_protocol_itw: Path
    asv_score_path: Path
    manifest_path: Path

    def required_paths(self) -> list[Path]:
        return [
            self.train_protocol,
            self.dev_protocol,
            self.eval_protocol_2019,
            self.eval_protocol_2021,
            self.eval_protocol_2021_la,
            self.eval_protocol_itw,
            self.asv_score_path,
            self.manifest_path,
        ]


@dataclass
class SafeEarViewPaths:
    root: Path
    asvspoof_view: ASVspoofViewPaths
    train_tsv: Path
    dev_tsv: Path
    test_tsv_2019: Path
    test_tsv_2021: Path
    manifest_path: Path


def _is_prepared(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _pack_digest(data_view: TaskPack) -> str:
    payload = {
        "task_id": data_view.task_id,
        "variant": data_view.variant,
        "train_rows": data_view.train_rows,
        "dev_rows": data_view.dev_rows,
        "test_rows": data_view.test_rows,
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return actual == expected


def _label_to_protocol_token(label: str) -> str:
    normalized = (label or "").strip().lower()
    if normalized in {"bonafide", "bona_fide", "1"}:
        return "bonafide"
    return "spoof"


def _source_for_row(row: dict[str, str], attack_index: int) -> str:
    label = _label_to_protocol_token(row.get("label", ""))
    if label == "bonafide":
        return "bonafide"
    return ATTACK_TYPES[attack_index % len(ATTACK_TYPES)]


def _speaker_for_row(row: dict[str, str], fallback_idx: int) -> str:
    speaker_id = (row.get("speaker_id") or "").strip()
    return speaker_id or f"spk_{fallback_idx:06d}"


def _link_split_audio(
    rows: list[dict[str, str]],
    dataset_root: Path,
    split_dir: Path,
    extension: str,
) -> list[dict[str, str]]:
    total = len(rows)
    split_name = split_dir.parent.name
    linked_rows: list[dict[str, str]] = []
    iterator = progress_iter(rows, total=total, desc=f"link {split_name}", unit="file", leave=False)
    for row in iterator:
        src = dataset_root / row["audio_path"]
        dst = split_dir / f"{row['sample_id']}.{extension}"
        _safe_symlink(src, dst)
        linked_rows.append({**row, "linked_audio_path": str(dst)})
    return linked_rows


def _write_protocol(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spoof_index = 0
    lines = []
    for idx, row in enumerate(rows):
        label = _label_to_protocol_token(row.get("label", ""))
        source = _source_for_row(row, spoof_index)
        spoof_index += 1 if label == "spoof" else 0
        speaker = _speaker_for_row(row, idx)
        lines.append(f"{speaker} {row['sample_id']} - {source} {label}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_dummy_asv_scores(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "utt_tar_1 target 0.95",
        "utt_tar_2 target 0.82",
        "utt_non_1 nontarget -0.45",
        "utt_non_2 nontarget -0.22",
        "utt_spoof_1 spoof -0.71",
        "utt_spoof_2 spoof -0.54",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_asvspoof_view(
    data_view: TaskPack,
    run_dir: Path,
    dataset_root: Path,
    *,
    root: Path | None = None,
    train_track: str = "LA",
    eval_track: str = "DF",
    audio_extension: str = "flac",
) -> ASVspoofViewPaths:
    root = root or (run_dir / "view" / "asvspoof")
    database_root = root / "database"
    protocols_root = root / "protocols"

    train_dir = database_root / f"ASVspoof2019_{train_track}_train" / audio_extension
    dev_dir = database_root / f"ASVspoof2019_{train_track}_dev" / audio_extension
    eval2019_dir = database_root / f"ASVspoof2019_{train_track}_eval" / audio_extension
    eval2021_dir = database_root / f"ASVspoof2021_{eval_track}_eval" / audio_extension
    eval2021_la_dir = database_root / "ASVspoof2021_LA_eval" / audio_extension
    itw_dir = database_root / "release_in_the_wild"

    train_protocol = protocols_root / f"ASVspoof_{train_track}_cm_protocols" / f"ASVspoof2019.{train_track}.cm.train.trn.txt"
    dev_protocol = protocols_root / f"ASVspoof_{train_track}_cm_protocols" / f"ASVspoof2019.{train_track}.cm.dev.trl.txt"
    eval_protocol_2019 = protocols_root / f"ASVspoof_{train_track}_cm_protocols" / f"ASVspoof2019.{train_track}.cm.eval.trl.txt"
    eval_protocol_2021 = protocols_root / f"ASVspoof_{eval_track}_cm_protocols" / f"ASVspoof2021.{eval_track}.cm.eval.trl.txt"
    eval_protocol_2021_la = protocols_root / "ASVspoof_LA_cm_protocols" / "ASVspoof2021.LA.cm.eval.trl.txt"
    eval_protocol_itw = protocols_root / "in_the_wild.txt"
    asv_score_path = database_root / f"ASVspoof2019_{train_track}_asv_scores" / f"ASVspoof2019.{train_track}.asv.eval.gi.trl.scores.txt"
    manifest_path = root / "view_manifest.json"

    view_paths = ASVspoofViewPaths(
        root=root,
        database_root=database_root,
        protocols_root=protocols_root,
        train_protocol=train_protocol,
        dev_protocol=dev_protocol,
        eval_protocol_2019=eval_protocol_2019,
        eval_protocol_2021=eval_protocol_2021,
        eval_protocol_2021_la=eval_protocol_2021_la,
        eval_protocol_itw=eval_protocol_itw,
        asv_score_path=asv_score_path,
        manifest_path=manifest_path,
    )
    manifest_payload = {
        "format": "asvspoof_view_v2",
        "train_track": train_track,
        "eval_track": eval_track,
        "audio_extension": audio_extension,
        "pack_digest": _pack_digest(data_view),
    }
    if _is_prepared(view_paths.required_paths()) and _manifest_matches(manifest_path, manifest_payload):
        LOGGER.info("reusing ASVspoof view for %s/%s from %s", data_view.task_id, data_view.variant, root)
        return view_paths
    if root.exists():
        LOGGER.info("rebuilding ASVspoof view for %s/%s because cached contents changed", data_view.task_id, data_view.variant)
        shutil.rmtree(root)

    LOGGER.info(
        "preparing ASVspoof view for %s/%s (%d train, %d dev, %d test)",
        data_view.task_id,
        data_view.variant,
        len(data_view.train_rows),
        len(data_view.dev_rows),
        len(data_view.test_rows),
    )
    train_rows = _link_split_audio(data_view.train_rows, dataset_root, train_dir, audio_extension)
    dev_rows = _link_split_audio(data_view.dev_rows, dataset_root, dev_dir, audio_extension)
    test_rows_2019 = _link_split_audio(data_view.test_rows, dataset_root, eval2019_dir, audio_extension)
    test_rows_2021 = _link_split_audio(data_view.test_rows, dataset_root, eval2021_dir, audio_extension)
    _link_split_audio(data_view.test_rows, dataset_root, eval2021_la_dir, audio_extension)
    _link_split_audio(data_view.test_rows, dataset_root, itw_dir, "wav")

    _write_protocol(train_protocol, train_rows)
    _write_protocol(dev_protocol, dev_rows)
    _write_protocol(eval_protocol_2019, test_rows_2019)
    _write_protocol(eval_protocol_2021, test_rows_2021)
    _write_protocol(eval_protocol_2021_la, test_rows_2021)
    eval_protocol_itw.write_text("\n".join(row["sample_id"] + ".wav" for row in data_view.test_rows) + "\n", encoding="utf-8")

    _write_dummy_asv_scores(asv_score_path)
    _write_manifest(manifest_path, manifest_payload)

    return view_paths


def _write_safeear_tsv(tsv_path: Path, audio_root: Path, rows: list[dict[str, str]], *, relative_dir: str) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(audio_root)]
    lines.extend(f"{relative_dir}/{row['sample_id']}.flac" for row in rows)
    tsv_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_safeear_view(
    data_view: TaskPack,
    run_dir: Path,
    dataset_root: Path,
    *,
    root: Path | None = None,
) -> SafeEarViewPaths:
    root = root or (run_dir / "view" / "safeear")
    asvspoof = build_asvspoof_view(
        data_view,
        run_dir,
        dataset_root,
        root=root.parent / "asvspoof_LA_LA_flac",
        train_track="LA",
        eval_track="LA",
        audio_extension="flac",
    )
    tsv_root = root / "tsv"
    audio_root = asvspoof.database_root

    train_rel = "ASVspoof2019_LA_train/flac"
    dev_rel = "ASVspoof2019_LA_dev/flac"
    test_rel_2019 = "ASVspoof2019_LA_eval/flac"
    test_rel_2021 = "ASVspoof2021_LA_eval/flac"

    train_tsv = tsv_root / "train.tsv"
    dev_tsv = tsv_root / "dev.tsv"
    test_tsv_2019 = tsv_root / "eval_2019.tsv"
    test_tsv_2021 = tsv_root / "eval_2021.tsv"
    manifest_path = root / "view_manifest.json"

    existing_view = SafeEarViewPaths(
        root=root,
        asvspoof_view=asvspoof,
        train_tsv=train_tsv,
        dev_tsv=dev_tsv,
        test_tsv_2019=test_tsv_2019,
        test_tsv_2021=test_tsv_2021,
        manifest_path=manifest_path,
    )
    manifest_payload = {
        "format": "safeear_view_v2",
        "pack_digest": _pack_digest(data_view),
    }
    if _is_prepared([train_tsv, dev_tsv, test_tsv_2019, test_tsv_2021, manifest_path]) and _manifest_matches(
        manifest_path, manifest_payload
    ):
        LOGGER.info("reusing SafeEar view for %s/%s from %s", data_view.task_id, data_view.variant, root)
        return existing_view
    if root.exists():
        LOGGER.info("rebuilding SafeEar view for %s/%s because cached contents changed", data_view.task_id, data_view.variant)
        shutil.rmtree(root)

    for tsv_path, rows, relative_dir in (
        (train_tsv, data_view.train_rows, train_rel),
        (dev_tsv, data_view.dev_rows, dev_rel),
        (test_tsv_2019, data_view.test_rows, test_rel_2019),
        (test_tsv_2021, data_view.test_rows, test_rel_2021),
    ):
        _write_safeear_tsv(tsv_path, audio_root, rows, relative_dir=relative_dir)
    _write_manifest(manifest_path, manifest_payload)

    return existing_view


def write_normalized_scores(path: Path, rows: list[dict[str, Any]]) -> Path:
    write_csv(path, rows, fieldnames=["sample_id", "score", "label", "raw_path"])
    return path


def extract_score_rows(raw_score_path: Path) -> list[dict[str, Any]]:
    if not raw_score_path.exists():
        return []
    rows = []
    for line in raw_score_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        rows.append({
            "sample_id": Path(parts[0]).stem,
            "score": parts[-1],
            "label": "",
            "raw_path": parts[0],
        })
    return rows


def extract_csv_score_rows(raw_score_path: Path) -> list[dict[str, Any]]:
    if not raw_score_path.exists():
        return []
    with raw_score_path.open("r", encoding="utf-8", newline="") as f:
        return [
            {
                "sample_id": Path(r["audio_path"]).stem,
                "score": r["score"],
                "label": r.get("label", ""),
                "raw_path": r["audio_path"],
            }
            for r in csv.DictReader(f)
        ]
