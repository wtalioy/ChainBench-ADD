#!/usr/bin/env python3
"""Shard a packaged ChainBench dataset into Hugging Face-safe tar.zst archives."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

DEFAULT_SPLITS = ("train", "dev", "test")
DEFAULT_COMPRESSION = 8
DEFAULT_MAX_SHARD_GIB = 15.0
MANIFEST_NAME = "release-manifest.json"


@dataclass(slots=True)
class FileEntry:
    relative_path: str
    size_bytes: int


@dataclass(slots=True)
class ShardPlan:
    archive_name: str
    split: str
    kind: str
    file_count: int
    total_size_bytes: int
    members: list[str]


def format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.2f} GiB"


def manifest_shard_record(plan: ShardPlan, *, output_dir: Path | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "archive_name": plan.archive_name,
        "split": plan.split,
        "kind": plan.kind,
        "file_count": plan.file_count,
        "total_size_bytes": plan.total_size_bytes,
    }
    if output_dir is not None:
        archive_path = output_dir / plan.archive_name
        if archive_path.exists():
            record["archive_size_bytes"] = archive_path.stat().st_size
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/ChainBench",
        help="Path to the packaged dataset root that should become downloadable.",
    )
    parser.add_argument(
        "--output-dir",
        default="release/ChainBench",
        help="Directory where tar.zst shards and the release manifest will be written.",
    )
    parser.add_argument(
        "--max-shard-gib",
        type=float,
        default=DEFAULT_MAX_SHARD_GIB,
        help="Maximum uncompressed payload size per shard in GiB.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION,
        help="zstd compression level passed to tar -I.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the release layout without writing archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing .tar.zst shards and manifest in the output directory first.",
    )
    return parser.parse_args()


def iter_files(root: Path, *, relative_to: Path, desc: str) -> Iterable[FileEntry]:
    candidates = sorted(root.rglob("*"))
    for path in tqdm(candidates, desc=desc, unit="path", dynamic_ncols=True):
        if path.is_file():
            yield FileEntry(relative_path=str(path.relative_to(relative_to)), size_bytes=path.stat().st_size)


def split_into_shards(
    files: list[FileEntry],
    *,
    archive_prefix: str,
    split: str,
    kind: str,
    max_shard_bytes: int,
) -> list[ShardPlan]:
    if not files:
        return []
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")

    shards: list[ShardPlan] = []
    current_members: list[str] = []
    current_size = 0

    for entry in files:
        if entry.size_bytes > max_shard_bytes:
            raise ValueError(
                f"Single file exceeds shard limit: {entry.relative_path} "
                f"({entry.size_bytes} bytes > {max_shard_bytes} bytes)"
            )
        next_size = current_size + entry.size_bytes
        if current_members and next_size > max_shard_bytes:
            index = len(shards)
            shards.append(
                ShardPlan(
                    archive_name=f"{archive_prefix}-{index:03d}.tar.zst",
                    split=split,
                    kind=kind,
                    file_count=len(current_members),
                    total_size_bytes=current_size,
                    members=list(current_members),
                )
            )
            current_members = []
            current_size = 0
        current_members.append(entry.relative_path)
        current_size += entry.size_bytes

    if current_members:
        index = len(shards)
        shards.append(
            ShardPlan(
                archive_name=f"{archive_prefix}-{index:03d}.tar.zst",
                split=split,
                kind=kind,
                file_count=len(current_members),
                total_size_bytes=current_size,
                members=list(current_members),
            )
        )
    return shards


def build_release_plan(dataset_root: Path, max_shard_bytes: int) -> list[ShardPlan]:
    plan: list[ShardPlan] = []
    print(f"Planning release layout for {dataset_root}")

    root_files = []
    for child in sorted(dataset_root.iterdir()):
        if child.name in DEFAULT_SPLITS:
            continue
        if child.is_file():
            root_files.append(
                FileEntry(relative_path=str(child.relative_to(dataset_root.parent)), size_bytes=child.stat().st_size)
            )
        elif child.is_dir():
            root_files.extend(iter_files(child, relative_to=dataset_root.parent, desc=f"Scan root/{child.name}"))

    if root_files:
        total_size = sum(item.size_bytes for item in root_files)
        plan.append(
            ShardPlan(
                archive_name="ChainBench-root.tar.zst",
                split="root",
                kind="metadata",
                file_count=len(root_files),
                total_size_bytes=total_size,
                members=[item.relative_path for item in root_files],
            )
        )
        print(
            f"Planned metadata archive: ChainBench-root.tar.zst "
            f"({len(root_files)} files, {format_gib(total_size)})"
        )

    for split in DEFAULT_SPLITS:
        split_root = dataset_root / split
        if not split_root.exists():
            continue
        split_files = list(iter_files(split_root, relative_to=dataset_root.parent, desc=f"Scan {split}"))
        split_plan = split_into_shards(
            split_files,
            archive_prefix=f"ChainBench-{split}",
            split=split,
            kind="split",
            max_shard_bytes=max_shard_bytes,
        )
        split_bytes = sum(item.total_size_bytes for item in split_plan)
        plan.extend(split_plan)
        print(
            f"Planned {split}: {len(split_files)} files into {len(split_plan)} shard(s) "
            f"totalling {format_gib(split_bytes)}"
        )

    return plan


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for path in output_dir.glob("*.tar.zst"):
        path.unlink()
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()


def create_archive(*, dataset_parent: Path, output_dir: Path, plan: ShardPlan, compression_level: int) -> None:
    archive_path = output_dir / plan.archive_name
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        file_list_path = Path(handle.name)
        for member in plan.members:
            handle.write(member)
            handle.write("\n")
    try:
        compression = f"zstd -T0 -{compression_level}"
        subprocess.run(
            [
                "tar",
                "-I",
                compression,
                "-cf",
                str(archive_path),
                "-C",
                str(dataset_parent),
                "-T",
                str(file_list_path),
            ],
            check=True,
        )
    finally:
        file_list_path.unlink(missing_ok=True)


def write_manifest(
    *,
    output_dir: Path,
    dataset_root: Path,
    plan: list[ShardPlan],
    max_shard_bytes: int,
    compression_level: int,
) -> Path:
    manifest = {
        "dataset_name": dataset_root.name,
        "dataset_root": dataset_root.name,
        "extract_into": str(dataset_root.parent),
        "archive_format": "tar.zst",
        "compression": {"codec": "zstd", "level": compression_level},
        "max_shard_bytes": max_shard_bytes,
        "max_shard_gib": round(max_shard_bytes / (1024**3), 3),
        "shards": [manifest_shard_record(item, output_dir=output_dir) for item in plan],
        "suggested_upload": {
            "command": (
                "python - <<'PY'\n"
                "from huggingface_hub import HfApi\n"
                "api = HfApi()\n"
                "api.upload_large_folder(\n"
                f"    folder_path={str(output_dir)!r},\n"
                "    repo_id='Lioy/ChainBench',\n"
                "    repo_type='dataset',\n"
                "    num_workers=16,\n"
                ")\n"
                "PY"
            )
        },
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program not found on PATH: {name}")


def main() -> int:
    args = parse_args()
    require_program("tar")
    require_program("zstd")

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    output_dir = Path(args.output_dir).resolve()
    max_shard_bytes = int(math.floor(args.max_shard_gib * (1024**3)))
    if max_shard_bytes <= 0:
        raise ValueError("--max-shard-gib must be positive")

    plan = build_release_plan(dataset_root, max_shard_bytes)
    if not plan:
        raise RuntimeError(f"No files found under dataset root: {dataset_root}")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    total_size = sum(item.total_size_bytes for item in plan)
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "output_dir": str(output_dir),
                "archive_count": len(plan),
                "total_input_size_bytes": total_size,
                "archives": [manifest_shard_record(item) for item in plan],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run:
        return 0

    archive_progress = tqdm(plan, desc="Create archives", unit="archive", dynamic_ncols=True)
    for item in archive_progress:
        archive_progress.set_postfix_str(f"{item.archive_name} ({format_gib(item.total_size_bytes)})")
        create_archive(
            dataset_parent=dataset_root.parent,
            output_dir=output_dir,
            plan=item,
            compression_level=args.compression_level,
        )

    manifest_path = write_manifest(
        output_dir=output_dir,
        dataset_root=dataset_root,
        plan=plan,
        max_shard_bytes=max_shard_bytes,
        compression_level=args.compression_level,
    )
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
