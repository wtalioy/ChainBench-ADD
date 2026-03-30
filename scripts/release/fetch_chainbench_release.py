#!/usr/bin/env python3
"""Download and extract a sharded ChainBench release from the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tqdm.auto import tqdm

MANIFEST_NAME = "release-manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="Lioy/ChainBench",
        help="Hugging Face dataset repository containing the release shards.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        help="Hub repository type passed to snapshot_download.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision to download.",
    )
    parser.add_argument(
        "--local-dir",
        default="release/ChainBench-download",
        help="Local directory used to store the downloaded shards.",
    )
    parser.add_argument(
        "--extract-parent",
        default="data",
        help="Parent directory where the dataset root should be recreated.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional local manifest path. If set, skip Hub download and extract from local files.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives after extraction.",
    )
    return parser.parse_args()


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program not found on PATH: {name}")


def download_release(repo_id: str, repo_type: str, revision: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for download mode. Install it with "
            "`pip install huggingface_hub` or pass --manifest to use local files."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading release shards from {repo_id}@{revision} into {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[MANIFEST_NAME, "*.tar.zst"],
    )
    manifest_path = local_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Downloaded repo does not contain {MANIFEST_NAME}")
    return manifest_path


def extract_archive(archive_path: Path, extract_parent: Path) -> None:
    subprocess.run(
        [
            "tar",
            "-I",
            "zstd -T0 -d",
            "-xvf",
            str(archive_path),
            "-C",
            str(extract_parent),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    require_program("tar")
    require_program("zstd")

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
    else:
        manifest_path = download_release(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=Path(args.local_dir).resolve(),
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_specs = manifest.get("shards")
    if not isinstance(shard_specs, list) or not shard_specs:
        raise RuntimeError("Manifest does not contain any shards")

    extract_parent = Path(args.extract_parent).resolve()
    extract_parent.mkdir(parents=True, exist_ok=True)

    archives_dir = manifest_path.parent
    extraction_progress = tqdm(shard_specs, desc="Extract archives", unit="archive", dynamic_ncols=True)
    for shard in extraction_progress:
        archive_name = str(shard["archive_name"])
        archive_path = archives_dir / archive_name
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing archive referenced by manifest: {archive_path}")
        extraction_progress.set_postfix_str(archive_name)
        extract_archive(archive_path, extract_parent)

    if not args.keep_archives and not args.manifest:
        shutil.rmtree(archives_dir)

    dataset_root = extract_parent / str(manifest["dataset_root"])
    print(f"Ready: {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
