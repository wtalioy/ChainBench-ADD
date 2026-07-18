"""Shared baseline runner abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chainbench.lib.conda import conda_run_python_command
from chainbench.lib.proc import run_command_streaming

from ..tasks import TaskPack

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = WORKSPACE_ROOT / "chainbench"


@dataclass
class BaselineRunResult:
    ok: bool
    returncode: int
    scores_path: Path | None = None
    model_path: Path | None = None
    raw_output_path: Path | None = None
    command: list[str] | None = None
    log_path: Path | None = None


class BaselineRunner:
    name = "baseline"
    checkpoint_patterns: tuple[str, ...] = ("best.pth", "best.pt", "checkpoint.pt", "model.pt")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.repo_path = Path(config["repo_path"])
        self.conda_env = str(config["conda_env"])

    def prepare_view(self, data_view: TaskPack, run_dir: Path, dataset_root: Path) -> dict[str, Any]:
        raise NotImplementedError

    def train(self, prepared_view: dict[str, Any], run_dir: Path) -> BaselineRunResult:
        raise NotImplementedError

    def evaluate(self, prepared_view: dict[str, Any], run_dir: Path, checkpoint: Path | None) -> BaselineRunResult:
        raise NotImplementedError

    def normalize_scores(self, prepared_view: dict[str, Any], run_dir: Path, raw_output_path: Path) -> Path:
        raise NotImplementedError

    def find_checkpoint(self, run_dir: Path) -> Path | None:
        for pattern in self.checkpoint_patterns:
            matches = sorted(run_dir.rglob(pattern))
            if matches:
                return matches[0]
        return None

    def _command_prefix(self) -> list[str]:
        return conda_run_python_command(self.conda_env)

    def _run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        extra_env: dict[str, str] | None = None,
    ) -> BaselineRunResult:
        env = {"PYTHONPATH": f"{WORKSPACE_ROOT}:{PACKAGE_ROOT}", **dict(self.config.get("env", {}))}
        if extra_env:
            env.update(extra_env)
        returncode = run_command_streaming(command, cwd=cwd, log_path=log_path, env=env, tee_output=False)
        return BaselineRunResult(ok=returncode == 0, returncode=returncode, command=command, log_path=log_path)
