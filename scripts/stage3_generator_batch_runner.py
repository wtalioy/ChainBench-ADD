#!/usr/bin/env python3
"""Persistent Stage-3 generator batch runner.

Loads one spoof generator once, then processes a JSONL job list sequentially.
This avoids paying model initialization cost for every single parent sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from common_logging import format_elapsed, get_logger, progress_bar, setup_logging


LOGGER = get_logger("stage3-runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Adapter key to use.")
    parser.add_argument("--repo-path", required=True, help="Generator repo root.")
    parser.add_argument("--config-path", required=True, help="Adapter config JSON.")
    parser.add_argument("--jobs-path", required=True, help="JSONL jobs file.")
    parser.add_argument("--results-path", required=True, help="JSONL results file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Emit aggregate progress every N jobs. Default auto-selects by batch size.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jobs(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def map_qwen_language(language: str) -> str:
    return {"zh": "Chinese", "en": "English"}.get(language, "Auto")


def resolve_local_or_hf_model_dir(repo_path: Path, local_path_str: str, hf_repo_id: str | None) -> Path:
    local_path = Path(local_path_str)
    if not local_path.is_absolute():
        local_path = repo_path / local_path
    if local_path.exists():
        if not local_path.is_dir():
            raise NotADirectoryError(f"Model path exists but is not a directory: {local_path}")
        if not hf_repo_id:
            return local_path

        # Local directory exists: ensure it is complete/consistent with the HF snapshot.
        LOGGER.info("verify model snapshot %s -> %s", hf_repo_id, local_path)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to verify/download HF model snapshot {hf_repo_id} into existing directory {local_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return local_path
    if not hf_repo_id:
        raise FileNotFoundError(f"Local model directory not found: {local_path}")

    LOGGER.info("download model %s -> %s", hf_repo_id, local_path)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_path


class AdapterRunner:
    def __init__(self, repo_path: Path, config: dict[str, Any]) -> None:
        self.repo_path = repo_path
        self.config = config

    def setup(self) -> None:
        raise NotImplementedError

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class Qwen3CloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        dtype_name = self.config.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)
        self.model = Qwen3TTSModel.from_pretrained(
            self.config["model_path"],
            device_map=self.config.get("device", "cuda:0"),
            dtype=dtype,
            attn_implementation=self.config.get("attn_implementation", "flash_attention_2"),
        )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        import soundfile as sf

        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavs, sr = self.model.generate_voice_clone(
            text=job["text"],
            language=map_qwen_language(job["language"]),
            ref_audio=job["prompt_audio_path"],
            ref_text=job["prompt_text"],
            **self.config.get("generation_kwargs", {}),
        )
        sf.write(output_path, wavs[0], sr)
        return {"sample_rate": sr}


class CosyVoice3CloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        sys.path.insert(0, str(self.repo_path / "third_party" / "Matcha-TTS"))
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        self.torch = torch
        model_dir = resolve_local_or_hf_model_dir(
            self.repo_path,
            self.config["model_path"],
            self.config.get("hf_repo_id"),
        )
        self.model = AutoModel(model_dir=str(model_dir))

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        import torchaudio

        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = job["prompt_text"]
        prefix = self.config.get("prepend_prompt_prefix", "")
        if prefix:
            prompt_text = f"{prefix}{prompt_text}"

        outputs = list(
            self.model.inference_zero_shot(
                job["text"],
                prompt_text,
                job["prompt_audio_path"],
                stream=bool(self.config.get("stream", False)),
                speed=float(self.config.get("speed", 1.0)),
                text_frontend=bool(self.config.get("text_frontend", True)),
            )
        )
        speech = self.torch.cat([item["tts_speech"] for item in outputs], dim=1).cpu()
        torchaudio.save(str(output_path), speech, self.model.sample_rate)
        return {"sample_rate": self.model.sample_rate}


class SparkTTSCloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        import torch
        from cli.SparkTTS import SparkTTS

        device_name = self.config.get("device", "cuda:0")
        self.torch = torch
        self.device = torch.device(device_name)
        model_dir = resolve_local_or_hf_model_dir(
            self.repo_path,
            self.config["model_dir"],
            self.config.get("hf_repo_id"),
        )
        self.model = SparkTTS(model_dir, self.device)

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        import soundfile as sf

        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav = self.model.inference(
            job["text"],
            prompt_speech_path=Path(job["prompt_audio_path"]),
            prompt_text=job["prompt_text"],
            temperature=float(self.config.get("temperature", 0.8)),
            top_k=int(self.config.get("top_k", 50)),
            top_p=float(self.config.get("top_p", 0.95)),
        )
        sf.write(output_path, wav, samplerate=self.model.sample_rate)
        return {"sample_rate": self.model.sample_rate}


class F5TTSCloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path / "src"))
        from f5_tts.api import F5TTS

        self.model = F5TTS(
            model=self.config.get("model", "F5TTS_v1_Base"),
            ckpt_file=self.config.get("ckpt_file", ""),
            vocab_file=self.config.get("vocab_file", ""),
            device=self.config.get("device"),
            hf_cache_dir=self.config.get("hf_cache_dir"),
            vocoder_local_path=self.config.get("vocoder_local_path"),
        )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _, sr, _ = self.model.infer(
            ref_file=job["prompt_audio_path"],
            ref_text=job["prompt_text"],
            gen_text=job["text"],
            target_rms=float(self.config.get("target_rms", 0.1)),
            cross_fade_duration=float(self.config.get("cross_fade_duration", 0.15)),
            sway_sampling_coef=float(self.config.get("sway_sampling_coef", -1)),
            cfg_strength=float(self.config.get("cfg_strength", 2.0)),
            nfe_step=int(self.config.get("nfe_step", 32)),
            speed=float(self.config.get("speed", 1.0)),
            fix_duration=self.config.get("fix_duration"),
            remove_silence=bool(self.config.get("remove_silence", False)),
            file_wave=str(output_path),
            seed=self.config.get("seed"),
        )
        return {"sample_rate": sr}


class VoxCPMCloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path / "src"))
        import soundfile as sf
        from voxcpm import VoxCPM

        self.sf = sf
        if self.config.get("model_path"):
            self.model = VoxCPM(
                voxcpm_model_path=self.config["model_path"],
                zipenhancer_model_path=self.config.get("zipenhancer_model_path"),
                enable_denoiser=bool(self.config.get("denoise", False)),
            )
        else:
            self.model = VoxCPM.from_pretrained(
                hf_model_id=self.config.get("hf_model_id", "openbmb/VoxCPM1.5"),
                cache_dir=self.config.get("cache_dir"),
                local_files_only=bool(self.config.get("local_files_only", False)),
                load_denoiser=bool(self.config.get("denoise", False)),
                zipenhancer_model_id=self.config.get("zipenhancer_model_path"),
            )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav = self.model.generate(
            text=job["text"],
            prompt_wav_path=job["prompt_audio_path"],
            prompt_text=job["prompt_text"],
            cfg_value=float(self.config.get("cfg_value", 2.0)),
            inference_timesteps=int(self.config.get("inference_timesteps", 10)),
            normalize=bool(self.config.get("normalize", False)),
            denoise=bool(self.config.get("denoise", False)),
        )
        sr = self.model.tts_model.sample_rate
        self.sf.write(output_path, wav, sr)
        return {"sample_rate": sr}


class IndexTTS2CloneRunner(AdapterRunner):
    def setup(self) -> None:
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))
        from indextts.infer_v2 import IndexTTS2

        cfg_path = self.repo_path / self.config.get("cfg_path", "checkpoints/config.yaml")
        model_dir = resolve_local_or_hf_model_dir(
            self.repo_path,
            self.config["model_dir"],
            self.config.get("hf_repo_id"),
        )
        self.model = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_fp16=bool(self.config.get("use_fp16", False)),
            device=self.config.get("device"),
            use_cuda_kernel=self.config.get("use_cuda_kernel"),
            use_deepspeed=bool(self.config.get("use_deepspeed", False)),
            use_accel=bool(self.config.get("use_accel", False)),
            use_torch_compile=bool(self.config.get("use_torch_compile", False)),
        )

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.infer(
            spk_audio_prompt=job["prompt_audio_path"],
            text=job["text"],
            output_path=str(output_path),
            verbose=bool(self.config.get("verbose", False)),
        )
        return {"sample_rate": None}


RUNNER_REGISTRY: dict[str, type[AdapterRunner]] = {
    "qwen3_clone": Qwen3CloneRunner,
    "cosyvoice3_clone": CosyVoice3CloneRunner,
    "sparktts_clone": SparkTTSCloneRunner,
    "f5tts_clone": F5TTSCloneRunner,
    "voxcpm_clone": VoxCPMCloneRunner,
    "indextts2_clone": IndexTTS2CloneRunner,
}


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    repo_path = Path(args.repo_path).resolve()
    config = load_json(Path(args.config_path))
    jobs = load_jobs(Path(args.jobs_path))
    results_path = Path(args.results_path)
    if results_path.exists():
        results_path.unlink()

    runner_cls = RUNNER_REGISTRY.get(args.adapter)
    if runner_cls is None:
        raise RuntimeError(f"Unsupported adapter: {args.adapter}")

    LOGGER.info("boot adapter=%s repo=%s", args.adapter, repo_path)
    startup_started_at = time.monotonic()
    runner = runner_cls(repo_path=repo_path, config=config)
    runner.setup()
    total_jobs = len(jobs)
    progress_every = args.progress_every if args.progress_every > 0 else (1 if total_jobs <= 10 else 5 if total_jobs <= 50 else 25)
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    started_at = time.monotonic()
    LOGGER.info(
        "ready in %s | jobs=%d",
        format_elapsed(time.monotonic() - startup_started_at),
        total_jobs,
    )

    for idx, job in enumerate(jobs, start=1):
        job_started_at = time.monotonic()
        LOGGER.info(
            "%s %d/%d starting sample=%s speaker=%s split=%s",
            progress_bar(idx - 1, total_jobs),
            idx,
            total_jobs,
            job["sample_id"],
            job["speaker_id"],
            job["split"],
        )
        output_path = Path(job["output_path"])
        if output_path.exists():
            skipped_count += 1
            append_jsonl(
                results_path,
                {
                    "job_id": job["job_id"],
                    "status": "skipped_existing",
                    "output_path": job["output_path"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                },
            )
            LOGGER.info(
            "%s %d/%d skip  | %s | sample=%s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - job_started_at),
                job["sample_id"],
                ok_count,
                skipped_count,
                failed_count,
            )
            continue

        try:
            meta = runner.run_job(job)
            ok_count += 1
            append_jsonl(
                results_path,
                {
                    "job_id": job["job_id"],
                    "status": "ok",
                    "output_path": job["output_path"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                    "meta": meta,
                },
            )
            LOGGER.success(
                "%s %d/%d done  | %s | sample=%s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - job_started_at),
                job["sample_id"],
                ok_count,
                skipped_count,
                failed_count,
            )
        except Exception as exc:
            failed_count += 1
            append_jsonl(
                results_path,
                {
                    "job_id": job["job_id"],
                    "status": "failed",
                    "output_path": job["output_path"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            LOGGER.error(
                "%s %d/%d fail  | %s | sample=%s | %s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - job_started_at),
                job["sample_id"],
                f"{type(exc).__name__}: {exc}",
                ok_count,
                skipped_count,
                failed_count,
            )

        if idx <= 3 or idx % progress_every == 0 or idx == total_jobs:
            LOGGER.info(
                "%s %d/%d pulse | %s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - started_at),
                ok_count,
                skipped_count,
                failed_count,
            )

    LOGGER.success(
        "complete | adapter=%s | jobs=%d | elapsed=%s | ok=%d skip=%d fail=%d",
        args.adapter,
        total_jobs,
        format_elapsed(time.monotonic() - started_at),
        ok_count,
        skipped_count,
        failed_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
