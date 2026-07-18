# ChainBench-ADD

[![Hugging Face%20-%20ChainBench](https://img.shields.io/badge/🤗%20Hugging%20Face%20-%20ChainBench-blue)](https://huggingface.co/datasets/Lioy/ChainBench)

ChainBench-ADD is a delivery-aware audio deepfake benchmark built around **matched clean parents**, **structured post-generation delivery chains**, and **protocol-aware evaluation**. The repository contains the full five-stage construction pipeline, the metadata annotation/export step, and a unified baseline evaluation pipeline for the five benchmark tasks introduced in the ChainBench-ADD paper.

## Installation

### Clone and pull submodules

```bash
git clone https://github.com/wtalioy/ChainBench-ADD.git
cd ChainBench-ADD
git submodule update --init --recursive

wget -O xlsr2_300m.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt
ln -s xlsr2_300m.pt baselines/Nes2Net_ASVspoof_ITW/xlsr2_300m.pt
ln -s xlsr2_300m.pt baselines/SLSforASVspoof-2021-DF/xlsr2_300m.pt
```

### Create the main orchestration environment

The main environment drives stages 1, 2, 4, 5, and the eval orchestrator.

```bash
conda create -n chainbench-add python=3.11
conda activate chainbench-add
pip install -e .
```

### Install system dependencies

The pipeline expects these tools to be available on `PATH`:

- `ffmpeg`
- `ffprobe`

If you don't have these dependencies, run:

```bash
sudo apt install ffmpeg
```

### Prepare generator/baseline environments

Under current implementation, each generator/baseline runs inside its own configured conda environment (specified in `config/stage3.json` and `config/eval.json`). You can refer to the README of each generator/baseline for installation instructions.

### Optional workspace root override

By default, paths resolve relative to the repo root. To run from elsewhere:

```bash
export CHAINBENCH_ROOT=/abs/path/to/ChainBench-ADD
```

## Basic Usage

### Build the benchmark from scratch

```bash
chainbench stage1 --config config/stage1.json
chainbench stage2 --config config/stage2.json
chainbench stage3 --config config/stage3.json
chainbench stage4 --config config/stage4.json
chainbench stage5 --config config/stage5.json
```

### Download the benchmark dataset

```bash
chainbench fetch
```

### Run evaluation

```bash
chainbench eval --config config/eval.json
```

Helpful modes:

```bash
chainbench eval --config config/eval.json --dry-run
chainbench eval --config config/eval.json --eval-only
chainbench eval --config config/eval.json --sample-ratio 0.1
chainbench eval --config config/eval.json --tasks in_chain_detection
```

## Documentation

- [Construction Guide](docs/construction.md) — detailed guide to the five-stage benchmark construction pipeline.
- [Evaluation Guide](docs/evaluation.md) — detailed guide to the metadata-driven evaluation pipeline and metrics.
