# E2Former

Official implementation of **E2Former**, an efficient and scalable E(3)-equivariant transformer for molecular energy and force prediction.

[Paper (arXiv:2501.19216)](https://arxiv.org/abs/2501.19216) | [License: MIT](./LICENSE)

![E2Former architecture](assets/fig2.png)

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Training](#training)
- [Validation](#validation)
- [Molecular Dynamics Rollout](#molecular-dynamics-rollout)
- [Command Reference](#command-reference)
- [Configuration Guide](#configuration-guide)
- [Reproducibility Tips](#reproducibility-tips)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

E2Former targets large-scale molecular modeling with a design that couples E(3)-equivariance and efficient attention. The codebase is built around FairChem-style configuration and training workflows, with support for single-node single-GPU and multi-GPU execution.

For model-design details, see [`model_architecture.md`](./model_architecture.md).

## Highlights

- Linear-scaling tensor-product design with Wigner-6j-based operations.
- Equivariant attention blocks for geometry-aware molecular learning.
- Config-driven experiments across OC22/S2EF-style benchmarks.
- Practical training utilities, including checkpoint resume and tmux-based launch.

## Repository Layout

```text
E2Former/
├── configs/          # Training, dataset, and experiment YAMLs
├── docs/             # Additional documentation and website assets
├── src/              # Core model implementation
│   ├── core/
│   ├── layers/
│   ├── models/
│   ├── utils/
│   └── wigner6j/
├── main.py           # Main FairChem-compatible training entrypoint
├── start_exp.py      # Background launcher (tmux-based)
├── test_e2former.py  # Model smoke/equivariance checks
├── simulate.py       # MD rollout entrypoint
└── env.yml           # Conda environment definition
```

## Requirements

- Linux with NVIDIA GPU recommended
- Python 3.10
- PyTorch 2.4.1 + CUDA 12.1 (defined in `env.yml`)

## Installation

1. Create and activate the environment:

```bash
conda env create -f env.yml
conda activate e2former
```

Or with `mamba`:

```bash
mamba env create -f env.yml
conda activate e2former
```

2. Install `fairchem-core` (required by `main.py`):

```bash
git clone https://github.com/FAIR-Chem/fairchem.git
pip install -e fairchem/packages/fairchem-core
```

3. Optional development setup:

```bash
pre-commit install
```

4. Optional MD rollout dependency (`simulate.py`):

```bash
git clone https://github.com/kyonofx/MDsim.git
pip install -e MDsim
```

## Data Setup

Training configs expect FairChem-style datasets (typically LMDB). Before launching training:

- Update `dataset.train.src` and `dataset.val.src` in your YAML.
- Ensure referenced normalization and linear-reference files exist.
- Start from one of the provided templates:
  - `configs/oc22/s2ef/e2former/e2former.yaml`
  - `configs/example_config_EScAIP.yml`

## Training

### Single GPU

```bash
python main.py \
  --mode train \
  --num-gpus 1 \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --run-dir ./runs \
  --identifier e2former_oc22
```

### Resume from checkpoint

```bash
python main.py \
  --mode train \
  --num-gpus 1 \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --run-dir ./runs \
  --identifier e2former_oc22 \
  --checkpoint /path/to/checkpoint.pt
```

### Background launch (tmux)

```bash
python start_exp.py \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --mode train \
  --cvd 0 \
  --run-dir ./runs \
  --identifier e2former_bg
```

### Multi-GPU (single node)

```bash
torchrun --standalone --nproc_per_node 4 main.py \
  --distributed \
  --num-gpus 4 \
  --mode train \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --run-dir ./runs \
  --identifier e2former_4gpu
```

## Validation

Run the smoke/equivariance check:

```bash
python test_e2former.py
```

## Molecular Dynamics Rollout

After installing MDsim, run:

```bash
python simulate.py \
  --simulation_config_yml <SIMULATION_YML> \
  --model_dir <MODEL_DIR> \
  --model_config_yml <MODEL_CONFIG_YML> \
  --identifier <RUN_NAME>
```

## Command Reference

Common CLI arguments used in this repository:

- `--config-yml`: experiment config YAML path.
- `--mode`: execution mode (for example, `train`).
- `--run-dir`: directory for checkpoints and logs.
- `--identifier`: experiment name used in logs/artifacts.
- `--checkpoint`: checkpoint path for resume/inference.
- `--num-gpus`: number of GPUs used by `main.py`.
- `--distributed`: enable distributed training flow.

## Configuration Guide

Key knobs for stability and performance:

- `model.backbone.max_neighbors`, `model.backbone.max_radius`
- `model.backbone.num_layers`, `model.backbone.num_attn_heads`
- `model.backbone.attn_type`, `model.backbone.atten_name`
- `model.backbone.use_fp16_backbone`, `model.backbone.use_compile`
- `optim.batch_size`, `optim.eval_batch_size`, `optim.lr_initial`

## Reproducibility Tips

- Keep each run tied to one explicit config file and `--identifier`.
- Log checkpoint paths used for resume/inference.
- Run `python test_e2former.py` before long experiments.
- Scale batch size and `max_neighbors` gradually when moving to larger GPUs.

## Citation

If you use this repository, please cite:

```bibtex
@article{li2025e2former,
  title={E2Former: A Linear-time Efficient and Equivariant Transformer for Scalable Molecular Modeling},
  author={Li, Yunyang and Huang, Lin and Ding, Zhihao and Wang, Chu and Wei, Xinran and Yang, Han and Wang, Zun and Liu, Chang and Shi, Yu and Jin, Peiran and others},
  journal={arXiv preprint arXiv:2501.19216},
  year={2025}
}
```

## License

This project is released under the [MIT License](./LICENSE).

## Acknowledgments

E2Former builds on the FairChem ecosystem and prior work in equivariant molecular machine learning.
