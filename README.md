# \[NeurIPS 2025 Spotlight\] E2Former

### An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products





[Paper (OpenReview)](https://openreview.net/forum?id=ls5L4IMEwt) • [Paper (arXiv)](https://arxiv.org/abs/2501.19216) • [Model Architecture](model_architecture.md) • [Quick start](#quick-start) • [Citation](#citation)

[![NeurIPS 2025 Spotlight](https://img.shields.io/badge/NeurIPS%202025-Spotlight-1f6feb)](https://openreview.net/forum?id=ls5L4IMEwt)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-informational)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red)](#requirements)

E2Former is an E(3)-equivariant molecular foundation model for energy and force prediction. It combines efficient Wigner-6j-based tensor products with equivariant attention to improve scalability while preserving geometric symmetries.

This repository is adapted from EScAIP and FairChem.

The repository includes:

- `src/`: core E2Former model components (layers, equivariant blocks, wrappers)
- `configs/`: experiment, dataset, and optimization YAMLs
- `main.py`: FairChem-style training entrypoint
- `start_exp.py`: background launcher for tmux-based jobs
- `simulate.py`: molecular dynamics rollout entrypoint

## Method overview

<p align="center">
  <img src="assets/fig2.png" width="900" alt="E2Former architecture with Wigner-6j convolution and equivariant transformer blocks for molecular modeling." />
</p>
<p align="center">
  <em>Figure. E2Former architecture for scalable E(3)-equivariant molecular modeling.</em>
</p>

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Where key code lives](#where-key-code-lives)
- [Repository layout](#repository-layout)
- [Configuration guide](#configuration-guide)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

## Requirements

- Linux with NVIDIA GPU recommended
- Python 3.10
- PyTorch 2.4.1 + CUDA 12.1 (from `env.yml`)

## Installation

### 1) Create the environment

```bash
conda env create -f env.yml
conda activate e2former
```

Or with mamba:

```bash
mamba env create -f env.yml
conda activate e2former
```

### 2) Install `fairchem-core` dependency

```bash
git clone https://github.com/FAIR-Chem/fairchem.git
pip install -e fairchem/packages/fairchem-core
```

### 3) Optional development tooling

```bash
pre-commit install
```

### 4) Optional MD dependency (`simulate.py`)

```bash
git clone https://github.com/kyonofx/MDsim.git
pip install -e MDsim
```

## Quick start

### A) Configure data and experiment YAML

Use one of the provided templates:

- `configs/oc22/s2ef/e2former/e2former.yaml`

Before training, update dataset paths in your config:

- `dataset.train.src`
- `dataset.val.src`

### B) Train E2Former

Single GPU:

```bash
python main.py \
  --mode train \
  --num-gpus 1 \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --run-dir ./runs \
  --identifier e2former_oc22
```

Resume from checkpoint:

```bash
python main.py \
  --mode train \
  --num-gpus 1 \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --run-dir ./runs \
  --identifier e2former_oc22 \
  --checkpoint /path/to/checkpoint.pt
```

Background run via tmux:

```bash
python start_exp.py \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --mode train \
  --cvd 0 \
  --run-dir ./runs \
  --identifier e2former_bg
```

Multi-GPU (single node):

```bash
torchrun --standalone --nproc_per_node 4 main.py \
  --distributed \
  --num-gpus 4 \
  --mode train \
  --config-yml configs/oc22/s2ef/e2former/e2former.yaml \
  --run-dir ./runs \
  --identifier e2former_4gpu
```

### C) Validate and run MD rollout

Smoke/equivariance check:

```bash
python test_e2former.py
```

Molecular dynamics rollout:

```bash
python simulate.py \
  --simulation_config_yml <SIMULATION_YML> \
  --model_dir <MODEL_DIR> \
  --model_config_yml <MODEL_CONFIG_YML> \
  --identifier <RUN_NAME>
```

## Where key code lives

- E2Former model (FairChem-facing wrapper): `src/models/E2Former_wrapper.py`  
  Main class: `E2FormerBackbone`
- E2Former core architecture (transformer stack): `src/models/e2former_main.py`  
  Main class: `E2former`
- Wigner-6j convolution primitive: `src/wigner6j/tensor_product.py`  
  Main class: `FullyConnectedTensorProductWigner6j`
- Arbitrary-order E(3)-equivariant tensor product used by attention: `src/wigner6j/tensor_product.py`  
  Main class: `E2TensorProductArbitraryOrder`
- Attention modules that call the arbitrary-order tensor product: `src/layers/attention/orders.py`

## Repository layout

```text
.
├── configs/                   # Training, dataset, and experiment YAMLs
├── src/                       # E2Former implementation
│   ├── core/
│   ├── layers/
│   ├── models/
│   ├── utils/
│   └── wigner6j/
├── assets/                    # Figures used in README/docs
├── main.py                    # Main training/inference entrypoint
├── start_exp.py               # tmux-based background training launcher
├── test_e2former.py           # Smoke/equivariance test script
├── simulate.py                # Molecular dynamics rollout entrypoint
├── model_architecture.md      # Architecture notes
└── env.yml                    # Conda environment file
```

## Configuration guide

Common high-impact settings:

- `model.backbone.max_neighbors`, `model.backbone.max_radius`
- `model.backbone.num_layers`, `model.backbone.num_attn_heads`
- `model.backbone.attn_type`, `model.backbone.atten_name`
- `model.backbone.use_fp16_backbone`, `model.backbone.use_compile`
- `optim.batch_size`, `optim.eval_batch_size`, `optim.lr_initial`

## Documentation

Start with:

- `model_architecture.md`
- `configs/oc22/s2ef/e2former/e2former.yaml`
- `test_e2former.py`

## Citation

If you use E2Former in your work, please cite:

OpenReview page: <https://openreview.net/forum?id=ls5L4IMEwt>

```bibtex
@inproceedings{li2025eformer,
  title={E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products},
  author={Yunyang Li and Lin Huang and Zhihao Ding and Xinran Wei and Chu Wang and Han Yang and Zun Wang and Chang Liu and Yu Shi and Peiran Jin and Tao Qin and Mark Gerstein and Jia Zhang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=ls5L4IMEwt}
}
```

## License

This project is released under the [MIT License](LICENSE).
