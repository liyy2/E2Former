# E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products

This repository contains the official implementation of E2Former, an equivariant neural network interatomic potential based on efficient attention mechanisms and E(3)-equivariant operations.


<img width="769" height="541" alt="fig2-Apr14 (4)" src="https://github.com/user-attachments/assets/4b3e89c8-f9f5-4848-afe9-76f133d7ce62" />




At its core, E2Former utilizes **Wigner 6j convolution** for efficient and accurate tensor product operations, enabling the model to capture complex geometric interactions while preserving physical symmetries.

E2Former achieves state-of-the-art performance on molecular property prediction tasks by efficiently scaling attention mechanisms while preserving important physical symmetries. The architecture incorporates both invariant and equivariant features through a carefully designed transformer-based architecture that operates on atomic graphs. The model demonstrates superior performance on challenging benchmarks including MD17, MD22, OC20, and SPICE datasets, achieving chemical accuracy for energy and force predictions.

## Key Features

- **Wigner 6j Convolution Core**: Leverages Wigner 6j symbols for efficient E(3)-equivariant tensor products ([arXiv:2501.19216](https://arxiv.org/pdf/2501.19216))
- **E(3)-Equivariant Architecture**: Maintains rotational and translational equivariance through spherical harmonics and tensor products
- **Modular Design**: Separated components for easy customization and extension


## Installation

### Step 1: Install mamba solver for conda (optional but recommended)

```bash
conda install mamba -n base -c conda-forge
```


### Step 2: Create and activate the environment

```bash
mamba env create -f env.yml
conda activate gotennet
```

Or if you prefer using conda directly:

```bash
conda env create -f env.yml
conda activate gotennet
```

### Step 3: Install FairChem core package

```bash
git submodule update --init --recursive
pip install -e fairchem/packages/fairchem-core
```

### Step 4: Install pre-commit hooks (for contributors)

```bash
pre-commit install
```

## Model Architecture


## Training

### Single GPU Training

```bash
python main.py --mode train --config-yml {CONFIG} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

### Background Training

Use `start_exp.py` to start a training run in the background:

```bash
python start_exp.py --config-yml {CONFIG} --cvd {GPU_NUM} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

### Multi-GPU Training (same node)

```bash
torchrun --standalone --nproc_per_node={N} main.py --distributed --num-gpus {N} {...}
```

## Testing

Run the E2Former test suite to verify the installation:

```bash
python test_e2former.py
```

This will test the model with different batch sizes and verify equivariance properties.

For quick performance sanity checks, compare wall-clock throughput across attention kernels and orders on the same system. You should observe the expected scaling improvements from the 6j-based implementation ([arXiv:2501.19216](https://arxiv.org/pdf/2501.19216)).

## Molecular Dynamics Simulation

### Setup

Install the MDSim package:

```bash
pip install -e MDsim
```

### Running Simulations

```bash
python simulate.py --simulation_config_yml {SIM_CONFIG} --model_dir {CHECKPOINT_DIR} --model_config_yml {MODEL_CONFIG} --identifier {IDENTIFIER}
```

### Example: MD22 Simulation

```bash
python simulate.py \
    --simulation_config_yml configs/s2ef/MD22/datasets/DHA/simulation.yml \
    --model_dir checkpoints/MD22_DHA/ \
    --model_config_yml configs/s2ef/MD22/E2Former/DHA.yml \
    --identifier test_simulation
```

### Analyzing Results

```bash
PYTHONPATH=./ python scripts/analyze_rollouts_md17_22.py \
    --md_dir checkpoints/MD22_DHA/md_sim_test_simulation \
    --gt_traj /data/md22/md22_AT-AT.npz \
    --xlim 25
```

## Configuration

### Key Configuration Options

- **Attention Configuration**:
  - **Attention Type** (`attn_type`): Choose attention order complexity
    - `zero-order`: Simplest, scalar attention only
    - `first-order`: Includes vector features  
    - `second-order`: Includes tensor features
    - `all-order`: Combines all orders with gating
  - **Alpha Computation** (`tp_type`):
    - `QK_alpha`: Query-Key attention (standard transformer-style)
    - `dot_alpha`: Equiformer-style attention with spherical harmonics
    - `dot_alpha_small`: Memory-efficient variant of dot_alpha
  - **Kernel Implementation**: 
    - `math`: PyTorch default, supports all datatypes and gradient forces
    - `memory_efficient`: Memory-optimized kernel, supports fp32/fp16
    - `flash`: Flash attention kernel, fp16 only, best performance

- **Model Variants**:
  - Set `with_cluster: true` for E2formerCluster variant
  - Configure `encoder: dit` for DIT encoder, or `encoder: transformer` for standard transformer encoder

- **Equivariant Settings**:
  - `irreps_node_embedding`: Irreducible representations for node features (e.g., "128x0e+128x1e+128x2e")
  - `irreps_head`: Irreps for attention heads (e.g., "32x0e+32x1e+32x2e")
  - `lmax`: Maximum angular momentum for spherical harmonics
  - `num_layers`: Number of transformer blocks

### Example Configuration

```yaml
model:
  backbone:
    irreps_node_embedding: "128x0e+128x1e+128x2e"
    num_layers: 8
    encoder: dit
    with_cluster: false
    attn_type: "first-order"
    max_neighbors: 20
    max_radius: 6.0
```

See [`configs/example_config_E2Former.yml`](configs/example_config_E2Former.yml) for a detailed configuration example.

## Project Structure

```
src/
├── models/                      # Main model implementations
│   ├── E2Former_wrapper.py     # Model wrapper and data preprocessing
│   ├── e2former.py             # Original E2Former implementation
│   └── e2former_modular.py     # Refactored modular version
├── layers/                      # Neural network layers
│   ├── attention/              # Modular attention system (NEW)
│   │   ├── base.py            # Base attention class
│   │   ├── sparse.py          # Sparse attention implementation
│   │   ├── cluster.py         # Cluster-aware attention
│   │   ├── orders.py          # Attention order implementations
│   │   ├── alpha.py           # Alpha computation modules
│   │   ├── utils.py           # Shared utilities
│   │   └── compat.py          # Backward compatibility
│   ├── blocks.py               # Transformer blocks
│   ├── embeddings.py           # Embedding networks
│   ├── interaction_blocks.py  # Molecular interactions
│   ├── dit.py                  # DIT encoder blocks
│   └── moe.py                  # Mixture of experts
├── core/                        # Base classes and utilities
│   ├── module_utils.py        # Core utility functions
│   └── e2former_utils.py      # E2Former specific utilities
├── configs/                     # Configuration management
│   └── E2Former_configs.py    # Configuration dataclasses
└── wigner6j/                   # Wigner 6j symbols
    └── tensor_product.py      # E(3)-equivariant operations
```


## Important Notes

- **Gradient Forces**: When using gradient-based force calculations, disable `torch.compile` as it doesn't support second-order gradients
- **Memory Management**: Adjust `max_num_nodes_per_batch` for optimal GPU memory usage
- **FP16 Training**: Use `use_fp16_backbone` or AutoMixedPrecision for improved performance
- **Attention Types**: The same channels must be used across all irreps orders (e.g., "128x0e+128x1e+128x2e")
- **Complexity expectations**: Training/inference time should scale primarily with the number of nodes rather than edges due to node-local 6j recoupling.

## Citation

If you find E2Former useful in your research, please consider citing:

```bibtex
@article{li2025e2former,
  title={E2Former: A Linear-time Efficient and Equivariant Transformer for Scalable Molecular Modeling},
  author={Li, Yunyang and Huang, Lin and Ding, Zhihao and Wang, Chu and Wei, Xinran and Yang, Han and Wang, Zun and Liu, Chang and Shi, Yu and Jin, Peiran and others},
  journal={arXiv preprint arXiv:2501.19216},
  year={2025}
}
```




## Acknowledgments

E2Former builds upon several excellent works in the field of neural network interatomic potentials and equivariant neural networks. We particularly acknowledge the FairChem framework for providing the foundation for this implementation.
