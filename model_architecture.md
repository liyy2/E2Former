# E2Former Architecture

E2Former is an E(3)-equivariant transformer for molecular energy/force modeling with an efficiency-first design centered on Wigner-6j-based tensor product factorization.

![E2Former architecture](assets/fig2.png)

## Design goals

| Goal | E2Former design choice |
|---|---|
| Preserve geometric equivariance | Irreps-based features + spherical harmonics + equivariant tensor products |
| Improve scalability | Wigner-6j recoupling to reduce expensive edge-indexed contractions |
| Keep training practical | Config-driven control of radius graph density, attention order, precision, and kernels |

## End-to-end pipeline

1. Build a radius graph from atom positions (`max_radius`, `max_neighbors`, optional PBC).
2. Encode atoms and geometry into equivariant feature tensors.
3. Apply stacked equivariant attention/interaction blocks.
4. Aggregate/read out node-level representations.
5. Predict system-level energy and atom-level forces through model heads.

## Core mechanism: Wigner-6j factorization

In conventional equivariant message passing, dominant tensor-product work is often edge-indexed:

\[
m_i = \sum_{j \in \mathcal{N}(i)} \mathrm{TP}(h_j, Y(\mathbf{r}_{ij}))
\]

E2Former uses Wigner-6j recoupling to reorganize contraction order and reuse intermediate structure, shifting dominant work toward node-centric operations where possible. Practically, this is the key reason E2Former can scale better while remaining equivariant.

## Code map

| Component | File | Primary class / symbol |
|---|---|---|
| FairChem-facing model wrapper | `src/models/E2Former_wrapper.py` | `E2FormerBackbone` |
| Core E2Former stack | `src/models/e2former_main.py` | `E2former` |
| Attention order variants | `src/layers/attention/orders.py` | `ZeroOrderAttention`, `FirstOrderAttention`, `SecondOrderAttention`, `AllOrderAttention` |
| Wigner-6j tensor-product primitives | `src/wigner6j/tensor_product.py` | `FullyConnectedTensorProductWigner6j` |
| Arbitrary-order equivariant TP | `src/wigner6j/tensor_product.py` | `E2TensorProductArbitraryOrder` |

## Inputs and outputs

| Interface | Expected fields / behavior |
|---|---|
| Input structure | Atomic positions (`pos`), atomic numbers (`atomic_numbers`), optional periodic cell (`cell`) and flags (`pbc`) |
| Batching | `process_batch_data` in `src/models/E2Former_wrapper.py` builds padded batch tensors/masks |
| Primary outputs | Energy + forces via configured heads (`model.heads`) |
| Optional behavior | Decoupled energy/force path (`decouple_EF`) and attention/kernel variants via config |

## Key configuration knobs

| Config key(s) | Impact |
|---|---|
| `model.backbone.max_radius`, `model.backbone.max_neighbors` | Graph sparsity, memory, and speed |
| `model.backbone.irreps_node_embedding` | Representation order/width and equivariant capacity |
| `model.backbone.num_layers`, `model.backbone.num_attn_heads` | Model depth/capacity vs. throughput |
| `model.backbone.attn_type`, `model.backbone.tp_type` | Attention/tensor-product behavior |
| `model.backbone.use_fp16_backbone`, `model.backbone.use_compile` | Performance and memory optimization |
| `optim.batch_size`, `optim.lr_initial` | Training stability and hardware utilization |

## Practical notes

- Keep `pbc_max_radius` consistent with `max_radius` for consistent PBC/non-PBC behavior.
- Start tuning with graph density knobs (`max_neighbors`, `max_radius`) and batch size first.
- If memory is tight, reduce neighbor density before reducing representation order.

## Related files

- Usage and commands: [`README.md`](README.md)
- Main OC22 config template: [`configs/oc22/s2ef/e2former/e2former.yaml`](configs/oc22/s2ef/e2former/e2former.yaml)
- Smoke/equivariance test: [`test_e2former.py`](test_e2former.py)
