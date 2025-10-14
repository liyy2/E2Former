# -*- coding: utf-8 -*-
"""
E2Former main model implementation.

This module defines the `E2former` class, an E(3)-equivariant transformer for
atomistic graph modeling. It combines attention mechanisms with spherical
harmonics-based tensor products to predict molecular properties while
maintaining rotational and translational equivariance. The model supports
periodic boundary conditions (PBC) and non-PBC systems.

Highlights:
- Equivariant message passing using spherical harmonics up to configurable order
  `lmax` inferred from `irreps_node_embedding`.
- Flexible radial basis options: Gaussian smearing, Gaussian RBF, or Bessel.
- Edge-degree embeddings that incorporate distance, direction, and local degree.
- Optional decoupling of energy and force heads.

Usage sketch:
    - Prepare `batched_data` with keys described in `E2former.forward`.
    - Optionally provide `token_embedding` per-atom; otherwise a learnable atom
      type embedding is used.
    - Call the model to obtain per-atom scalar features and equivariant vectors.
"""

import math
import warnings
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from torch import logical_not, nn

# Import from modular files
from ..core.base_modules import (
    create_trans_block,
    init_embeddings,
    no_weight_decay,
)
from ..layers.attention import E2TensorProductArbitraryOrder
from ..layers.embeddings import (
    EdgeDegreeEmbeddingNetwork_higherorder,
    EdgeDegreeEmbeddingNetwork_eqv2,
)
from ..layers.blocks import construct_radius_neighbor
from ..core.module_utils import (
    GaussianRadialBasisLayer,
    GaussianSmearing,
    get_normalization_layer,
)

# FairChem imports
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis

# Constants
DEFAULT_ATOM_TYPE_COUNT = 256


def get_powers(vec, coeffs, lmax):
    """Compute spherical harmonics powers for an input vector field.

    Args:
        vec: Tensor of shape [..., 3] representing Cartesian vectors.
        coeffs: Sequence of scale coefficients as returned by
            `E2TensorProductArbitraryOrder.get_coeffs()`; length must be
            `lmax + 1`.
        lmax: Maximum spherical harmonics order to compute (non-negative int).

    Returns:
        List[Tensor]: A list of tensors of length `lmax + 1` where the element
        at index `l` has shape [..., (2l+1), 1] and contains the spherical
        harmonics Y_lm (scaled by the corresponding `coeffs[l]`). The first
        element (l=0) is a constant term with shape [..., 1, 1].
    """
    out_powers = [
        coeffs[0] * torch.ones_like(vec.narrow(-1, 0, 1).unsqueeze(dim=-1))
    ]
    # Y is pos. Precompute spherical harmonics for all orders
    for i in range(1, lmax + 1):
        out_powers.append(
            coeffs[i]
            * e3nn.o3.spherical_harmonics(
                i, vec, normalize=False, normalization="integral"
            ).unsqueeze(-1)
        )

    return out_powers


class E2former(torch.nn.Module):
    """E2Former: E(3)-equivariant transformer for atomistic graphs.

    This model operates on batched atomic structures with positions and atom
    types, constructing radius-based neighborhoods and applying a stack of
    equivariant transformer blocks. It produces per-atom scalar features (order
    0) and, when present in the representation, equivariant vector/tensor
    features (orders > 0). Optionally, a decoupled energy/force head can be
    used.

    Key configuration:
    - `irreps_node_embedding`: Spherical tensor representation string (e.g.,
      "128x0e+128x1e+128x2e"). Must contain a `0e` part for scalar channels.
    - `basis_type`: Radial basis type: "gaussiansmear", "gaussian", or "bessel".
    - `edge_embedtype`: Controls edge-degree embedding variant: includes
      "default", "highorder", "elec", or "eqv2".
    - `attn_type`/`tp_type`/`ffn_type`: Control attention and tensor-product
      variants in the transformer blocks.

    Args:
        irreps_node_embedding: Irreps string or `o3.Irreps` for node features.
        num_layers: Number of transformer blocks.
        pbc_max_radius: Maximum cutoff for PBC neighbor search; must equal
            `max_radius`.
        max_neighbors: Max neighbors per node used in attention/message passing.
        max_radius: Spatial cutoff radius for neighborhood construction.
        basis_type: Radial basis type ("gaussiansmear", "gaussian", or
            "bessel").
        number_of_basis: Number of radial basis functions.
        num_attn_heads: Number of attention heads per block.
        attn_scalar_head: Scalar head width per attention head.
        irreps_head: Irreps string for attention/tensor product heads.
        rescale_degree: Whether to scale features by degree.
        nonlinear_message: Enable nonlinear message projection.
        norm_layer: Normalization layer identifier (e.g., "rms_norm_sh").
        alpha_drop: Dropout rate used in attention weights.
        proj_drop: Dropout rate on projections.
        out_drop: Dropout rate on block outputs.
        drop_path_rate: Stochastic depth rate.
        atom_type_cnt: Size of the learnable atom-type embedding table used when
            `token_embedding` is not provided at call time.
        tp_type: Tensor product variant (e.g., "QK_alpha").
        attn_type: Attention variant (e.g., "first-order").
        edge_embedtype: Edge-degree embedding variant.
        attn_biastype: Strategy for attention bias parameters.
        ffn_type: Feed-forward network variant inside each block.
        add_rope: Whether to add rotary positional encodings.
        time_embed: Expect and use time embeddings in forward pass.
        sparse_attn: Enable sparse attention within blocks.
        dynamic_sparse_attn_threthod: Threshold to enable dynamic sparse
            attention.
        avg_degree: Estimated average graph degree; used by embeddings.
        force_head: Reserved argument for external force head (currently unused
            here; forces produced via blocks or decoupled head).
        decouple_EF: If True, use a separate block for energy/force outputs.
        **kwargs: Accepted for forward-compatibility; unused here.
    """
    
    def __init__(
        self,
        irreps_node_embedding="128x0e+128x1e+128x2e",
        num_layers=6,
        pbc_max_radius=15,
        max_neighbors=20,
        max_radius=15.0,
        basis_type="gaussiansmear",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+32x1e+32x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="rms_norm_sh",  # the default is deprecated
        alpha_drop=0.1,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.1,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        tp_type="QK_alpha",
        attn_type="first-order",
        edge_embedtype="default",
        attn_biastype="share",  # add
        ffn_type="default",
        add_rope=True,
        time_embed=False,
        sparse_attn=False,
        dynamic_sparse_attn_threthod=1000,
        avg_degree=23.01,
        force_head=None,
        decouple_EF=False,
        **kwargs,
    ):
        """Initialize the E2former module with the provided configuration.

        See the class docstring for a detailed description of all arguments.
        """
        super().__init__()
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.pbc_max_radius = pbc_max_radius  #
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.add_rope = add_rope
        self.time_embed = time_embed
        self.sparse_attn = sparse_attn
        self.dynamic_sparse_attn_threthod = dynamic_sparse_attn_threthod
        
        if pbc_max_radius != max_radius:
            raise ValueError("Please ensure these two radius equal for pbc and non-pbc generalize")
            
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.decouple_EF = decouple_EF
        
        if "0e" not in self.irreps_node_embedding:
            raise ValueError("Sorry, the irreps node embedding must have 0e embedding")

        self.unifiedtokentoembedding = nn.Linear(
            self.irreps_node_embedding[0][0], self.irreps_node_embedding[0][0]
        )

        self.default_node_embedding = torch.nn.Embedding(
            atom_type_cnt, self.irreps_node_embedding[0][0]
        )

        self._node_scalar_dim = self.irreps_node_embedding[0][0]
        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )

        ## this is for f( r_ij )
        self.basis_type = basis_type
        self.attn_biastype = attn_biastype
        self.heads2basis = nn.Linear(
            self.num_attn_heads, self.number_of_basis, bias=True
        )
        
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(
                self.number_of_basis, cutoff=self.max_radius
            )
        elif self.basis_type == "gaussiansmear":
            self.rbf = GaussianSmearing(
                self.number_of_basis, cutoff=self.max_radius, basis_width_scalar=2
            )
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis,
                cutoff=self.max_radius,
                rbf={"name": "spherical_bessel"},
            )
        else:
            raise ValueError(f"Invalid basis_type: '{self.basis_type}'. Expected 'gaussiansmear' or 'bessel'")

        # edge embedding network
        if (
            "default" in edge_embedtype
            or "highorder" in edge_embedtype
            or "elec" in edge_embedtype
        ):
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_higherorder(
                self.irreps_node_embedding,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                time_embed=self.time_embed,
                use_atom_edge=True,
                use_layer_norm="wolayernorm" not in edge_embedtype,
            )
        elif "eqv2" in edge_embedtype:
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_eqv2(
                self.irreps_node_embedding,
                avg_degree,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                lmax=len(self.irreps_node_embedding) - 1,
                time_embed=self.time_embed,
            )
        else:
            raise ValueError(f"Invalid edge_embedtype: '{edge_embedtype}'. Please check edge embedtype")

        # Create transformer blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            blk = create_trans_block(
                irreps_node_embedding=self.irreps_node_embedding,
                number_of_basis=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type=self.attn_type,
                ffn_type=ffn_type,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                max_radius=max_radius,
                layer_id=i,
                is_last_layer=(i == self.num_layers - 1),
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
            )
            self.blocks.append(blk)

        self.energy_force_block = None
        if self.decouple_EF:
            self.energy_force_block = create_trans_block(
                irreps_node_embedding=self.irreps_node_embedding,
                number_of_basis=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type=self.attn_type,
                ffn_type=ffn_type,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
                max_radius=max_radius,
                layer_id=0,
                is_last_layer=True,
                force_attn_type="first-order",
            )

        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.lmax = len(self.irreps_node_embedding) - 1
        
        self.norm_tmp = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.norm_final = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        
        if len(self.irreps_node_embedding) == 1:
            self.f_linear = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim),
                nn.LayerNorm(self.scalar_dim),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, 3 * self.scalar_dim),
            )

        self.apply(self._init_weights)

    def reset_parameters(self):
        """Reset learnable parameters, if implemented by submodules.

        Currently emits a warning because this composite model relies on
        submodules' own initialization and does not implement a global reset.
        """
        warnings.warn("Sorry, output model not implement reset parameters")

    def _init_weights(self, m):
        """Initialize module weights for linear and layer norm layers.

        Args:
            m: A submodule possibly of type `torch.nn.Linear` or
                `torch.nn.LayerNorm`.
        """
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameter names exempt from weight decay for optimizers.

        Returns:
            Set[str]: Names of parameters that should not use weight decay.
        """
        return no_weight_decay(self)

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        mixed_attn_bias=None,
        padding_mask: torch.Tensor = None,
        pbc_expand_batched: Optional[Dict] = None,
        time_embed: Optional[torch.Tensor] = None,
        return_node_irreps=False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batched_data: A dictionary with at least the following keys:
                - `pos` (torch.Tensor): [B, L, 3] atomic coordinates.
                - `atom_masks` (torch.BoolTensor): [B, L] True for valid atoms.
                - `atomic_numbers` (torch.LongTensor): [B, L] atomic numbers.
                - `pbc` (torch.BoolTensor): [B] indicates PBC per system.
              Additional keys may be added and consumed by downstream blocks.
            token_embedding: Optional per-atom embedding of shape [B, L, D]. If
                provided, it is projected by `unifiedtokentoembedding`. If None,
                an internal learnable embedding indexed by `atomic_numbers` is
                used instead.
            mixed_attn_bias: Optional external attention bias (currently unused
                in this implementation path, accepted for API compatibility).
            padding_mask: Optional [B, L] mask; if provided it is ignored here
                in favor of `atom_masks` from `batched_data`.
            pbc_expand_batched: Optional dictionary required when `pbc` is True
                containing:
                - `outcell_index` (torch.LongTensor): [B, L2] mapping to base
                  cell indices for expanded atoms.
                - `expand_pos` (torch.Tensor): [B, L2, 3] expanded positions.
                - `expand_mask` (torch.BoolTensor): [B, L2] False for valid
                  expanded atoms, True for padding.
            time_embed: Optional time embedding tensor broadcastable to nodes;
                only used if `self.time_embed` is True.
            return_node_irreps: If True, also returns internal irreps tensors
                for analysis/diagnostics.

        Returns:
            - node_attr (torch.Tensor): [B, L, C0] scalar (0e) per-atom features.
            - node_vec (torch.Tensor): [B, L, 3, C0] equivariant vectors derived
              from higher-order irreps if present; zeros if not applicable.

        When `return_node_irreps` is True, additionally returns:
            - node_irreps (torch.Tensor): [B, L, (lmax+1)^2, C0] stacked order
              components before the final normalization.
            - node_irreps_his (torch.Tensor): Same shape as `node_irreps`,
              snapshot from the penultimate block after temporary normalization.
        """

        # =====================================================================
        # Step 1: Data Preparation and Tensor Setup
        # =====================================================================
        print("Using e2former backbone to train the model")
        # Extract data type and device for consistent tensor operations
        tensortype = self.default_node_embedding.weight.dtype
        device = batched_data["pos"].device
        
        # Get batch dimensions: B = batch size, L = max atoms per molecule
        B, L = batched_data["pos"].shape[:2]

        # Extract atomic positions and create padding mask
        # padding_mask: True where atoms are padded (not real)
        node_pos = batched_data["pos"]
        # padding_mask = ~batched_data["atom_masks"]
        padding_mask = batched_data["padding_mask"]
        
        # Set padded positions to a large value (999.0) to avoid interference
        # This ensures padded atoms don't contribute to neighbor calculations
        node_pos = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3),  # Expand mask to 3D
            999.0,  # Sentinel value for padded positions
            node_pos  # Keep original positions for real atoms
        )

        # =====================================================================
        # Step 2: Optional Time Embedding Processing
        # =====================================================================
        
        # Handle optional time embedding 
        if (time_embed is not None) and self.time_embed:
            time_embed = time_embed.to(dtype=tensortype)
        else:
            time_embed = None

        # =====================================================================
        # Step 3: Flatten Batch Data for Efficient Processing
        # =====================================================================
        
        # Create mask for real atoms (inverse of padding mask)
        node_mask = logical_not(padding_mask)
        
        # Extract atomic numbers for real atoms only
        atomic_numbers = batched_data["atomic_numbers"].reshape(B, L)[node_mask]
        
        # Create pointer array for batch indexing
        # ptr[i] indicates the start index of batch i in the flattened arrays
        # ptr[i+1] - ptr[i] gives the number of real atoms in batch i
        ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(torch.sum(node_mask, dim=-1), dim=-1),
            ],
            dim=0,
        )
        
        # Flatten positions to include only real atoms
        # f_node_pos shape: [total_real_atoms, 3]
        f_node_pos = node_pos[node_mask]
        f_N1 = f_node_pos.shape[0]  # Total number of real atoms across all batches
        
        # Create batch indices for each real atom
        # f_batch[i] indicates which batch atom i belongs to
        f_batch = torch.arange(B).reshape(B, 1).repeat(1, L).to(device)[node_mask]

        # =====================================================================
        # Step 4: Initialize Variables for Periodic Boundary Conditions (PBC)
        # =====================================================================
        
        # Default values for non-PBC (molecular) systems
        expand_node_mask = node_mask  # Initially same as node_mask
        expand_node_pos = node_pos    # Initially same as node_pos
        
        # Create indices mapping each atom to its original cell
        # Shape: [B, L] - each element is the atom's index in the original unit cell
        outcell_index = torch.arange(L).unsqueeze(dim=0).repeat(B, 1).to(device)
        
        # Flattened versions for efficient processing
        f_exp_node_pos = f_node_pos  # Expanded positions (same as original for molecules)
        f_outcell_index = torch.arange(len(f_node_pos)).to(device)  # Simple indices
        
        # System type: 0 for molecules, 1 for periodic systems
        mol_type = 0  
        L2 = L  # L2 will be larger than L for PBC due to cell expansion
        
        # =====================================================================
        # Step 5: Handle Periodic Boundary Conditions (PBC)
        # =====================================================================
        
        if torch.any(batched_data["pbc"]):
            # Switch to periodic system mode
            mol_type = 1
            
            # Update dimensions to account for expanded cells
            # L2: total atoms including periodic images
            L2 = pbc_expand_batched["outcell_index"].shape[1]
            
            # Map each expanded atom back to its original cell atom
            outcell_index = pbc_expand_batched["outcell_index"]
            
            # Get expanded positions including periodic images
            expand_node_pos = pbc_expand_batched["expand_pos"].float()
            
            # Mark padded expanded positions with sentinel value
            expand_node_pos[
                pbc_expand_batched["expand_mask"]
            ] = 999  # Ensures padded atoms don't affect neighbor calculations
            
            # Create mask for real expanded atoms
            expand_node_mask = logical_not(pbc_expand_batched["expand_mask"])

            # Flatten expanded positions for efficient processing
            f_exp_node_pos = expand_node_pos[expand_node_mask]
            
            # Map flattened expanded atoms to their original cell atoms
            # Adding ptr offsets to correctly index into flattened arrays
            f_outcell_index = (outcell_index + ptr[:B, None])[
                expand_node_mask
            ]  # Maps expanded atoms -> original atoms in flattened representation

        # Store system type for later use
        batched_data["mol_type"] = mol_type

        # =====================================================================
        # Step 6: Construct Neighbor Graph
        # =====================================================================
        
        # Build neighbor graph based on distance cutoff
        # This finds all atom pairs within max_radius distance
        neighbor_info = construct_radius_neighbor(
            node_pos, node_mask,
            expand_node_pos, expand_node_mask,
            radius=self.max_radius,
            outcell_index=outcell_index,
            max_neighbors=self.max_neighbors
        )
        batched_data.update(neighbor_info)
        
        # Extract edge information from neighbor graph
        f_edge_vec = neighbor_info["f_edge_vec"]      # Edge vectors between atoms
        f_dist = neighbor_info["f_dist"]              # Edge distances
        f_poly_dist = neighbor_info["f_poly_dist"]    # Polynomial edge distances
        f_attn_mask = neighbor_info["f_attn_mask"]    # Attention mask for valid edges
        
        # Compute radial basis functions for distance encoding
        # Shape: [num_edges, num_neighbors, num_basis]
        f_dist_embedding = self.rbf(f_dist)

        # =====================================================================
        # Step 7: Atom Embedding
        # =====================================================================
        
        # Create initial atom embeddings
        # Two pathways: use provided token embeddings or default atomic embeddings
        if token_embedding is not None:
            # Use pre-computed token embeddings (e.g., from an encoder)
            f_atom_embedding = self.unifiedtokentoembedding(
                token_embedding[node_mask]
            )  # Transform and flatten: [B, L, D] => [total_atoms, D]
        else:
            # Use default learnable embeddings based on atomic numbers
            f_atom_embedding = self.default_node_embedding(atomic_numbers)

        # =====================================================================
        # Step 8: Compute Spherical Harmonics Powers
        # =====================================================================
        
        # Get coefficients for spherical harmonics computation
        coeffs = E2TensorProductArbitraryOrder.get_coeffs()
        
        # Pre-compute spherical harmonics powers for positions and edges
        # These are used for E(3)-equivariant operations throughout the network
        batched_data.update(
            {
                "f_exp_node_pos": f_exp_node_pos,
                "f_outcell_index": f_outcell_index,
                "Y_powers": get_powers(f_node_pos, coeffs, self.lmax),        # Node SH powers
                "exp_Y_powers": get_powers(f_exp_node_pos, coeffs, self.lmax), # Expanded node SH powers
                "edge_vec_powers": torch.cat(get_powers(f_edge_vec, coeffs, self.lmax), dim=-2),  # Edge SH powers
            }
        )
        
        # =====================================================================
        # Step 9: Edge Degree Embedding
        # =====================================================================
        
        # Compute edge degree embeddings that incorporate neighbor information
        # This creates equivariant features based on local atomic environment
        edge_degree_embedding_dense = self.edge_deg_embed_dense(
            f_atom_embedding,
            f_node_pos,
            f_dist,
            edge_scalars=f_dist_embedding,
            edge_vec=f_edge_vec,
            batch=None,
            attn_mask=f_attn_mask,
            atomic_numbers=atomic_numbers,
            batched_data=batched_data,
            time_embed=time_embed,
        )

        # Initialize node irreducible representations (irreps)
        # Shape: [num_atoms, (lmax+1)^2, hidden_dim]
        f_node_irreps = edge_degree_embedding_dense
        
        # Add skip connection for scalar (l=0) features
        f_node_irreps[:, 0, :] = f_node_irreps[:, 0, :] + f_atom_embedding
        
        # Initialize tensor for storing intermediate representations
        node_irreps_his = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )

        # =====================================================================
        # Step 10: Forward Through Transformer Blocks
        # =====================================================================
        
        # Process through each transformer block sequentially
        # Each block performs E(3)-equivariant attention and updates node features
        for i, blk in enumerate(self.blocks):
            f_node_irreps, f_dist_embedding = blk(
                node_pos=f_node_pos,
                node_irreps=f_node_irreps,
                edge_dis=f_dist,
                poly_dist=f_poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=f_attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,        # Rotary position embeddings
                sparse_attn=self.sparse_attn,  # Sparse attention optimization
                batch=f_batch,
            )
            
            # Save penultimate layer representations for potential auxiliary tasks
            if i == len(self.blocks) - 2:
                node_irreps_his[node_mask] = self.norm_tmp(
                    f_node_irreps
                )

        # =====================================================================
        # Step 11: Final Normalization and Feature Extraction
        # =====================================================================
        
        # Apply final normalization to node features
        f_node_irreps_final = self.norm_final(f_node_irreps)
        
        # Initialize output tensors with zeros (for padded positions)
        node_irreps = torch.zeros(
            (B, L, (self.lmax + 1) ** 2, self._node_scalar_dim), device=device
        )
        node_irreps[node_mask] = f_node_irreps  # Fill in real atom features

        # Initialize scalar (energy) and vector (force) output tensors
        node_attr = torch.zeros((B, L, self._node_scalar_dim), device=device)  # Energy features
        node_vec = torch.zeros((B, L, 3, self._node_scalar_dim), device=device)  # Force features
        
        # =====================================================================
        # Step 12: Extract Energy and Force Features
        # =====================================================================
        
        if not self.decouple_EF:
            # Coupled energy-force mode: directly extract from irreps
            # l=0 (scalar) features for energy
            node_attr[node_mask] = f_node_irreps_final[:, 0]
            
            # l=1 (vector) features for forces (if available)
            if f_node_irreps_final.shape[1] > 1:
                node_vec[node_mask] = f_node_irreps_final[:, 1:4]  # Extract x,y,z components
        else:
            # Decoupled energy-force mode: use separate processing blocks
            # This can provide better accuracy for force predictions
            
            # Process scalar features through dedicated energy block
            node_attr[node_mask] = self.energy_force_block.ffn_s2(f_node_irreps_final)[
                :, 0
            ]
            
            # Process vector features through dedicated force block with attention
            node_vec[node_mask] = self.energy_force_block.ga(
                node_pos=f_node_pos,
                node_irreps_input=f_node_irreps_final,
                edge_dis=f_dist,
                poly_dist=f_poly_dist,
                edge_vec=f_edge_vec,
                attn_weight=f_dist_embedding,
                atomic_numbers=atomic_numbers,
                attn_mask=f_attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
                sparse_attn=self.sparse_attn,
            )[0][:, 1:4]  # Extract vector components
            
        # =====================================================================
        # Step 13: Return Results
        # =====================================================================
        
        # Return all intermediate representations if requested (for analysis/debugging)
        if return_node_irreps:
            return node_attr, node_vec, node_irreps, node_irreps_his

        return node_attr, node_vec