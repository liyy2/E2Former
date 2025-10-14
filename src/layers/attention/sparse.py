# -*- coding: utf-8 -*-
"""
E2Former Sparse Attention Module

Refactored implementation using modular components.
"""

import torch
from torch import nn

# Import base class and utilities
from .base import BaseE2Attention
from .utils import DEFAULT_ATOM_TYPE_COUNT
from .alpha import create_alpha_module
from .orders import create_attention_order

# Import required modules
from ...core.module_utils import get_normalization_layer


class E2AttentionArbOrder_sparse(BaseE2Attention):
    """
    E2Former sparse attention mechanism with arbitrary order support.
    
    This class inherits from BaseE2Attention and uses modular components
    for alpha computation and attention order implementation.
    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        alpha_drop=0.1,
        rescale_degree=False,
        nonlinear_message=False,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="first-order",
        add_rope=True,
        layer_id=0,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=DEFAULT_ATOM_TYPE_COUNT,
        norm_layer="identity",
        **kwargs,
    ):
        # Initialize base class
        super().__init__(
            irreps_node_input=irreps_node_input,
            attn_weight_input_dim=attn_weight_input_dim,
            num_attn_heads=num_attn_heads,
            attn_scalar_head=attn_scalar_head,
            irreps_head=irreps_head,
            alpha_drop=alpha_drop,
            tp_type=tp_type,
            attn_type=attn_type,
            norm_layer=norm_layer,
            atom_type_cnt=atom_type_cnt,
            **kwargs,
        )
        
        # Store additional parameters
        self.neighbor_weight = neighbor_weight
        self.layer_id = layer_id
        self.add_rope = add_rope
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.proj_drop = proj_drop
        self.irreps_origin = irreps_origin
        
        # Create alpha computation module using factory
        self.alpha_module = create_alpha_module(
            self.tp_type,
            self.irreps_node_input,
            self.num_attn_heads,
            self.attn_scalar_head,
            self.attn_weight_input_dim,
            self.edge_channel_list,
            self.lmax,
        )
        
        # Create attention order module using factory
        self.attention_order_module = create_attention_order(
            self.attn_type,
            self.irreps_node_input,
            self.irreps_head,
            self.num_attn_heads,
            self.edge_channel_list,
            self.lmax,
            self.attn_weight_input_dim,
        )
        
        # Initialize normalization layers
        self.norm_1 = get_normalization_layer(
            norm_layer,
            lmax=self.lmax,
            num_channels=self.irreps_node_output[0][0],
        )
        self.norm_2 = get_normalization_layer(
            norm_layer,
            lmax=self.lmax,
            num_channels=self.irreps_node_output[0][0],
        )

    def forward(
        self,
        node_pos: torch.Tensor,
        node_irreps_input: torch.Tensor,
        edge_dis: torch.Tensor,
        edge_vec: torch.Tensor,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        poly_dist=None,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        Forward pass for sparse attention.
        
        Args:
            node_pos: Node positions (f_N1, 3)
            node_irreps_input: Node features (f_N1, irreps_dim, hidden)
            edge_dis: Edge distances (f_N1, topK)
            edge_vec: Edge vectors (f_N1, topK, 3)
            attn_weight: Attention weights (f_N1, topK, attn_weight_input_dim)
            atomic_numbers: Atomic numbers (f_N1,)
            poly_dist: Polynomial distance weights
            attn_mask: Attention mask (f_N1, topK, 1)
            batch: Batch indices
            batched_data: Additional batched data
        
        Returns:
            node_output: Updated node features
            attn_weight: Unchanged attention weights
        """
        
        f_N1 = node_irreps_input.shape[0]
        topK = attn_weight.shape[1]
        f_sparse_idx_node = batched_data["f_sparse_idx_node"]

        print("f_N1", f_N1)
        print("topK",topK)
        print("attn_weight",attn_weight)
        print("f_sparse_idx_node",f_sparse_idx_node)
        
        # Mask attention weights
        attn_weight = attn_weight.masked_fill(attn_mask, 0)
        edge_feature = attn_weight.sum(dim=1)  # (f_N1, attn_weight_input_dim)
        
        # Compute edge features
        
        x_edge, src_node, tgt_node = self.compute_edge_features(
            attn_weight, atomic_numbers, f_N1, topK, f_sparse_idx_node
        )
        print("x_edge", x_edge)

        # Compute alpha weights using alpha module
        alpha = self.alpha_module(
            x_edge=x_edge,
            node_irreps_input=node_irreps_input,
            edge_vec=edge_vec,
            f_sparse_idx_node=f_sparse_idx_node,
        )
        
        # Apply softmax normalization
        alpha = self.apply_softmax(alpha, poly_dist, attn_mask)
        
        # Store original alpha for all-order attention
        alpha_org = alpha
        
        # Apply attention with specific order
        if self.attn_type == "all-order":
            # All-order requires edge_feature and node_irreps_input for gating
            node_output = self.attention_order_module(
                alpha=alpha_org,
                value=node_irreps_input,
                x_edge=x_edge,
                node_pos=node_pos,
                edge_dis=edge_dis,
                batched_data=batched_data,
                edge_feature=edge_feature,
                node_irreps_input=node_irreps_input,
            )
        else:
            # For other attention types, use the standard forward
            node_output = self.attention_order_module(
                alpha=alpha,
                value=node_irreps_input,
                x_edge=x_edge,
                node_pos=node_pos,
                edge_dis=edge_dis,
                batched_data=batched_data,
            )
        
        return node_output, attn_weight