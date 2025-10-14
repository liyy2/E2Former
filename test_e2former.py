#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Working test for E2Former - providing data in the exact format expected
"""

import torch
import numpy as np
from torch_geometric.data import Data
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the E2Former wrapper
from src.models.E2Former_wrapper import E2FormerBackbone

def create_proper_batch_data(batch_size=2, max_atoms_per_mol=10, device='cpu'):
    """
    Create data in the exact format the model expects after process_batch_data.
    This mimics what process_batch_data should output.
    """
    # Generate variable length molecules
    atom_counts = []
    for b in range(batch_size):
        atom_counts.append(np.random.randint(5, max_atoms_per_mol + 1))
    
    max_nodes = max(atom_counts)
    
    # Create padded batch tensors
    batched_pos = torch.zeros((batch_size, max_nodes, 3), device=device)
    batched_atomic_numbers = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    atom_masks = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    
    # Fill in actual data for each molecule
    for b in range(batch_size):
        n_atoms = atom_counts[b]
        
        # Random positions
        batched_pos[b, :n_atoms] = torch.rand(n_atoms, 3, device=device) * 10.0
        
        # Random atomic numbers (H to Ne)
        batched_atomic_numbers[b, :n_atoms] = torch.randint(1, 11, (n_atoms,), device=device)
        
        # Mark valid atoms
        atom_masks[b, :n_atoms] = True
    
    # Create cell and PBC info
    cell = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 15.0
    pbc = torch.ones(batch_size, 3, dtype=torch.bool, device=device)
    
    # Create the data object with ALL the keys the model expects
    data = Data()
    
    # Keys that the model expects after process_batch_data AND that are accessed directly
    data.pos = batched_pos  # [batch_size, max_nodes, 3]
    data.atomic_numbers = batched_atomic_numbers  # [batch_size, max_nodes]
    data.atom_masks = atom_masks  # [batch_size, max_nodes]
    data.cell = cell  # [batch_size, 3, 3]
    data.pbc = pbc  # [batch_size, 3]
    
    # Additional keys that might be needed
    data.charge = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=device)
    data.multiplicity = torch.ones(batch_size, max_nodes, dtype=torch.long, device=device)
    
    # Keys created by process_batch_data that might be needed
    data.token_id = batched_atomic_numbers  # Same as atomic_numbers
    data.masked_token_type = batched_atomic_numbers.clone()
    data.padding_mask = ~atom_masks  # Inverse of atom_masks
    data.graph_padding_mask = ~atom_masks
    data.num_atoms = torch.tensor(atom_counts, dtype=torch.long, device=device)
    data.is_protein = torch.zeros((batch_size, max_nodes, 1), dtype=torch.bool, device=device)
    data.position_ids = torch.arange(max_nodes, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    # Create batch indices for compatibility
    batch_indices = []
    for b in range(batch_size):
        batch_indices.extend([b] * atom_counts[b])
    data.node_batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
    
    return data, atom_counts

def test_e2former():
    """
    Test E2Former with properly formatted data.
    """
    print("=" * 60)
    print("E2Former Working Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create model configuration
    config = {
        # Global configuration
        'regress_forces': False,
        
        # E2Former configuration (no encoder)
        
        # PBC and radius configuration
        'pbc_max_radius': 6.0,
        'pbc_expanded_num_cell_per_direction': 1,
        'expanded_token_cutoff': 512,
        'max_neighbors': 32,
        
        # E2Former backbone configuration
        'irreps_node_embedding': '128x0e+128x1e+128x2e',
        'num_layers': 2,
        'max_radius': 6.0,
        'basis_type': 'gaussian',
        'number_of_basis': 32,
        'num_attn_heads': 4,
        'attn_scalar_head': 16,
        'irreps_head': '32x0e+32x1e+32x2e',
        'rescale_degree': False,
        'nonlinear_message': False,
        'norm_layer': 'rms_norm_sh',
        'alpha_drop': 0.0,
        'proj_drop': 0.0,
        'out_drop': 0.0,
        'drop_path_rate': 0.0,
        'tp_type': 'dot_alpha_small',
        'attn_type': 'all-order',
        'edge_embedtype': 'default',
        'attn_biastype': 'share',
        'ffn_type': 'default',
        'add_rope': False,
        'time_embed': False,
        'sparse_attn': False,
        'dynamic_sparse_attn_threthod': 1000,
        'force_head': None,
        
        # Important: disable torch.compile to avoid the compilation issues
        'use_compile': False,
    }
    
    print("\nInitializing E2Former model...")
    model = E2FormerBackbone(**config).to(device)
    print("✓ Model initialized successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Test with different configurations
    test_configs = [
        (1, 5),   # Single molecule
        (2, 8),   # Batch of 2
        (3, 6),   # Batch of 3
    ]
    
    for batch_size, max_atoms in test_configs:
        print(f"\n" + "=" * 60)
        print(f"Testing with batch_size={batch_size}, max_atoms={max_atoms}")
        
        # Create properly formatted data
        data, atom_counts = create_proper_batch_data(
            batch_size=batch_size, 
            max_atoms_per_mol=max_atoms, 
            device=device
        )
        
        print(f"\nData Info:")
        print(f"  Batch size: {batch_size}")
        print(f"  Atoms per molecule: {atom_counts}")
        print(f"  pos shape: {data.pos.shape}")
        print(f"  atomic_numbers shape: {data.atomic_numbers.shape}")
        print(f"  atom_masks shape: {data.atom_masks.shape}")
        
        # Forward pass
        print("\nRunning forward pass...")
        
        model.eval()
        with torch.no_grad():
            try:
                # Since we're providing data in the exact format expected after process_batch_data,
                # and it has 3D pos, process_batch_data should just return it as-is
                output = model(data)
                
                print("✓ Forward pass successful!")
                
                # Check output
                if isinstance(output, dict):
                    print("\nOutput keys:", list(output.keys())[:5])
                    
                    for key in ['node_features', 'node_irreps', 'node_vec_features']:
                        if key in output and isinstance(output[key], torch.Tensor):
                            print(f"  {key}: shape {output[key].shape}")
                            
            except Exception as e:
                print(f"✗ Error: {e}")
                # Print only relevant part of traceback
                import traceback
                tb_lines = traceback.format_exc().split('\n')
                # Find the most relevant error line
                for line in tb_lines:
                    if 'Error' in line or 'line' in line:
                        print(line)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_e2former()