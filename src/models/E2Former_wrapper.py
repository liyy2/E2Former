# -*- coding: utf-8 -*-
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3

from torch_geometric.data import Data


from .e2former_main import E2former
from ..core.base_modules import no_weight_decay
from ..layers.blocks import construct_radius_neighbor
from ..configs.E2Former_configs import E2FormerConfigs
from ..core.module_utils import CellExpander,GaussianLayer_Edgetype,polynomial


from ..utils.graph_utils import compilable_scatter, unpad_results,RandomRotate
from ..utils.nn_utils import init_linear_weights
from ..utils.base_utils import registry,init_configs


_AVG_NUM_NODES = 77.81317


def process_batch_data(data, max_nodes=None):
    """
    Process raw batch data into padded batched format with masks.
    
    Converts PyTorch Geometric style flattened data into batched tensors
    with padding for efficient batch processing.

    Args:
        data: Input data containing pos, cell, atomic_numbers, etc.
            Expected format: PyG Data object with flattened tensors
        max_nodes: Maximum number of nodes for padding. If None, uses maximum in batch.

    Returns:
        Data: Batched data object with padded tensors and masks
            - pos: [num_graphs, max_nodes, 3] - Atomic positions
            - cell: [num_graphs, 3, 3] - Unit cell vectors
            - token_id: [num_graphs, max_nodes] - Atomic numbers
            - padding_mask: [num_graphs, max_nodes] - True for padded atoms
    """
    # Early return if already in batched format
    if len(data.pos.shape) == 3:
        return data

    # =========================================================================
    # Extract Batch Information
    # =========================================================================
    
    batch_idx = data.batch  # Maps each atom to its graph
    num_graphs = data.ptr.size(0) - 1  # Number of graphs in batch

    # Determine maximum nodes for padding
    if max_nodes is None:
        max_nodes = max([data.ptr[i + 1] - data.ptr[i] for i in range(num_graphs)])

    # =========================================================================
    # Initialize Output Tensors with Padding
    # =========================================================================
    
    # Position tensor: padded with zeros
    batched_pos = torch.zeros((num_graphs, max_nodes, 3), device=data.pos.device)
    
    # Atomic numbers: padded with zeros (invalid atomic number)
    batched_token_id = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.long, device=data.atomic_numbers.device
    )
    
    # Masked token types for potential masked language modeling
    masked_token_type = torch.zeros(
        (num_graphs, max_nodes), dtype=torch.long, device=data.atomic_numbers.device
    )
    
    # Padding mask: True indicates padded (invalid) atoms
    padding_mask = torch.ones(
        (num_graphs, max_nodes), dtype=torch.bool, device=data.pos.device
    )
    # =========================================================================
    # Handle Periodic Boundary Conditions
    # =========================================================================
    
    if "pbc" not in data:
        # Default PBC settings for surface catalysis (2D periodic)
        # [1, 1, 0] means periodic in x and y, non-periodic in z
        pbc = (
            torch.tensor([[1, 1, 0]]).repeat(num_graphs, 1).to(data.pos.device)
        )  # Default for Open Catalyst datasets
    else:
        pbc = data.pbc

    # =========================================================================
    # Additional Metadata
    # =========================================================================
    
    # Protein flag (for potential biomolecular applications)
    is_protein = torch.zeros(
        (num_graphs, max_nodes, 1), dtype=torch.bool, device=data.pos.device
    )
    
    # Count actual atoms per graph
    num_atoms = torch.tensor(
        [data.ptr[i + 1] - data.ptr[i] for i in range(num_graphs)],
        dtype=torch.long,
        device=data.pos.device,
    )

    # =========================================================================
    # Process Each Graph in the Batch
    # =========================================================================
    
    for i in range(num_graphs):
        # Get slice indices for current graph
        start_idx = data.ptr[i]
        end_idx = data.ptr[i + 1]
        num_nodes = end_idx - start_idx

        # Copy atomic positions
        batched_pos[i, :num_nodes] = data.pos[start_idx:end_idx]

        # Copy atomic numbers (used as token IDs)
        batched_token_id[i, :num_nodes] = data.atomic_numbers[start_idx:end_idx]
        
        # Handle masked token types (for masked language modeling tasks)
        if "masked_token_type" in data:
            masked_token_type[i, :num_nodes] = data.masked_token_type[start_idx:end_idx]
        else:
            # Default: use actual atomic numbers
            masked_token_type[i, :num_nodes] = data.atomic_numbers[start_idx:end_idx]

        # Mark real atoms in mask (False = real atom, True = padding)
        padding_mask[i, :num_nodes] = False

    # =========================================================================
    # Construct Batched Data Dictionary
    # =========================================================================
    
    batched_data = {
        # Core tensors
        "pos": batched_pos,                    # [num_graphs, max_nodes, 3] - Positions
        "cell": data.cell,                     # [num_graphs, 3, 3] - Unit cells
        "token_id": batched_token_id,          # [num_graphs, max_nodes] - Atomic numbers
        "atomic_numbers": batched_token_id,
        "masked_token_type": masked_token_type,# [num_graphs, max_nodes] - For MLM tasks
        "padding_mask": padding_mask,          # [num_graphs, max_nodes] - Padding indicators
        "pbc": pbc,                           # [num_graphs, 3] - PBC flags per dimension
        
        # Optional dataset metadata
        "subset_name": None if "subset_name" not in data else data.subset_name,
        "forces_subset_name": None if "forces_subset_name" not in data else data.forces_subset_name,
        
        # Additional flags
        "is_protein": is_protein,              # [num_graphs, max_nodes, 1] - Protein flag
        
        # Position IDs for potential positional encoding (currently unused)
        "position_ids": torch.arange(max_nodes)
        .unsqueeze(dim=0)
        .repeat(num_graphs, 1),                # [num_graphs, max_nodes]
        
        # Batch information
        "num_atoms": num_atoms,                # [num_graphs] - Actual atom counts
        "node_batch": batch_idx,               # [num_nodes] - Graph assignment
        "graph_padding_mask": padding_mask,    # [num_graphs, max_nodes] - Same as padding_mask
    }

    batched_data = Data(**batched_data)

    return batched_data


@registry.register_model("PSM_ESCAIP_backbone")
class E2FormerBackbone(nn.Module):
    """
    Physics Science Module backbone model integrated with EScAIP framework.

    This model combines the PSM architecture with EScAIP's configuration and processing
    pipeline, enabling it to work within the EScAIP framework while maintaining PSM's
    unique architectural features.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize E2Former Backbone model.
        
        This wrapper handles data preprocessing and coordinates the E2Former decoder.
        It processes molecular/crystal structures and predicts energy and forces.
        
        Args:
            **kwargs: Configuration parameters including:
                - pbc_max_radius: Maximum radius for periodic boundary conditions
                - max_neighbors: Maximum number of neighbors per atom
                - backbone_config: E2Former decoder configuration
                - global_cfg: Global training configuration
        """
        super().__init__()
        self.kwargs = kwargs
        
        # =====================================================================
        # Configuration Loading
        # =====================================================================
        
        # Initialize configuration from kwargs using dataclass validation
        cfg = init_configs(E2FormerConfigs, kwargs)
        self.cfg = cfg
        self.global_cfg = cfg.global_cfg

        # =====================================================================
        # Training Configuration
        # =====================================================================
        
        # Whether to predict forces in addition to energy
        self.regress_forces = cfg.global_cfg.regress_forces

        # PSM (Protein Structure Model) specific configuration
        # TODO: Integrate PSM config with EScAIP config system
        self.psm_config = cfg.psm_config

        # =====================================================================
        # Periodic Boundary Conditions (PBC) Setup
        # =====================================================================
        
        # Initialize cell expander for handling periodic systems
        # This creates periodic images of atoms for proper neighbor calculations
        self.cell_expander = CellExpander(
            self.kwargs.get("pbc_max_radius", 5.),  # Cutoff radius for PBC
            self.kwargs.get("expanded_token_cutoff", 512),  # deprecated parameter
            self.kwargs.get("pbc_expanded_num_cell_per_direction", 4),  # Cells to expand in each direction
            self.kwargs.get("pbc_max_radius", 5.),  # Duplicate parameter (should be cleaned up)
        )

        # =====================================================================
        # Embedding Layers
        # =====================================================================
        
        # Extract embedding dimension from E2Former decoder configuration
        # This ensures compatibility between wrapper and decoder
        self.fea_dim = o3.Irreps(cfg.backbone_config.irreps_node_embedding)[0][0]
        
        # Initialize learnable embeddings for atomic properties
        self.embedding = nn.Embedding(256, self.fea_dim)  # Atomic number embeddings (up to element 256)
        self.embedding_charge = nn.Embedding(30, self.fea_dim)  # Charge state embeddings (-10 to +19)
        self.embedding_multiplicity = nn.Embedding(30, self.fea_dim)  # Spin multiplicity embeddings (0 to 29)

        # =====================================================================
        # Additional Configuration Parameters
        # =====================================================================
        
        # Parameters for potential future electron density features
        self.uniform_center_count = 5  # Number of uniform centers for density
        self.sph_grid_channel = 8      # Channels for spherical grid representation
        
        # =====================================================================
        # Initialize E2Former Decoder
        # =====================================================================
        print("e2former use config like follows: \n", cfg.backbone_config)
        self.decoder = E2former(**vars(cfg.backbone_config))
        # Enable high precision matrix multiplication if not using fp16
        if not self.global_cfg.use_fp16_backbone:
            torch.set_float32_matmul_precision("high")

        # Configure logging and compilation
        torch._logging.set_logs(recompiles=True)
        # print("compiled:", self.global_cfg.use_compile)

        # Set up forward function with optional compilation
        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )



    def compiled_forward(
        self,
        batched_data,
        **kwargs,
    ):
        """
        Forward pass implementation that can be compiled with torch.compile.
        """
        # Enable gradient computation for forces if needed
        use_grad = (
            True  # self.global_cfg.regress_forces and not self.global_cfg.direct_force
        )
        batched_data["pos"].requires_grad_(use_grad)

        batched_data = process_batch_data(batched_data, None)
        # Generate embeddings
        atomic_numbers = batched_data["atomic_numbers"]
        # padding_mask = ~batched_data["atom_masks"]
        padding_mask = batched_data["padding_mask"]
        pos = batched_data["pos"]
        batched_data["pos"] = torch.where(
            padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3),
            999.0,
            batched_data["pos"].float(),
        )
        bsz, L = batched_data["pos"].shape[:2]
        # with torch.cuda.amp.autocast(enabled=self.global_cfg.use_fp16_backbone):
        with nullcontext():
            # Handle periodic boundary conditions
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                pbc_expand_batched = self.cell_expander.expand_includeself(
                    pos,
                    None,
                    batched_data["pbc"],
                    batched_data["num_atoms"],
                    atomic_numbers,
                    batched_data["cell"],
                    neighbors_radius=(
                        self.kwargs["max_neighbors"],
                        self.kwargs["pbc_max_radius"],
                    ),
                    use_local_attention=False,  # use_local_attention,
                    use_grad=use_grad,
                    padding_mask=padding_mask,
                )
                # dist: B*tgt_len*src_len
                
                pbc_expand_batched["expand_pos"][
                    pbc_expand_batched["expand_mask"]
                ] = 999  # set expand node pos padding to 9999


            else:
                pbc_expand_batched = None

            token_embedding = self.embedding(atomic_numbers) 
                # self.embedding_charge(torch.clip(batched_data["charge"],-10,10)+10) + \
                    # self.embedding_multiplicity(torch.clip(batched_data["multiplicity"],0,20))

            # =====================================================================
            # Step 4: Forward Pass Through E2Former Decoder
            # =====================================================================
            
            # Pass preprocessed data directly to E2Former decoder
            # The decoder handles all E(3)-equivariant transformations
            (
                node_features,
                node_vec_features,
                node_irreps,
                node_irreps_his,
            ) = self.decoder(
                batched_data,
                token_embedding,
                None,
                padding_mask,
                pbc_expand_batched=pbc_expand_batched,
                return_node_irreps=True,
            )

        # =====================================================================
        # Step 5: Flatten Node Features (PyG Style)
        # =====================================================================
        
        # Convert from batched format [B, N, ...] to flattened format [total_atoms, ...]
        # This is compatible with PyTorch Geometric (PyG) style processing
        # Padded nodes are filtered out during flattening
        (
            node_features_flatten,
            node_vec_features_flatten,
            node_irreps_flatten,
            node_irreps_his_flatten,
        ) = self.flatten_node_features(
            node_features,
            node_vec_features,
            node_irreps,
            node_irreps_his,
            ~padding_mask,  # Invert mask: True for real atoms
        )

        # =====================================================================
        # Step 6: Return Results
        # =====================================================================
        
        # Return both batched and flattened representations
        return {
            # Batched format outputs [B, N, ...]
            "node_irrepsBxN": node_irreps,              # Full irreducible representations
            "node_featuresBxN": node_features,          # Scalar features (energy)
            "node_vec_featuresBxN": node_vec_features,  # Vector features (forces)
            "data": batched_data,                       # Original input data with updates
            
            # Flattened format outputs [total_atoms, ...]
            "node_irreps": node_irreps_flatten,         # Flattened irreps
            "node_features": node_features_flatten,      # Flattened scalar features
            "node_vec_features": node_vec_features_flatten,  # Flattened vector features
            "node_irreps_his": node_irreps_his_flatten, # Historical irreps (penultimate layer)
        }

    def flatten_node_features(
        self,
        node_features,
        node_vec_features,
        node_irreps,
        node_irreps_his,
        padding_mask,
    ):
        """
        Flatten batched node features to PyG-style format.
        
        Converts from batched format [B, N, ...] to flattened format [total_atoms, ...]
        by removing padding and concatenating all real atoms.
        
        Args:
            node_features: Scalar features [B, N, D]
            node_vec_features: Vector features [B, N, 3, D]
            node_irreps: Irreducible representations [B, N, (lmax+1)^2, D]
            node_irreps_his: Historical irreps from penultimate layer
            padding_mask: Boolean mask, True for real atoms
        
        Returns:
            Tuple of flattened features containing only real atoms
        """
        
        # Reshape to merge batch and node dimensions
        flat_node_irreps = node_irreps.view(
            -1, node_irreps.size(-2), node_irreps.size(-1)
        )  # [B*N, (lmax+1)^2, D]
        
        flat_node_irreps_his = node_irreps_his.view(
            -1, node_irreps_his.size(-2), node_irreps_his.size(-1)
        )  # [B*N, (lmax+1)^2, D]
        
        flat_node_features = node_features.view(-1, node_features.size(-1))  # [B*N, D]
        
        flat_node_vec_features = node_vec_features.view(
            -1, node_vec_features.size(-2), node_vec_features.size(-1)
        )  # [B*N, 3, D]
        
        # Flatten the mask
        flat_mask = padding_mask.view(-1)  # [B*N]
        
        # Filter out padded nodes using the mask
        valid_node_irreps = flat_node_irreps[flat_mask]      # [num_real_atoms, (lmax+1)^2, D]
        valid_node_irreps_his = flat_node_irreps_his[flat_mask]  # [num_real_atoms, (lmax+1)^2, D]
        valid_node_features = flat_node_features[flat_mask]  # [num_real_atoms, D]
        valid_node_vec_features = flat_node_vec_features[flat_mask]
        return (
            valid_node_features,
            valid_node_vec_features,
            valid_node_irreps,
            valid_node_irreps_his,
        )

    def forward(
        self,
        data: torch_geometric.data.Batch,
        node_embedding=None,
        # aa_mask=None,
        # padding_mask=None,
        *args,
        **kwargs,
    ):
        """
        Main forward pass of the model.
        """
        # PSM handles preprocessing internally
        return self.forward_fn(data, token_embedding=node_embedding)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Returns parameters that should not use weight decay.
        """
        return no_weight_decay(self)
        # return no_weight_decay

    def test_equivariant(self, original_data):
        # assume batch size is 1
        assert (
            original_data.batch.max() == 0
        ), "batch size must be 1 for test_equivariant"
        self.eval()  # this is very important
        data_2 = original_data.clone().cpu()
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_2, matrix, inv_matrix = transform(data_2)
        data_2 = data_2.to(original_data.pos.device)
        data_list = [original_data, data_2]
        data_list.ptr = torch.tensor(
            [
                0,
                original_data.pos.size(0),
                original_data.pos.size(0) + data_2.pos.size(0),
            ],
            device=original_data.pos.device,
        )
        results = self.compiled_forward(data_list)
        combined_node_features = results["node_features"]
        # split the node features into two parts
        node_features_1 = combined_node_features[: original_data.pos.size(0)]
        node_features_2 = combined_node_features[original_data.pos.size(0) :]

        assert node_features_1.allclose(
            node_features_2, rtol=1e-2, atol=1e-2
        ), "node features are not equivariant"

        node_vec_features_1 = results["node_vec_features"][: original_data.pos.size(0)]
        node_vec_features_2 = results["node_vec_features"][original_data.pos.size(0) :]
        # rotate the node vec features
        node_vec_features_1 = torch.einsum(
            "bsd, sj -> bjd", node_vec_features_1, matrix.to(node_vec_features_1.device)
        )
        assert node_vec_features_1.allclose(
            node_vec_features_2, rtol=1e-2, atol=1e-2
        ), "node vec features are not equivariant"

class E2FormerHeadBase(nn.Module):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        # self.molecular_graph_cfg = backbone.molecular_graph_cfg
        # self.gnn_cfg = backbone.gnn_cfg
        # self.reg_cfg = backbone.reg_cfg

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("E2Former_easy_energy_head")
class E2FormerEasyEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)
        self.use_amp = False

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.linear(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,
                index=data.node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return {"energy": energy_output}


@registry.register_model("E2Former_easy_force_head")
class E2FormerEasyForceHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)
        self.use_amp = False

        self.post_init()

    def compiled_forward(
        self, node_features, node_vec_features, data
    ):
        # get force direction from node vector features
        force_direction = self.linear(node_vec_features).squeeze(-1)  # (num_nodes, 3)

        # get output force
        return force_direction

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            node_features=emb["node_features"],
            node_vec_features=emb["node_vec_features"],
            data=emb["data"],
        )

        return {"forces": force_output}


@registry.register_model("E2Former_grad_energy_force_head")
class E2FormerGradEnergyHead(E2FormerHeadBase):
    def __init__(self, backbone: E2FormerBackbone):
        super().__init__(backbone)
        self.linear = nn.Linear(backbone.decoder.scalar_dim, 1, bias=False)

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data):
        energy_output = self.linear(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")
        # the shape of energy_output is [num_nodes, 1]
        # the shape of data.node_batch is [num_nodes]
        # the shape of data.graph_padding_mask is [num_graphs, num_nodes]
        # the shape of data.node_batch is [num_nodes]
        # dim size is the number of graphs
        number_of_graphs = data.node_batch.max() + 1
        energy_output = (
            compilable_scatter(
                src=energy_output,
                index=data.node_batch,
                dim_size=number_of_graphs,
                dim=0,
                reduce="sum",
            )
            / _AVG_NUM_NODES
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )

        forces_output = (
            -1
            * torch.autograd.grad(
                energy_output.sum(), data.pos, create_graph=self.training
            )[0]
        )

        return {"energy": energy_output, "forces": forces_output}

