# -*- coding: utf-8 -*-
"""
Cell Utils module for E2Former.
"""

import warnings
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
import logging
import numpy as np

@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


# follow PSM


class CellExpander:
    def __init__(
        self,
        cutoff=10.0,
        expanded_token_cutoff=512,
        pbc_expanded_num_cell_per_direction=10,
        pbc_multigraph_cutoff=10.0,
    ):
        self.cells = []
        for i in range(
            -pbc_expanded_num_cell_per_direction,
            pbc_expanded_num_cell_per_direction + 1,
        ):
            for j in range(
                -pbc_expanded_num_cell_per_direction,
                pbc_expanded_num_cell_per_direction + 1,
            ):
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                ):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    self.cells.append([i, j, k])

        self.cells = torch.tensor(self.cells)

        self.cell_mask_for_pbc = self.cells != 0

        self.candidate_cells = torch.tensor(
            [
                [i, j, k]
                for i in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
                for j in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                )
            ]
        )

        self.cutoff = cutoff

        self.expanded_token_cutoff = expanded_token_cutoff

        self.pbc_multigraph_cutoff = pbc_multigraph_cutoff

        self.pbc_expanded_num_cell_per_direction = pbc_expanded_num_cell_per_direction

        self.conflict_cell_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i != 0 or j != 0 or k != 0:
                        self.conflict_cell_offsets.append([i, j, k])
        self.conflict_cell_offsets = torch.tensor(self.conflict_cell_offsets)  # 26 x 3

        conflict_to_consider = self.cells.unsqueeze(
            1
        ) - self.conflict_cell_offsets.unsqueeze(
            0
        )  # num_expand_cell x 26 x 3
        conflict_to_consider_mask = (
            ((conflict_to_consider * self.cells.unsqueeze(1)) >= 0)
            & (torch.abs(conflict_to_consider) <= self.cells.unsqueeze(1).abs())
        ).all(
            dim=-1
        )  # num_expand_cell x 26
        conflict_to_consider_mask &= (
            (conflict_to_consider <= pbc_expanded_num_cell_per_direction)
            & (conflict_to_consider >= -pbc_expanded_num_cell_per_direction)
        ).all(
            dim=-1
        )  # num_expand_cell x 26
        self.conflict_to_consider_mask = conflict_to_consider_mask

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)

    def _get_cell_tensors(self, cell, use_local_attention):
        # fitler impossible offsets according to cell size and cutoff
        def _get_max_offset_for_dim(cell, dim):
            lattice_vec_0 = cell[:, dim, :]
            lattice_vec_1_2 = cell[
                :, torch.arange(3, dtype=torch.long, device=cell.device) != dim, :
            ]
            normal_vec = torch.cross(
                lattice_vec_1_2[:, 0, :], lattice_vec_1_2[:, 1, :], dim=-1
            )
            normal_vec = normal_vec / normal_vec.norm(dim=-1, keepdim=True)
            cutoff = self.pbc_multigraph_cutoff if use_local_attention else self.cutoff

            max_offset = int(
                torch.max(
                    torch.ceil(
                        cutoff
                        / torch.abs(torch.sum(normal_vec * lattice_vec_0, dim=-1))
                    )
                )
            )
            return max_offset

        max_offsets = []
        for i in range(3):
            try:
                max_offset = _get_max_offset_for_dim(cell, i)
            except Exception as e:
                logging.warning(f"{e} with cell {cell}")
                max_offset = self.pbc_expanded_num_cell_per_direction
            max_offsets.append(max_offset)
        max_offsets = torch.tensor(max_offsets, device=cell.device)
        self.cells = self.cells.to(device=cell.device)
        self.cell_mask_for_pbc = self.cell_mask_for_pbc.to(device=cell.device)
        mask = (self.cells.abs() <= max_offsets).all(dim=-1)
        selected_cell = self.cells[mask, :]
        return selected_cell, self.cell_mask_for_pbc[mask, :], mask

    def _get_conflict_mask(self, cell, pos, atoms):
        batch_size, max_num_atoms = pos.size()[:2]
        self.conflict_cell_offsets = self.conflict_cell_offsets.to(device=pos.device)
        self.conflict_to_consider_mask = self.conflict_to_consider_mask.to(
            device=pos.device
        )
        offset = torch.bmm(
            self.conflict_cell_offsets.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(dtype=cell.dtype),
            cell,
        )  # batch_size x 26 x 3
        expand_pos = (pos.unsqueeze(1) + offset.unsqueeze(2)).reshape(
            batch_size, -1, 3
        )  # batch_size x max_num_atoms x 3, batch_size x 26 x 3 -> batch_size x (26 x max_num_atoms) x 3
        expand_dist = (pos.unsqueeze(2) - expand_pos.unsqueeze(1)).norm(
            p=2, dim=-1
        )  # batch_size x max_num_atoms x (26 x max_num_atoms)

        expand_atoms = atoms.repeat(
            1, self.conflict_cell_offsets.size()[0]
        )  # batch_size x (26 x max_num_atoms)
        atoms_identical_mask = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(
            1
        )  # batch_size x max_num_atoms x (26 x max_num_atoms)

        conflict_mask = (
            ((expand_dist < 1e-5) & atoms_identical_mask)
            .any(dim=1)
            .reshape(batch_size, -1, max_num_atoms)
        )  # batch_size x 26 x max_num_atoms
        all_conflict_mask = (
            torch.bmm(
                self.conflict_to_consider_mask.unsqueeze(0)
                .to(dtype=cell.dtype)
                .repeat(batch_size, 1, 1),
                conflict_mask.to(dtype=cell.dtype),
            )
            .long()
            .bool()
        )  # batch_size x num_expand_cell x 26, batch_size x 26 x max_num_atoms -> batch_size x num_expand_cell x max_num_atoms
        return all_conflict_mask

    def check_conflict(self, pos, atoms, pbc_expand_batched):
        # ensure that there's no conflict in the expanded atoms
        # a conflict means that two atoms (or special tokens) share both the same position and token type
        expand_pos = pbc_expand_batched["expand_pos"]
        all_pos = torch.cat([pos, expand_pos], dim=1)
        num_expanded_atoms = all_pos.size()[1]
        all_dist = (all_pos.unsqueeze(1) - all_pos.unsqueeze(2)).norm(p=2, dim=-1)
        outcell_index = pbc_expand_batched[
            "outcell_index"
        ]  # batch_size x expanded_max_num_atoms
        all_atoms = torch.cat(
            [atoms, torch.gather(atoms, dim=-1, index=outcell_index)], dim=-1
        )
        atom_identical_mask = all_atoms.unsqueeze(1) == all_atoms.unsqueeze(-1)
        full_mask = torch.cat([atoms.eq(0), pbc_expand_batched["expand_mask"]], dim=-1)
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(-1), False
        )
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(1), False
        )
        conflict_mask = (all_dist < 1e-5) & atom_identical_mask
        conflict_mask[
            :,
            torch.arange(num_expanded_atoms, device=all_pos.device),
            torch.arange(num_expanded_atoms, device=all_pos.device),
        ] = False
        assert ~(
            conflict_mask.any()
        ), f"{all_dist[conflict_mask]} {all_atoms[conflict_mask.any(dim=-2)]}"

    def expand(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        pair_token_type,
        use_local_attention=True,
        use_grad=False,
    ):
        with torch.set_grad_enabled(use_grad):
            pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.size()[:2]
            cell_tensor, cell_mask, selected_cell_mask = self._get_cell_tensors(
                cell, use_local_attention
            )

            if not use_local_attention:
                all_conflict_mask = self._get_conflict_mask(cell, pos, atoms)
                all_conflict_mask = all_conflict_mask[:, selected_cell_mask, :].reshape(
                    batch_size, -1
                )
            # if expand_includeself:
            #     cell_tensor = torch.cat([torch.zeros((1,3),device = cell_tensor.device),cell_tensor],dim = 0)
            #     cell_mask = torch.cat([torch.ones((1,3),device = cell_mask.device).bool(),cell_mask],dim = 0)

            cell_tensor = (
                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(
                2
            )  # B x num_expand_cell x T x 3
            expand_pos = expand_pos.view(
                batch_size, -1, 3
            )  # B x (num_expand_cell x T) x 3

            # eliminate duplicate atoms of expanded atoms, comparing with the original unit cell
            expand_dist = torch.norm(
                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
            )  # B x T x (num_expand_cell x T)
            expand_atoms = atoms.repeat(1, num_expanded_cell)
            expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)
            expand_mask = (
                expand_dist
                < (self.pbc_multigraph_cutoff if use_local_attention else self.cutoff)
            ) & (
                (expand_dist > 1e-5) | ~expand_atom_identical
            )  # B x T x (num_expand_cell x T)
            expand_mask = torch.masked_fill(
                expand_mask, atoms.eq(0).unsqueeze(-1), False
            )
            expand_mask = torch.sum(expand_mask, dim=1) > 0
            if not use_local_attention:
                expand_mask = expand_mask & (~all_conflict_mask)
            expand_mask = expand_mask & (
                ~(atoms.eq(0).repeat(1, num_expanded_cell))
            )  # B x (num_expand_cell x T)

            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (num_expand_cell x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            threshold_num_expanded_token = torch.clamp(
                self.expanded_token_cutoff - num_atoms, min=0
            )

            max_expand_len = torch.max(expand_len)

            # cutoff within expanded_token_cutoff tokens
            need_threshold = expand_len > threshold_num_expanded_token
            if need_threshold.any():
                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
                expand_dist_mask = (
                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)
                ).repeat(1, 1, num_expanded_cell)
                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
                min_expand_dist = min_expand_dist.masked_fill_(
                    ~cell_mask.unsqueeze(1), np.inf
                )
                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]

                need_threshold_distances = min_expand_dist[
                    need_threshold
                ]  # B x (num_expand_cell x T)
                threshold_num_expanded_token = threshold_num_expanded_token[
                    need_threshold
                ]
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0]

                threshold_dist = torch.gather(
                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1)
                )

                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
                expand_mask[need_threshold] &= new_expand_mask
                expand_len = torch.sum(expand_mask, dim=-1)
                max_expand_len = torch.max(expand_len)

            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]

            expand_pair_token_type = torch.gather(
                pair_token_type,
                dim=2,
                index=outcell_index.unsqueeze(1)
                .unsqueeze(-1)
                .repeat(1, max_num_atoms, 1, pair_token_type.size()[-1]),
            )
            expand_node_type_edge = torch.cat(
                [pair_token_type, expand_pair_token_type], dim=2
            )

            if use_local_attention:
                dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(p=2, dim=-1)
                expand_dist_compress = (
                    pos.unsqueeze(2) - expand_pos_compressed.unsqueeze(1)
                ).norm(p=2, dim=-1)
                local_attention_weight = self.polynomial(
                    torch.cat([dist, expand_dist_compress], dim=2),
                    cutoff=self.pbc_multigraph_cutoff,
                )
                is_periodic = pbc.any(dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                expand_mask = mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                )
                full_mask = torch.cat([atoms.eq(0), expand_mask], dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    full_mask.unsqueeze(1), 0.0
                )
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": expand_mask,
                    "local_attention_weight": local_attention_weight,
                    "expand_node_type_edge": expand_node_type_edge,
                }
            else:
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": mask_after_k_persample(
                        batch_size, max_expand_len, expand_len
                    ),
                    "local_attention_weight": None,
                    "expand_node_type_edge": expand_node_type_edge,
                }

            expand_pos_no_offset = torch.gather(
                pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            offset = expand_pos_compressed - expand_pos_no_offset
            init_expand_pos_no_offset = torch.gather(
                init_pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            init_expand_pos = init_expand_pos_no_offset + offset
            init_expand_pos = init_expand_pos.masked_fill(
                pbc_expand_batched["expand_mask"].unsqueeze(-1),
                0.0,
            )

            pbc_expand_batched["init_expand_pos"] = init_expand_pos

            # # self.check_conflict(pos, atoms, pbc_expand_batched)
            # print(f"local attention weight {local_attention_weight.numel()} zero:{torch.sum(local_attention_weight==0)}")
            # # print(torch.sum(local_attention_weight==0,dim = 1)==(local_attention_weight.shape[1]))
            # print("N1+N2, ",local_attention_weight.shape[2],torch.sum(
            #     torch.sum(local_attention_weight==0,dim = 1)==local_attention_weight.shape[1])/(local_attention_weight.shape[0]*1.0))

            return pbc_expand_batched

    def expand_includeself(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        neighbors_radius,
        pair_token_type=None,
        use_local_attention=True,
        use_grad=False,
        padding_mask=None,
    ):
        with torch.set_grad_enabled(use_grad):
            pos = torch.where(
                padding_mask.unsqueeze(dim=-1).repeat(1, 1, 3), 999.0, pos.float()
            )
            # pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.size()[:2]
            cell_tensor, cell_mask, selected_cell_mask = self._get_cell_tensors(
                cell, use_local_attention
            )

            cell_tensor = torch.cat(
                [torch.zeros((1, 3), device=cell_tensor.device), cell_tensor], dim=0
            )
            # self.cell_mask_for_pbc = self.cells != 0
            cell_mask = torch.cat(
                [torch.zeros((1, 3), device=cell_mask.device).bool(), cell_mask], dim=0
            )

            cell_tensor = (
                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(
                2
            )  # B x num_expand_cell x T x 3
            expand_pos = expand_pos.view(
                batch_size, -1, 3
            )  # B x (num_expand_cell x T) x 3

            # eliminate duplicate atoms of expanded atoms, comparing with the original unit cell
            expand_dist = torch.norm(
                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
            )  # B x T x (num_expand_cell x T)
            # expand_atoms = atoms.repeat(1, num_expanded_cell)
            # expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)

            if neighbors_radius[0] is None or neighbors_radius[0]>=expand_pos.shape[1]:
                expand_mask = (expand_dist < neighbors_radius[1])
            else:
                values, _ = torch.topk(
                    expand_dist, neighbors_radius[0] + 1, dim=-1, largest=False
                )
                expand_mask = (
                    expand_dist <= (values[:, :, neighbors_radius[0]].unsqueeze(dim=-1))
                ) & (expand_dist < neighbors_radius[1])
                # & (expand_dist > 1e-5)
                #     (
                #     (expand_dist > 1e-5) | ~expand_atom_identical
                # )  # B x T x (num_expand_cell x T)
            
            expand_mask = (
                expand_mask
                & (~padding_mask.repeat(1, num_expanded_cell).unsqueeze(1))
                & (~(atoms.eq(0).unsqueeze(-1)))
            )

            expand_mask = torch.sum(expand_mask, dim=1) > 0
            # if not use_local_attention:
            #     expand_mask = expand_mask & (~all_conflict_mask)
            expand_mask = expand_mask & (
                ~(atoms.eq(0).repeat(1, num_expanded_cell))
            )  # B x (num_expand_cell x T)

            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (num_expand_cell x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            threshold_num_expanded_token = torch.clamp(
                self.expanded_token_cutoff - num_atoms * 0, min=0
            )

            max_expand_len = torch.max(expand_len)

            # cutoff within expanded_token_cutoff tokens
            need_threshold = expand_len > threshold_num_expanded_token
            if need_threshold.any():
                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
                expand_dist_mask = (
                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)
                ).repeat(1, 1, num_expanded_cell)
                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
                min_expand_dist = min_expand_dist.masked_fill_(
                    ~cell_mask.unsqueeze(1), np.inf
                )
                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]

                need_threshold_distances = min_expand_dist[
                    need_threshold
                ]  # B x (num_expand_cell x T)
                threshold_num_expanded_token = threshold_num_expanded_token[
                    need_threshold
                ]
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0]

                threshold_dist = torch.gather(
                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1).long()
                )

                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
                expand_mask[need_threshold] &= new_expand_mask
                expand_len = torch.sum(expand_mask, dim=-1)
                max_expand_len = torch.max(expand_len)
                print("expand_len",expand_len)
                print("expand_mask",expand_mask)

            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]
            pbc_expand_batched = {
                "expand_pos": expand_pos_compressed,
                "outcell_index": outcell_index,
                "expand_mask": mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                ),
                "local_attention_weight": None,
                # "expand_node_type_edge": expand_node_type_edge,
            }
            return pbc_expand_batched





