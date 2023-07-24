from typing import Union, List, Optional, Tuple
from fusedrug.data.protein.structure.structure_io import (
    load_protein_structure_features,
)
from itertools import combinations
import torch
from collections import defaultdict
import numpy as np


class ProteinComplex:
    """
    Holds a protein complex (or a subset of it)
    Allows to construct a fake (usually "negative") pair coming from different pdb_id entries

    call `add(...)` to accumulate more chains into it
    """

    def __init__(self) -> None:
        self.chains_data = {}  # maps from chain description (e.g. ('7vux', 'A')) to
        self.flattened_data = {}

    def add(
        self, pdb_id: str, chain_ids: Optional[List[Union[str, int]]] = None
    ) -> None:
        """
        Args:
            pdb_id: for example '7vux'
            chain_ids: provide None (default) to load all chains
                provide a list of chain identifiers to select which are loaded.
                    use str to use chain_id
                    use int to load chain at (zero based) index
        """
        assert isinstance(chain_ids, list) or (chain_ids is None)
        loaded_chains = load_protein_structure_features(
            pdb_id,
            pdb_id=pdb_id if len(pdb_id) == 4 else None,
            chain_id=chain_ids,
        )

        for k, d in loaded_chains.items():
            self.chains_data[(pdb_id, k)] = d

    def flatten(
        self,
        chains_descs: List[Tuple[str, str]],
        inter_chain_index_extra_offset: int = 1,
    ) -> None:
        """
        concats each feature from multiple chains into a single sequence.
        This is useful, for example, before supplying it to a model, which expects a tensor that contains info on the complex (or complex subset, e.g. pair)
        """

        concat_seq = []
        concat_feats = defaultdict(list)
        next_start_residue_index = 0
        for i, chain_desc in enumerate(chains_descs):
            seq = self.chains_data[chain_desc]["aa_sequence_str"]
            feats = self.chains_data[chain_desc]
            concat_seq.append(seq)
            for k, d in feats.items():
                concat_feats[k].append(d)

            length = feats["aatype"].shape[0]
            concat_feats["residue_index"].append(
                torch.arange(length) + next_start_residue_index
            )
            concat_feats["chain_index"].append(torch.full((length,), fill_value=i))

            next_start_residue_index += length + inter_chain_index_extra_offset

        for k, d in concat_feats.items():
            if isinstance(d[0], str):
                concat_elem = ",".join(d)
            else:
                concat_elem = torch.concat(d, dim=0)
            self.flattened_data[k] = concat_elem

        self.flattened_data["aa_sequence_str"] = "".join(concat_seq)

        self.chains_descs_for_flatten = chains_descs

    def spatial_crop(
        self, crop_size: int = 256, distance_threshold: float = 10.0, eps: float = 1e-6
    ) -> None:
        """
        Spatial crop of a pair of chains which favors interacting residues.
        Note - you must call "flatten" (with only two chain descriptor) prior to calling this method.

        The code is heavily influenced from the spatial crop done in RF2
        """
        if not hasattr(self, "chains_descs_for_flatten"):
            raise Exception("You must call flatten() method first.")

        if len(self.chains_descs_for_flatten) != 2:
            raise Exception("")
        chains_lengths = [
            self.chains_data[chain_desc]["aatype"].shape[0]
            for chain_desc in self.chains_descs_for_flatten
        ]

        xyz = self.flattened["atom14_gt_positions"]
        mask = self.flattened["atom14_gt_exists"]

        carbo_alpha_atom_index = 1

        cond = (
            torch.cdist(
                xyz[: chains_lengths[0], carbo_alpha_atom_index],
                xyz[chains_lengths[0] :, 1],
                p=2,
            )
            < distance_threshold
        )
        # only keep residue for which both ground truth masks show it's legit (was actually experimentally determined)
        cond = torch.logical_and(
            cond,
            mask[: chains_lengths[0], None, carbo_alpha_atom_index]
            * mask[None, chains_lengths[0] :, carbo_alpha_atom_index],
        )
        i, j = torch.where(cond)
        ifaces = torch.cat([i, j + chains_lengths[0]])
        if len(ifaces) < 1:
            raise Exception("No interface residues!")

        # pick a random interface residue
        cnt_idx = ifaces[np.random.randint(len(ifaces))]

        # calculate distance from this residue to all other residues, and add an increasing epsilon
        # it seems that:
        # 1. the increasing epsilon would favor the N-terminus
        # 2. the increasing epsilon would favor the first chain
        # NOTE: we can modify it to not favor the first chain, while still favoring the N-terminus
        dist = (
            torch.cdist(
                xyz[:, carbo_alpha_atom_index],
                xyz[cnt_idx, carbo_alpha_atom_index][None],
                p=2,
            ).reshape(-1)
            + torch.arange(len(xyz), device=xyz.device) * eps
        )
        # make sure that only residues with actual resolved position participate
        cond = mask[:, carbo_alpha_atom_index] * mask[cnt_idx, carbo_alpha_atom_index]
        dist[~cond.bool()] = 999999.9
        _, idx = torch.topk(dist, crop_size, largest=False)

        # sel, _ = torch.sort(sel[idx]) #this was the original, I wonder why they got "sel" from outside, maybe related to homo-oligomers?
        sel, _ = torch.sort(idx)

        self.apply_selection(sel)

        print("remember to save a PDB to visualize this!")

    def apply_selection(self, selection: torch.Tensor) -> None:
        """
        Applies selection described as indices in 'selection'

        Note - only affects the flattened data!
        """
        if not hasattr(self, "chains_descs_for_flatten"):
            raise Exception("You must call flatten() method first.")

        self.flattened["aa_sequence_str"] = "".join(
            np.array(list(self.flattened["aa_sequence_str"]))[selection].tolist()
        )

        for k, d in self.flattened.items():
            if k in ["resolution", "pdb_id", "chain_id"]:
                continue
            self.flattened[k] = d[selection]

    def findInteractingChains(
        self,
        distance_threshold: float = 10.0,
        min_interacting_residues_count: int = 4,
        assume_not_interacting_if_from_different_pdb_ids: bool = True,
        verbose: bool = True,
    ) -> List[Tuple[Tuple[str, str], str]]:
        """
        returns a list of non-redundant pair tuples, e.g.:
        [(('7vux', 'A'), ('7vux', 'H')),
         (('7vux', 'A'), ('7vux', 'L')),
         (('7vux', 'H'), ('7vux', 'L'))]

        Args:
            distance_threshold: the maximum distance that 2 residues are considered interacting
            min_interacting_residues_count: minimal amount of interacting residues to decide that two chains are interacting
            assume_not_interacting_if_from_different_pdb_ids: if two chains arrive from different PDBs skip check and assume that they are not interacting.
                the default is True
                this is because this is likely to be used for (fake) negative pairs
            verbose:
                printing amount

        Returns:
            a list of elements, each element is a tuple with two chains descriptors, describing the interacting chains pair
        """
        ans = []

        all_pairs = 0
        interacting_pairs = 0

        for comb in combinations(self.chains_data.keys(), 2):
            chain_1_desc = comb[0]
            chain_1_pdb_id, chain_1_chain_id = chain_1_desc

            chain_2_desc = comb[1]
            chain_2_pdb_id, chain_2_chain_id = chain_2_desc

            if assume_not_interacting_if_from_different_pdb_ids:
                if chain_1_pdb_id != chain_2_pdb_id:
                    if verbose:
                        print(
                            "Not same pdb_id and assume_not_interacting_if_from_different_pdb_ids=True, skipping"
                        )
                    continue
            print(comb)
            if check_interacting(
                xyz_1=self.chains_data[chain_1_desc]["atom14_gt_positions"],
                mask_1=self.chains_data[chain_1_desc]["atom14_gt_exists"],
                #
                xyz_2=self.chains_data[chain_2_desc]["atom14_gt_positions"],
                mask_2=self.chains_data[chain_2_desc]["atom14_gt_exists"],
                #
                distance_threshold=distance_threshold,
                min_interacting_residues_count=min_interacting_residues_count,
            ):
                if verbose:
                    print(f"chains {chain_1_desc} and {chain_2_desc} are interacting!")
                ans.append(comb)
                interacting_pairs += 1
            else:
                if verbose:
                    print(f"chains {comb[0]} and {comb[1]} are NOT interacting!")

            all_pairs += 1

        if verbose:
            print(
                f"out of total {all_pairs} pairs, {interacting_pairs} were interacting"
            )

        return ans

    #    def check_interacting(self, chain_id_1, chain_id)

    def spatialCrop(self) -> None:

        raise NotImplementedError()


def check_interacting(
    xyz_1: torch.Tensor,
    mask_1: torch.Tensor,
    xyz_2: torch.Tensor,
    mask_2: torch.Tensor,
    distance_threshold: float,
    min_interacting_residues_count: int,
    p_norm: float = 2.0,
    carbon_alpha_only: bool = True,
    verbose: bool = True,
) -> bool:

    if not carbon_alpha_only:
        raise Exception("carbon_alpha_only=False is not supported yet")

    carbon_alpha_index = 1
    # calculate distances between all pairs,
    cond = (
        torch.cdist(xyz_1[:, carbon_alpha_index], xyz_2[:, carbon_alpha_index], p_norm)
        <= distance_threshold
    )
    # create a mask with True only where both residues have ground truth coordinate info
    cond = torch.logical_and(
        cond, mask_1[:, None, carbon_alpha_index] * mask_2[None, :, carbon_alpha_index]
    )

    found_interacting_residues_count = cond.sum()
    ans = found_interacting_residues_count >= min_interacting_residues_count
    if verbose:
        print(f"found {found_interacting_residues_count} interacting residues")
    return ans
