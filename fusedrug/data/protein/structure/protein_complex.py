from typing import Union, List, Optional, Tuple
from fusedrug.data.protein.structure.structure_io import (
    get_mmcif_native_full_name,
    get_chain_native_features,
)
from itertools import combinations
import torch
from collections import defaultdict


class ProteinComplex:
    """
    Holds a protein complex (or a subset of it)
    Allows to construct a fake (usually "negative") pair coming from different pdb_id entries

    call `add(...)` to accumulate more chains into it
    """

    def __init__(self) -> None:
        self.chains_data = {}

    def add(
        self, pdb_id: str, chain_ids: Optional[List[Union[str, int]]] = None
    ) -> None:
        """
        Args:
            chain_ids: provide None (default) to load all chains
                provide a list of chain identifiers to select which are loaded.
                    use str to use chain_id
                    use int to load chain at (zero based) index
        """
        self.filename = get_mmcif_native_full_name(pdb_id)

        assert isinstance(chain_ids, list) or (chain_ids is None)
        loaded_chains = get_chain_native_features(
            self.filename,
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
            seq = self.chains_data[chain_desc]["gt_sequence"]
            feats = self.chains_data[chain_desc]["gt_mmcif_feats"]
            concat_seq.append(seq)
            for k, d in feats.items():
                concat_feats[k].append(d)

            length = feats["aatype"].shape[0]
            concat_feats["residue_index"].append(
                torch.arange(length) + next_start_residue_index
            )
            concat_feats["chain_index"].append(torch.full((length,), fill_value=i))

            next_start_residue_index += length + inter_chain_index_extra_offset

        flattened_chain_data = {}
        flattened_chain_data["flattened"] = {}
        flattened_chain_data["flattened"]["gt_sequence"] = "".join(concat_seq)
        #
        flattened_chain_data["flattened"]["gt_mmcif_feats"] = {}
        for k, d in concat_feats.items():
            if isinstance(d[0], str):
                concat_elem = ",".join(d)
            else:
                concat_elem = torch.concat(d, dim=0)
            flattened_chain_data["flattened"]["gt_mmcif_feats"][k] = concat_elem

        self.chains_data = flattened_chain_data

    def findInteractingChains(
        self,
        distance_threshold: float = 7.0,
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
                xyz_1=self.chains_data[chain_1_desc]["gt_mmcif_feats"][
                    "atom14_gt_positions"
                ],
                mask_1=self.chains_data[chain_1_desc]["gt_mmcif_feats"][
                    "atom14_gt_exists"
                ],
                #
                xyz_2=self.chains_data[chain_2_desc]["gt_mmcif_feats"][
                    "atom14_gt_positions"
                ],
                mask_2=self.chains_data[chain_2_desc]["gt_mmcif_feats"][
                    "atom14_gt_exists"
                ],
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
