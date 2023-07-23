from typing import Union, List, Optional, Tuple
from fusedrug.data.protein.structure.structure_io import (
    get_mmcif_native_full_name,
    get_chain_native_features,
)
from itertools import combinations
import torch


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
        loaded_chains = get_chain_native_features(self.filename, chain_id=chain_ids)

        for k, d in loaded_chains.items():
            self.chains_data[(pdb_id, k)] = d

    def concatAllChainsToSingleSequence(
        self, index_offset_between_chains: int = 2
    ) -> None:
        """
        concats each feature from multiple chains into a single sequence.
        This is useful, for example, before supplying it to a model, which expects a tensor that contains info on the complex (or complex subset, e.g. pair)
        """
        raise NotImplementedError()

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
        """
        ans = []

        # protein["atom14_atom_exists"] = residx_atom14_mask
        # protein["atom14_gt_exists"] = residx_atom14_gt_mask
        # protein["atom14_gt_positions"] = residx_atom14_gt_positions
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
        < distance_threshold
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
