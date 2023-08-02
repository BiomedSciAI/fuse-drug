from typing import Union, List, Optional, Tuple, Dict
from fusedrug.data.protein.structure.structure_io import (
    load_protein_structure_features,
    flexible_save_pdb_file,
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

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.chains_data = {}  # maps from chain description (e.g. ('7vux', 'A')) to
        self.flattened_data = {}

    def add(
        self,
        pdb_id: str,
        chain_ids: Optional[List[Union[str, int]]] = None,
        load_protein_structure_features_overrides: Dict = None,
        min_chain_residues_count: int = 10,
        max_residue_type_part: float = 0.5,
        allow_dna_or_rna_in_complex: bool = False,
    ) -> None:
        """
        Args:
            pdb_id: for example '7vux'
            chain_ids: provide None (default) to load all chains
                provide a list of chain identifiers to select which are loaded.
                    use str to use chain_id
                    use int to load chain at (zero based) index
            load_protein_structure_features_overrides: a dictionary with optional args to override args to send to load_protein_structure_features_overrides
            min_chain_residues_count: any chain with less than this amount of residues will be dropped. Set to None to keep all
            max_residue_type_part: any chain that contains a residue type that is at least this part will be dropped.
                For example, if max_residue_type_part is set to 0.5 (default) then if there exists a residue type that is at least 50% of the total residues,
                the chain will be dropped. This helps to filter out, for example, cases in which a list of Oxygen residues are defined as a peptide chain, or some "degenerate" cases.

        """
        assert isinstance(chain_ids, list) or (chain_ids is None)

        if load_protein_structure_features_overrides is None:
            load_protein_structure_features_overrides = {}

        ans = load_protein_structure_features(
            pdb_id,
            pdb_id=pdb_id if len(pdb_id) == 4 else None,
            chain_id=chain_ids,
            also_return_mmcif_object=True,
            **load_protein_structure_features_overrides,
        )

        if ans is None:
            if self.verbose:
                print(f"ProteinComplex::add could not load pdb_id={pdb_id}")
            return

        loaded_chains, mmcif_object = ans

        if not allow_dna_or_rna_in_complex:
            if mmcif_object.info["rna_or_dna_only_sequences_count"] > 0:
                if self.verbose:
                    print(
                        f'dna or rna sequences are not allowed, and detected {mmcif_object.info["rna_or_dna_only_sequences_count"]}'
                    )
                return

        # min_chain_residues_count:int = 10,
        # max_residue_type_part:float = 0.5,

        for k, d in loaded_chains.items():
            if min_chain_residues_count is not None:
                if len(d["aa_sequence_str"]) < min_chain_residues_count:
                    if self.verbose:
                        print(
                            f"chain {k} is too small, less than {min_chain_residues_count}"
                        )
                    continue
            if max_residue_type_part is not None:
                most_frequent_residue_part = d["aatype"].unique(return_counts=True)[
                    1
                ].max() / len(d["aatype"])
                if most_frequent_residue_part > max_residue_type_part:
                    if self.verbose:
                        print(
                            f"chain {k} dropped because it had {most_frequent_residue_part*100:.2f}% same residue id."
                        )
                    continue

            self.chains_data[(pdb_id, k)] = d

    def flatten(
        self,
        chains_descs: Optional[List[Tuple[str, str]]] = None,
        inter_chain_index_extra_offset: int = 1,
    ) -> None:
        """
        concats each feature from multiple chains into a single sequence.
        This is useful, for example, before supplying it to a model, which expects a tensor that contains info on the complex (or complex subset, e.g. pair)
        """

        concat_seq = []
        concat_feats = defaultdict(list)
        next_start_residue_index = 0
        if chains_descs is None:
            chains_descs = [desc for desc in self.chains_data.keys()]

        self.flattened_chain_parts = []

        for i, chain_desc in enumerate(chains_descs):
            seq = self.chains_data[chain_desc]["aa_sequence_str"]
            feats = self.chains_data[chain_desc]
            concat_seq.append(seq)
            for k, d in feats.items():
                concat_feats[k].append(d)

            length = feats["aatype"].shape[0]
            # print(f'debug: {i} {chain_desc} length={length} ')
            concat_feats["residue_index"].append(
                torch.arange(length) + next_start_residue_index
            )
            concat_feats["chain_index"].append(torch.full((length,), fill_value=i))

            next_start_residue_index += length + inter_chain_index_extra_offset

            ###
            prev_real_end = (
                self.flattened_chain_parts[-1][1]
                if len(self.flattened_chain_parts) > 0
                else 0
            )
            self.flattened_chain_parts += [(prev_real_end, prev_real_end + length)]

        for k, d in concat_feats.items():
            if isinstance(d[0], str):
                concat_elem = ",".join(d)
            else:
                concat_elem = torch.concat(d, dim=0)
                # print(f'flatten - post concat: {k} shape={concat_elem.shape}')
            self.flattened_data[k] = concat_elem

        self.flattened_data["aa_sequence_str"] = "".join(concat_seq)

        self.chains_descs_for_flatten = chains_descs
        # cumsums = np.cumsum([d['aatype'].shape[0] for k,d in self.chains_data.items()])

    def spatial_crop(
        self,
        *,
        chains_descs: Optional[List[Tuple[str, str]]] = None,
        crop_size: int = 256,
        distance_threshold: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        """
        Spatial crop of a pair of chains which favors interacting residues.
        Note - you must call "flatten" (with only two chain descriptor) prior to calling this method.

        The code is heavily influenced from the spatial crop done in RF2
        """
        if chains_descs is None:
            chains_descs = [desc for desc in self.chains_data.keys()]

        carbo_alpha_atom_index = 1

        # randomize the order of the chains, the FIRST chain will be considered "anchor" and the method will favor
        # adding residues from itself and interacting residues from other chains as well
        # so, for example, if it created random chains order [3,1,0,2,4] it will favor adding residues which are part of cross-chain interaction in 3-1, 3-0, 3-2, 3-4
        # and will NOT favor adding residues pairs that are part of, e.g. 2-4 interaction  ("2-4" means interaction between chain 2 and chain 4)
        chains_order = np.random.permutation(len(chains_descs))
        # print('DEBUG! return to random permutation!')
        # chains_order = np.arange(len(chains_lengths))

        chains_descs = [chains_descs[i] for i in chains_order]

        self.flatten(
            chains_descs=chains_descs,
        )

        # xyz = []
        # mask = []
        # for chain_index in chains_order:
        #     start, end = self.flattened_chain_parts[chain_index]
        #     # print(chain_index, "->", end - start)
        #     xyz += [orig_xyz[start:end]]
        #     mask += [orig_mask[start:end]]
        # xyz = torch.cat(xyz)
        # mask = torch.cat(mask)

        xyz = self.flattened_data["atom14_gt_positions"]
        mask = self.flattened_data["atom14_gt_exists"]

        first_chain_part = self.flattened_chain_parts[chains_order[0]]
        first_chain_length = first_chain_part[1] - first_chain_part[0]

        cond = (
            torch.cdist(
                xyz[:first_chain_length, carbo_alpha_atom_index],
                xyz[first_chain_length:, 1],
                p=2,
            )
            < distance_threshold
        )
        # only keep residue for which both ground truth masks show it's legit (was actually experimentally determined)
        cond = torch.logical_and(
            cond,
            mask[:first_chain_length, None, carbo_alpha_atom_index]
            * mask[None, first_chain_length:, carbo_alpha_atom_index],
        )
        i, j = torch.where(cond)
        ifaces = torch.cat([i, j + first_chain_length])
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

    def spatial_crop_supporting_only_pair(
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

        xyz = self.flattened_data["atom14_gt_positions"]
        mask = self.flattened_data["atom14_gt_exists"]

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

        self.flattened_data["aa_sequence_str"] = "".join(
            np.array(list(self.flattened_data["aa_sequence_str"]))[selection].tolist()
        )

        for k, d in self.flattened_data.items():
            if k in ["resolution", "pdb_id", "chain_id", "aa_sequence_str"]:
                continue
            self.flattened_data[k] = d[selection]

    def save_flattened_to_pdb(self, out_pdb_filename: str) -> None:
        flexible_save_pdb_file(
            save_path=out_pdb_filename,
            xyz=self.flattened_data[
                "atom14_gt_positions"
            ],  # note - it's "flattened" in the sense of data originating from multiple chains, it still has [residues, 14, 3] shape!
            sequence=self.flattened_data["aatype"],
            residues_mask=self.flattened_data["atom14_gt_exists"].max(dim=-1)[0],
        )

    def has_chains_pair_with_identical_sequence(self, verbose: bool = True) -> bool:
        for comb in combinations(self.chains_data.keys(), 2):
            chain_1_desc = comb[0]
            chain_2_desc = comb[1]

            if (
                self.chains_data[chain_1_desc]["aa_sequence_str"]
                == self.chains_data[chain_2_desc]["aa_sequence_str"]
            ):
                if verbose:
                    print(
                        f"Found chain pair with coordinates info and identical sequences: {comb}"
                    )
                return True

        return False

    def remove_duplicates(self, method: str = "coordinates") -> None:
        """
        Removes duplicate chains (keeps only one from the duplicates)

        Args:
            method: use 'coordinates' to remove duplicates based on 3d coordinates
                    use 'sequence' to remove duplicates based on 1d amino-acid residues sequence
        """
        assert False, "not implemented yet"
        assert method in ["coordinates", "sequence"]

        for comb in combinations(self.chains_data.keys(), 2):
            print(comb)
            if (
                self.chains_data[comb[0]]["atom14_gt_positions"].shape
                != self.chains_data[comb[1]]["atom14_gt_positions"].shape
            ):
                continue
            if (
                self.chains_data[comb[0]]["atom14_gt_positions"]
                == self.chains_data[comb[1]]["atom14_gt_positions"]
            ).all():
                print(f"{comb[0]} and {comb[1]} are the same based on method={method}")

    def calculate_chains_interaction_info(
        self,
        *,
        distance_threshold: float = 10.0,
        min_interacting_residues_count: int = 4,
        minimal_chain_length: int = 10,
        assume_not_interacting_if_from_different_pdb_ids: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[Tuple[Tuple[str, str], str]]]:
        """
        returns a a dictionary mapping interaction type
        (one of:
            'interacting_pairs',
            'too_few_residues_interacting_pairs',
            non_interacting_pairs'
            )
            into a list of non-redundant pair tuples, e.g.:
                [(('7vux', 'A'), ('7vux', 'H')),
                (('7vux', 'A'), ('7vux', 'L')),
                (('7vux', 'H'), ('7vux', 'L'))]

        It is based on the following heuristics:
        1. If a chain pair pass some thresholds related to the "amount" of interaction, the pair is added to 'interacting_pairs'
        2. If a chain pair has at least one residues pair that are in proximity, but not enough residues are interacting, the pair is added to 'too_few_residues_interacting_pairs''
        3. If there is more than one pair of chains that have known structure and IDENTICAL sequence, no 'non_interacting_pairs' are outputted.
            The rationale for that is that is symmetry exists, the fact that two chains are not seen interacting in the PDB, it does not mean they do not interact.
            As an extreme example see this magnificant Homo 1356-mer : https://www.rcsb.org/structure/3j3q
           ELSE:
               if 0 residue pairs pass the defined thresholds, then the pair is added to 'non_interacting_pairs'


        Args:
            distance_threshold: the maximum distance that 2 residues are considered interacting
            min_interacting_residues_count: minimal amount of interacting residues to decide that two chains are interacting
            assume_not_interacting_if_from_different_pdb_ids: if two chains arrive from different PDBs skip check and assume that they are not interacting.
                the default is True
                this is because this is likely to be used for (fake) negative pairs
            verbose:
                printing amount

        Returns:
            a dictionary containing both 'interacting_pairs' and 'non_interacting_pairs' entries, each containing
                a list of elements, each element is a tuple with two chains descriptors, describing the interacting chains pair
        """
        interacting_pairs = []
        too_few_residues_interacting_pairs = []
        non_interacting_pairs = []

        likely_oligomer = self.has_chains_pair_with_identical_sequence()

        for comb in combinations(self.chains_data.keys(), 2):
            chain_1_desc = comb[0]
            chain_1_pdb_id, chain_1_chain_id = chain_1_desc

            chain_2_desc = comb[1]
            chain_2_pdb_id, chain_2_chain_id = chain_2_desc

            if assume_not_interacting_if_from_different_pdb_ids:
                if chain_1_pdb_id != chain_2_pdb_id:
                    if verbose:
                        print(
                            "Not same pdb_id so assuming a non-interacting chain pair"
                        )
                    non_interacting_pairs.append(comb)
                    continue
            print(comb)
            if (
                self.chains_data[chain_1_desc]["atom14_gt_positions"].shape[0]
                < minimal_chain_length
            ):
                continue  # one of the elements is too small, skipping
            if (
                self.chains_data[chain_2_desc]["atom14_gt_positions"].shape[0]
                < minimal_chain_length
            ):
                continue  # one of the elements is too small, skipping

            interacting_residues_count = calculate_number_of_interacting_residues(
                xyz_1=self.chains_data[chain_1_desc]["atom14_gt_positions"],
                mask_1=self.chains_data[chain_1_desc]["atom14_gt_exists"],
                #
                xyz_2=self.chains_data[chain_2_desc]["atom14_gt_positions"],
                mask_2=self.chains_data[chain_2_desc]["atom14_gt_exists"],
                #
                distance_threshold=distance_threshold,
            )

            if interacting_residues_count == 0:
                if not likely_oligomer:
                    if verbose:
                        print(f"chains {comb[0]} and {comb[1]} are NOT interacting")
                    non_interacting_pairs.append(comb)
            elif interacting_residues_count < min_interacting_residues_count:
                if not likely_oligomer:
                    if verbose:
                        print(
                            f"chains {chain_1_desc} and {chain_2_desc} have some interaction, but it's below the defined min_interacting_residues_count={min_interacting_residues_count} threshold"
                        )
                    too_few_residues_interacting_pairs.append(comb)
            elif interacting_residues_count >= min_interacting_residues_count:
                if verbose:
                    print(f"chains {chain_1_desc} and {chain_2_desc} are interacting")
                interacting_pairs.append(comb)

        if verbose:
            print(
                f"total interacting_pairs={len(interacting_pairs)} too_few_residues_interacting_pairs={len(too_few_residues_interacting_pairs)} non_interacting_pairs={len(non_interacting_pairs)}"
            )

        return dict(
            interacting_pairs=interacting_pairs,
            too_few_residues_interacting_pairs=too_few_residues_interacting_pairs,
            non_interacting_pairs=non_interacting_pairs,
        )


def calculate_number_of_interacting_residues(
    *,
    xyz_1: torch.Tensor,
    mask_1: torch.Tensor,
    xyz_2: torch.Tensor,
    mask_2: torch.Tensor,
    distance_threshold: float,
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

    if (found_interacting_residues_count / cond.numel()) > 0.9:
        raise Exception(
            "A suspicious case in which more than 90% of the residues pairs are interacting, possibly the same chain twice in same position?"
        )
    return found_interacting_residues_count


if __name__ == "__main__":
    comp = ProteinComplex()

    chain_ids = None

    # pdb_id = "7vux"

    # pdb_id = "1fvm"
    # pdb_id = "2ohi"

    # pdb_id = "3j3q"
    # chain_ids = ['gG','j1']

    # pdb_id = "6enu"
    # pdb_id = "1A2W" # Homo 2-mer
    # pdb_id = "1a0r" # Homo 2-mer
    # pdb_id = "1xbp" # had RNA
    # pdb_id = "4r4f" # antibody + target + peptide interacting with the target (HIV related)
    # pdb_id = "7vux" #ananas, for some reason, shows symmetry group C2 - is it because it "knows" that antibodies have double of each chain type? or just wrong?
    # pdb_id = "2ZS0" #symmetry is shown on PDB website, but not in what we load or in what pyMOL shows be default - negative-pairs may be wrong!
    # pdb_id = "3idx" #Ab with target - looks like no direct contact between light chain and the target
    # pdb_id = "3vxn" # 3 chains, one is a tiny peptide (10 residues long) - shown by default as just backbone or something like that in pyMOL
    # pdb_id = "3qt1" # complex with multiple parts - RNA polymerase II variant containing A Chimeric RPB9-C11 subunit
    pdb_id = "4hna"

    comp.add(
        pdb_id,
        chain_ids=chain_ids,
    )

    # comp.remove_duplicates(method="coordinates")

    comp.calculate_chains_interaction_info()
    # comp.flatten()

    for i in range(10):
        crop_size = 256
        comp.spatial_crop(crop_size=crop_size)
        comp.save_flattened_to_pdb(
            f"./spatial_crop_{pdb_id}_residues_{crop_size}_try_{i}.pdb"
        )
