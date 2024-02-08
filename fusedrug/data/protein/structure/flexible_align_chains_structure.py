from jsonargparse import CLI
from typing import List, Union, Dict, Tuple
from Bio import Align
from tiny_openfold.utils.superimposition import superimpose

# from fusedrug.data.protein.structure.protein_complex import ProteinComplex
from fusedrug.data.protein.structure.structure_io import (
    load_pdb_chain_features,
    protein_utils,
    # flexible_save_pdb_file,
    save_structure_file,
)
import numpy as np
from warnings import warn


def flexible_align_chains_structure(
    dynamic_ordered_chains: Union[List[Tuple], str],
    apply_rigid_transformation_to_dynamic_chain_ids: Union[List[Tuple], str],
    static_ordered_chains: Union[List[Tuple], str],
    output_pdb_filename_extentionless: str,
    minimal_matching_sequence_level_chunk: int = 8,
    ###chain_id_type:str = "author_assigned",
) -> None:
    """
    Finds and applies a rigid transformation to align between chains (or sets of chains)
    Searches first for sequence level alignment, and then uses the matching subset to find the rigid transformation

    IMPORTANT: if you provide multiple chains, the order matters and should be consistent with the order in static_ordered_chains
     otherwise you might get nonsensical alignment !

    Args:

        dynamic_ordered_chains: the chains from `pdb_dynamic` that we want to move.
            either a list, for example: [ ('7vux', 'H'), ('/some/path/blah.pdb','N'), ...] #each tuple is [pdb id or filename, chain_id]
            or a string, for example: "7vux^H@/some/path/blah.pdb^N   #^ seprates between the different tuples and ^ separates between the tuple elements
            IMPORTANT: if you provide multiple, the order matters and should be consistent with the order in static_ordered_chains otherwise you might get nonsensical alignment !

        apply_rigid_transformation_to_dynamic_chain_ids:
            either a list, for example: [ ('7vux', 'H'), ('/some/path/blah.pdb','N'), ...] #each tuple is [pdb id or filename, chain_id]
            or a string, for example: "7vux^H@/some/path/blah.pdb^N   #^ seprates between the different tuples and ^ separates between the tuple elements

            the found transformation will be applied to these changed, and these chains will be stored in the location that `output_pdb_filename` defines
            It can be identical to dynamic_ordered_chains, or it can be different.
            A use case in which making it different can make sense is to align heavy+light chains of a candidate antibody to the heavy chain of a reference


        static_ordered_chains: the chains from `pdb_static` that we want to align the dynamic part to.
            IMPORTANT: if you provide multiple, the order matters and should be consistent with the order in dynamic_ordered_chains otherwise you might get nonsensical alignment !
            either a list, for example: [ ('7vux', 'H'), ('/some/path/blah.pdb','N'), ...] #each tuple is [pdb id or filename, chain_id]
            or a string, for example: "7vux^H@/some/path/blah.pdb^N   #^ seprates between the different tuples and ^ separates between the tuple elements

        output_pdb_filename: the chains from pdb_dynamic that are selected and moved will be saved into this pdb file

        minimal_matching_sequence_level_chunk: the minimal size in which a chunk of matching aligned sequence will be used for the 3d alignment.
            The motivation for this is to avoid "nonsense" matches scattered all over the sequence, resulting in (very) suboptimal alignment

    """

    dynamic_ordered_chains = _to_list(dynamic_ordered_chains)
    apply_rigid_transformation_to_dynamic_chain_ids = _to_list(
        apply_rigid_transformation_to_dynamic_chain_ids
    )
    static_ordered_chains = _to_list(static_ordered_chains)

    dynamic_chains: Dict[str, protein_utils.Protein] = {}
    for pdb_file, chain_id in dynamic_ordered_chains:
        dynamic_chains[chain_id] = load_pdb_chain_features(pdb_file, chain_id)

    apply_rigid_on_dynamic_chains: Dict[str, protein_utils.Protein] = {}
    for pdb_file, chain_id in apply_rigid_transformation_to_dynamic_chain_ids:
        apply_rigid_on_dynamic_chains[chain_id] = load_pdb_chain_features(
            pdb_file, chain_id
        )

    static_chains: Dict[str, protein_utils.Protein] = {}
    for pdb_file, chain_id in static_ordered_chains:
        static_chains[chain_id] = load_pdb_chain_features(pdb_file, chain_id)

    attributes = [
        "atom14_gt_positions",
        "atom14_gt_exists",
        "aasequence_str",
        "aatype",
        # "residue_index",
    ]

    # concatanate
    dynamic_concat = {
        attribute: _concat_elements_from_dict(dynamic_chains, attribute)
        for attribute in attributes
    }

    static_concat = {
        attribute: _concat_elements_from_dict(static_chains, attribute)
        for attribute in attributes
    }

    # calculate alignment in sequence space
    dynamic_indices, static_indices = get_alignment_indices(
        dynamic_concat["aasequence_str"],
        static_concat["aasequence_str"],
        minimal_matching_sequence_level_chunk=minimal_matching_sequence_level_chunk,
    )

    # dynamic_indices = dynamic_indices[:50]
    # static_indices = static_indices[:50]

    # extract seq-level matching atoms coordinates
    dynamic_matching = _apply_indices(dynamic_concat, dynamic_indices)
    static_matching = _apply_indices(static_concat, static_indices)

    # calculate the rigid transformation to translate from the starting pose of the dynamic onto the static

    combined_mask = np.logical_and(
        dynamic_matching["atom14_gt_exists"].astype(bool),
        static_matching["atom14_gt_exists"].astype(bool),
    )
    # orig_atom_pos_shape = dynamic_matching["atom14_gt_positions"].shape
    _, rmsd, rot_matrix, trans_matrix = superimpose(
        static_matching["atom14_gt_positions"].reshape(-1, 3),
        dynamic_matching["atom14_gt_positions"].reshape(-1, 3),
        combined_mask.reshape(-1),
        verbose=True,
    )

    assert rot_matrix.shape == (1, 3, 3)
    rot_matrix = rot_matrix[0]

    assert trans_matrix.shape == (1, 3)
    trans_matrix = trans_matrix[0]

    assert len(rmsd.shape) == 0

    if rmsd > 6.0:
        warn(
            f"flexible_align_chains_structure: got a pretty high rmsd={rmsd} in alignment. Either the structures are very different or the sequence alignment was suboptimal."
        )

    # apply the rigid transformation on the chains described in `apply_rigid_transformation_to_dynamic_chain_ids` argument
    transformed_dynamic_atom_pos = {}
    for chain_id, prot in apply_rigid_on_dynamic_chains.items():
        _atom_pos_orig_shape = prot["atom14_gt_positions"].shape
        _atom_pos_flat = prot["atom14_gt_positions"].reshape(-1, 3)
        _atom_pos_flat_transformed = np.dot(_atom_pos_flat, rot_matrix) + trans_matrix
        _atom_pos_transformed = _atom_pos_flat_transformed.reshape(
            *_atom_pos_orig_shape
        )
        transformed_dynamic_atom_pos[chain_id] = _atom_pos_transformed

        # transformed_dynamic_atom_pos[chain_id] = prot['atom14_gt_positions']

    save_structure_file(
        output_filename_extensionless=output_pdb_filename_extentionless,
        pdb_id="unknown",
        chain_to_atom14=transformed_dynamic_atom_pos,
        chain_to_aa_str_seq={
            chain_id: apply_rigid_on_dynamic_chains[chain_id]["aasequence_str"]
            for chain_id in apply_rigid_on_dynamic_chains.keys()
        },
        chain_to_aa_index_seq={
            chain_id: apply_rigid_on_dynamic_chains[chain_id]["aatype"]
            for chain_id in apply_rigid_on_dynamic_chains.keys()
        },
        save_cif=False,
        mask=None,  # TODO: check
    )

    # apply_on_atom_pos = apply_rigid_on_dynamic_concat['atom_positions']
    # apply_on_atom_pos_flat = apply_on_atom_pos.reshape(-1,3)

    # transformed_flat = np.dot(apply_on_atom_pos_flat, rot_matrix) + trans_matrix
    # transformed = transformed_flat.reshape()

    # superimposed = superimposed_flat.reshape(*orig_atom_pos_shape)

    # flexible_save_pdb_file(
    #     xyz: torch.Tensor,
    #     sequence: torch.Tensor,
    #     residues_mask: torch.Tensor,
    #     save_path: str,
    #     model: int = 0,
    #     init_chain: str = "A",
    #     only_save_backbone: bool = False,
    #     b_factors: Optional[torch.Tensor] = None,

    dynamic_matching = dynamic_matching
    static_matching = static_matching


def _apply_indices(x: Dict, indices: np.ndarray) -> Tuple[str, np.ndarray]:
    ans = {}
    for k, d in x.items():
        if isinstance(d, str):
            ans[k] = "".join(d[i] for i in indices)
        else:
            ans[k] = d[indices]
    return ans


def get_alignment_indices(
    target: str, query: str, minimal_matching_sequence_level_chunk: int
) -> Tuple[np.ndarray, np.ndarray]:
    aligner = Align.PairwiseAligner()

    ###https://biopython.org/docs/1.75/api/Bio.Align.html#Bio.Align.PairwiseAlignment
    ### https://github.com/biopython/biopython/blob/master/Bio/Align/substitution_matrices/data/README.txt
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

    alignments = aligner.align(target, query)
    alignment = alignments[0]

    target_indices = []
    query_indices = []

    for (target_start, target_end), (query_start, query_end) in zip(*alignment.aligned):
        if target_end - target_start >= minimal_matching_sequence_level_chunk:
            target_indices.extend(list(range(target_start, target_end)))
            query_indices.extend(list(range(query_start, query_end)))

    target_indices = np.array(target_indices)
    query_indices = np.array(query_indices)

    return target_indices, query_indices


def _concat_elements_from_dict(
    input_dict: Dict, attribute: str
) -> Union[str, np.ndarray]:
    # elements = [getattr(p, attribute) for (_, p) in input_dict.items()]
    elements = [p[attribute] for (_, p) in input_dict.items()]
    ans = _concat_elements(elements)
    return ans


def _concat_elements(elements: List[Union[str, np.ndarray]]) -> Union[str, np.ndarray]:
    assert len(elements) > 0
    if isinstance(elements[0], str):
        return "".join(elements)

    ans = np.concatenate(elements, axis=0)
    return ans


def _to_list(x: Union[str, List]) -> List:
    if isinstance(x, str):
        x = x.split("@")
        x = [tuple(curr.split("^")) for curr in x]
    assert isinstance(x, list)

    for element in x:
        assert len(element) == 2
    return x


if __name__ == "__main__":
    CLI(flexible_align_chains_structure)


####usage examples

"""
python $MY_GIT_REPOS/fuse-drug/fusedrug/data/protein/structure/flexible_align_chains_structure.py  \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^A \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^A@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^B \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/PD1_7VUX_antibody_heavy_chain_from_equalized_reference_complex.pdb^H \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/output_aligned_candidate_antibody_dimer_only_H_for_alignment



python $MY_GIT_REPOS/fuse-drug/fusedrug/data/protein/structure/flexible_align_chains_structure.py  \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^A@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^B \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^A@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^B \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/PD1_7VUX_antibody_heavy_chain_from_equalized_reference_complex.pdb^H@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/PD1_7VUX_antibody_light_chain_from_equalized_reference_complex.pdb^L \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/output_aligned_candidate_antibody_dimer_used_both_LH_for_alignment




"""
