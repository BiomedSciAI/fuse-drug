from jsonargparse import CLI
from typing import List, Union, Dict, Tuple

# from fusedrug.data.protein.structure.protein_complex import ProteinComplex
from fusedrug.data.protein.structure.structure_io import (
    pdb_to_openfold_protein,
    protein_utils,
)
import numpy as np


def flexible_align_chains_structure(
    dynamic_ordered_chains: Union[List[Tuple], str],
    apply_rigid_transformation_to_dynamic_chain_ids: Union[List[Tuple], str],
    static_ordered_chains: Union[List[Tuple], str],
    output_pdb_filename: str,
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

        chain_id_type: either "author_assigned" (default) or "pdb_assigned"

    """

    dynamic_ordered_chains = _to_list(dynamic_ordered_chains)
    apply_rigid_transformation_to_dynamic_chain_ids = _to_list(
        apply_rigid_transformation_to_dynamic_chain_ids
    )
    static_ordered_chains = _to_list(static_ordered_chains)

    dynamic_chains: Dict[str, protein_utils.Protein] = {}
    for pdb_file, chain_id in dynamic_ordered_chains:
        dynamic_chains[chain_id] = pdb_to_openfold_protein(pdb_file, chain_id)

    static_chains: Dict[str, protein_utils.Protein] = {}
    for pdb_file, chain_id in static_ordered_chains:
        static_chains[chain_id] = pdb_to_openfold_protein(pdb_file, chain_id)

    attributes = [
        "atom_positions",
        "aasequence_str",
        "aatype",
        "atom_mask",
        "residue_index",
    ]

    # concatanate
    dynamic_concat = {
        chain_id: {
            attribute: _concat_elements_from_dict(dynamic_chains, attribute)
            for attribute in attributes
        }
        for chain_id in dynamic_chains.keys()
    }
    static_concat = {
        chain_id: {
            attribute: _concat_elements_from_dict(static_chains, attribute)
            for attribute in attributes
        }
        for chain_id in static_chains.keys()
    }

    dynamic_concat = dynamic_concat
    static_concat = static_concat


def _concat_elements_from_dict(
    input_dict: Dict, attribute: str
) -> Union[str, np.ndarray]:
    elements = [getattr(p, attribute) for (_, p) in input_dict.items()]
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
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb \
    "A" \
    "A,B" \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/PD1_7VUX_antibody_heavy_chain_from_equalized_reference_complex.pdb \
    'H' \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/output_aligned_candidate_antibody_dimer_used_only_H_for_alignment.pdb



python $MY_GIT_REPOS/fuse-drug/fusedrug/data/protein/structure/flexible_align_chains_structure.py  \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^A@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^B \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^A@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/antibody_dimer_candidate_with_indels_NOT_aligned.pdb^B \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/PD1_7VUX_antibody_heavy_chain_from_equalized_reference_complex.pdb^H@$MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/PD1_7VUX_antibody_light_chain_from_equalized_reference_complex.pdb^L \
    $MY_GIT_REPOS/fuse-drug/fusedrug/tests_data/structure/protein/flexible_align/output_aligned_candidate_antibody_dimer_used_both_LH_for_alignment.pdb




"""
