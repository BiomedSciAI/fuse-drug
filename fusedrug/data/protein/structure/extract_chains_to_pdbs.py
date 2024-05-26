from jsonargparse import CLI
from fusedrug.data.protein.structure.structure_io import (
    load_pdb_chain_features,
    save_structure_file,
)
from typing import Optional


def main(
    *,
    input_pdb_path: str,
    orig_name_chains_to_extract: str,
    output_pdb_path_extensionless: str,
    output_chain_ids_to_extract: Optional[str] = None,
) -> None:
    """

    Takes an input PDB files and splits it into separate files, one per describe chain, allowing to rename the chains if desired

    Args:
    input_pdb_path:
    input_chain_ids_to_extract: '_' separated chain ids
    output_chain_ids_to_extract: '_' separated chain ids
        if not provided, will keep original chain ids

    """

    orig_name_chains_to_extract = orig_name_chains_to_extract.split("_")
    if output_chain_ids_to_extract is None:
        output_chain_ids_to_extract = orig_name_chains_to_extract.split("_")
    else:
        output_chain_ids_to_extract = output_chain_ids_to_extract.split("_")

    assert len(orig_name_chains_to_extract) > 0
    assert len(orig_name_chains_to_extract) == len(output_chain_ids_to_extract)
    assert len(orig_name_chains_to_extract[0]) == 1

    loaded_chains = {}
    for orig_chain_id in orig_name_chains_to_extract:
        loaded_chains[orig_chain_id] = load_pdb_chain_features(
            input_pdb_path, orig_chain_id
        )

    mapping = dict(zip(orig_name_chains_to_extract, output_chain_ids_to_extract))

    loaded_chains_mapped = {
        mapping[chain_id]: data for (chain_id, data) in loaded_chains.items()
    }

    save_structure_file(
        output_filename_extensionless=output_pdb_path_extensionless,
        pdb_id="unknown",
        chain_to_atom14={
            chain_id: data["atom14_gt_positions"]
            for (chain_id, data) in loaded_chains_mapped.items()
        },
        chain_to_aa_str_seq={
            chain_id: data["aasequence_str"]
            for (chain_id, data) in loaded_chains_mapped.items()
        },
        chain_to_aa_index_seq={
            chain_id: data["aatype"]
            for (chain_id, data) in loaded_chains_mapped.items()
        },
        save_cif=False,
        mask=None,  # TODO: check
    )


if __name__ == "__main__":
    CLI(main)
