from jsonargparse import CLI
from fusedrug.data.protein.structure.structure_io import (
    load_pdb_chain_features,
    save_structure_file,
)
from typing import Optional, Sequence
from os.path import isfile, join, dirname, basename
import os
import sys
import subprocess
import threading


def main(
    *,
    input_pdb_path: str,
    input_scfv_chain_id: str,
    output_pdb_path_extensionless: str,
    output_heavy_chain_id: Optional[str] = "H",
    output_light_chain_id: Optional[str] = "L",
    passthrough_chains: Optional[str] = None,
    cleanup_temp_files: bool = True,
) -> None:
    """

    Takes an input PDB file and allows to split scfv within it to 2 separate chains.
    This is useful for modifying such PDB to be used in follow up steps that assume such separate chains for heavy and light chain.

    It allows also to "passthrough" additional chains to maintain a "full" PDB.

    Args:
    input_pdb_path:

    passthrough_chains: optional, will be "pass through", '_' separated if you want multiple

    """

    if passthrough_chains is not None:
        passthrough_chains = passthrough_chains.split("_")

    loaded_scfv = load_pdb_chain_features(input_pdb_path, input_scfv_chain_id)

    scfv_seq = loaded_scfv["aasequence_str"]

    safety = f"_{os.getpid()}_{threading.get_ident()}"

    scfv_sequence_filename = join(
        dirname(input_pdb_path),
        f"sequence_info_{input_scfv_chain_id}_"
        + basename(input_pdb_path)
        + safety
        + ".txt",
    )

    if not isfile(scfv_sequence_filename):
        with open(scfv_sequence_filename, "wt") as f:
            f.write(f">scfv_{input_scfv_chain_id}:...\n{scfv_seq}\n")

    # run anarci:
    anarci_executable = join(dirname(sys.executable), "ANARCI")
    if not isfile(anarci_executable):
        raise Exception(
            f"ANARCI binary not found in {dirname(sys.executable)}. check installation. You can install it in your env like this: conda install -c bioconda abnumber"
        )

    anarci_output_filename = join(
        dirname(input_pdb_path),
        f"anarci_output_{input_scfv_chain_id}_"
        + basename(input_pdb_path)
        + safety
        + ".txt",
    )

    if not isfile(anarci_output_filename):
        subprocess.run(
            [
                anarci_executable,
                "-i",
                scfv_sequence_filename,
                "-o",
                anarci_output_filename,
            ]
        )
    # parse anarci outputs  and obtain separate heavy and light chains:
    heavy_chain, light_chain = split_heavy_light_chain_from_anarci_output(
        anarci_output_filename
    )

    if 0 == len(heavy_chain):
        raise Exception("ANARCI could not find the heavy chain domain")

    if 0 == len(light_chain):
        raise Exception("ANARCI could not find the light chain domain")

    # cleanup
    if cleanup_temp_files:
        os.remove(scfv_sequence_filename)
        os.remove(anarci_output_filename)

    heavy_start = scfv_seq.find(heavy_chain)
    assert heavy_start >= 0

    light_start = scfv_seq.find(light_chain)
    assert light_start >= 0

    chain_to_atom14 = {
        output_heavy_chain_id: loaded_scfv["atom14_gt_positions"][
            heavy_start : heavy_start + len(heavy_chain)
        ],
        output_light_chain_id: loaded_scfv["atom14_gt_positions"][
            light_start : light_start + len(light_chain)
        ],
    }

    chain_to_aa_str_seq = {
        output_heavy_chain_id: loaded_scfv["aasequence_str"][
            heavy_start : heavy_start + len(heavy_chain)
        ],
        output_light_chain_id: loaded_scfv["aasequence_str"][
            light_start : light_start + len(light_chain)
        ],
    }

    chain_to_aa_index_seq = {
        output_heavy_chain_id: loaded_scfv["aatype"][
            heavy_start : heavy_start + len(heavy_chain)
        ],
        output_light_chain_id: loaded_scfv["aatype"][
            light_start : light_start + len(light_chain)
        ],
    }

    if passthrough_chains is not None:
        for chain_id in passthrough_chains:
            curr_loaded_chain_data = load_pdb_chain_features(input_pdb_path, chain_id)

            chain_to_atom14[chain_id] = curr_loaded_chain_data["atom14_gt_positions"]
            chain_to_aa_str_seq[chain_id] = curr_loaded_chain_data["aasequence_str"]
            chain_to_aa_index_seq[chain_id] = curr_loaded_chain_data["aatype"]

    saved_files = save_structure_file(
        output_filename_extensionless=output_pdb_path_extensionless,
        pdb_id="unknown",
        chain_to_atom14=chain_to_atom14,
        chain_to_aa_str_seq=chain_to_aa_str_seq,
        chain_to_aa_index_seq=chain_to_aa_index_seq,
        save_cif=False,
        mask=None,  # TODO: check
    )

    assert len(saved_files) == 1
    print(f"saved {saved_files}")

    return saved_files[0]


def split_heavy_light_chain_from_anarci_output(filename: str) -> list[Sequence[str]]:
    # parses ANARCI output on a fasta file of a single heavy and light chain domains
    heavy_chain = []
    light_chain = []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            else:
                parts = line.split()
                residue = parts[-1]
                if residue == "-":
                    continue
                if line.startswith("H"):
                    heavy_chain.append(residue)
                elif line.startswith("L"):
                    light_chain.append(residue)

    heavy_chain = "".join(heavy_chain)
    light_chain = "".join(light_chain)

    return heavy_chain, light_chain


if __name__ == "__main__":
    CLI(main, as_positional=False)
