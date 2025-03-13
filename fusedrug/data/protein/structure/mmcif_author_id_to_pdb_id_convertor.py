from fusedrug.data.protein.structure.structure_io import load_protein_structure_features
from Bio import PDB
import io
from typing import Any, Mapping, Sequence, Dict

from tiny_openfold.data.mmcif_parsing import (
    mmcif_loop_to_list,
    mmcif_loop_to_dict,
    _handle_residue_id_duplication,
    _get_atom_site_list,
)


def get_author_assigned_pdb_id_to_pdb_assigned_pdb_id_convertor(
    *,
    mmcif_path: str,
    pdb_id: str,
) -> Dict:
    features, mmcif_object, mmcif2dict = load_protein_structure_features(
        pdb_id_or_filename=mmcif_path, pdb_id=pdb_id
    )
    convertion = mmcif_object.author_assigned_chain_id_to_pdb_assigned_chain_id
    return convertion


def _get_author_assigned_pdb_id_to_pdb_assigned_pdb_id_convertor_helper(
    *,
    mmcif_string: str,
    handle_residue_id_duplication: bool = False,
    quiet_parsing: bool = True,
) -> dict:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate
    Returns:
      A ParsingResult.
    """

    # errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=quiet_parsing)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure("", handle)
        _ = next(full_structure.get_models())  # get the first structure
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        # header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        _ = _get_all_chains(
            parsed_info=parsed_info,
            handle_residue_id_duplication=handle_residue_id_duplication,
        )

        mmcif_to_author_chain_id = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

    except Exception as e:
        print(e)
        # return None
        raise

    # flip the direction, getting "author assigned pdb id" to "pdb assigned pdb id" dictionary

    author_assigned_to_pdb_assigned = {
        c2: c1 for (c1, c2) in mmcif_to_author_chain_id.items()
    }

    return author_assigned_to_pdb_assigned


def _get_all_chains(
    *,
    parsed_info: Mapping[str, Any],
    handle_residue_id_duplication: bool = False,
) -> Sequence[str]:
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list("_entity_poly_seq.", parsed_info)

    if handle_residue_id_duplication:
        entity_poly_seqs = _handle_residue_id_duplication(
            entity_id="_entity_poly_seq.entity_id",
            residue_num="_entity_poly_seq.num",
            data=entity_poly_seqs,
            logic="keep_last",
        )

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins.
    _ = mmcif_loop_to_dict("_chem_comp.", "_chem_comp.id", parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list("_struct_asym.", parsed_info)

    # entity_to_mmcif_chains = collections.defaultdict(list)
    # for struct_asym in struct_asyms:
    #     chain_id = struct_asym["_struct_asym.id"]
    #     entity_id = struct_asym["_struct_asym.entity_id"]
    #     entity_to_mmcif_chains[entity_id].append(chain_id)

    all_seen_chain_ids = []
    for struct_asym in struct_asyms:
        chain_id = struct_asym["_struct_asym.id"]
        all_seen_chain_ids.append(chain_id)
