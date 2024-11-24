from typing import Optional, Dict, Union, List, Tuple
import gzip
import io
import os
import torch
from copy import deepcopy
import pathlib
from tqdm import trange
import numpy as np
from Bio.PDB import *  # noqa: F401, F403
from Bio.PDB import StructureBuilder
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser, Select
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio import PDB


from tiny_openfold.data import data_transforms
from tiny_openfold.utils.tensor_utils import tree_map
import tiny_openfold.np.protein as protein_utils
from tiny_openfold.np.residue_constants import restype_3to1
from tiny_openfold.data import (
    data_pipeline,
    mmcif_parsing,
)
from tiny_openfold.np import residue_constants as rc
from tiny_openfold.data.mmcif_parsing import MmcifObject

# from omegafold.utils.protein_utils import residue_constants as rc

from fusedrug.data.protein.structure.utils import (
    aa_sequence_from_aa_integers,
    get_structure_file_type,
    residx_to_3,
)

# https://bioinformatics.stackexchange.com/questions/14101/extract-residue-sequence-from-pdb-file-in-biopython-but-open-to-recommendation

# TODO: split this into pdb related and mmcif related functions


def save_structure_file(
    *,  # prevent positional args
    output_filename_extensionless: str,
    pdb_id: str,
    chain_to_atom14: Dict[str, torch.Tensor],
    # optional args
    chain_to_aa_str_seq: Optional[Dict[str, str]] = None,
    chain_to_aa_index_seq: Optional[Dict[str, torch.Tensor]] = None,
    save_pdb: bool = True,
    save_cif: bool = True,
    b_factors: Optional[Dict[str, torch.Tensor]] = None,
    reference_cif_filename: Optional[str] = None,
    mask: Optional[Dict[str, List]] = None,
    shorten_chain_ids_if_needed:bool = True,
) -> List[str]:
    """
    A helper function allowing to save single or multi chain structure into pdb and/or mmcif format.

    Args:
        output_filename_extensionless:str - the name of the output file, without extension. For example: /tmp/my_pred
        pdb_id:str - pdb_id of the structure
        chain_to_atom14:Dict[str,torch.Tensor] - a dictionary mapping from chain_id to atom14 (heavy atom) [..., 14, 3] tensor

        chain_to_aa_str_seq:Optional[Dict[str,str]] - a dictionary mapping from chain_id to amino acid string
        chain_to_aa_index_seq:Optional[Dict[str,torch.Tensor]] - a dictionary mapping from chain_id to residues (as integers tensors)
        save_pdb:bool - should it store pdb format
        save_cif - should it store mmCIF format (newer, and no length limits)
        b_factors -
        reference_cif_filename:Optional[str] - for mmCIF outputs you must provide an mmCIF reference file (you can use the ground truth one)
        mask: - an optional dictionary mapping chain_id to *residue-level* mask

    Returns:
        A list with paths for all saved files
    """
    assert save_pdb or save_cif
    assert len(chain_to_atom14) > 0
    if chain_to_aa_index_seq is not None:
        assert len(chain_to_aa_index_seq) == len(chain_to_atom14)
        sorted_chain_ids = sorted(list(chain_to_aa_index_seq.keys()))

    if chain_to_aa_str_seq is not None:
        assert len(chain_to_aa_str_seq) == len(chain_to_atom14)
        sorted_chain_ids = sorted(list(chain_to_aa_str_seq.keys()))

    assert (chain_to_aa_str_seq is not None) or (chain_to_aa_index_seq is not None)

    def fix_too_long_chain_name_if_needed(chain_id):
        ans = chain_id
        if len(chain_id) > 1:
            ans = chain_id[:1]
            print(
                f"WARNING: shortening too long chain_id from {chain_id} to {ans}"
            )
        return ans

    if shorten_chain_ids_if_needed:
        chain_to_atom14 = {fix_too_long_chain_name_if_needed(chain_id):val for (chain_id, val) in chain_to_atom14.items()}
        if b_factors is not None:
            chain_to_atom14 = {fix_too_long_chain_name_if_needed(chain_id):val for (chain_id, val) in b_factors.items()}
        chain_to_aa_index_seq = {fix_too_long_chain_name_if_needed(chain_id):val for (chain_id, val) in chain_to_aa_index_seq.items()}        

    all_saved_files = []

    if save_cif:
        if len(chain_to_aa_str_seq) > 1:
            if reference_cif_filename is not None:
                out_multimer_cif_filename = (
                    output_filename_extensionless + "_multimer_chain_ids_"
                )
                out_multimer_cif_filename += (
                    "_".join(sorted(list(chain_to_aa_str_seq.keys()))) + ".cif"
                )

                print(f"saving MULTImer cif file {out_multimer_cif_filename}")
                save_updated_mmcif_file(
                    reference_mmcif_file=reference_cif_filename,
                    pdb_id=pdb_id,
                    chain_to_aa_seq=chain_to_aa_str_seq,
                    chain_to_atom14=chain_to_atom14,
                    output_mmcif_filename=out_multimer_cif_filename,
                )

                all_saved_files.append(out_multimer_cif_filename)
            else:
                print(
                    f'not writing multimer file because "reference_cif_filename" was not provided for {reference_cif_filename}'
                )

    all_masks = {}
    for chain_id in sorted_chain_ids:
        pos_atom14 = chain_to_atom14[chain_id]
        if mask is not None:
            curr_mask = mask[chain_id]
        else:
            curr_mask = torch.full((pos_atom14.shape[0],), fill_value=True)

        all_masks[chain_id] = curr_mask

        # if save_pdb: #save individual pdb per chain
        #     out_pdb = output_filename_extensionless + "_chain_" + chain_id + ".pdb"
        #     print(f"Saving structure file to {out_pdb}")
          

        #     flexible_save_pdb_file(
        #         xyz=pos_atom14,
        #         b_factors=b_factors[chain_id]
        #         if b_factors is not None
        #         else None,  # torch.tensor([100.0] * pos_atom14.shape[0]),
        #         sequence=chain_to_aa_index_seq[chain_id],
        #         residues_mask=curr_mask,
        #         save_path=out_pdb,                
        #         model=0,
        #     )

        #     all_saved_files.append(out_pdb)

        if save_cif:
            out_cif = output_filename_extensionless + "_chain_" + chain_id + ".cif"
            if reference_cif_filename is not None:
                print(f"Saving structure file to {out_cif}")
                save_updated_mmcif_file(
                    reference_mmcif_file=reference_cif_filename,
                    pdb_id=pdb_id,
                    chain_to_aa_seq={chain_id: chain_to_aa_str_seq[chain_id]},
                    chain_to_atom14={chain_id: chain_to_atom14[chain_id]},
                    output_mmcif_filename=out_cif,
                )

                all_saved_files.append(out_cif)
            else:
                print(
                    f'not writing chain cif file because no "reference_cif_filename" was provided for {reference_cif_filename}'
                )
    
    
    if shorten_chain_ids_if_needed:
        all_masks = {fix_too_long_chain_name_if_needed(chain_id):val for (chain_id, val) in all_masks.items()}

    if save_pdb:
        #save a pdb with (potentially) multiple chains
        flexible_save_pdb_file(
            xyz=chain_to_atom14,
            b_factors=b_factors,            
            sequence=chain_to_aa_index_seq,
            residues_mask=all_masks,
            save_path=output_filename_extensionless,
            model=0,
        )

    all_saved_files.append(output_filename_extensionless)
    
    return all_saved_files


def load_protein_structure_features(
    pdb_id_or_filename: str,
    *,
    pdb_id: Optional[str] = None,
    chain_id: Optional[Union[Union[str, int], List[Union[str, int]]]] = None,
    chain_id_type: str = "author_assigned",
    device: str = "cpu",
    max_allowed_file_size_mbs: float = None,
) -> Union[Tuple[str, dict], None]:
    """
    Extracts ground truth features from a given pdb_id or filename.
    Note - only mmCIF is tested (using pdb will trigger an exception)

    pdb_id_or_filename: pdb_id (example: '7vux') or a full put to a file which may be .gz compressed (e.g. /some/path/to/7vux.pdb.gz)

    pdb_id: pdb id - for example '7vux'.  If you already provided pdb_id_or_filename as a pdb id (e.g. '7vux') you don't need to supply it

    chain_id: you have multiple options here:
        * a single character (example: 'A')
        * an integer (zero-based) and then the i-th chain in the *sorted* list of available chains will be used
        * A list of any combination of integers and singel characters, for example: ['A',2,'H']
            in this case the answer will be a dictionary mapping from chain_id name to the processed info
        * None - in this case all chains will be loaded, and a dictionary mapping chain_id names to processed info will be returned

    chain_id_type: one of the allowed options "author_assigned" or "pdb_assigned"
        "author_assigned" means that the provided chain_id is using the chain id that the original author who uploaded to PDB assigned to it.
        "pdb_assigned" means that the provided chain_d is using the chain id that PDB dataset assigned.

    device:
    """

    assert chain_id_type in ["author_assigned", "pdb_assigned"]
    assert isinstance(chain_id, (int, str, List)) or (chain_id is None)

    if is_pdb_id(pdb_id_or_filename):
        pdb_id_or_filename = pdb_id_or_filename.lower()
        pdb_id = pdb_id_or_filename
    else:
        if not is_pdb_id(pdb_id):
            raise Exception(
                "pdb_id_or_filename was deduced to be a path to a file, in such case you must provide pdb_id as well"
            )
        pdb_id = pdb_id.lower()

    native_structure_filename = get_mmcif_native_full_name(pdb_id_or_filename)

    if max_allowed_file_size_mbs is not None:
        if (
            os.path.getsize(native_structure_filename) / (10**6)
            > max_allowed_file_size_mbs
        ):
            print(
                f"file is too big for requested threshold of {max_allowed_file_size_mbs} mbs! file={native_structure_filename}"
            )
            return None

    return_dict = True
    if isinstance(chain_id, list):
        for curr in chain_id:
            assert isinstance(curr, (str, int))
        chain_ids = chain_id
    elif chain_id is None:
        chain_ids = None
    else:
        return_dict = False
        chain_ids = [chain_id]  # "listify"

    ans = {}

    structure_file_format = get_structure_file_type(native_structure_filename)
    if structure_file_format == "pdb":
        assert False, "not tested for a long time now"
        try:
            chains_names = get_available_chain_ids_in_pdb(native_structure_filename)
        except Exception as e:
            print(e)
            print(
                f"Had an issue with a protein, skipping it. Protein file:{native_structure_filename}"
            )
            return None
    elif structure_file_format == "cif":
        try:
            mmcif_object, mmcif_dict = parse_mmcif(
                native_structure_filename,
                unique_file_id=pdb_id,
                quiet_parsing=True,
                also_return_mmcif_dict=True,
            )
            chains_names = list(mmcif_object.chain_to_seqres.keys())
        except Exception as e:
            print(e)
            print(f"Had an issue reading {native_structure_filename}")
            return None

        if chain_ids is None:
            chain_ids = chains_names
    else:
        assert False

    if structure_file_format == "pdb":
        if chain_id_type == "pdb_assigned":
            raise Exception(
                "chain_id_type=pdb_assigned  is not supported yet for PDB, only for mmCIF"
            )
        gt_data = pdb_to_openfold_protein(native_structure_filename, chain_id=chain_id)

        gt_mmcif_feats = dict(
            aatype=gt_data.aatype,
            all_atom_positions=gt_data.atom_positions,
            all_atom_mask=gt_data.atom_mask,
        )

        gt_mmcif_feats = {k: torch.tensor(d) for (k, d) in gt_mmcif_feats.items()}

    elif structure_file_format == "cif":

        for chain_id in chain_ids:

            if isinstance(chain_id, int):
                chains_names = sorted(chains_names)
                if (chain_id < 0) or (chain_id >= len(chains_names)):
                    raise Exception(
                        f"chain_id(int)={chain_id} is out of bound for the options: {chains_names}"
                    )
                chain_id = chains_names[chain_id]  # taking the i-th chain

            if chain_id_type == "pdb_assigned":
                use_chain_id = (
                    mmcif_object.pdb_assigned_chain_id_to_author_assigned_chain_id[
                        chain_id
                    ]
                )  # convert from pdb assigned to author assigned chain id
            else:
                use_chain_id = chain_id

            gt_all_mmcif_feats = get_chain_data(mmcif_object, chain_id=use_chain_id)

            # move to device a selected subset
            gt_mmcif_feats = {
                k: gt_all_mmcif_feats[k]
                for k in [
                    "aatype",
                    "all_atom_positions",
                    "all_atom_mask",
                    "all_atom_bfactors",
                    "resolution",
                    "residue_index",
                    "chain_index",
                    "all_atom_bfactors",
                ]
            }

            to_tensor = lambda t: torch.tensor(np.array(t)).to(device)
            gt_mmcif_feats = tree_map(to_tensor, gt_mmcif_feats, np.ndarray)

            # as make_atom14_masks & make_atom14_positions seems to expect indices and not one-hots !
            gt_mmcif_feats["aatype"] = gt_mmcif_feats["aatype"].argmax(axis=-1)

            gt_mmcif_feats["aa_sequence_str"] = gt_all_mmcif_feats["aa_sequence_str"]

            ans[chain_id] = gt_mmcif_feats
    else:
        assert False

    for chain_id, data in ans.items():
        data = calculate_additional_features(data)
        data["pdb_id"] = pdb_id
        data["chain_id"] = chain_id

    if return_dict:
        final_ans = ans
    else:
        final_ans = ans[chain_id]

    final_ans = (final_ans, mmcif_object, mmcif_dict)

    return final_ans


def get_chain_native_features(
    native_structure_filename: str,
    chain_id: Optional[Union[Union[str, int], List[Union[str, int]]]] = None,
    pdb_id: str = "dummy",
    chain_id_type: str = "author_assigned",
    device: str = "cpu",
) -> Union[Tuple[str, dict], None]:
    raise Exception(
        '"get_chain_native_features()" is deprecated, please switch to "load_protein_structure_features()"'
    )


def calculate_additional_features(gt_mmcif_feats: Dict) -> Dict:
    gt_mmcif_feats = data_transforms.atom37_to_frames(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.get_backbone_frames(gt_mmcif_feats)

    gt_mmcif_feats = data_transforms.make_atom14_masks(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.make_atom14_positions(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.make_atom14_bfactors(gt_mmcif_feats)

    # for reference, remember .../openfold/openfold/data/input_pipeline.py
    # data_transforms.make_atom14_masks
    # data_transforms.make_atom14_positions
    # data_transforms.atom37_to_frames
    # data_transforms.atom37_to_torsion_angles(""),
    gt_mmcif_feats = data_transforms.make_pseudo_beta_no_curry(gt_mmcif_feats)
    # data_transforms.get_backbone_frames
    # data_transforms.get_chi_angles
    return gt_mmcif_feats


def aa_sequence_from_pdb(pdb_filename: str) -> Dict[str, str]:
    """
    Extracts the amino-acid sequence from the input pdb file.

    Returns a dictionary that maps chain_id to sequence
    """
    structure = structure_from_pdb(pdb_filename)
    return aa_sequence_from_pdb_structure(structure)


def aa_sequence_from_pdb_structure(structure: Structure) -> dict:
    # iterate each model, chain, and residue
    # printing out the sequence for each chain
    chains = {}

    for model in structure:
        for chain in model:
            seq = "".join(
                [
                    restype_3to1[residue.resname]
                    for residue in chain
                    if residue.resname in restype_3to1
                ]
            )
            chains[chain.id] = seq

    return chains


def aa_sequence_coord_from_pdb_structure(structure: Structure) -> dict:
    # iterate each model, chain, and residue
    # printing out the sequence coordinates for the atoms in each chain
    chains = {}

    for model in structure:
        for chain in model:
            chains[chain.id] = [
                np.asarray([atom.coord for atom in residue.get_atoms()])
                for residue in chain
                if residue.resname in restype_3to1
            ]
    return chains


def structure_from_pdb(pdb_filename: str) -> Structure:
    pdb_filename = get_pdb_native_full_name(pdb_filename)
    text = read_file_raw_string(pdb_filename)

    pdb_fh = io.StringIO(text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_fh)
    return structure


def load_pdb_chain_features(
    filename: str,
    chain_id: Optional[str] = None,
    also_return_openfold_protein: bool = False,
) -> Dict:
    prot = pdb_to_openfold_protein(
        filename,
        chain_id,
    )

    features = convert_openfold_protein_to_dict(prot)
    features = calculate_additional_features(features)

    if also_return_openfold_protein:
        return features, prot
    return features


def pdb_to_openfold_protein(
    filename: str,
    chain_id: Optional[str] = None,
) -> protein_utils.Protein:
    """
    Loads data from the pdb file - which includes the atoms positions, atom mask, the AA sequence.
    """
    filename = get_pdb_native_full_name(filename)
    text = read_file_raw_string(filename)
    protein = protein_utils.from_pdb_string(text, chain_id=chain_id)

    protein.aasequence_str = aa_sequence_from_aa_integers(protein.aatype)

    return protein

    # with open(filename,'rt') as f:
    #     return protein_utils.from_pdb_string(f.read(), chain_id=chain_id)


def convert_openfold_protein_to_dict(
    prot: protein_utils.Protein, to_torch: bool = True
) -> Dict:
    """
    Note: Aligning with the mmcif code expected names
    """

    names_mapping = {  # Protin to expected keys in dict
        "atom_positions": "all_atom_positions",
        "aatype": "aatype",
        "atom_mask": "all_atom_mask",
        #'residue_index' :  'residue_index',
        "b_factors": "all_atom_bfactors",
        #'chain_index' :  ,
        #'remark' :  ,
        #'parents' :  ,
        #'parents_chain_index' :  ,
        "aasequence_str": "aasequence_str",
    }

    ans = {}
    for from_name, to_name in names_mapping.items():
        ans[to_name] = getattr(prot, from_name)
        if to_torch and (not isinstance(ans[to_name], str)):
            ans[to_name] = torch.from_numpy(ans[to_name])

    return ans


def get_available_chain_ids_in_pdb(filename: str) -> List[str]:
    """
    Will return all available chain ids in a pdb file, performs some filtering to get protein chains and not other chains
    """
    filename = get_pdb_native_full_name(filename)
    pdb_str = read_file_raw_string(filename)

    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    ans = [chain.id for chain in model]
    return ans


def pdb_to_biopython_structure(
    pdb_filename: str, unique_name: str = "bananaphone"
) -> Structure:
    """
    Reads a pdb file and outputs a biopython structure
    """
    pdb_filename = get_pdb_native_full_name(pdb_filename)
    parser = PDBParser()
    structure = parser.get_structure(unique_name, pdb_filename)
    return structure


def extract_single_chain(full_structure: Structure, chain_id: str) -> Structure:
    """
    Creates and returns a biopython structure containing only the request chain_id

    Note:
    Currently doing it in a non optimal way, loading the full structure, and then
    constructing a new biopython structure adding only the requested chain
    There is probably a better way ...

    Another approach could be to use Select like shown here: https://stackoverflow.com/questions/11685716/how-to-extract-chains-from-a-pdb-file
    but that means saving the pdb to disk and I prefer to do it in memory.
    """

    seen_chain_ids = []
    selected_chain = None
    for check_chain in full_structure.get_chains():
        seen_chain_ids.append(check_chain.id)
        if check_chain.id == chain_id:
            # print('found ', chain_id)
            selected_chain = check_chain
            break

    if selected_chain is None:
        raise Exception(
            f"could not find request chain_id={chain_id}  available chains are {seen_chain_ids}"
        )

    return create_single_chain_structure(selected_chain)


def create_single_chain_structure(selected_chain: Chain) -> Structure:
    single_chain_model = Model(id="bananaphone", serial_num="999")
    single_chain_model.add(selected_chain)

    single_chain_structure = Structure("bananaphone")
    single_chain_structure.add(single_chain_model)

    return single_chain_structure


def is_pdb_id(pdb_id: str) -> bool:
    """
    Checks if the given string is a pdb id
    Currently only checks for string length.
    NOTE: in the future pdb ids will be 8 characters long
    """
    return 4 == len(pdb_id)


def get_pdb_native_full_name(pdb_id: str, strict: bool = False) -> str:
    """
    Uses the PDB_DIR environment variable to get a full path filename from pdb_id
    """
    if not is_pdb_id(pdb_id):
        if strict:
            raise Exception(
                f"pdb_id is expected to be 4 letters long, but got {pdb_id} - note, 8 letters pdb id is not supported yet."
            )
        else:
            return pdb_id
    if "PDB_DIR" not in os.environ:
        raise Exception(
            "Please set an environment var named PDB_DIR pointing to the downloaded pdb structures. You can use localpdb (pip installable) to maintain such directory."
        )
    pdb_dir = os.environ["PDB_DIR"]
    ans = f"{pdb_dir}/pdb/{pdb_id[1:3]}/pdb{pdb_id}.ent.gz"
    return ans


def read_file_raw_string(filename: str) -> str:
    """
    Gets a raw string of a file, supports gz compression
    """
    use_open = open
    if filename.endswith(".gz"):
        use_open = gzip.open
    with use_open(filename, "rt") as f:
        loaded = f.read()
    return loaded


def save_trajectory_to_pdb_file(
    traj_xyz: torch.Tensor,
    sequence: torch.Tensor,
    residues_mask: torch.Tensor,
    save_path: str,
    traj_b_factors: torch.Tensor = None,
    init_chain: str = "A",
    verbose: bool = False,
) -> None:
    """
    Stores a trajectory into a single PDB file.

    Args:

    traj_xyz: a torch tensor of shape [trajectory steps, residues num, atoms num, 3]
    sequence: the amino acid of the pos14, represented by integers (see tiny_openfold.np.residue_constants)
    residues_mask: *residue* level mask (not atoms level!)
    save_path: the path to save the pdb file
    traj_b_factors: the b_factors of the amino acids - it can represent per residue: 1. Measurement accuracy in ground truth lab experiment or 2. Model prediction certainty
        optional - will be set to 100.0 for all elements if not provided.
    init_chain: chain id to use when saving to file

    Returns:
        None
    """

    if traj_b_factors is None:
        traj_b_factors = torch.full(traj_xyz.shape[:2], fill_value=100.0)

    builder = StructureBuilder.StructureBuilder()
    builder.init_structure(0)

    use_range_func = trange if verbose else range

    for model in use_range_func(traj_xyz.shape[0]):
        builder.init_model(model)
        builder.init_chain(init_chain)
        builder.init_seg("    ")

        # extract current frame/step info in the trajectory
        xyz = traj_xyz[model]
        b_factors = traj_b_factors[model]

        if torch.is_tensor(residues_mask):
            residues_mask = residues_mask.bool()
        else:
            residues_mask = residues_mask.astype(bool)

        for i, (aa_idx, p_res, b, m_res) in enumerate(
            zip(sequence, xyz, b_factors, residues_mask)
        ):
            if not m_res:
                continue
            aa_idx = aa_idx.item()
            if torch.is_tensor(p_res):
                p_res = p_res.clone().detach().cpu()  # fixme: this looks slow
            if aa_idx == 21:
                continue
            try:
                three = residx_to_3(aa_idx)
            except IndexError:
                continue
            builder.init_residue(three, " ", int(i), icode=" ")
            for j, (atom_name,) in enumerate(
                zip(rc.restype_name_to_atom14_names[three])
            ):  # why is zip used here?
                if (len(atom_name) > 0) and (len(p_res) > j):
                    builder.init_atom(
                        atom_name,
                        p_res[j].tolist(),
                        b.item(),
                        1.0,
                        " ",
                        atom_name.join([" ", " "]),
                        element=atom_name[0],
                    )
    structure = builder.get_structure()
    io = PDB.PDBIO()
    io.set_structure(structure)
    os.makedirs(pathlib.Path(save_path).parent, exist_ok=True)
    io.save(save_path)


def flexible_save_pdb_file(
    *,
    xyz: Dict[str, torch.Tensor], #chain_id to tensor
    sequence: Dict[str, torch.Tensor], #chain_id to tensor
    residues_mask: Dict[str, torch.Tensor], #chain_id to tensor
    save_path: str,
    model: int = 0,
    only_save_backbone: bool = False,
    b_factors: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None, #chain_id to tensor
) -> None:
    """
    saves a PDB file containing the provided coordinates.

    "xyz" coordinates and "mask" should be aligned with tiny_openfold.np.residue_constants.restype_name_to_atom14_names
    you don't have to provide 14 atoms per mask, but the order should be aligned.

    Example 1 - full heavy atoms info - you can provide:
        xyz of shape [residues_num, 14, 3]

    Example 2 - backbone only - you can provide
        xyz of shape [residues_num, 4, 3]

        in this case, only "N", "CA", "C", "O" atoms will be saved to the PDB
        (But it will still output amino acid type into the PDB, based on what you provided as sequence)
        Note - if you don't know it, you can provide only Lysine for the entire "sequence" argument

        Alternatively, when you want to save only the backbone coordinates, you may provide the full 14 atoms info for both xyz and mask,
        and supply only_save_backbone=True which is effectively the same as supplying
            xyz = xyz[:,:4, :]
            mask = mask[:, :4, :]



    Support both atom14 and

    To learn more about pdb format see: https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/introduction


    Args:
        pos14: the atom14 representation of the coordinates
        b_factors: the b_factors of the amino acids - it can represent per residue: 1. Measurement accuracy in ground truth lab experiment or 2. Model prediction certainty
        sequence: the amino acid of the pos14
        residues_mask: *residue* level mask (not atoms level!)
        save_path: the path to save the pdb file
        model: the model id of the pdb file
        init_chain: chain id to use when saving to file
        only_save_backbone: only consider the first 4 atoms
            (Note - amino acid type name will still be output to the PDB based on the 'sequence' arg!)

    return:
        None

    """

    #either all are dicts (which supports multiple chains) all or are tensors (which means a single chain)
    assert isinstance(xyz, dict) and isinstance(sequence, dict) and isinstance(residues_mask ,dict) and  ( (b_factors is None) or isinstance(b_factors, dict))

    assert list(xyz.keys()) == list(sequence.keys())
    assert list(xyz.keys()) == list(residues_mask.keys())
    if b_factors is not None:
        assert list(xyz.keys()) == list(b_factors.keys())
        
    if only_save_backbone:
        print(
            "flexible_save_pdb_file:: only output backbone requested, will store coordinates only for the first 4 atoms in atom14 convention order."
        )
        for k in xyz.keys():        
            xyz[k] = xyz[k][:, :4, ...]

        assert xyz[k].shape[1] in [
            4,
            14,
            37,
        ], f"xyz shape is allowed to be 14 (all heavy atoms) or 4 (only BB), got xyz.shap={xyz.shape}"

    if b_factors is None:
        # b_factors = torch.tensor([100.0] * xyz.shape[0])
        b_factors = {}
        for k in xyz.keys():
            b_factors[k] = torch.zeros((xyz[k].shape[:-1]))

    builder = StructureBuilder.StructureBuilder()
    builder.init_structure(0)
    builder.init_model(model)
    for chain_id in xyz.keys():
        builder.init_chain(chain_id)
        builder.init_seg("    ")
        if torch.is_tensor(residues_mask[chain_id]):
            residues_mask[chain_id] = residues_mask[chain_id].bool()

        if torch.is_tensor(xyz[chain_id]):
            xyz[chain_id] = xyz[chain_id].clone().detach().cpu()

        for i, (aa_idx, p_res, b, m_res) in enumerate(
            zip(sequence[chain_id], xyz[chain_id], b_factors[chain_id], residues_mask[chain_id])
        ):
            if not m_res:
                continue
            aa_idx = aa_idx.item()

            if aa_idx == 21:  # is this X ? (unknown/special)
                continue
            try:
                three = residx_to_3(aa_idx)
            except IndexError:
                continue
            builder.init_residue(three, " ", int(i), icode=" ")

            if xyz[chain_id].shape[1] == 37:
                atom_names = rc.atom_types
            else:
                atom_names = rc.restype_name_to_atom14_names[three]

            residue_atom_names = rc.residue_atoms[three]

            for j, (atom_name,) in enumerate(zip(atom_names)):  # why is zip used here?
                if (
                    (len(atom_name) > 0)
                    and (len(p_res) > j)
                    and atom_name in residue_atom_names
                ):
                    builder.init_atom(
                        atom_name,
                        p_res[j].tolist(),
                        b[j].item(),
                        1.0,
                        " ",
                        atom_name.join([" ", " "]),
                        element=atom_name[0],
                    )
    structure = builder.get_structure()
    io = PDB.PDBIO()
    io.set_structure(structure)
    os.makedirs(pathlib.Path(save_path).parent, exist_ok=True)
    io.save(save_path)
    pass


def save_pdb_file(
    pos14: torch.Tensor,
    b_factors: torch.Tensor,
    sequence: torch.Tensor,
    mask: torch.Tensor,
    save_path: str,
    model: int = 0,
    init_chain: str = "A",
) -> None:
    """
    saves the pos14 as a pdb file

    To learn more about pdb format see: https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/introduction


    Args:
        pos14: the atom14 representation of the coordinates
        b_factors: the b_factors of the amino acids - it can represent per residue: 1. Measurement accuracy in ground truth lab experiment or 2. Model prediction certainty
        sequence: the amino acid of the pos14
        mask: the validity of the atoms
        save_path: the path to save the pdb file
        model: the model id of the pdb file
        init_chain: chain id to use when saving to file

    return:
        None

    """
    assert len(pos14.shape) == 3
    assert pos14.shape[1] == 14

    assert len(mask.shape) == 1  # 2
    # assert mask.shape[1] == 14

    flexible_save_pdb_file(
        xyz=pos14,
        b_factors=b_factors,
        sequence=sequence,
        residues_mask=mask,
        save_path=save_path,
        model=model,
        init_chain=init_chain,
    )


# code heavily inspired on alpha/open fold data_modules.py


def parse_mmcif(
    filename: str,
    unique_file_id: str,
    handle_residue_id_duplication: bool = True,
    quiet_parsing: bool = False,
    raise_exception_on_error: bool = True,
    also_return_mmcif_dict: bool = False,
) -> MmcifObject:
    """
    filename: path for the mmcif file to load (can be .gz compressed)
        may also be an open file handle
    unique_file_id: a unique_id for this file

    returns an MmcifObject
    """
    if isinstance(filename, str):
        filename = get_mmcif_native_full_name(filename)
        raw_mmcif_str = read_file_raw_string(filename)
    else:
        raw_mmcif_str = filename.read()

    mmcif_object = mmcif_parsing.parse(
        file_id=unique_file_id,
        catch_all_errors=False,
        mmcif_string=raw_mmcif_str,
        handle_residue_id_duplication=handle_residue_id_duplication,
        quiet_parsing=quiet_parsing,
        also_return_mmcif_dict=also_return_mmcif_dict,
    )

    if also_return_mmcif_dict:
        mmcif_object, _raw_mmcif_dict = mmcif_object

    # https://biopython-cn.readthedocs.io/zh_CN/latest/en/chr11.html#reading-an-mmcif-file

    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with at the alignment stage.

    if mmcif_object.mmcif_object is None:
        if raise_exception_on_error:
            # raise list(mmcif_object.errors.values())[0]
            err_vals = list(mmcif_object.errors.values())
            if len(err_vals) == 1:
                raise Exception(err_vals[0])
            else:
                raise Exception([Exception(_) for _ in err_vals])
        else:
            # this might be slow when iterating on many items
            raise Exception([Exception(_) for _ in err_vals])
            return None  # keeping this here in case someone comments out the exception raising.

    mmcif_object = mmcif_object.mmcif_object

    if not also_return_mmcif_dict:
        return mmcif_object
    else:
        return mmcif_object, _raw_mmcif_dict


def get_chain_data(
    mmcif: mmcif_parsing.MmcifObject,
    chain_id: Union[str, int],
) -> dict:
    """
    Assembles features for a specific chain in an mmCIF object.
            if chain_id is str, it is used

    chain_id: author assigned chain id.
        For more details in author assigned chain id vs. pdb assigned chain id see https://www.rcsb.org/docs/general-help/identifiers-in-pdb




    """

    mmcif_feats = data_pipeline.make_mmcif_features(mmcif, chain_id)
    mmcif_feats["aa_sequence_str"] = mmcif.chain_to_seqres[chain_id]
    # input_sequence = mmcif.chain_to_seqres[chain_id]

    # return dict(
    #     mmcif_feats=mmcif_feats,
    #     input_sequence=input_sequence,
    # )

    return mmcif_feats


def load_mmcif_features(filename: str, pdb_id: str, chain_id: str) -> Tuple[dict, str]:
    """
    Features in the style that *Fold use
    """
    filename = get_mmcif_native_full_name(filename)
    mmcif_data = parse_mmcif(filename, "bananaphone")
    chains_names = list(mmcif_data.chain_to_seqres.keys())
    if chain_id not in chains_names:
        raise Exception(
            f"Error requested chain_id={chain_id} not found in available chains {chains_names}"
        )
    gt_all_mmcif_feats = get_chain_data(mmcif_data, chain_id=chain_id)
    gt_sequence = gt_all_mmcif_feats["aa_sequence_str"]

    gt_mmcif_feats = {
        k: gt_all_mmcif_feats[k]
        for k in ["aatype", "all_atom_positions", "all_atom_mask"]
    }

    to_tensor = lambda t: torch.tensor(np.array(t))  # .cude()
    gt_mmcif_feats = tree_map(to_tensor, gt_mmcif_feats, np.ndarray)

    # as make_atom14_masks & make_atom14_positions seems to expect indices and not one-hots !
    gt_mmcif_feats["aatype"] = gt_mmcif_feats["aatype"].argmax(axis=-1)

    gt_mmcif_feats = data_transforms.make_atom14_masks(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.make_atom14_positions(gt_mmcif_feats)

    for k in gt_all_mmcif_feats.keys():
        if k not in gt_mmcif_feats:
            gt_mmcif_feats[k] = gt_all_mmcif_feats[k]

    # consider adding gt_sequence directly to the returned features, but be careful because it might cause a mismatch
    # with some different item, because when the 3d data is parse some residues can be skipped
    return gt_mmcif_feats, gt_sequence


def save_updated_mmcif_file(
    *,  # prevent positional args
    reference_mmcif_file: str,
    pdb_id: str,
    chain_to_aa_seq: Dict[str, str],
    chain_to_atom14: Dict[str, np.ndarray],
    output_mmcif_filename: str,
) -> None:
    """ """

    biopython_structure = create_biopython_structure(
        reference_mmcif_file=reference_mmcif_file,
        pdb_id=pdb_id,
        chain_to_aa_seq=chain_to_aa_seq,
        chain_to_atom14=chain_to_atom14,
    )

    store_chains = list(chain_to_aa_seq.keys())

    biopython_structure_to_mmcif_file(
        biopython_structure=biopython_structure,
        store_chains=store_chains,
        output_mmcif_filename=output_mmcif_filename,
    )


def create_biopython_structure(
    *,  # prevent positional args
    reference_mmcif_file: str,
    pdb_id: str,
    chain_to_aa_seq: Dict[str, str],
    chain_to_atom14: Dict[str, np.ndarray],
    quiet: bool = True,
) -> Structure:
    """
    Creates a biopython structure with updated info, which can then be use to save to file (for example, in mmCIF format),
        requires a reference mmcif file to serve as "template"

    Args:

    reference_mmcif_file: the reference ground truth mmcif file (it may be compressed with .gz)
    output_mmcif_file: the output file, in which both the ground truth and the prediction models will be found
    pdb_id: the pdb id (currently 4 characters, in the future they intend to increase it to 8 chars)
    chain_aa_seq: a dictionary mapping between chain_id and the aa sequence (a string in which each character is an amino-acid)
    chain_to_atom14: a dictionary mapping between chain_id and a tensor of the shape [residues_num, 14, 3]
        14 represents the maximum amount of heavy atoms in each residue, and 3 cartesean coordinates
    """

    # mmcif_string = read_file_raw_string(reference_mmcif_file)
    parser = MMCIFParser(QUIET=quiet)
    if reference_mmcif_file.endswith(".gz"):
        use_open = gzip.open
    else:
        use_open = open
    with use_open(reference_mmcif_file, "rt") as fh:
        ref_structure = parser.get_structure(pdb_id, fh)
    # parser = MMCIFParser(QUIET=False)
    # handle = io.StringIO(mmcif_string)
    # structure = parser.get_structure(pdb_id, mmcif_string)
    # deepcopy?
    # first_model_structure = next(full_structure.get_models()) #take the first

    # go over the

    # pred_structure = Structure(pdb_id)
    pred_model = Model(id="prediction@" + pdb_id, serial_num="0")  # 999')

    # restypes = rc.restypes + ["X"]
    # res_1to3 = lambda r: rc.restype_1to3.get(restypes[r], "UNK")
    # atom_types = openfold_rc.atom_types

    # pred_chains = {}

    atom_idx = -1

    for chain_id, seq in chain_to_aa_seq.items():
        if any([x not in rc.restype_1to3 for x in seq]):
            raise ValueError("Invalid aatypes.")

        ref_chain = [c for c in ref_structure.get_chains() if c.id == chain_id]
        assert 1 == len(ref_chain), f"could not find chain_id={chain_id} !"
        ref_chain = ref_chain[0]

        # pred_chain = Chain('prediction@'+ref_chain.id)
        pred_chain = Chain(ref_chain.id)

        # go over all atoms:
        # already_defined_in_this_chain = set()

        # for res_i, (ref_residue, r_1char) in enumerate(zip(ref_chain.get_residues(), seq)):
        for res_i, r_1char in enumerate(seq):
            # ref_atoms = list(ref_residue.get_atoms())
            # pred_residue_pos = pred_pos_all_atom_with_res_structure[res_i]
            pred_residue_pos = chain_to_atom14[chain_id][res_i]
            res_3char = rc.restype_1to3[r_1char]

            atom14_atoms = rc.restype_name_to_atom14_names[res_3char]

            pred_residue = Residue(
                # ref_residue.id,
                # ref_residue.resname,
                # ref_residue.segid,
                id=(" ", res_i, " "),  # hetfield, resseq, icode
                resname=res_3char,
                segid=" ",
            )

            # for pred_atom_name, pred_atom_pos, ref_atom in zip(atom14_atoms, pred_residue_pos, ref_atoms):
            for pred_atom_name, pred_atom_pos in zip(atom14_atoms, pred_residue_pos):
                if pred_atom_name == "":
                    continue

                # assert pred_atom_name == ref_atom.id
                # ref_atom.set_coord(pred_atom_pos.tolist())

                atom_idx += 1

                pred_atom = Atom(
                    pred_atom_name,  # ref_atom.name,
                    pred_atom_pos.tolist(),
                    100.0,  # ref_atom.bfactor, #TODO: add confidence here instead
                    1.0,  # ref_atom.occupancy,
                    " ",  # ref_atom.altloc,
                    pred_atom_name,  # ref_atom.fullname,
                    atom_idx,  # ref_atom.serial_number,
                    element=pred_atom_name[0],  # element = ref_atom.element,
                    # pqr_charge = ref_atom.pqr_charge,
                    # radius = ref_atom.radius,
                )

                pred_residue.add(pred_atom)

                # already_defined_in_this_chain.add(ref_atom.name)

            pred_chain.add(pred_residue)

        # pred_chains[ref_chain.id] = pred_chain

        pred_model.add(pred_chain)

        # also add the reference chain
        # pred_model.add(ref_chain)

    # for chain_id, chain in pred_chains.items():
    #     #pred_structure.add(chain)
    #     pred_model.add(chain)

    pred_structure = Structure(pdb_id)
    pred_structure.add(pred_model)
    # pred_structure.add([m for m in ref_structure][0]) #take the first model

    return pred_structure


def biopython_structure_to_mmcif_file(
    biopython_structure: Structure,
    store_chains: List[str],
    output_mmcif_filename: str,
    verbose: int = 0,
) -> None:
    io = MMCIFIO()
    os.makedirs(os.path.dirname(output_mmcif_filename), exist_ok=True)
    io.set_structure(biopython_structure)
    io.save(output_mmcif_filename, select=ChainsSelector(store_chains))
    if verbose > 0:
        print(f"saved biopython structure into: {output_mmcif_filename}")


class ChainsSelector(Select):
    def __init__(self, keep_chains: List[str]):
        self._keep_chains = deepcopy(keep_chains)

    def __repr__(self) -> str:
        """Represent the output as a string for debugging."""
        return f"selects only the chains: {self._keep_chains}"

    # def accept_model(self, model):
    #     """Overload this to reject models for output."""
    #     return 1

    def accept_chain(self, chain: Chain) -> bool:
        """Overload this to reject chains for output."""
        return chain.id in self._keep_chains
        # return 1

    # def accept_residue(self, residue):
    #     """Overload this to reject residues for output."""
    #     return 1

    # def accept_atom(self, atom):
    #     """Overload this to reject atoms for output."""
    #     return 1


def get_mmcif_native_full_name(pdb_id: str, strict: bool = False) -> str:
    if not is_pdb_id(pdb_id):
        if strict:
            raise Exception(
                f"pdb_id is expected to be 4 letters long, but got {pdb_id} - note, 8 letters pdb id is not supported yet."
            )
        else:
            return pdb_id
    if "PDB_DIR" not in os.environ:
        raise Exception(
            "Please set an environment var named PDB_DIR pointing to the downloaded pdb structures. You can use localpdb (pip installable) to maintain such directory."
        )
    pdb_dir = os.environ["PDB_DIR"]
    ans = f"{pdb_dir}/mmCIF/{pdb_id[1:3]}/{pdb_id}.cif.gz"
    return ans


def load_mmcif_biopython(
    filename: str, structure_name: str = "bananaphone"
) -> Structure:
    if is_pdb_id(filename):
        filename = get_mmcif_native_full_name(filename)
    print(f"loading: {filename}")
    use_open = gzip.open if filename.endswith(".gz") else open
    parser = MMCIFParser()
    with use_open(filename, "rt") as fh:
        structure = parser.get_structure(structure_name, fh)
    return structure
