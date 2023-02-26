from openfold.np.residue_constants import restype_3to1
from Bio.PDB import PDBParser
import openfold.np.protein as protein_utils
#from openfold.np import residue_constants as rc
from omegafold.utils.protein_utils import residue_constants as rc
#from Bio.PDB import *
import gzip
import nglview as nv
from fusedrug.tests_data import get_tests_data_dir
from os.path import join
from Bio.PDB import *
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from typing import Optional, Dict, List
import os
import io
#from fusedrug.data.protein.structure.utils import aa_sequence_from_aa_integers
#https://bioinformatics.stackexchange.com/questions/14101/extract-residue-sequence-from-pdb-file-in-biopython-but-open-to-recommendation
import torch
from Bio import PDB as PDB
from Bio.PDB import StructureBuilder
import pathlib


from typing import Dict, Optional, List
import torch

from typing import Optional, Dict, Union
import gzip
import io
from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    #templates,
)
#from openfold.np import residue_constants as openfold_rc
from omegafold.utils import residue_constants as omegafold_rc
import numpy as np
from Bio.PDB import MMCIFParser, Select, MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from typing import List
from copy import deepcopy
import os
import torch

from openfold.utils.superimposition import superimpose
from openfold.data import data_transforms
from openfold.utils.tensor_utils import tree_map

#from openfold.np import residue_constants as rc

from fusedrug.data.protein.structure.utils import aa_sequence_from_aa_integers, get_structure_file_type
    
def save_structure_file(*, #prevent positional args
    output_filename_extensionless:str, 
    pdb_id:str,     
    chain_to_atom14:Dict[str,torch.Tensor],

    #optional args
    chain_to_aa_str_seq:Optional[Dict[str,str]] = None,
    chain_to_aa_index_seq:Optional[Dict[str,torch.Tensor]] = None,
    save_pdb = True,
    save_cif = True,
    b_factors:Optional[Dict[str,torch.Tensor]] = None,
    reference_cif_filename:Optional[str] = None,
    mask:Optional[List] = None,

    ):
        assert save_pdb or save_cif
        assert len(chain_to_atom14) > 0
        if chain_to_aa_index_seq is not None:
            assert len(chain_to_aa_index_seq) == len(chain_to_atom14)
            sorted_chain_ids = sorted(list(chain_to_aa_index_seq.keys()))

        if chain_to_aa_str_seq is not None:
            assert len(chain_to_aa_str_seq) == len(chain_to_atom14)
            sorted_chain_ids = sorted(list(chain_to_aa_str_seq.keys()))

        assert (chain_to_aa_str_seq is not None) or (chain_to_aa_index_seq is not None)
        
        if save_cif:
            if len(chain_to_aa_str_seq) > 1:
                if reference_cif_filename is not None:
                    out_multimer_cif_filename = output_filename_extensionless+'_multimer_chain_ids_'
                    out_multimer_cif_filename += '_'.join(sorted(list(chain_to_aa_str_seq.keys())))+'.cif'

                    print(f'saving MULTImer cif file {out_multimer_cif_filename}')
                    save_updated_mmcif_file(
                        reference_mmcif_file = reference_cif_filename,
                        pdb_id = pdb_id, 
                        chain_to_aa_seq = chain_to_aa_str_seq,
                        chain_to_atom14 = chain_to_atom14,     
                        output_mmcif_filename = out_multimer_cif_filename,
                    )                
                else:
                    print(f'not writing multimer file because "reference_cif_filename" was not provided for {reference_cif_filename}')


        for chain_id in sorted_chain_ids:
            pos_atom14  = chain_to_atom14[chain_id]

            if save_pdb:
                out_pdb = output_filename_extensionless+'_chain_'+chain_id+'.pdb'
                print(f"Saving structure file to {out_pdb}")

                potentially_fixed_chain_id = chain_id 
                if len(potentially_fixed_chain_id)>1:
                    potentially_fixed_chain_id = chain_id[:1]
                    print(f'WARNING: shortening too long chain_id from {chain_id} to {potentially_fixed_chain_id}')

                save_pdb_file(
                    pos14 = pos_atom14,
                    b_factors = b_factors[chain_id] if b_factors is not None else torch.tensor([100.0] * pos_atom14.shape[0]),
                    sequence = chain_to_aa_index_seq[chain_id],
                    mask = mask if mask is not None else torch.full((pos_atom14.shape[0],), fill_value=True),
                    save_path = out_pdb,
                    init_chain = potentially_fixed_chain_id,
                    model=0,
                )

            if save_cif:
                out_cif = output_filename_extensionless+'_chain_'+chain_id+'.cif'            
                if reference_cif_filename is not None:
                    print(f"Saving structure file to {out_cif}")
                    save_updated_mmcif_file(
                        reference_mmcif_file = reference_cif_filename,
                        pdb_id = pdb_id, 
                        chain_to_aa_seq = {chain_id:chain_to_aa_str_seq[chain_id]},
                        chain_to_atom14 = {chain_id:chain_to_atom14[chain_id]},
                        output_mmcif_filename = out_cif,
                    )
                else:
                    print(f'not writing chain cif file because no "reference_cif_filename" was provided for {reference_cif_filename}')



def get_chain_native_features(native_structure_filename:str, chain_id:str, pdb_id:str, device='cpu'):
    '''

    chain_id:
        can be a single character (example: 'A')
        or an integer (zero-based) and then the i-th chain in the *sorted* list of available chains will be used
    '''

    structure_file_format = get_structure_file_type(native_structure_filename)
    if structure_file_format=='pdb':
        assert False, 'not tested for a long time now'
        try:
            chains_names = get_available_chain_ids_in_pdb(native_structure_filename)                                 
        except Exception as e:
            print(e)
            print(f'Had an issue with a protein, skipping it. Protein file:{native_structure_filename}')
            return None
    elif structure_file_format=='cif':
        try:
            mmcif_object, chains_names = parse_mmcif(native_structure_filename, unique_file_id=pdb_id, quiet_parsing=True)
        except Exception as e:
            print(e)
            print(f'Had an issue reading {native_structure_filename}')
            return None
    else:
        assert False

    
    if structure_file_format=='pdb':            
        gt_data = pdb_to_openfold_protein(native_structure_filename, chain_id=chain_id)               
        gt_sequence = gt_data.aasequence_str

        gt_mmcif_feats = dict(
            aatype = gt_data.aatype,
            all_atom_positions = gt_data.atom_positions,
            all_atom_mask = gt_data.atom_mask,
        )

        gt_mmcif_feats = {k:torch.tensor(d) for (k,d) in gt_mmcif_feats.items()}                                                                                                      

    elif structure_file_format=='cif':            
        
        if isinstance(chain_id, int):                
            chains_names = sorted(chains_names)
            if (chain_id<0) or (chain_id>=len(chains_names)):
                raise Exception(f'chain_id(int)={chain_id} is out of bound for the options: {chains_names}')
            chain_id = chains_names[chain_id] #taking the i-th chain

        gt_data = get_chain_data(mmcif_object, chain_id=chain_id)
        gt_all_mmcif_feats = gt_data['mmcif_feats']
        gt_sequence = gt_data['input_sequence']
        #move to device
        gt_mmcif_feats = {
            k: gt_all_mmcif_feats[k] for k in ['aatype', 'all_atom_positions', 'all_atom_mask']
        }
        
        to_tensor = lambda t: torch.tensor(np.array(t)).to(device)
        gt_mmcif_feats = tree_map(to_tensor, gt_mmcif_feats, np.ndarray)
        #as make_atom14_masks & make_atom14_positions seems to expect indices and not one-hots !
        gt_mmcif_feats['aatype'] = gt_mmcif_feats['aatype'].argmax(axis=-1)

        
    else:
        assert False

    gt_mmcif_feats = data_transforms.atom37_to_frames(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.get_backbone_frames(gt_mmcif_feats)
    
    gt_mmcif_feats = data_transforms.make_atom14_masks(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.make_atom14_positions(gt_mmcif_feats)

    
    #for reference, remember .../openfold/openfold/data/input_pipeline.py
    #data_transforms.make_atom14_masks,
    # data_transforms.make_atom14_positions,
    # data_transforms.atom37_to_frames,
    # data_transforms.atom37_to_torsion_angles(""),
    # data_transforms.make_pseudo_beta(""),
    # data_transforms.get_backbone_frames,
    # data_transforms.get_chi_angles,

    gt_mmcif_feats['pdb_id'] = pdb_id
    gt_mmcif_feats['chain_id'] = chain_id

    return gt_sequence, gt_mmcif_feats




def aa_sequence_from_pdb(pdb_filename:str) -> Dict[str,str]:
    """
    Extracts the amino-acid sequence from the input pdb file.

    Returns a dictionary that maps chain_id to sequence
    """

    assert False, "Not implemented fully yet - in 7kpj it crashes on attempt to treat HOH as a residue. Should probably just filter protein chains"
    pdb_filename = get_pdb_native_full_name(pdb_filename)
    text = read_file_raw_string(pdb_filename)

    pdb_fh = io.StringIO(text)
    
    parser = PDBParser(QUIET=False)
    structure = parser.get_structure('struct', pdb_fh)
    # iterate each model, chain, and residue
    # printing out the sequence for each chain

    chains = {}

    for model in structure:
        for chain in model:
            #print(chain.name)
            seq = ''.join([restype_3to1[residue.resname] for residue in chain])
            chains[chain.id] = seq
            #print('>some_header\n',''.join(seq))
    return chains

def pdb_to_openfold_protein(filename:str, chain_id:Optional[str]=None) -> protein_utils.Protein:
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


def get_available_chain_ids_in_pdb(filename:str) -> List[str]:    
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

    

def pdb_to_biopython_structure(pdb_filename:str, unique_name:str='bananaphone') -> Structure:
    """
    Reads a pdb file and outputs a biopython structure
    """
    pdb_filename = get_pdb_native_full_name(pdb_filename)
    parser = PDBParser()
    structure = parser.get_structure(unique_name, pdb_filename)
    return structure

def extract_single_chain(full_structure:Structure, chain_id:str) -> Structure:
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
            #print('found ', chain_id)
            selected_chain = check_chain
            break
    
    if selected_chain is None:
        raise Exception(f'could not find request chain_id={chain_id}  available chains are {seen_chain_ids}')
    
    single_structure_model = Model(id='bananaphone', serial_num='999')
    single_structure_model.add(selected_chain)
    
    single_chain_structure = Structure('bananaphone')    
    single_chain_structure.add(single_structure_model)
    
    return single_chain_structure


def is_pdb_id(pdb_id:str):
    return 4==len(pdb_id)

def get_pdb_native_full_name(pdb_id, strict=False):
    if not is_pdb_id(pdb_id):
        if strict:
            raise Exception(f'pdb_id is expected to be 4 letters long, but got {pdb_id} - note, 8 letters pdb id is not supported yet.')
        else:
            return pdb_id
    if 'PDB_DIR' not in os.environ:
        raise Exception('Please set an environment var named PDB_DIR pointing to the downloaded pdb structures. You can use localpdb (pip installable) to maintain such directory.')
    pdb_dir = os.environ['PDB_DIR']
    ans = f'{pdb_dir}/pdb/{pdb_id[1:3]}/pdb{pdb_id}.ent.gz'        
    return ans


def read_file_raw_string(filename:str) -> str:
    use_open = open
    if filename.endswith('.gz'):
        use_open = gzip.open
    with use_open(filename, 'rt') as f:
        loaded = f.read()
    return loaded

def save_pdb_file(
        pos14: torch.Tensor,
        b_factors: torch.Tensor,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        save_path: str,
        model: int = 0,
        init_chain: str = 'A'
) -> None:
    """
    saves the pos14 as a pdb file

    Args:
        pos14: the atom14 representation of the coordinates
        b_factors: the b_factors of the amino acids
        sequence: the amino acid of the pos14
        mask: the validity of the atoms
        save_path: the path to save the pdb file
        model: the model id of the pdb file
        init_chain

    return:
        the structure saved to ~save_path

    """
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure(0)
    builder.init_model(model)
    builder.init_chain(init_chain)
    builder.init_seg('    ')
    for i, (aa_idx, p_res, b, m_res) in enumerate(
            zip(sequence, pos14, b_factors, mask.bool())
    ):
        if not m_res:
            continue
        aa_idx = aa_idx.item()
        p_res = p_res.clone().detach().cpu()
        if aa_idx == 21:
            continue
        try:
            three = rc.residx_to_3(aa_idx)
        except IndexError:
            continue
        builder.init_residue(three, " ", int(i), icode=" ")
        for j, (atom_name,) in enumerate(
                zip(rc.restype_name_to_atom14_names[three])
        ):
            if len(atom_name) > 0:
                builder.init_atom(
                    atom_name, p_res[j].tolist(), b.item(), 1.0, ' ',
                    atom_name.join([" ", " "]), element=atom_name[0]
                )
    structure = builder.get_structure()
    io = PDB.PDBIO()
    io.set_structure(structure)
    os.makedirs(pathlib.Path(save_path).parent, exist_ok=True)
    io.save(save_path)    



#code based on alpha/open fold data_modules.py

def read_file_raw_string(filename:str) -> str:
    use_open = open
    if filename.endswith('.gz'):
        use_open = gzip.open
    with use_open(filename, 'rt') as f:
        loaded = f.read()
    return loaded

def parse_mmcif(filename:str, unique_file_id:str, handle_residue_id_duplication:bool=True, quiet_parsing:bool=False):
    """
    filename: path for the mmcif file to load (can be .gz compressed)
    unique_file_id: a unique_id for this file
    """
    filename = get_mmcif_native_full_name(filename)
    raw_mmcif_str = read_file_raw_string(filename)

    mmcif_object = mmcif_parsing.parse(
        file_id=unique_file_id, mmcif_string=raw_mmcif_str,
        handle_residue_id_duplication=handle_residue_id_duplication,
        quiet_parsing = quiet_parsing,
    )

    #https://biopython-cn.readthedocs.io/zh_CN/latest/en/chr11.html#reading-an-mmcif-file

    #this is inefficient as I'm reading the mmcif file again, but choosing to do this to keep openfold code unmodified for now
    handle = io.StringIO(raw_mmcif_str)
    mmcif_dict = MMCIF2Dict.MMCIF2Dict(handle)

    entities_details = mmcif_parsing.mmcif_loop_to_list('_entity.', mmcif_dict)

    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with at the alignment stage.
    if(mmcif_object.mmcif_object is None):
        #raise list(mmcif_object.errors.values())[0]
        err_vals = list(mmcif_object.errors.values())
        if len(err_vals) == 1:
            raise Exception(err_vals[0])
        else:
            raise Exception([Exception(_) for _ in err_vals])

    mmcif_object = mmcif_object.mmcif_object

    return mmcif_object, list(mmcif_object.chain_to_seqres.keys())

    #it does too much - templates search and MSA
    # data = self.data_pipeline.process_mmcif(
    #     mmcif=mmcif_object,
    #     alignment_dir=alignment_dir,
    #     chain_id=chain_id,
    #     alignment_index=alignment_index
    # )

def get_chain_data(
    mmcif: mmcif_parsing.MmcifObject, 
    chain_id: Union[str,int],
    ) -> dict:
        """
        Assembles features for a specific chain in an mmCIF object.               
                if chain_id is str, it is used
                
                
        """           

        mmcif_feats = data_pipeline.make_mmcif_features(mmcif, chain_id)       
        input_sequence = mmcif.chain_to_seqres[chain_id]

        return dict(
            mmcif_feats=mmcif_feats,
            input_sequence=input_sequence,
        )


def load_mmcif_features(filename, pdb_id:str, chain_id:str):
    """
    Features in the style that *Fold use
    """
    filename = get_mmcif_native_full_name(filename)
    mmcif_data, chains_names = parse_mmcif(filename, 'bananaphone')   
    if chain_id not in chains_names:
        raise Exception(f'Error requested chain_id={chain_id} not found in available chains {chains_names}')
    gt_data = get_chain_data(mmcif_data, chain_id=chain_id)
    gt_sequence = gt_data['input_sequence']
    gt_all_mmcif_feats = gt_data['mmcif_feats']
    
    gt_mmcif_feats = {
                k: gt_all_mmcif_feats[k] for k in ['aatype', 'all_atom_positions', 'all_atom_mask']
            }
    
    to_tensor = lambda t: torch.tensor(np.array(t)) #.cude()
    gt_mmcif_feats = tree_map(to_tensor, gt_mmcif_feats, np.ndarray)
    
    #as make_atom14_masks & make_atom14_positions seems to expect indices and not one-hots !
    gt_mmcif_feats['aatype'] = gt_mmcif_feats['aatype'].argmax(axis=-1)

    gt_mmcif_feats = data_transforms.make_atom14_masks(gt_mmcif_feats)
    gt_mmcif_feats = data_transforms.make_atom14_positions(gt_mmcif_feats)   
    
    for k in gt_all_mmcif_feats.keys():
        if k not in gt_mmcif_feats:
            gt_mmcif_feats[k] = gt_all_mmcif_feats[k]
    
    
    #consider adding gt_sequence directly to the returned features, but be careful because it might cause a mismatch 
    # with some different item, because when the 3d data is parse some residues can be skipped
    return gt_mmcif_feats, gt_sequence


def save_updated_mmcif_file(
    *, #prevent positional args
    reference_mmcif_file:str,
    pdb_id:str,
    chain_to_aa_seq:Dict[str,str],
    chain_to_atom14:Dict[str, np.ndarray],
    output_mmcif_filename:str):
    """
    """

    biopython_structure = create_biopython_structure(
        reference_mmcif_file=reference_mmcif_file,
        pdb_id=pdb_id,
        chain_to_aa_seq=chain_to_aa_seq,
        chain_to_atom14=chain_to_atom14,        
    )

    store_chains = list(chain_to_aa_seq.keys())

    biopython_structure_to_mmcif_file(biopython_structure=biopython_structure, store_chains=store_chains, output_mmcif_filename=output_mmcif_filename)


def create_biopython_structure(
    *, #prevent positional args
    reference_mmcif_file:str,
    pdb_id:str,
    chain_to_aa_seq:Dict[str,str],
    chain_to_atom14:Dict[str, np.ndarray],    
    quiet:bool = True,
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

    #mmcif_string = read_file_raw_string(reference_mmcif_file)
    parser = MMCIFParser(QUIET=quiet)
    if reference_mmcif_file.endswith('.gz'):
        use_open = gzip.open
    else:
        use_open = open
    with use_open(reference_mmcif_file, 'rt') as fh:
        ref_structure = parser.get_structure(pdb_id, fh)
    #parser = MMCIFParser(QUIET=False)  
    #handle = io.StringIO(mmcif_string)
    #structure = parser.get_structure(pdb_id, mmcif_string)    
    #deepcopy?
    #first_model_structure = next(full_structure.get_models()) #take the first

    #go over the 

    #pred_structure = Structure(pdb_id)
    pred_model = Model(id='prediction@'+pdb_id, serial_num='0') #999')

    restypes = omegafold_rc.restypes + ["X"]
    #res_1to3 = lambda r: omegafold_rc.restype_1to3.get(restypes[r], "UNK")
    #atom_types = openfold_rc.atom_types

    #pred_chains = {}

    atom_idx = -1
    
    for chain_id, seq in chain_to_aa_seq.items():
        if any([x not in omegafold_rc.restype_1to3 for x in seq]):
            raise ValueError("Invalid aatypes.")
        
        ref_chain = [c for c in ref_structure.get_chains() if c.id == chain_id]
        assert 1 == len(ref_chain), f'could not find chain_id={chain_id} !'
        ref_chain = ref_chain[0]

        #pred_chain = Chain('prediction@'+ref_chain.id)
        pred_chain = Chain(ref_chain.id)

        #go over all atoms:
        #already_defined_in_this_chain = set()

        #for res_i, (ref_residue, r_1char) in enumerate(zip(ref_chain.get_residues(), seq)):
        for res_i, r_1char in enumerate(seq):
            #ref_atoms = list(ref_residue.get_atoms())
            #pred_residue_pos = pred_pos_all_atom_with_res_structure[res_i]
            pred_residue_pos = chain_to_atom14[chain_id][res_i]
            res_3char = omegafold_rc.restype_1to3[r_1char]

            atom14_atoms = omegafold_rc.restype_name_to_atom14_names[res_3char]

            pred_residue = Residue(
                # ref_residue.id,
                # ref_residue.resname,
                # ref_residue.segid,

                id = (' ', res_i, ' '),  #hetfield, resseq, icode
                resname = res_3char,
                segid = ' '
            )

            #for pred_atom_name, pred_atom_pos, ref_atom in zip(atom14_atoms, pred_residue_pos, ref_atoms):
            for pred_atom_name, pred_atom_pos in zip(atom14_atoms, pred_residue_pos):
                if pred_atom_name == '':
                    continue 
                
                #assert pred_atom_name == ref_atom.id
                #ref_atom.set_coord(pred_atom_pos.tolist())

                atom_idx += 1

                pred_atom = Atom(
                    pred_atom_name, #ref_atom.name, 
                    pred_atom_pos.tolist(),
                    100.0, #ref_atom.bfactor, #TODO: add confidence here instead
                    1.0, #ref_atom.occupancy,
                    ' ', #ref_atom.altloc,
                    pred_atom_name, #ref_atom.fullname,
                    atom_idx, #ref_atom.serial_number,
                    element = pred_atom_name[0], ##element = ref_atom.element,
                    # pqr_charge = ref_atom.pqr_charge,
                    # radius = ref_atom.radius,
                )

                pred_residue.add(pred_atom)

                #already_defined_in_this_chain.add(ref_atom.name)

            pred_chain.add(pred_residue)

        #pred_chains[ref_chain.id] = pred_chain

        pred_model.add(pred_chain)

        #also add the reference chain
        #pred_model.add(ref_chain)


    # for chain_id, chain in pred_chains.items():
    #     #pred_structure.add(chain)
    #     pred_model.add(chain)

    pred_structure = Structure(pdb_id)    
    pred_structure.add(pred_model)
    #pred_structure.add([m for m in ref_structure][0]) #take the first model

    return pred_structure

def biopython_structure_to_mmcif_file(biopython_structure, store_chains:List[str], output_mmcif_filename:str,    
    verbose:int=0,
):
    io = MMCIFIO()
    os.makedirs(os.path.dirname(output_mmcif_filename), exist_ok=True)    
    io.set_structure(biopython_structure)
    io.save(output_mmcif_filename, select=ChainsSelector(store_chains))
    if verbose>0:
        print(f'saved biopython structure into: {output_mmcif_filename}')



class ChainsSelector(Select):
    def __init__(self, keep_chains:List[str]):
        self._keep_chains = deepcopy(keep_chains)

    def __repr__(self):
        """Represent the output as a string for debugging."""
        return f'selects only the chains: {self._keep_chains}'

    # def accept_model(self, model):
    #     """Overload this to reject models for output."""
    #     return 1

    def accept_chain(self, chain):
        """Overload this to reject chains for output."""
        return chain.id in self._keep_chains
        #return 1

    # def accept_residue(self, residue):
    #     """Overload this to reject residues for output."""
    #     return 1

    # def accept_atom(self, atom):
    #     """Overload this to reject atoms for output."""
    #     return 1


def is_pdb_id(pdb_id:str):
    return 4==len(pdb_id)

def get_mmcif_native_full_name(pdb_id, strict=False):
    if not is_pdb_id(pdb_id):
        if strict:
            raise Exception(f'pdb_id is expected to be 4 letters long, but got {pdb_id} - note, 8 letters pdb id is not supported yet.')
        else:
            return pdb_id
    if 'PDB_DIR' not in os.environ:
        raise Exception('Please set an environment var named PDB_DIR pointing to the downloaded pdb structures. You can use localpdb (pip installable) to maintain such directory.')
    pdb_dir = os.environ['PDB_DIR']
    ans = f'{pdb_dir}/mmCIF/{pdb_id[1:3]}/{pdb_id}.cif.gz'        
    return ans

def load_mmcif_biopython(filename:str, structure_name:str='bananaphone'):
    if is_pdb_id(filename):
        filename = get_mmcif_native_full_name(filename)
    print(f'loading: {filename}')
    use_open = gzip.open if filename.endswith('.gz') else open
    parser = MMCIFParser()
    with use_open(filename, 'rt') as fh:
        structure = parser.get_structure(structure_name, fh)
    return structure

