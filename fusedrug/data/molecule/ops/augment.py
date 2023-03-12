import numpy as np
from rdkit import Chem
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id


class SmilesRandomizeAtomOrder(OpBase):
    """
    Randomizes the order of a smiles string representation of a molecule (while preserving the molecule structure)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key: str = "data.input.ligand"):
        mol = sample_dict[key]

        if not isinstance(mol, Chem.rdchem.Mol):
            sid = get_sample_id(sample_dict)
            raise Exception(
                f"sample_id={sid}: expected key_in={key} to point to RDKit mol, but instead got {type(mol)}. Note - you can use SmilesToRDKitMol Op to convert a smiles string to RDKit mol"
            )

        mol = randomize_rdkit_mol_atoms_order(mol)
        sample_dict[key] = mol

        return sample_dict


def randomize_smiles_atom_order(smiles_str):
    """
    based on https://github.com/XinhaoLi74/SmilesPE/blob/19d4775f664ea3cf5e4dd6592942e8d66032bbe7/SmilesPE/learner.py#L20

    shuffle the order of atoms while preserving the molecule entity
    """
    mol = Chem.MolFromSmiles(smiles_str, sanitize=False)
    mol = randomize_rdkit_mol_atoms_order(mol)

    # https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles
    return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True, kekuleSmiles=False)


def randomize_rdkit_mol_atoms_order(mol):
    if not isinstance(mol, Chem.rdchem.Mol):
        raise Exception(f"Expected mol to be Chem.rdchem.Mol but got {type(mol)}")
    ans = list(range(mol.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(mol, ans)
    return nm
