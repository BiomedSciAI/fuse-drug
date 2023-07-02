from typing import Optional
from rdkit import Chem
from rdkit.Chem.rdmolops import SanitizeFlags
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id


class SmilesToRDKitMol(OpBase):
    """
    Converts a smiles string into Chem.rdchem.Mol molecule representation
    """

    def __init__(self, sanitize: bool = False, verbose: int = 0, **kwargs: dict):
        super().__init__(**kwargs)
        self._verbose = verbose
        self._sanitize = sanitize

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str = "data.input.ligand_str",
        key_out: str = "data.input.ligand",
    ) -> NDict:
        smiles_seq = sample_dict[key_in]
        if not isinstance(smiles_seq, str):
            raise Exception(
                f"Expected key_in={key_in} to point to a string, and instead got a {type(smiles_seq)}"
            )

        try:
            mol = Chem.MolFromSmiles(smiles_seq, sanitize=self._sanitize)
            sample_dict[key_out] = mol
        except:
            if self._verbose > 0:
                sid = get_sample_id(sample_dict)
                print(
                    f"ERROR in sample_id={sid}: The following smiles string could not be loaded by Chem.MolFromSmiles: {smiles_seq} - dropping sample."
                )
            return None

        return sample_dict


class RDKitMolToSmiles(OpBase):
    def __init__(
        self,
        verbose: int = 1,
        kwargs_SmilesFromMol: Optional[dict] = None,
        **kwargs: dict,
    ):
        """
        Creates an Op which converts Chem.rdchem.Mol to smiles string

        IMPORANT - you can provide kwargs_SmilesFromMol to pass custom kwargs to Chem.SmilesFromMol
        For example - when using atom-order shuffling augmentation, this is critical, because the default behavior is to **cannonize** the smiles str,
            which defeats the purpose of shuffling the atoms order...

        Args: kwargs_SmilesFromMol - provide a dictionary of kwargs to pass to SmilesFromMol.
            if None is provided, defaults to kwargs_SmilesFromMol=dict(canonical=False)

        """

        super().__init__(**kwargs)
        self._verbose = verbose
        self._kwargs_SmilesFromMol = kwargs_SmilesFromMol
        if self._kwargs_SmilesFromMol is None:
            self._kwargs_SmilesFromMol = dict(canonical=False)

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str = "data.input.ligand",
        key_out: str = "data.input.ligand_str",
    ) -> NDict:
        mol = sample_dict[key_in]
        if not isinstance(mol, Chem.rdchem.Mol):
            raise Exception(
                f"expected key_in={key_in} to point to RDKit mol, but instead got {type(mol)}. Note - you can use SmilesToRDKitMol Op to convert a smiles string to RDKit mol"
            )

        try:
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles
            smiles_str = Chem.MolToSmiles(mol, **self._kwargs_SmilesFromMol)
            sample_dict[key_out] = smiles_str
        except:
            if self._verbose > 0:
                sid = get_sample_id(sample_dict)
                print(
                    f"ERROR in sample_id={sid}: RDKitMolToSmiles could not process it - dropping sample."
                )
            return None

        return sample_dict


class SanitizeMol(OpBase):
    """
    Discards the sample (returns None) if the smiles string representation seems to be invalid
    """

    def __init__(
        self,
        sanitize_flags: SanitizeFlags = Chem.rdmolops.SANITIZE_NONE,
        verbose: int = 1,
        **kwargs: dict,
    ):
        """
        Args:
            sanitize_flags: use flags from https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.SanitizeFlags
                for example - sanitize_flags=SANITIZE_KEKULIZE | SANITIZE_CLEANUP | SANITIZE_PROPERTIES
            verbose: set to 0 to silence
        """
        super().__init__(**kwargs)
        self._verbose = verbose
        self._sanitize_flags = sanitize_flags

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        mol = sample_dict[key]

        if not isinstance(mol, Chem.rdchem.Mol):
            raise Exception(
                f"expected key_in={key} to point to RDKit mol, but instead got {type(mol)}. Note - you can use SmilesToRDKitMol Op to convert a smiles string to RDKit mol"
            )

        try:
            # note - it modifies the mol inplace
            Chem.SanitizeMol(mol, self._sanitize_flags)
        except:
            if self._verbose > 0:
                sid = get_sample_id(sample_dict)
                print(
                    f"The following smiles string did not pass SanitizeSmiles: {Chem.MolToSmiles(mol)} - dropping sample. Sample ID = {sid}"
                )
            return None

        return sample_dict
