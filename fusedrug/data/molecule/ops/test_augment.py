import unittest
from rdkit import Chem

from fuse.utils.ndict import NDict
from fuse.data.pipelines.pipeline_default import PipelineDefault

from fusedrug.data.molecule.ops import SmilesRandomizeAtomOrder, SmilesToRDKitMol, RDKitMolToSmiles


class TestMoleculesAugmentations(unittest.TestCase):
    """
    tests 'SmilesRandomizeAtomOrder' augmentation op
    """

    def test_smiles_random_order(self) -> None:
        """
        unit-tests SmilesRandomizeAtomOrder op
        """

        # create a sample
        sample = NDict()
        orig_mol = "COc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1"
        sample["data.input.ligand"] = orig_mol

        # create a pipeline with aug
        pipeline = PipelineDefault(
            "test_pipeline",
            [
                (SmilesToRDKitMol(), dict(key_in="data.input.ligand", key_out="data.input.ligand")),
                (SmilesRandomizeAtomOrder(), dict(key="data.input.ligand")),  # augment
                (RDKitMolToSmiles(), dict(key_in="data.input.ligand", key_out="data.input.ligand")),
            ],
        )

        # process augmentations
        aug_mol = pipeline(sample)["data.input.ligand"]
        aug_mol2 = pipeline(sample)["data.input.ligand"]

        # check that the all three represent the same molecule
        assert Chem.CanonSmiles(orig_mol) == Chem.CanonSmiles(aug_mol)
        assert Chem.CanonSmiles(orig_mol) == Chem.CanonSmiles(aug_mol2)


if __name__ == "__main__":
    unittest.main()
