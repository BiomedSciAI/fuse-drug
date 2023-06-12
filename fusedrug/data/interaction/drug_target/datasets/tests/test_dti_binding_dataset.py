import os
import unittest
import random

from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import dti_binding_dataset


class TestDTIBindingDatasetFuseWrapper(unittest.TestCase):
    """
    Unit test for dti_binding_dataset
    """

    def setUp(self) -> None:
        self.pairs_columns_to_extract = ["ligand_id", "target_id", "activity_value", "activity_label"]
        self.pairs_rename_columns = {
            "activity_value": "ground_truth_activity_value",
            "activity_label": "ground_truth_activity_label",
        }
        self.ligands_columns_to_extract = ["canonical_smiles"]
        self.ligands_rename_columns = {"canonical_smiles": "ligand_str"}
        self.targets_columns_to_extract = ["canonical_aa_sequence"]
        self.targets_rename_columns = {"canonical_aa_sequence": "target_str"}

    def test_small_dti_binding_dataset(self) -> None:
        """
        Test using small portion of the data.

        see ./test_data/README.md for more details about the subset of the data
        """

        # data paths
        test_data_dir = os.path.join(os.path.dirname(__file__), "data")
        small_pairs_tsv = os.path.join(test_data_dir, "small_pairs.tsv")
        small_ligands_tsv = os.path.join(test_data_dir, "small_ligands.tsv")
        small_targets_tsv = os.path.join(test_data_dir, "small_targets.tsv")

        # test logic
        self.ds_tester(
            pairs_tsv=small_pairs_tsv,
            ligands_tsv=small_ligands_tsv,
            targets_tsv=small_targets_tsv,
            pairs_columns_to_extract=self.pairs_columns_to_extract,
            pairs_rename_columns=self.pairs_rename_columns,
            ligands_columns_to_extract=self.ligands_columns_to_extract,
            ligands_rename_columns=self.ligands_rename_columns,
            targets_columns_to_extract=self.targets_columns_to_extract,
            targets_rename_columns=self.targets_rename_columns,
        )

    def test_full_dti_binding_dataset(self) -> None:
        """
        Test DTIBindingDataset with full tsv data.

        By default this test will be skipped due directory permissions and high memory usage.
        If you wish to use this test, please set 'TEST_FULL_BINDING_DATASET' environment variable to True/1.
        """

        if "BINDING_DATA_DIR_PATH" not in os.environ:
            self.skipTest("Missing env variable 'BINDING_DATA_DIR_PATH'")

        base_data_dir = os.environ["BINDING_DATA_PATH"]

        full_pairs_tsv = os.path.join(
            base_data_dir, "pubchem_13-07-2022@native@single_protein_target@affinity_pairs_v0.1.tsv"
        )
        full_ligands_tsv = os.path.join(base_data_dir, "pubchem_13-07-2022@native@single_protein_target@ligands.tsv")
        full_targets_tsv = os.path.join(base_data_dir, "pubchem_13-07-2022@native@single_protein_target@targets.tsv")

        # test logic
        self.ds_tester(
            pairs_tsv=full_pairs_tsv,
            ligands_tsv=full_ligands_tsv,
            targets_tsv=full_targets_tsv,
            pairs_columns_to_extract=self.pairs_columns_to_extract,
            pairs_rename_columns=self.pairs_rename_columns,
            ligands_columns_to_extract=self.ligands_columns_to_extract,
            ligands_rename_columns=self.ligands_rename_columns,
            targets_columns_to_extract=self.targets_columns_to_extract,
            targets_rename_columns=self.targets_rename_columns,
        )

    def ds_tester(self, pairs_tsv: str, ligands_tsv: str, targets_tsv: str, **kwargs: dict) -> None:
        """
        helper function that operates all the test logic (since the two tests differ only by their data paths).
        """

        # creating ds

        ds = dti_binding_dataset(pairs_tsv=pairs_tsv, ligands_tsv=ligands_tsv, targets_tsv=targets_tsv, **kwargs)

        rand_idx = random.choice(list(ds.dynamic_pipeline.ops[0]._data.keys()))

        rand_sample = ds[rand_idx]
        print(rand_sample)


if __name__ == "__main__":
    unittest.main()
