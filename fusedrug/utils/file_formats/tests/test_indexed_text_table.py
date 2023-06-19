import unittest
from fusedrug.utils.file_formats import IndexedTextTable
from fusedrug.tests_data import get_tests_data_dir
import os


class TestIndexedTextTable(unittest.TestCase):
    def test_no_headline(self) -> None:
        ligands_smi_path = os.path.join(
            get_tests_data_dir(), "ligands_1k_no_columns_headline.smi"
        )
        itt = IndexedTextTable(
            ligands_smi_path,
            seperator="\t",
            columns_names=(
                "smiles_string",
                "smiles_id",
            ),
            first_row_is_columns_names=False,
            id_column_idx=1,
            allow_access_by_id=True,
            force_recreate_index=True,
        )

        mol_id, mol_data = itt[100]
        mol_id_2, mol_data_2 = itt[mol_id]

        self.assertEqual(mol_id, mol_id_2)
        self.assertTrue(all(mol_data == mol_data_2))

    def test_with_headline(self) -> None:
        ligands_smi_path = os.path.join(
            get_tests_data_dir(), "ligands_1k_with_columns_headline.smi"
        )
        itt = IndexedTextTable(
            ligands_smi_path,
            seperator="\t",
            first_row_is_columns_names=True,
            id_column_idx=1,
            allow_access_by_id=True,
            force_recreate_index=True,
        )

        mol_id_1, mol_data_1 = itt[100]
        mol_id_2, mol_data_2 = itt[mol_id_1]

        self.assertEqual(mol_id_1, mol_id_2)
        self.assertTrue(all(mol_data_1 == mol_data_2))

        ## now with providing custom columns names
        itt_custom_col_names = IndexedTextTable(
            ligands_smi_path,
            seperator="\t",
            first_row_is_columns_names=True,
            columns_names=(
                "banoni",
                "banani_id",
            ),
            id_column_idx=1,
            allow_access_by_id=True,
        )

        mol_id_3, mol_data_3 = itt_custom_col_names[100]
        mol_id_4, mol_data_4 = itt_custom_col_names[mol_id_3]

        self.assertEqual(mol_id_3, mol_id_4)
        self.assertTrue(all(mol_data_3 == mol_data_4))

        self.assertEqual(mol_id_1, mol_id_4)

        self.assertEqual(mol_data_1[0], mol_data_4[0])
        self.assertEqual(mol_data_1[1], mol_data_4[1])


if __name__ == "__main__":
    unittest.main()
