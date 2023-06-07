import unittest
from fusedrug.utils.file_formats import IndexedFasta
from fusedrug.tests_data import get_tests_data_dir
import os


class TestIndexedFasta(unittest.TestCase):
    def test_indexed_fasta(self) -> None:
        file_path = os.path.join(get_tests_data_dir(), "example_viral_proteins.fasta")
        ifasta = IndexedFasta(file_path)

        seq_by_index = str(ifasta[10])
        seq_by_name = str(ifasta[ifasta[10].name])

        self.assertEqual(len(ifasta), 895)
        self.assertEqual(seq_by_index, seq_by_name)


if __name__ == "__main__":
    unittest.main()
