import unittest
from fusedrug.utils.file_formats import IndexedTextFile
from fusedrug.tests_data import get_tests_data_dir
import os

class TestIndexedTextFile(unittest.TestCase):
    def test_no_headline(self):
        ligands_smi_path = os.path.join(get_tests_data_dir(), 'ligands_1k_no_columns_headline.smi')
        itf = IndexedTextFile(ligands_smi_path,
            force_recreate_index=True)

        self.assertEqual(itf[100], 
            'CC[C@@](O)(c1cc2C(=O)N([C@@H](CC(O)=O)c3ccc(Cl)cc3)[C@](OC)(c2c(F)c1)c1ccc(Cl)cc1)C1(F)CCOCC1\t(3S)-3-(4-chlorophenyl)-3-[(1R)-1-(4-chlorophenyl)-7-fluoro-5-[1-(4-fluorooxan-4-yl)-1-hydroxypropyl]-1-methoxy-3-oxo-2,3-dihydro-1H-isoindol-2-yl]propanoic Acid::US10544132, Example 102::US10544132, Example 110\n')

    def test_with_headline(self):
        ligands_smi_path = os.path.join(get_tests_data_dir(), 'ligands_1k_with_columns_headline.smi')
        itf = IndexedTextFile(ligands_smi_path,
            force_recreate_index=True)

        self.assertEqual(itf[0], 
            'smiles_string\tsmiles_id\n')
        self.assertEqual(itf[100], 
            'CCO[C@]1(N([C@@H](CC(O)=O)c2ccc(Cl)cc2)C(=O)c2cc(cc(F)c12)[C@](O)(CC)C1(F)CCOCC1)c1ccc(Cl)cc1\tUS10544132, Example 111\n')
        self.assertEqual(itf[101],
            'CC[C@@](O)(c1cc2C(=O)N([C@@H](CC(O)=O)c3ccc(Cl)cc3)[C@](OC)(c2c(F)c1)c1ccc(Cl)cc1)C1(F)CCOCC1\t(3S)-3-(4-chlorophenyl)-3-[(1R)-1-(4-chlorophenyl)-7-fluoro-5-[1-(4-fluorooxan-4-yl)-1-hydroxypropyl]-1-methoxy-3-oxo-2,3-dihydro-1H-isoindol-2-yl]propanoic Acid::US10544132, Example 102::US10544132, Example 110\n'
        )

        #no with existing index
        itf = IndexedTextFile(ligands_smi_path)

        self.assertEqual(itf[0], 
            'smiles_string\tsmiles_id\n')
        self.assertEqual(itf[100], 
            'CCO[C@]1(N([C@@H](CC(O)=O)c2ccc(Cl)cc2)C(=O)c2cc(cc(F)c12)[C@](O)(CC)C1(F)CCOCC1)c1ccc(Cl)cc1\tUS10544132, Example 111\n')
        self.assertEqual(itf[101],
            'CC[C@@](O)(c1cc2C(=O)N([C@@H](CC(O)=O)c3ccc(Cl)cc3)[C@](OC)(c2c(F)c1)c1ccc(Cl)cc1)C1(F)CCOCC1\t(3S)-3-(4-chlorophenyl)-3-[(1R)-1-(4-chlorophenyl)-7-fluoro-5-[1-(4-fluorooxan-4-yl)-1-hydroxypropyl]-1-methoxy-3-oxo-2,3-dihydro-1H-isoindol-2-yl]propanoic Acid::US10544132, Example 102::US10544132, Example 110\n'
        )

            
if __name__ == '__main__':
    unittest.main()
