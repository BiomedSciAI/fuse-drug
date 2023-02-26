import unittest
from fusedrug.utils.file_formats import SDFIterableDataset
from fusedrug.tests_data import get_tests_data_dir
import os
from rdkit import Chem

class TestSDFIterableDataset(unittest.TestCase):
    def test_sdf_iteration(self):
        file_path = os.path.join(get_tests_data_dir(), 'tiny_chembl_30.sdf')
        it = SDFIterableDataset(file_path)

        mols_so_far = 0
        mols_ok = 0
        for mol in iter(it):
            mols_so_far += 1
            try:
                smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True, kekuleSmiles=False)
                mols_ok += 1
            except:
                pass

        print('total molecules=', mols_so_far)
        print(f'{mols_ok}/{mols_so_far} {(mols_ok/mols_so_far)*100.0:.2f}')
        
        self.assertEqual(mols_ok, 19)
        self.assertEqual(mols_so_far, 19)
            
if __name__ == '__main__':
    unittest.main()

