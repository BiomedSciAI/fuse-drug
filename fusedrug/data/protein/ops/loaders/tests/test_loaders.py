import unittest
from fusedrug.data.protein.ops.loaders.fasta_loader import FastaLoader
import os
from fusedrug import get_tests_data_dir
from fuse.data import create_initial_sample

class TestLoaders(unittest.TestCase):

    def test_fasta_loader(self):
        fasta_file_loc = os.path.join(get_tests_data_dir(), 'example_viral_proteins.fasta')
        op = FastaLoader(fasta_file_loc=fasta_file_loc)

        sample_1 = create_initial_sample(100)    
        sample_1 = op(sample_1)

        sample_2 = create_initial_sample('YP_009047135.1')
        sample_2 = op(sample_1)

        self.assertEqual(sample_1['data.gt.seq'], sample_2['data.gt.seq'])


        banana=123


if __name__ == '__main__':
    unittest.main()
