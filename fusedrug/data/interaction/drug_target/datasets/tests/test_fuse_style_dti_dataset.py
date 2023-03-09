import unittest
from fusedrug.data.interaction.drug_target.datasets.fuse_style_dti import DTIDataset


class TestFuseStyleDTIDataset(unittest.TestCase):
    def test_dataset(self):
        """
        Sanity check
        """

        TRAIN_URL = "https://raw.githubusercontent.com/samsledje/ConPLex/main/dataset/BindingDB/train.csv"

        dataset = DTIDataset.dataset(data_path=TRAIN_URL)
        sample = dataset[42]
