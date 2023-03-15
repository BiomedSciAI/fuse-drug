import unittest
import os
from fusedrug.data.interaction.drug_target.datasets.fuse_style_dti import DTIDataset


class TestFuseStyleDTIDataset(unittest.TestCase):
    def test_dataset(self) -> None:
        """
        Sanity check
        """
        train_data_path = os.path.join(os.environ["BINDINGDB_SMALL"], "train.csv")

        dataset = DTIDataset.dataset(data_path=train_data_path)
        sample = dataset[42]


if __name__ == "__main__":
    unittest.main()
