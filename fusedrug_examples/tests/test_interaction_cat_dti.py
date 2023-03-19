import unittest
import os
from fusedrug_examples.interaction.drug_target.affinity_prediction.cat_dti.runner import main
from omegaconf import OmegaConf
from pathlib import Path
import tempfile
import shutil


class InterCatDtiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()

    def test_cat_dti(self) -> None:
        config_path = Path(__file__, "../../interaction/drug_target/affinity_prediction/cat_dti/config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.paths.root_dir = os.path.join(self.root, "test_cat_dti")
        cfg.params.train.num_epochs = 1
        main(cfg)

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
