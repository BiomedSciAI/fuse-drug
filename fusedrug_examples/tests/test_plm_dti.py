import os
import unittest
from omegaconf import OmegaConf
from pathlib import Path

# from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.runner import (
#     main,
# )
import tempfile
import shutil


class PLMDTITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()

    @unittest.skip("'dgl' lib causing issues")
    def test_main(self) -> None:
        config_path = Path(
            __file__,
            "../../interaction/drug_target/affinity_prediction/PLM_DTI/configs/config.yaml",
        )
        cfg = OmegaConf.load(config_path)
        cfg.trainer.epochs = 2
        cfg.paths.results = os.path.join(self.root, "test_plm_dti")
        main(cfg)  # noqa

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
