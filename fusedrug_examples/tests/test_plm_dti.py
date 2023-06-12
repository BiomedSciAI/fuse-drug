import os
import unittest
from omegaconf import OmegaConf
from pathlib import Path
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.runner import main
import tempfile
import shutil


class PLMDTITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()

    @unittest.skip(
        "Doesn't support Lightning >=2.0.0"
    )  # Support for `validation_epoch_end` has been removed in v2.0.0. `PLM_DTI_Module` implements this method. You can use the `on_validation_epoch_end` hook instead.
    def test_main(self) -> None:
        config_path = Path(
            __file__, "../../interaction/drug_target/affinity_prediction/PLM_DTI/configs/train_config.yaml"
        )
        cfg = OmegaConf.load(config_path)
        cfg.trainer.epochs = 2
        cfg.paths.results = os.path.join(self.root, "test_plm_dti")
        main(cfg)

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
