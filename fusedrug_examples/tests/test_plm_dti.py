import unittest
from omegaconf import OmegaConf
from pathlib import Path
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.runner import main, CONFIGS_DIR

class PLMDTITestCase(unittest.TestCase):
    def test_main(self):
        config_path = Path(__file__, "../../interaction/drug_target/affinity_prediction/PLM_DTI/configs/train_config.yaml")
        cfg = OmegaConf.load(config_path)
        main(cfg)

if __name__ == "__main__":
    unittest.main()