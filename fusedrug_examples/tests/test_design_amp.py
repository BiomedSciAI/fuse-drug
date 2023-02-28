import unittest
import fusedrug_examples.design.amp.classifier.main_classifier_train as main_classifier_train
import fusedrug_examples.design.amp.design.main_design_train as main_design_train
from omegaconf import OmegaConf
from pathlib import Path
class DesignAMPTestCase(unittest.TestCase):

    def test_classifier(self):
        config_path = Path(__file__, "../../design/amp/classifier/config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.train.trainer_kwargs.max_epochs = 1
        cfg.data.data_loader.num_workers=16
        cfg.data.batch_size = 32
        cfg.data.num_batches = 10
        main_classifier_train.main(cfg)

    def test_design(self):
        config_path = Path(__file__, "../../design/amp/design/config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.train.num_iter = 1
        cfg.train.trainer_kwargs.max_epochs = 1
        cfg.data.data_loader.num_workers=16
        cfg.data.batch_size = 32
        cfg.data.num_batches = 10
        main_design_train.main(cfg)


if __name__ == "__main__":
    unittest.main()