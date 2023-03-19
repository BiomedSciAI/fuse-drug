import unittest
import fusedrug_examples.design.amp.classifier.main_classifier_train as main_classifier_train
import fusedrug_examples.design.amp.design.main_design_train as main_design_train
from omegaconf import OmegaConf
from pathlib import Path
import tempfile
import shutil
import os
import time
from fuse.utils.file_io.file_io import create_dir


class DesignAMPTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.start_time = time.time()

    def test_classifier(self) -> None:
        config_path = Path(__file__, "../../design/amp/classifier/config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.data.data_loader.num_workers = 16
        cfg.train.trainer_kwargs.max_epochs = 1
        cfg.data.batch_size = 32
        cfg.data.num_batches = 10
        cfg.data.peptides_datasets.uniprot_raw_data_path_reviewed = None
        cfg.data.peptides_datasets.uniprot_raw_data_path_not_reviewed = None
        cfg.root = os.path.join(self.root, "cls")
        create_dir(cfg.root)
        main_classifier_train.main(cfg)

    def test_design(self) -> None:
        config_path = Path(__file__, "../../design/amp/design/config.yaml")
        cfg = OmegaConf.load(config_path)
        cfg.train.num_iter = 1
        cfg.train.trainer_kwargs.max_epochs = 1
        cfg.data.data_loader.num_workers = 16
        cfg.data.batch_size = 32
        cfg.data.num_batches = 10
        cfg.data.peptides_datasets.uniprot_raw_data_path_reviewed = None
        cfg.data.peptides_datasets.uniprot_raw_data_path_not_reviewed = None
        cfg.root = os.path.join(self.root, "design")
        create_dir(cfg.root)
        main_design_train.main(cfg)

    def tearDown(self) -> None:
        # Delete temporary directories
        shutil.rmtree(self.root)
        t = time.time() - self.start_time
        print("%s: %.3f" % (self.id(), t))


if __name__ == "__main__":
    unittest.main()
