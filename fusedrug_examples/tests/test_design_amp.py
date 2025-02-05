import unittest
import fusedrug_examples.design.amp.classifier.main_classifier_train as main_classifier_train
import fusedrug_examples.design.amp.design.main_design_train as main_design_train
from omegaconf import OmegaConf
from pathlib import Path
import tempfile
import shutil
import os
from fuse.utils.file_io.file_io import create_dir
import torch
from fuse.utils.tests.decorators import skipIfMultiple


@skipIfMultiple(
    ("define environment variable 'CINC_TEST_DATA_PATH' to run this test",),
    (not torch.cuda.is_available(), "No GPU is available"),
    (True, "removing this test until we solve the GLIBCXX_3.4.29 not found issue"),
)
class DesignAMPTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()

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


if __name__ == "__main__":
    unittest.main()
