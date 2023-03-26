import unittest
from omegaconf import OmegaConf
from pathlib import Path
import torch
import tempfile
import os
import shutil
from hydra.core.utils import setup_globals
from fusedrug_examples.interaction.drug_target.affinity_prediction.bimodal_mca.runner import run_train_and_val
from fusedrug_examples.interaction.drug_target.affinity_prediction.bimodal_mca.data import AffinityDataModule
from fusedrug_examples.interaction.drug_target.affinity_prediction.bimodal_mca.model import AffinityPredictorModule


class BimodalMCATestCase(unittest.TestCase):
    def setUp(self) -> None:
        # load and modify config file
        self.config_path = Path(
            __file__, "../../interaction/drug_target/affinity_prediction/bimodal_mca/configs/train_config.yaml"
        )
        self.cfg = OmegaConf.load(self.config_path)
        self.cfg.data.lightning_data_module.num_workers = 0  # TODO change to 10
        os.environ["BIMCA_RESULTS"] = tempfile.mkdtemp()
        setup_globals()  # defines resolver 'now'

    def test_data_and_model(self) -> None:
        """
        basic test
        """
        # get device
        device = torch.device("cuda")

        # get data
        lightning_data = AffinityDataModule(**self.cfg["data"]["lightning_data_module"])
        train_dl = lightning_data.train_dataloader()
        batch = next(iter(train_dl))

        # split and move data to device
        smiles = batch["data.input.tokenized_ligand"].to(device=device)
        proteins = batch["data.input.tokenized_protein"].to(device=device)

        # create module and move to device
        lightning_module = AffinityPredictorModule(**self.cfg["model"])
        lightning_module.to(device)

        # forward pass
        lightning_module.forward(smiles, proteins)

    def test_runner(self) -> None:
        """
        end2end test - run a short runner
        """
        # short run
        self.cfg.trainer.max_epochs = 1
        run_train_and_val(self.cfg)

    def tearDown(self) -> None:
        shutil.rmtree(os.environ["BIMCA_RESULTS"])


if __name__ == "__main__":
    unittest.main()
