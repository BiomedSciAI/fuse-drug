import unittest
from omegaconf import OmegaConf
from pathlib import Path
import torch

from fusedrug_examples.interaction.drug_target.affinity_prediction.bimodal_mca.runner import run_train_and_val
from fusedrug_examples.interaction.drug_target.affinity_prediction.bimodal_mca.data import AffinityDataModule
from fusedrug_examples.interaction.drug_target.affinity_prediction.bimodal_mca.model import AffinityPredictorModule


class BimodalMCATestCase(unittest.TestCase):
    def setUp(self) -> None:
        print(__file__)
        self.config_path = Path(
            __file__, "../../interaction/drug_target/affinity_prediction/bimodal_mca/configs/train_config.yaml"
        )
        self.cfg = OmegaConf.load(self.config_path)

    def test_data_and_model(self) -> None:
        device = torch.device("cuda")

        lightning_data = AffinityDataModule(**self.cfg["data"]["lightning_data_module"])
        train_dl = lightning_data.train_dataloader()

        batch = next(iter(train_dl))

        lightning_module = AffinityPredictorModule(**self.cfg["model"])

        lightning_module.model

        # curren
        device = next(lightning_module.model.parameters()).device
        smiles = batch["data.input.tokenized_ligand"].to(device=device)
        proteins = batch["data.input.tokenized_protein"].to(device=device)
        print("-" * 10)
        print(smiles)
        print(device)
        print("-" * 10)
        lightning_module.forward(smiles, proteins)

        # lightning_module.training_step(batch, 0)

    def test_runner(self) -> None:
        self.cfg.trainer.max_epochs = 1

        run_train_and_val(self.cfg)


if __name__ == "__main__":
    unittest.main()
