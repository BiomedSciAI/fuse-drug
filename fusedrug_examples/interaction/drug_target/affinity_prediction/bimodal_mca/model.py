import pytorch_lightning as pl
from paccmann_predictor.models.bimodal_mca import BimodalMCA
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

class AffinityPredictorModule(pl.LightningModule):
    def __init__(self,        
                 **model_params : dict, #will be passed to BimodalMCA        
                ) -> None:
        """
        Affinity Predictor            
        """
        super().__init__()

        assert 'learning_rate' in model_params

        self.learning_rate = model_params['learning_rate']
        
        self.model_params = model_params
        self.save_hyperparameters()

        self.model = BimodalMCA(self.model_params)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print('total model num_params=',self.num_params)        

    def forward(self, smiles: torch.Tensor, proteins: torch.Tensor) -> torch.Tensor:
        return self.model.forward(smiles, proteins)[0]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        
        smiles = batch['data.input.tokenized_ligand']
        proteins = batch['data.input.tokenized_protein']
        y = batch['data.gt.affinity_val']
        y_hat = self(smiles, proteins)
        return self.model.loss(y_hat, y)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        
        smiles = batch['data.input.tokenized_ligand']
        proteins = batch['data.input.tokenized_protein']
        y = batch['data.gt.affinity_val']
        
        y_hat = self(smiles, proteins)
        loss = self.model.loss(y_hat, y)
        y = y.cpu().detach().squeeze().tolist()
        y_hat = y_hat.cpu().detach().squeeze().tolist()
        rmse = np.sqrt(mean_squared_error(y_hat, y))
        pearson = pearsonr(y_hat, y)[0]
        spearman = spearmanr(y_hat, y)[0]
        self.log("val_loss", loss, batch_size=smiles.shape[0])
        self.log("val_rmse", rmse, batch_size=smiles.shape[0])
        self.log("val_pearson", pearson, batch_size=smiles.shape[0])
        self.log("val_spearman", spearman, batch_size=smiles.shape[0])

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        
        smiles = batch['data.input.tokenized_ligand']
        proteins = batch['data.input.tokenized_protein']
        y = batch['data.gt.affinity_val']
        y_hat = self(smiles, proteins)
        loss = self.model.loss(y_hat, y)

        y = y.cpu().detach().squeeze().tolist()
        y_hat = y_hat.cpu().detach().squeeze().tolist()
        rmse = np.sqrt(mean_squared_error(y_hat, y))
        pearson = pearsonr(y_hat, y)[0]
        spearman = spearmanr(y_hat, y)[0]
        self.log("test_loss", loss, batch_size=smiles.shape[0])
        self.log("test_rmse", rmse, batch_size=smiles.shape[0])
        self.log("test_pearson", pearson, batch_size=smiles.shape[0])
        self.log("test_spearman", spearman, batch_size=smiles.shape[0])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
