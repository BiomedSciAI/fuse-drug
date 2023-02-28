
import os
import torch
import tensorflow as tf

import pytorch_lightning as pl
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from paccmann_predictor.models.bimodal_mca import BimodalMCA
MODEL_FACTORY = {    
    'bimodal_mca': BimodalMCA,
}
import numpy as np
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style


class AffinityPredictorModule(pl.LightningModule):
    def __init__(self, 
        **model_params : dict, #will be passed to base model        
        ) -> None:
        '''
        Affinity Predictor

        Args:
            
        '''
        super().__init__()
        
        assert 'learning_rate' in model_params

        self.learning_rate = model_params['learning_rate']
        
        self.model_params = model_params
        self.base_model = self.model_params['base_model']
        self.save_hyperparameters()#ignore=['peptide_language_model','protein_language_model'])
        
        self.model = MODEL_FACTORY[self.base_model](self.model_params)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print('total model num_params=',self.num_params)
        

    def forward(self, smiles: torch.Tensor, proteins: torch.Tensor) -> torch.Tensor:
        return self.model.forward(smiles, proteins)[0]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        
        #smiles, proteins, y = batch
        smiles = batch['data.input.tokenized_ligand']
        proteins = batch['data.input.tokenized_protein']
        y = batch['data.gt.affinity_val']
        
        y_hat = self(smiles, proteins)
        return self.model.loss(y_hat, y)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        
        #smiles, proteins, y = batch

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
        
        #smiles, proteins, y = batch
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
