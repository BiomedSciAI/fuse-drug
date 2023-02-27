from fuse.utils import NDict
from fuse.utils.cpu_profiling import Timer
from fuse.data import OpBase
from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import DTIBindingDataset
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.Contrastive_PLM_DTI.src.featurizers import Featurizer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from fuse.data.utils.collates import CollateDefault

class FeaturizeTarget(OpBase):
    def __init__(self,
        dataset: DTIBindingDataset, 
        featurizer: Featurizer,
        device: str="cpu",
        num_workers: int=16,
        debug:bool=False,
        ):
        """
        Apply target featurizer on amino acid sequence
        """
        super().__init__()
        self.debug = debug
        if self.debug:
            print("debug mode. target featurizer will return a random dummy tensor")
            return  
        self.dataset = dataset
        self.featurizer = featurizer
        self._device = device
        # get unique target amino acid strings
        all_targets = []
        # helper loader to multiprocess obtaining the unique drugs
        loader = DataLoader( 
                    self.dataset,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=CollateDefault(skip_keys=("ground_truth_activity_value",)),
                    batch_size=1000
                )
        with Timer("Getting unique targets..."):
            for data in loader:
                all_targets += data['target_str']
            all_targets = list(set(all_targets))
        if self._device == "cuda":
            self.featurizer.cuda(self._device)
        # Not writing target featurizer to disk. will obtain the features during preload from the model directly
        #if not self.featurizer.path.exists():
        #    with Timer("Featurizing targets and writing to disk..."):
        #        self.featurizer.write_to_disk(all_targets)
        with Timer("Preloading featurized targets..."):
            self.featurizer.preload(all_targets, write_first=False)
        self.featurizer.cpu()

    def __call__(self, sample_dict: NDict, 
        key_out_target:str='data.input.target',
        ):
        
        """
        """
        if self.debug:
            sample_dict[key_out_target] = torch.randn(1024) # dummy input
            return sample_dict

        sample_dict[key_out_target] = self.featurizer(sample_dict[key_out_target]) 
     
        return sample_dict


        

        
