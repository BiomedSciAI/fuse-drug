from typing import Optional
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import (
    DTIBindingDataset,
)
from fuse.utils.cpu_profiling import Timer
import torch
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.Contrastive_PLM_DTI.src.featurizers import (
    Featurizer,
)
from torch.utils.data import DataLoader
from fuse.data.utils.collates import CollateDefault


class FeaturizeDrug(OpBase):
    def __init__(
        self,
        dataset: Optional[DTIBindingDataset],
        all_drugs: Optional[list],
        featurizer: Featurizer,
        device: str = "cpu",
        num_workers: int = 16,
        debug: bool = False,
    ):
        """
        Apply drug featurizer on SMILES string
        """
        super().__init__()
        self.debug = debug
        if self.debug:
            print("debug mode. drug featurizer will return a random dummy tensor")
            return
        self.dataset = dataset
        self.featurizer = featurizer
        self._device = device
        # get unique drug SMILES strings
        if all_drugs is None:
            all_drugs = []
            # helper loader to multiprocess obtaining the unique drugs
            loader = DataLoader(
                self.dataset,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=CollateDefault(skip_keys=("ground_truth_activity_value",)),
                batch_size=1000,
            )
            with Timer("Getting unique drugs..."):
                for data in loader:
                    all_drugs += data["ligand_str"]
                all_drugs = list(set(all_drugs))

        if self._device == "cuda":
            self.featurizer.cuda(self._device)
        # Not writing drug featurizer to disk due to its size. will obtain the morgan fingerprints dynamically
        # if not self.featurizer.path.exists():
        #    with Timer("Featurizing drugs and writing to disk..."):
        #        self.featurizer.write_to_disk(all_drugs)
        with Timer("Preloading featurized drugs..."):
            self.featurizer.preload(all_drugs, write_first=False)
        self.featurizer.cpu()

    def __call__(self, sample_dict: NDict, key_out_ligand: str = "data.input.ligand"):

        """ """
        if self.debug:
            sample_dict[key_out_ligand] = torch.randint(
                2, (2048,), dtype=torch.float32
            )  # dummy input
            return sample_dict

        sample_dict[key_out_ligand] = self.featurizer(sample_dict[key_out_ligand])

        return sample_dict
