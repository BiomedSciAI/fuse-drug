
# The Contrastive_PLM_DTI submodule is the repository found at https://github.com/samsledje/Contrastive_PLM_DTI
# and described in the paper "Adapting protein language models for rapid DTI prediction": https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf 
from Contrastive_PLM_DTI.src.data import (
    DTIDataModule,
    TDCDataModule,
    DUDEDataModule,
    EnzPredDataModule,
)
import torch
from omegaconf import open_dict # to be able to add new keys to hydra dictconfig
from Contrastive_PLM_DTI.src.utils import get_featurizer
from Contrastive_PLM_DTI.src.featurizers import Featurizer
from utils import get_task_dir
from fuse.data.datasets.dataset_wrap_seq_to_dict import DatasetWrapSeqToDict
from torch.utils.data import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.pipelines.pipeline_default import PipelineDefault
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
from typing import Union, List, Dict, Optional
from fusedrug.data.interaction.drug_target.loaders.dti_binding_dataset_loader import DTIBindingDatasetLoader
from fuse.utils.cpu_profiling import Timer
from fuse.data import DatasetDefault, PipelineDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fusedrug.utils.samplers.balanced_df_sampler import BalancedClassDataFrameSampler
from torch.utils.data import RandomSampler
from fusedrug.utils.samplers.subset_sampler import SubsetSampler
from fusedrug.data.molecule.ops.featurizer_ops import FeaturizeDrug
from fusedrug.data.protein.ops.featurizer_ops import FeaturizeTarget
from fuse.data.ops.ops_common import OpDeleteKeypaths, OpLookup
#from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import dti_binding_dataset
import dti_dataset
class BenchmarkDTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_data_dir: str,
        output_dir: str,
        pairs_tsv: str, 
        ligands_tsv: str, 
        targets_tsv: str,
        splits_tsv: str,
        train_folds: Union[List[str],str],
        val_folds: Union[List[str],str],
        test_folds: Union[List[str],str],
        class_label_to_idx: Dict[str,int],
        minibatches_per_epoch: int,
        validation_epochs: int,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        num_workers: int = 0,
        featurizer_debug_mode: bool = False,
    ):

        super().__init__()

        self._device = device
        self.data_dir = root_data_dir
        self.pairs_tsv = pairs_tsv
        self.ligands_tsv = ligands_tsv
        self.targets_tsv = targets_tsv
        self.splits_tsv = splits_tsv
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds
        self.class_label_to_idx = class_label_to_idx
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        self.batch_size = batch_size
        self.minibatches_per_epoch = minibatches_per_epoch
        self.validation_epochs = validation_epochs
        self.num_workers = num_workers
        self.featurizer_debug_mode = featurizer_debug_mode

    def _create_pipeline_desc(self, drug_featurizer_op, target_featurizer_op):

        pipeline_desc = [
            #load affinity sample
            #(drug_target_affinity_loader_op, 
            #    dict(
            #        key_out_ligand='data.drug',
            #        key_out_target='data.target',
            #        key_out_ground_truth_activity_value='data.activity_value',
            #        key_out_ground_truth_activity_label='data.label',
            #        )),
            (drug_featurizer_op,
                dict(key_out_ligand='data.drug')),
            (target_featurizer_op,
                dict(key_out_target='data.target')),
            (OpDeleteKeypaths(),
                dict(keypaths=['data.activity_value'])),
            (OpLookup(map=self.class_label_to_idx), dict(key_in='data.label', key_out='data.label')),
            (OpToTensor(), {'dtype': torch.float32, 'key': 'data.label'}),
        ]

        return pipeline_desc

    def _make_dataloader(self,
        phase:str,
        use_folds:Union[str,List[str]],
        pipeline_name:str,
    ):
        
        #dataset_loader_op = DTIBindingDatasetLoader(
        #    pairs_tsv=self.pairs_tsv,
        #    ligands_tsv=self.ligands_tsv,
        #    targets_tsv=self.targets_tsv,
        #    splits_tsv=self.splits_tsv,
        #    return_index=True if phase=='test' else False,
        #    use_folds=use_folds,
        #    keep_activity_labels=list(self.class_label_to_idx.keys()),
        #    cache_dir=Path(self.data_dir, 'PLM_DTI_cache'),
        #    force_dummy_constant_ligand_for_debugging=False,
        #    force_dummy_constant_target_for_debugging=False,
        #)

       
        #drug_featurizer_op = FeaturizeDrug(dataset=dataset_loader_op.dataset, 
        #                                   featurizer=self.drug_featurizer, debug=self.featurizer_debug_mode)
        #target_featurizer_op = FeaturizeTarget(dataset=dataset_loader_op.dataset, 
        #                                       featurizer=self.target_featurizer, debug=self.featurizer_debug_mode)
        #pipeline_desc = self._create_pipeline_desc(
        #    #drug_target_affinity_loader_op=dataset_loader_op,
        #    drug_featurizer_op=drug_featurizer_op,
        #    target_featurizer_op=target_featurizer_op,
        #)
        
        #with Timer('constructing dataset'):
        #    dataset = DatasetDefault(
        #        sample_ids=None,
        #        static_pipeline=None,
        #        dynamic_pipeline= PipelineDefault(name=pipeline_name, ops_and_kwargs=pipeline_desc),
        #    )

        #with Timer('creating dataset'):
        #    dataset.create()        
        #print(f'created dataset has length={len(dataset_loader_op.dataset)}') # dataset has no known len because it's created without explicit sample ids
        
        dataset, pairs_df = dti_dataset.dti_binding_dataset_with_featurizers(pairs_tsv=self.pairs_tsv, ligands_tsv=self.ligands_tsv, 
                        targets_tsv=self.targets_tsv, \
                        pairs_columns_to_extract=['ligand_id', 'target_id', 'activity_value', 'activity_label'], \
                        pairs_rename_columns={'activity_value': 'ground_truth_activity_value', 'activity_label': 'ground_truth_activity_label'}, \
                        ligands_columns_to_extract=['canonical_smiles'], \
                        ligands_rename_columns={'canonical_smiles': 'ligand_str'}, \
                        targets_columns_to_extract=['canonical_aa_sequence'], \
                        targets_rename_columns={'canonical_aa_sequence': 'target_str'}, \
                        drug_featurizer=self.drug_featurizer,
                        target_featurizer=self.target_featurizer,
                        featurizer_debug_mode=self.featurizer_debug_mode,
                        )

        if phase=='train': 
            with Timer("Initializing training sampler..."):
                batch_sampler = BalancedClassDataFrameSampler(df=pairs_df, label_column_name="activity_label", classes=list(self.class_label_to_idx.keys()), 
                                                counts=[self.batch_size//2, self.batch_size//2], shuffle=True, total_minibatches=self.minibatches_per_epoch)
            print("Done.")
            dl = DataLoader(dataset,
                            batch_sampler = batch_sampler,            
                            num_workers = self.num_workers,
                            collate_fn = CollateDefault()
                            )
        else:
            num_samples_per_epoch = self.validation_epochs*self.minibatches_per_epoch*self.batch_size if phase=='val' else None
            sampler = SubsetSampler(dataset=dataset, 
                                    sample_ids = list(dataset.dynamic_pipeline._ops_and_kwargs[0][0]._data.keys()),
                                    num_samples_per_epoch=num_samples_per_epoch, 
                                    shuffle=True)        
            dl = DataLoader(dataset,
                            batch_size = self.batch_size,
                            sampler = sampler,
                            num_workers = self.num_workers,
                            collate_fn = CollateDefault()
                            )

        return dl

    def prepare_data(self, test_mode=False):
        if not test_mode:
            self.train_dl = self.train_dataloader()
            self.val_dl = self.val_dataloader()
        self.test_dl = self.test_dataloader()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.data_train = self.train_dl.dataset
            self.data_val = self.val_dl.dataset

        if stage == "test" or stage is None:
            self.data_test = self.test_dl.dataset

    def train_dataloader(self):
        dl = self._make_dataloader(
            phase='train',
            use_folds=self.train_folds,
            pipeline_name='train_pipeline',
        )

        return dl
        
    def val_dataloader(self):
        dl = self._make_dataloader(
            phase='val',
            use_folds=self.val_folds,
            pipeline_name='val_pipeline',
        )

        return dl

    def test_dataloader(self):
        dl = self._make_dataloader(
            phase='test',
            use_folds=self.test_folds,
            pipeline_name='test_pipeline',
        )

        return dl


def get_dataloaders(cfg, device=torch.device("cpu"), contrastive=False, test_mode=False):
    if cfg.experiment.task.lower() == "benchmark":
        task_dir = cfg.experiment.dir
    else:
        task_dir = get_task_dir(cfg.experiment.task, orig_repo_name="Contrastive_PLM_DTI")

    if not contrastive:
        drug_featurizer = get_featurizer(cfg.model.drug_featurizer, save_dir=task_dir)
        target_featurizer = get_featurizer(cfg.model.target_featurizer, save_dir=task_dir)
        if cfg.experiment.task == "dti_dg":
            with open_dict(cfg): 
                cfg.model.classify = False
                cfg.trainer.watch_metric = "val/pcc"
            datamodule = TDCDataModule(
                task_dir,
                drug_featurizer,
                target_featurizer,
                device=device,
                seed=cfg.experiment.seed,
                batch_size=cfg.data.batch_size,
                shuffle=cfg.data.shuffle,
                num_workers=cfg.data.num_workers,
            )
            datamodule.prepare_data()
        elif cfg.experiment.task in EnzPredDataModule.dataset_list():
            with open_dict(cfg): 
                cfg.model.classify = True
                cfg.trainer.watch_metric = "validation.metrics.val/aupr"
            datamodule = EnzPredDataModule(
                task_dir,
                drug_featurizer,
                target_featurizer,
                device=device,
                seed=cfg.experiment.seed,
                batch_size=cfg.data.batch_size,
                shuffle=cfg.data.shuffle,
                num_workers=cfg.data.num_workers,
            )
            datamodule.prepare_data()
        elif cfg.experiment.task == "benchmark":
            with open_dict(cfg): 
                cfg.model.classify = True 
                cfg.trainer.watch_metric = "validation.metrics.val/aupr"
            datamodule = BenchmarkDTIDataModule(
                root_data_dir=cfg.benchmark_data.root_path,
                output_dir=task_dir,
                pairs_tsv=cfg.benchmark_data.pairs_tsv,
                ligands_tsv=cfg.benchmark_data.ligands_tsv,
                targets_tsv=cfg.benchmark_data.targets_tsv,
                splits_tsv=cfg.benchmark_data.splits_tsv,
                train_folds=cfg.benchmark_data.train_folds,
                class_label_to_idx=cfg.benchmark_data.class_label_to_idx,
                minibatches_per_epoch=cfg.benchmark_data.minibatches_per_epoch,
                validation_epochs=cfg.benchmark_data.validation_epochs,
                val_folds=cfg.benchmark_data.val_folds,
                test_folds=cfg.benchmark_data.test_folds,
                drug_featurizer=drug_featurizer,
                target_featurizer=target_featurizer,
                device=device,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                featurizer_debug_mode=cfg.benchmark_data.featurizer_debug_mode,
            )
            datamodule.prepare_data(test_mode=test_mode)
        else:
            with open_dict(cfg): 
                cfg.model.classify = True 
                cfg.trainer.watch_metric = "validation.metrics.val/aupr"
            datamodule = DTIDataModule(
                task_dir,
                drug_featurizer,
                target_featurizer,
                device=device,
                batch_size=cfg.data.batch_size,
                shuffle=cfg.data.shuffle,
                num_workers=cfg.data.num_workers,
            )
            datamodule.prepare_data()
        datamodule.setup(stage="test" if test_mode else None)
        
        with open_dict(cfg):
            cfg.model.drug_shape = drug_featurizer.shape
            cfg.model.target_shape = target_featurizer.shape
    else:
        task_dir = get_task_dir("DUDe", orig_repo_name="Contrastive_PLM_DTI")
        drug_featurizer = get_featurizer(cfg.model.drug_featurizer, save_dir=task_dir)
        target_featurizer = get_featurizer(cfg.model.target_featurizer, save_dir=task_dir)
        datamodule = DUDEDataModule(
            cfg.trainer.contrastive_split,
            drug_featurizer,
            target_featurizer,
            device=device,
            batch_size=cfg.data.contrastive_batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")


    if cfg.experiment.task == "benchmark":
        if not test_mode:
            train_dataloader = datamodule.train_dl
            valid_dataloader = datamodule.val_dl
        else:
            train_dataloader = None
            valid_dataloader = None
        test_dataloader = datamodule.test_dl
    else:
        pipeline_desc = [(OpToTensor(), {'dtype': torch.float32, 'key': 'data.label'})] # convert labels to float
        # FuseMedML wrapper
        # dataloader with batch dict collator:
        datamodule._loader_kwargs['collate_fn'] = CollateDefault()
        if not test_mode:
            datamodule.data_train = DatasetWrapSeqToDict(
                    name="PLM_DTI_train",
                    dataset=datamodule.data_train,
                    sample_keys=("data.drug", "data.target", "data.label"),
                    dynamic_pipeline=PipelineDefault(name='data_pipeline', ops_and_kwargs=pipeline_desc),
                )
            datamodule.data_train.create()

            datamodule.data_val = DatasetWrapSeqToDict(
                    name="PLM_DTI_val",
                    dataset=datamodule.data_val,
                    sample_keys=("data.drug", "data.target", "data.label"),
                    dynamic_pipeline=PipelineDefault(name='data_pipeline', ops_and_kwargs=pipeline_desc),
                )
            datamodule.data_val.create()

            train_dataloader = DataLoader(dataset=datamodule.data_train, 
                                    **datamodule._loader_kwargs)
        
            valid_dataloader = DataLoader(dataset=datamodule.data_val, 
                                        **datamodule._loader_kwargs)
        else:
            train_dataloader = None
            valid_dataloader = None

        datamodule.data_test = DatasetWrapSeqToDict(
                name="PLM_DTI_test",
                dataset=datamodule.data_test,
                sample_keys=("data.drug", "data.target", "data.label"),
                dynamic_pipeline=PipelineDefault(name='data_pipeline', ops_and_kwargs=pipeline_desc),
            )
        datamodule.data_test.create()
        
        test_dataloader = DataLoader(dataset=datamodule.data_test, 
                                    **datamodule._loader_kwargs)
        

    return train_dataloader, valid_dataloader, test_dataloader, cfg
