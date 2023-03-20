# The Contrastive_PLM_DTI submodule is the repository found at https://github.com/samsledje/Contrastive_PLM_DTI
# and described in the paper "Adapting protein language models for rapid DTI prediction": https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.Contrastive_PLM_DTI.src.data import (
    DTIDataModule,
    TDCDataModule,
    DUDEDataModule,
    EnzPredDataModule,
)
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.Contrastive_PLM_DTI.src.utils import (
    get_featurizer,
)
import torch
from omegaconf import open_dict  # to be able to add new keys to hydra dictconfig
from fusedrug_examples.interaction.drug_target.affinity_prediction.PLM_DTI.utils import get_task_dir
from fuse.data.datasets.dataset_wrap_seq_to_dict import DatasetWrapSeqToDict
from torch.utils.data import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.pipelines.pipeline_default import PipelineDefault


def get_dataloaders(
    cfg: dict, device: torch.device = torch.device("cpu"), contrastive: bool = False
) -> Tuple[DatasetWrapSeqToDict, DatasetWrapSeqToDict, DatasetWrapSeqToDict, dict]:
    if cfg.experiment.task.lower() == "ours":
        task_dir = cfg.benchmark_data.path
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
        elif cfg.experiment.task in EnzPredDataModule.dataset_list():
            with open_dict(cfg):
                cfg.model.classify = True
                cfg.trainer.watch_metric = "val/aupr"
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
        else:
            with open_dict(cfg):
                cfg.model.classify = True
                cfg.trainer.watch_metric = "val/aupr"
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
        datamodule.setup()

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

    pipeline_desc = [(OpToTensor(), {"dtype": torch.float32, "key": "data.label"})]  # convert labels to float

    # FuseMedML wrapper
    datamodule.data_train = DatasetWrapSeqToDict(
        name="PLM_DTI_train",
        dataset=datamodule.data_train,
        sample_keys=("data.drug", "data.target", "data.label"),
        dynamic_pipeline=PipelineDefault(name="data_pipeline", ops_and_kwargs=pipeline_desc),
    )
    datamodule.data_train.create()

    datamodule.data_val = DatasetWrapSeqToDict(
        name="PLM_DTI_val",
        dataset=datamodule.data_val,
        sample_keys=("data.drug", "data.target", "data.label"),
        dynamic_pipeline=PipelineDefault(name="data_pipeline", ops_and_kwargs=pipeline_desc),
    )
    datamodule.data_val.create()

    datamodule.data_test = DatasetWrapSeqToDict(
        name="PLM_DTI_test",
        dataset=datamodule.data_test,
        sample_keys=("data.drug", "data.target", "data.label"),
        dynamic_pipeline=PipelineDefault(name="data_pipeline", ops_and_kwargs=pipeline_desc),
    )
    datamodule.data_test.create()

    # dataloader with batch dict collator:
    datamodule._loader_kwargs["collate_fn"] = CollateDefault()
    train_dataloader = DataLoader(dataset=datamodule.data_train, **datamodule._loader_kwargs)

    valid_dataloader = DataLoader(dataset=datamodule.data_val, **datamodule._loader_kwargs)

    test_dataloader = DataLoader(dataset=datamodule.data_test, **datamodule._loader_kwargs)

    return train_dataloader, valid_dataloader, test_dataloader, cfg
