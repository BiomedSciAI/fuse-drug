import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict, Dict, Any
import copy
import torch.optim as optim
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from fuse.dl.losses.loss_default import LossDefault
from fusedrug.data.interaction.drug_target.datasets.fuse_style_dti import (
    DTIDataModule,
)
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
from fuse.dl.models.backbones.backbone_transformer import CrossAttentionTransformerEncoder
from fuse.dl.models.heads.heads_1D import Head1D
from fuse.dl.lightning.pl_funcs import start_clearml_logger
from fuse.utils.ndict import NDict
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAccuracy,
    MetricAUCROC,
    MetricROCCurve,
)
from fuse.eval.metrics.classification.metrics_thresholding_common import (
    MetricApplyThresholds,
)

from fuse.utils.file_io.file_io import create_dir, save_dataframe
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe
from fuse.eval.evaluator import EvaluatorDefault
from fuse.utils.utils_logger import fuse_logger_start


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    runs train -> inference -> evaluation pipeline using the "./config.yaml" file.

    :param cfg: Hydra's config object that the decorator supplies.
    """
    cfg = hydra.utils.instantiate(cfg)
    cfg = NDict(OmegaConf.to_object(cfg))
    cfg.print_tree(True)

    paths = cfg["paths"]
    train_params = NDict(cfg["params.train"])
    infer_params = NDict(cfg["params.infer"])

    if cfg["logging.log_clear_ml"]:
        start_clearml_logger(project_name="DTI", task_name=f"{cfg['logging.task_name']}")

    run_train(paths, train_params)
    run_infer(paths, infer_params)
    run_eval(paths)


def create_model(model_params: dict) -> nn.Module:
    """
    creates PyTorch model using cfg's model parameters

    :param model_params: cfg model parameters
    """

    # backbone model
    torch_model = CrossAttentionTransformerEncoder(**model_params)
    bb_model = ModelWrapSeqToDict(
        model=torch_model,
        model_inputs=("data.drug.tokenized", "data.target.tokenized"),
        model_outputs=("model.backbone_features",),
        post_forward_processing_function=partial(torch.mean, dim=1),
    )

    # classification head
    head = Head1D(
        mode="classification", num_outputs=2, conv_inputs=(("model.backbone_features", model_params["output_dim"]),)
    )

    model = nn.Sequential(bb_model, head)
    return model


def create_datamodule(paths: dict, params: dict):
    """
    create dti datamodule

    :param paths: paths dictionary
    :param params: training params
    """
    datamodule = DTIDataModule(
        train_data_path=paths["train_data_path"],
        val_data_path=paths["val_data_path"],
        test_data_path=paths["test_data_path"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        drug_fixed_size=params["data.drug_fixed_size"],
        target_fixed_size=params["data.target_fixed_size"],
    )

    return datamodule


def run_train(paths: Dict[str, str], params: Dict[str, Any]) -> None:
    """
    run train stage

    :param paths: paths dictionary
    :param params: training params
    """

    # start logger
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")

    lgr.info("Fuse Train", {"attrs": ["bold", "underline"]})
    lgr.info(f'model_dir={paths["model_dir"]}', {"color": "magenta"})
    lgr.info(f'cache_dir={paths["cache_dir"]}', {"color": "magenta"})

    # build PyTorch-Lightning's datamodule
    lgr.info("Datamodule:", {"attrs": "bold"})
    datamodule = create_datamodule(paths, params)
    lgr.info("Datamodule: Done", {"attrs": "bold"})

    # create PyTorch based model using fuse's wrapper
    lgr.info("Model:", {"attrs": "bold"})
    model = create_model(model_params=params["model"])
    lgr.info("Model: Done", {"attrs": "bold"})

    # instantiate CE loss function using Fuse's wrapper
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.head_0",
            target="data.label",
            callable=F.cross_entropy,
            weight=1.0,
        ),
    }

    # instantiate metric(s) using Fuse Eval package
    train_metrics = OrderedDict(
        [
            (
                "auc",
                MetricAUCROC(
                    pred="model.output.head_0",
                    target="data.label",
                    class_names=["0", "1"],
                ),
            ),
        ]
    )
    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    # choose best epoch source - could be based on loss(es)/metric(s)
    best_epoch_source = dict(monitor="validation.metrics.auc.macro_avg", mode="max")

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # create learning rate monitor for ClearML logs
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # create instance of PL module - FuseMedML generic version
    lgr.info("Train:", {"attrs": "bold"})
    pl_module = LightningModuleDefault(
        model_dir=paths["model_dir"],
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightning trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=params["num_epochs"],
        accelerator=params["accelerator"],
        devices=params["num_devices"],
        strategy=params["strategy"],
        callbacks=[lr_monitor] if params["log_lr"] else None,
    )

    # train!
    pl_trainer.fit(pl_module, datamodule=datamodule)
    lgr.info("Train: Done", {"attrs": "bold"})


def run_infer(paths: Dict[str, str], params: Dict[str, Any]) -> None:
    """
    run inference stage

    :param paths: paths dictionary
    :param params: inference params
    """

    # start logger
    create_dir(paths["infer_dir"])
    fuse_logger_start(output_path=paths["infer_dir"], console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")

    infer_file_path = os.path.join(paths["infer_dir"], paths["infer_filename"])
    ckpt_file_path = os.path.join(paths["model_dir"], paths["ckpt_filename"])

    lgr.info("Fuse Inference:", {"attrs": ["bold", "underline"]})
    lgr.info(f"infer_file_path={infer_file_path}", {"color": "magenta"})
    lgr.info(f"ckpt_file_path={ckpt_file_path}", {"color": "magenta"})

    # build PyTorch-Lightning's datamodule
    lgr.info("Datamodule:", {"attrs": "bold"})
    datamodule = create_datamodule(paths, params)
    lgr.info("Datamodule: Done", {"attrs": "bold"})

    # create PyTorch based model using fuse's wrapper (same model from the training phase)
    lgr.info("Model:", {"attrs": "bold"})
    model = create_model(model_params=params["model"])
    lgr.info("Model: Done", {"attrs": "bold"})

    # load model from checkpoint that was saved in the training phase and was determined by "best_epoch_source"
    pl_module = LightningModuleDefault.load_from_checkpoint(
        ckpt_file_path,
        model_dir=paths["model_dir"],
        model=model,
        map_location="cpu",
        strict=True,
    )

    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(["model.output.head_0", "data.label"])  # which keys to extract and dump into file

    # create a trainer instance
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=params["accelerator"],
        devices=params["num_devices"],
        max_epochs=0,
    )

    # predict !
    lgr.info("Predict:", {"attrs": "bold"})
    predictions = pl_trainer.predict(pl_module, datamodule=datamodule, return_predictions=True)
    lgr.info("Predict: Done", {"attrs": "bold"})

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file_path)

    lgr.info("Fuse Inference: Done", {"attrs": ["bold", "underline"]})


def run_eval(paths: Dict[str, str]) -> NDict:
    """
    run evaluation on the inference output from the inference stage.

    :param paths: paths dictionary
    :return: a Fuse's NDict dictionary that holds all the results.
    """
    # start logger
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    infer_file_path = os.path.join(paths["infer_dir"], paths["infer_filename"])

    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Eval", {"attrs": ["bold", "underline"]})
    lgr.info(f"infer_file_path={infer_file_path}", {"color": "magenta"})
    lgr.info(f'eval_dir={paths["eval_dir"]}', {"color": "magenta"})

    # instantiate metric(s) using Fuse Eval package
    metrics = OrderedDict(
        [
            (
                "op",
                MetricApplyThresholds(pred="model.output.head_0"),
            ),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target="data.label")),
            (
                "accuracy",
                MetricAccuracy(pred="results:metrics.op.cls_pred", target="data.label"),
            ),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.head_0",
                    target="data.label",
                    output_filename=os.path.join(paths["infer_dir"], "roc_curve.png"),
                ),
            ),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # evaluate !
    results = evaluator.eval(
        ids=None,
        data=infer_file_path,
        metrics=metrics,
        output_dir=paths["eval_dir"],
        silent=False,
    )
    lgr.info("Fuse Eval: Done", {"attrs": ["bold", "underline"]})

    return results


if __name__ == "__main__":
    main()
