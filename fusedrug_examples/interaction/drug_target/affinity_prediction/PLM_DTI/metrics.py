import torch, torchmetrics
from typing import OrderedDict
from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCPR, MetricAUCROC
def get_metrics(task):
    if task == "dti_dg":
        raise Exception("dti_dg task currently won't work. you need to use fuse.eval metrics")
        ## TODO: use fuse.eval metrics. currently the following will not work
        #val_metrics = {
        #    "val/mse": torchmetrics.MeanSquaredError,
        #    "val/pcc": torchmetrics.PearsonCorrCoef,
        #}
        #
        #test_metrics = {
        #    "test/mse": torchmetrics.MeanSquaredError,
        #    "test/pcc": torchmetrics.PearsonCorrCoef,
        #}
    else:

        train_metrics = OrderedDict(
            [
                ("train/aupr", MetricAUCPR(pred="model.output", target="data.label")),  
                ("train/auroc", MetricAUCROC(pred="model.output", target="data.label")),
            ]
        )

        val_metrics = OrderedDict(
            [
                ("val/aupr", MetricAUCPR(pred="model.output", target="data.label")),  
                ("val/auroc", MetricAUCROC(pred="model.output", target="data.label")),
            ]
        )

        test_metrics = OrderedDict(
            [
                ("test/aupr", MetricAUCPR(pred="model.output", target="data.label")),  
                ("test/auroc", MetricAUCROC(pred="model.output", target="data.label")),
            ]
        )

    return train_metrics, val_metrics, test_metrics


