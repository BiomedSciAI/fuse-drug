import torch, torchmetrics


def get_metrics(task):
    if task == "dti_dg":
        val_metrics = {
            "val/mse": torchmetrics.MeanSquaredError,
            "val/pcc": torchmetrics.PearsonCorrCoef,
        }

        test_metrics = {
            "test/mse": torchmetrics.MeanSquaredError,
            "test/pcc": torchmetrics.PearsonCorrCoef,
        }
    else:
        val_metrics = {
            "val/aupr": torchmetrics.classification.BinaryAveragePrecision,
            "val/auroc": torchmetrics.classification.BinaryAUROC,
        }

        test_metrics = {
            "test/aupr": torchmetrics.classification.BinaryAveragePrecision,
            "test/auroc": torchmetrics.classification.BinaryAUROC,
        }
    return val_metrics, test_metrics


def get_metrics_instances(metrics):
    metric_dict = {}
    for k, met_class in metrics.items():
        met_instance = met_class()
        met_instance.reset()
        metric_dict[k] = met_instance
    return metric_dict
