# BimodalMCA binding affinity predictor

This example implements FuseMedML-Drug based training, inference and evaluation of a drug interaction binding affinity prediction model. The model is a Bimodal Multiscale Convolutional Attention network (BiMCA) [[1]](#1), implemented originally in [[2]](#2).

## How to run:
1. Download the data from this [Box link](https://ibm.ent.box.com/s/xtml12cbd9bwb97odxykbl5qzzh147ml/folder/141340279248).
2. Set your local paths to the data and desired results location in `config/train_config.yaml` under the `paths` section. Note that by default it uses the environment variables `BIMCA_DATA` and `BIMCA_RESULTS`, but you can set any path manually instead.
3. Run `runner.py`.

## References
<a id="1">[1]</a>
Manica, M. et al.
Toward Explainable Anticancer Compound Sensitivity Prediction via Multimodal Attention-Based Convolutional Encoders.
Molecular Pharmaceutics 2019 16 (12), 4797-4806

<a id="2">[2]</a>
Paccmann predictor - https://github.com/PaccMann/paccmann_predictor
