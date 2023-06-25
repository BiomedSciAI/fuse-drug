
# Drug Target Interaction (DTI) using Cross Attention Transformer (CAT) Encoder

This implementation aims to predict the binding affinity of a small molecule (drug) and a protein (target) with a Cross Attention Transformer Encoder model while using [FuseMedML](https://github.com/BiomedSciAI/fuse-med-ml) tools.

## Run
```
python runner.py
```
## Config
Among other things, the configuration and the model itself enable to toggle easily between which sequence (drug/target/both) will be the context in the cross attention mechanism.
##### see `./config.yaml` for more

## Data
In this experiment we use the data directly from [ConPLex](https://github.com/samsledje/ConPLex) repository.
