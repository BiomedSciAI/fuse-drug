# fuse-drug
[FuseMedML](https://github.com/BiomedSciAI/fuse-med-ml) based molecular biochemistry library for drug discovery/repurposing

*fuse-drug* contains generic tools to facilitate working with datasets and representations for proteins, small molecules, and interactions. These include data loaders, processing and augmentation *fuse* style ops, utilities and more.  

It also contains end to end examples of data and model training pipelines, currently focused on the protein-ligand affinity prediction task. 

*fuse-drug* is a work in progress. It will gradually expand to cover more representations and tasks such as molecule property prediction, protein-protein interaction, generation and more.  
It will also extend the *fuse.eval* package to cover evaluation metrics specific to the biochemistry domain.

Coming soon: DrugDiscoveryFoundationBenchmarks - A repository for biochemical ML benchmarks, which includes tools for data curation and creation, creating different types of splits for model training, and application of evaluation metrics.


## Installation instructions

1. Install FuseMedML and its dependencies as described [here](https://github.com/BiomedSciAI/fuse-med-ml#option-1-install-from-source-recommended).

2. Install Fuse-Drug only (without examples) by running:
``` 
pip install -e .

# to also install development deps use:
pip install -e .[dev]
```
or:  

Install Fuse-Drug with examples by running:
```
pip install -e .[examples]
```

In case of a CUDA related error, we recommend working in a conda environment with Python>=3.9 (Create one by running `conda create -n ENV_NAME python=3.9`) and updating PyTorch following the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) **after** completing the above steps.
