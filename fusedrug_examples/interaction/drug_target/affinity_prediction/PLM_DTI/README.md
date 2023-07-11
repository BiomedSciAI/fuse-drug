PLM DTI model from the paper [Adapting protein language models for rapid DTI prediction](https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf).

This implementation is based on the original one by the authors, found [here](https://github.com/samsledje/Contrastive_PLM_DTI/). This implementation uses PyTorch Lightning with FuseMedML wrappers for the data, model and losses to facilitate working with `batch_dict`s rather than raw tensors.

In addition to the datasets used in the paper, we implement training and evaluation on a large DTI dataset that we curated and published [here](https://zenodo.org/record/8105617).  
In order to use this data:
1. Download and extract the large zip archive in the Zenodo link above.
2. Set the environment variable `DTI_BENCHMARK_DATA` to the local path to which you downloaded the file.
3. Run training using the config file `python runner.py --config_path=configs/config_dti_benchmark.yaml`. Note the namespace "benchmark_data" in the config in which you can select one of 4 different train/validation/test set splits: 
    * lenient - an entity unique to a set is a pair of ligand and target.
    * cold ligand - an entity unique to a set is a ligand (more restrictive than lenient)
    * cold target - an entity unique to a set is a target (more restrictive than lenient)
    * temporal - the sets are separated by the date range in which the assay was performed

To run training followed by test inference, simply run `python runner.py`.
Unless specified otherwise as an argument, the default parameter file found under `confings/config.yaml` is used. Note that you are expected to set the environment variable `DTI_RESULTS` to a local path of your choice. Results will be saved there.
