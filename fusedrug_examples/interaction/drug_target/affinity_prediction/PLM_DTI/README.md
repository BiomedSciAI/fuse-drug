PLM DTI model from the paper [Adapting protein language models for rapid DTI prediction](https://www.mlsb.io/papers_2021/MLSB2021_Adapting_protein_language_models.pdf).

This implementation is based on the original one by the authors, found [here](https://github.com/samsledje/Contrastive_PLM_DTI/). This implementation uses PyTorch Lightning with FuseMedML wrappers for the data, model and losses to facilitate working with `batch_dict`s rather than raw tensors.  

In addition to the datasets used in the paper, we plan to implement training and evaluation on our DTI benchmark (coming soon...). 

To run training followed by test inference, simply run `python runner.py`.
Unless specified otherwise as an argument, the default parameter file found under `confings/train_config.yaml` is used. Note that in it, under the "paths" and "our_date" namespaces, you are expected to have a couple of paths defined as environment variables in your system.