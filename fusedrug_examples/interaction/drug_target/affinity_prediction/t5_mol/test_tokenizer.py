# Taken from https://huggingface.co/course/chapter6/8?fw=pt
from datasets import load_dataset
from tokenizer.special_tokens import (
    get_special_tokens,
    special_mark_AA,
    special_wrap_input,
    strip_special_wrap,
)
import os

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import (
    dti_binding_dataset as dti_dataset,
)
import json
import os
from typing import Dict
from rdkit import Chem
import time

FORBIDDEN = set(["B", "O", "U", "X", "Z"])

"""
Yoel's base tokenizer code: fuse-drug/fusedrug/data/tokenizer/fast_tokenizer_learn.py
Yoel's protein tokenizer: fuse-drug/fusedrug/data/protein/tokenizer/build_protein_tokenizer_pair_encoding.py
Tokenizer use:
/zed_dti/zed_dti/fully_supervised/dti_classifier/data.py
Tokenizer ops: /fuse-drug/fusedrug/data/tokenizer/ops/fast_tokenizer_ops.py

Thoughts:
1. You can't simply train a tokenizer on AA sequences and then manually wrap all tokens with <> within the tokenizer json using BPE. This causes a loading error.
    It would seem that a vocabulary must first contain byte-level characters, and the merges part is where merging to more characters happens. Basically, for every 
    complex token (with >1 characters) in a vocabulary of the tokenizer, e.g. "ABC", there must be a line in merges that creates it from smaller tokens, e.g. "A BC" 
    (tokens "A" and "BC" must exist in the vocabulary).
    The resulting tokenizer is saved in fuse-drug/fusedrug_examples/interaction/drug_target/affinity_prediction/t5_mol/tokenizer/pretrained/simple_protein_tokenizer_wrapped_non_special_AA.json 
    This way, we have a vocabulary that must contain single-letter tokens. The only way around this is to introduce special tokens. 
    A version that only includes special tokens is saved here:
    fuse-drug/fusedrug_examples/interaction/drug_target/affinity_prediction/t5_mol/tokenizer/pretrained/simple_protein_tokenizer_wrapped_special_AA.json
    It was trained on wrapped AA sequences from TITAN, which resulted in the addition of single-letter tokens to the vocabulary. I manually removed them, keeping
    only the special ones as defined in tokenizer.special_tokens.get_special_tokens
2. It seems that the vocabulary is sorted by token frequency in the training data (or maybe it's just the merges part)?
3. RoBERTa and GPT use single letters from outside the english alphabet for special tokens. Maybe we should do the same for AAs?
4. Merging tokenizers:  Now I'll take one of the tokenizers from 1., and merge them with the SMILES tokenizer. 
    The merging should probably be done in such a way as to keep the indices of tokens of the main tokenizer, and adjust only the indices of the secondary vocabulary. This way, we can use
    pre-trained models trained using the main tokenizer.
    In order to do that, I need to take care of 3 entities in the jsons of the two tokenizers:
    - Special tokens:   
        This is problematic, because special tokens are found in the lower indices. It may be possible to add them in higher ones - need to check this.
        Another option is to only keep the special tokens of the main tokenizer (this would be AA, with all the tokens to be used in T5 and other models)
    - Vocabulary
        A working solution would be just to go over the secondary tokenizer vocabulary and add all its tokens (except the ones already in the main tokenizer)
    - Merges    
        The most naive approach for now - concatenate the two merges and hope for the best. A slightly less naive approach, as mentioned here: https://github.com/huggingface/tokenizers/issues/690,
        would be to merge the merges by order of token length.
    

"""


def aas_to_smiles(aas, sanitize=True):
    """
    Taken from pytoda.proteins.utils
    Converts an amino acid sequence (IUPAC) into SMILES.

    Args:
        aas (str): The amino acid sequence to be converted.
            Following IUPAC notation.
        sanitize (bool, optional): [description]. Defaults to True.

    Raises:
        TypeError: If aas is not a string.
        ValueError: If string cannot be converted to mol.

    Returns:
        smiles: SMILES string of the AA sequence.
    """
    if not isinstance(aas, str):
        raise TypeError(f"Provide string not {type(aas)}.")
    if len(set(aas).intersection(FORBIDDEN)) > 0:
        raise ValueError(
            f"Characters from: {FORBIDDEN} cant be parsed. Found one in: {aas}"
        )
    mol = Chem.MolFromFASTA(aas, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"Sequence could not be converted to SMILES: {aas}")
    smiles = Chem.MolToSmiles(mol)
    return smiles


TITAN_AA_PATH = (
    "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/public/epitopes.csv"
)
TITAN_SMILES_PATH = (
    "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/public/epitopes.smi"
)


def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]


"""
Given two tokenizers, combine them and create a new tokenizer
Usage: python combine_tokenizers.py --tokenizer1 ../config/en/roberta_8 --tokenizer2 ../config/hi/roberta_8 --save_dir ../config/en/en_hi/roberta_8
"""


def combine_tokenizers(path_primary: str, path_secondary: str, path_out: str) -> None:
    """Combines two tokenizers

    Args:
        args (_type_): _description_
    """
    # Load both the json files, take the union, and store it
    json1 = json.load(open(path_primary))
    json2 = json.load(open(path_secondary))

    assert (
        json1["model"]["type"] == json2["model"]["type"]
    )  # the merged models must be of the same type

    # Create a new special token vocabulary
    new_ST_vocab = json1[
        "added_tokens"
    ]  # This is redundant since we keep the special tokens from json1 anyway.

    found_merges = False
    # Create a new merges
    if "merges" in json1["model"]:
        new_merges = json1["model"]["merges"]
        found_merges = True
    else:
        new_merges = []

    # Add merges from the secondary tokenizer
    if "merges" in json2["model"]:
        found_merges = True
        for merge in json2["model"]["merges"]:
            if merge not in new_merges:
                new_merges.append(merge)
    if found_merges:
        json1["model"]["merges"] = new_merges

    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    for word in json1["model"]["vocab"].keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Add words from the secondary tokenizer
    for word in json2["model"]["vocab"].keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    json1["model"]["vocab"] = new_vocab

    # Make the output directory if necessary
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

    # Save the vocab
    with open(path_out, "w") as fp:
        json.dump(json1, fp, ensure_ascii=False, indent=2)

    # Instantiate the new tokenizer
    # tokenizer = Tokenizer.from_file(path_out)
    a = 1


# def main():
#     parser = argparse.ArgumentParser()

#     # Dataset Arguments
#     parser.add_argument("--tokenizer1", type=str, required=True, help="")
#     parser.add_argument("--tokenizer2", type=str, required=True, help="")
#     parser.add_argument("--save_dir", type=str, required=True, help="")

#     args = parser.parse_args()

#     combine_tokenizers(args)


def create_tokenizers(cfg_raw: Dict):
    vocab_data = pd.read_csv(
        TITAN_SMILES_PATH, sep="\t", header=None, names=["repr", "ID"]
    )
    AA_vocab_data = pd.read_csv(
        TITAN_AA_PATH, sep="\t", header=None, names=["repr", "ID"]
    )

    # Tokenizer example taken from https://huggingface.co/course/chapter6/8?fw=pt
    tokenizer = Tokenizer(models.BPE())
    tokenizer.model = models.BPE()
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # Significantly reduces the resulting vocabulary size
    AA_tokenizer = Tokenizer(models.BPE())

    special_tokens = get_special_tokens()
    trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=special_tokens)
    trainer_AA_no_special = trainers.BpeTrainer(vocab_size=40)
    trainer_AA = trainers.BpeTrainer(vocab_size=400, special_tokens=special_tokens)

    tokenizer.train_from_iterator(
        get_training_corpus(dataset=list(vocab_data["repr"])), trainer=trainer
    )
    # Normalizer (an alternative to special_mark_AA())
    AA_tokens = get_special_tokens(subset=["AA"])
    AA_normalizer = normalizers.Sequence(
        [normalizers.Replace(strip_special_wrap(x), x) for x in AA_tokens]
    )
    tmp = list(AA_vocab_data["repr"])
    t1 = time.perf_counter()
    wrapped_AA_vocab = [special_mark_AA(x) for x in tmp]
    dt = time.perf_counter() - t1
    print(f"Special wrapping took {dt} seconds")
    t1 = time.perf_counter()
    wrapped_AA_vocab = [AA_normalizer.normalize_str(x) for x in tmp]
    dt = time.perf_counter() - t1
    print(f"Normalizer wrapping took {dt} seconds")
    AA_tokenizer.train_from_iterator(
        get_training_corpus(dataset=wrapped_AA_vocab), trainer=trainer_AA
    )
    if not os.path.exists(
        os.path.dirname(cfg_raw["data"]["tokenizer"]["combined_tokenizer_path"])
    ):
        os.makedirs(
            os.path.dirname(cfg_raw["data"]["tokenizer"]["combined_tokenizer_path"])
        )
    tokenizer.save(path=cfg_raw["data"]["tokenizer"]["combined_tokenizer_path"])

    if not os.path.exists(
        os.path.dirname(cfg_raw["data"]["tokenizer"]["AA_tokenizer_path"])
    ):
        os.makedirs(os.path.dirname(cfg_raw["data"]["tokenizer"]["AA_tokenizer_path"]))
    AA_tokenizer.save(path=cfg_raw["data"]["tokenizer"]["AA_tokenizer_path"])

    print("Fin")


def test_tokenizers(cfg_raw: Dict):

    # Normalizer (an alternative to special_mark_AA())
    AA_tokens = get_special_tokens(subset=["AA"])
    AA_normalizer = normalizers.Sequence(
        [normalizers.Replace(strip_special_wrap(x), x) for x in AA_tokens]
    )

    tokenizer = Tokenizer.from_file(
        cfg_raw["data"]["tokenizer"]["combined_tokenizer_path"]
    )

    test_sequence = "ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT"
    encoding = tokenizer.encode(
        special_wrap_input("BINDING")
        + special_mark_AA(test_sequence)
        + special_wrap_input("SEP")
        + aas_to_smiles(test_sequence)
    )
    print(encoding.tokens)
    print(len(encoding.tokens))
    print(tokenizer.get_vocab_size())

    print(AA_normalizer.normalize_str(test_sequence))
    loaded_tokenizer_w_norm = Tokenizer.from_file(
        cfg_raw["data"]["tokenizer"]["combined_tokenizer_path"]
    )
    # tokenizer_w_norm.normalizer = AA_normalizer

    loaded_AA_tokenizer = Tokenizer.from_file(
        cfg_raw["data"]["tokenizer"]["AA_tokenizer_path"]
    )

    encoding_norm = tokenizer.encode(
        special_wrap_input("BINDING")
        + AA_normalizer.normalize_str(test_sequence)
        + special_wrap_input("SEP")
        + aas_to_smiles(test_sequence)
    )
    print(f"Combined tokenizer tokens ({len(encoding_norm.tokens)}):")
    print(encoding_norm.tokens)

    encoding_AA = loaded_AA_tokenizer.encode(AA_normalizer.normalize_str(test_sequence))
    print(f"AA tokenizer tokens ({len(encoding_AA.tokens)}):")
    print(encoding_AA.tokens)

    # Post-processor

    # decoder example
    print(tokenizer.decode(encoding_norm.ids, skip_special_tokens=False))
    print("Fin")


@hydra.main(config_path="./configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    cfg_raw = OmegaConf.to_object(cfg)
    # ppi_dataset, pairs_df = dti_dataset(**cfg_raw['data']['benchmarks']['TITAN_benchmark']['lightning_data_module'])
    # ppi_dataset, pairs_df = dti_dataset.dti_binding_dataset(

    #                     pairs_tsv=self.pairs_tsv,
    #                     ligands_tsv=self.ligands_tsv,
    #                     targets_tsv=self.targets_tsv,
    #                     splits_tsv=self.splits_tsv,
    #                     use_folds=['train1', 'train2', 'train3', 'train4', 'train5'],
    #                     pairs_columns_to_extract=['ligand_id', 'target_id', 'activity_label'],
    #                     pairs_rename_columns={'activity_label': 'data.label'},
    #                     ligands_columns_to_extract=['canonical_smiles', 'canonical_aa_sequence', 'full_canonical_aa_sequence'],
    #                     ligands_rename_columns={'canonical_smiles': 'ligand_str'},
    #                     targets_columns_to_extract=['canonical_smiles', 'canonical_aa_sequence', 'full_canonical_aa_sequence'],
    #                     targets_rename_columns={'canonical_aa_smiles': 'target_str'},

    #                     keep_activity_labels=list(self.class_label_to_idx.keys()),
    #                     cache_dir=Path(self.data_dir, 'PLM_DTI_cache'),
    #                     dynamic_pipeline=[(OpLookup(map=self.class_label_to_idx), dict(key_in='data.label', key_out='data.label')),
    #                                       (OpToTensor(), {'dtype': torch.float32, 'key': 'data.label'})],
    #                     )
    # create_tokenizers(cfg_raw=cfg_raw) #We don't want to run this with the same config, so as not to overwrite the tokenizer jsons we use
    combine_tokenizers(
        path_primary=cfg_raw["data"]["tokenizer"]["AA_tokenizer_path"],
        path_secondary=cfg_raw["data"]["tokenizer"]["SMILES_tokenizer_path"],
        path_out=cfg_raw["data"]["tokenizer"]["combined_tokenizer_path"],
    )
    test_tokenizers(cfg_raw=cfg_raw)


if __name__ == "__main__":
    os.environ["TITAN_DATA"] = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/"
    os.environ[
        "TITAN_RESULTS"
    ] = "/dccstor/fmm/users/vadimra/dev/output/TITAN_t5/08-02-2023/"
    main()

# Modus operandi:
# 1. Get input
# 2. For each part of the input that contains AA sequences apply normalizer
# 3. Tokenize
# 4. Apply task-dependent post-processing
