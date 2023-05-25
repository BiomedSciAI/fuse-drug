import hydra
from omegaconf import DictConfig, OmegaConf
from multi_tokenizer import ModularTokenizer
import os
from typing import Dict, Optional, Any, List
import collections
from collections.abc import Generator

from special_tokens import get_additional_tokens, get_special_tokens_dict
from tokenizers import (
    models,
    trainers,
    Tokenizer,
)
import pandas as pd


TITAN_AA_PATH = os.environ["TITAN_DATA"] + "/public/epitopes.csv"
TITAN_SMILES_PATH = os.environ["TITAN_DATA"] + "/public/epitopes.smi"


def test_tokenizer(
    t_inst: ModularTokenizer, cfg_raw: Dict, mode: Optional[str] = ""
) -> None:
    TypedInput = collections.namedtuple(
        "TypedInput", ["input_type", "input_string", "max_len"]
    )
    input_strings = [
        TypedInput("AA", "<BINDING>ACDEFGHIJKLMNPQRSUVACDEF", 10),
        TypedInput("SMILES", "CCCHHCCCHC", 4),
        TypedInput("AA", "EFGHEFGHEFGH", 5),
        TypedInput("SMILES", "C=H==CC=HCCC", None),
    ]
    # TODO: named tuples with specific properties, e.g. max_len for every input, not for input type
    # Test general encoding: (per-tokenizer truncation works)
    enc = t_inst.encode_list(
        typed_input_list=input_strings,
        max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
    )
    print(f"encoded tokens: {enc.tokens}")
    # Test overall padding: (global padding works)
    enc_pad = t_inst.encode_list(
        typed_input_list=input_strings,
        max_len=50,
    )
    assert (
        len(enc_pad.ids) == 50
    ), f"Didn't pad to the expected number of tokens, mode: {mode}"
    # Test overall cropping: (global truncation works)
    enc_trunc = t_inst.encode_list(
        typed_input_list=input_strings,
        max_len=15,
    )
    assert (
        len(enc_trunc.ids) == 15
    ), f"Didn't truncate to the expected number of tokens, mode: {mode}"

    decoded_tokens = t_inst.decode(ids=list(enc_pad.ids))
    assert (
        "".join(enc_pad.tokens) == decoded_tokens
    ), f"decoded tokens do not correspond to the original tokens, mode: {mode}"
    decoded_tokens_no_special = t_inst.decode(
        ids=list(enc_pad.ids), skip_special_tokens=True
    )
    print(f"decoded tokens: {decoded_tokens}")
    print(f"decoded tokens, no special tokens: {decoded_tokens_no_special}")
    test_token = "<PAD>"
    id1 = t_inst.token_to_id(test_token)
    id2 = t_inst.token_to_id(token=test_token, t_type="SMILES")
    print(f"{test_token} encodes to {id1} (default) and {id2} (smiles)")
    test_token_dec = t_inst.id_to_token(id1)
    print(f"{id1} decodes to {test_token_dec} ")
    assert test_token == test_token_dec, "id_to_token(token_to_id) is not consistent"
    max_id = t_inst.get_max_id()
    tokenizer_vocab_size = t_inst.get_vocab_size()
    print(f"tokanizer max id is {max_id}, and vocabulary size: {tokenizer_vocab_size}")
    added_vocab = t_inst.get_added_vocab()
    print(f"Found {len(added_vocab)} added tokens")
    # a = 1


def create_base_AA_tokenizer(cfg_raw: Dict[str, Any]) -> None:
    def get_training_corpus(dataset: List) -> Generator:
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]

    AA_vocab_data = pd.read_csv(
        TITAN_AA_PATH, sep="\t", header=None, names=["repr", "ID"]
    )

    # Tokenizer example taken from https://huggingface.co/course/chapter6/8?fw=pt

    ############################## unwrapped AA tokenizer: AAs are treated as letters
    unwrapped_AA_tokenizer = Tokenizer(models.BPE())
    added_tokens = get_additional_tokens(subset=["special", "task"])
    initial_alphabet = get_additional_tokens(subset="AA")
    trainer_AA = trainers.BpeTrainer(
        vocab_size=100, special_tokens=added_tokens, initial_alphabet=initial_alphabet
    )
    unwrapped_AA_vocab = list(AA_vocab_data["repr"])
    unwrapped_AA_tokenizer.train_from_iterator(
        get_training_corpus(dataset=unwrapped_AA_vocab),
        trainer=trainer_AA,
    )
    # unwrapped_AA_tokenizer.add_special_tokens(special_token_dict)
    for d in cfg_raw["data"]["tokenizer"]["tokenizers_info"]:
        if "AA" == d["name"]:
            AA_json_path = d["json_path"]
    if not os.path.exists(os.path.dirname(AA_json_path)):
        os.makedirs(os.path.dirname(AA_json_path))

    if os.path.exists(AA_json_path):
        raise Exception(
            f"{AA_json_path} already exists. Make sure you want to override  and then comment this exception"
        )
    unwrapped_AA_tokenizer.save(path=AA_json_path)
    print("Fin")


@hydra.main(
    config_path="./configs", config_name="tokenizer_config_personal", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    # create_base_AA_tokenizer(
    #     cfg_raw=cfg_raw
    # )  # uncomment if a new AA tokenizer is needed. Note - be really careful about it as this will override any existing tokenizer
    special_tokens_dict = get_special_tokens_dict()
    cfg_tokenizer: Dict[str, Any] = cfg_raw["data"]["tokenizer"]
    t_mult = ModularTokenizer(**cfg_tokenizer, special_tokens_dict=special_tokens_dict)
    # t_mult.save_jsons(tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"]) #This is a less preferable way to save a tokenizer

    t_mult.save(path=cfg_raw["data"]["tokenizer"]["out_path"])

    test_tokenizer(t_mult, cfg_raw)

    print("Fin")


if __name__ == "__main__":
    main()
