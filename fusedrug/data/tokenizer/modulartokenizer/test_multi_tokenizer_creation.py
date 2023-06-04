import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
import os
from typing import Dict, Optional, Any, List
from collections.abc import Generator
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import TypedInput

from special_tokens import get_additional_tokens, get_special_tokens_dict
from tokenizers import (
    models,
    trainers,
    Tokenizer,
)
import pandas as pd
import traceback
import copy

TITAN_AA_PATH = os.environ["TITAN_DATA"] + "/public/epitopes.csv"
TITAN_SMILES_PATH = os.environ["TITAN_DATA"] + "/public/epitopes.smi"


def test_tokenizer(t_inst: ModularTokenizer, cfg_raw: Dict, mode: Optional[str] = "") -> None:
    input_strings = [
        TypedInput("AA", "<BINDING>ACDEFGHIJKLMNPQRSUVACDEF", 10),
        TypedInput("SMILES", "CCCHHCCCHC", 4),
        TypedInput("AA", "EFGHEFGHEFGH", 5),
        TypedInput("SMILES", "C=H==CC=HCCC", None),
    ]
    # TODO: named tuples with specific properties, e.g. max_len for every input, not for input type
    # Test general encoding: (per-tokenizer truncation works)
    t_inst = copy.deepcopy(t_inst)

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
    assert len(enc_pad.ids) == 50, f"Didn't pad to the expected number of tokens, mode: {mode}"

    t_inst.enable_padding(length=70, pad_token="<PAD>")
    # Test overall padding: (global padding works)
    enc_pad = t_inst.encode_list(
        typed_input_list=input_strings,
    )
    assert len(enc_pad.ids) == 70, f"Didn't pad to the expected number of tokens, mode: {mode}"

    # Test overall cropping: (global truncation works)
    enc_trunc = t_inst.encode_list(
        typed_input_list=input_strings,
        max_len=15,
    )
    assert len(enc_trunc.ids) == 15, f"Didn't truncate to the expected number of tokens, mode: {mode}"

    decoded_tokens = t_inst.decode(ids=list(enc_pad.ids))
    assert (
        "".join(enc_pad.tokens) == decoded_tokens
    ), f"decoded tokens do not correspond to the original tokens, mode: {mode}"
    decoded_tokens_no_special = t_inst.decode(ids=list(enc_pad.ids), skip_special_tokens=True)
    print(f"decoded tokens: {decoded_tokens}")
    print(f"decoded tokens, no special tokens: {decoded_tokens_no_special}")

    # test id_to_token and token_to_id
    # for a special token
    test_token = "<PAD>"
    print(
        f"Trying to encode special token {test_token} using token_to_id without supplying context (subtokenizer type). This should succeed since special tokens have unique IDs."
    )
    id1 = t_inst.token_to_id(test_token)
    id2 = t_inst.token_to_id(token=test_token, t_type="SMILES")
    print(f"{test_token} encodes to {id1} (default) and {id2} (SMILES)")
    test_token_dec = t_inst.id_to_token(id1)
    print(f"{id1} decodes to {test_token_dec} ")
    assert test_token == test_token_dec, "id_to_token(token_to_id) is not consistent"

    # for a regular token
    test_token = "C"
    print(
        f"Trying to encode regular token {test_token} using token_to_id without supplying context (subtokenizer type). This should fail, since there are several ids C encodes to."
    )
    try:
        id1 = t_inst.token_to_id(test_token)
    except:
        print("As expected, this did not work. Got the following exception:")
        traceback.print_exc()
    print(f"Now trying to encode token {test_token} using token_to_id using specific subtokenizer:")
    id1 = t_inst.token_to_id(t_type="AA", token=test_token)
    id2 = t_inst.token_to_id(token=test_token, t_type="SMILES")
    print(f"{test_token} encodes to {id1} (AA) and {id2} (SMILES)")
    test_token_dec = t_inst.id_to_token(id1)
    print(f"{id1} decodes to {test_token_dec} ")
    assert test_token == test_token_dec, "id_to_token(token_to_id) is not consistent"

    test_token = "abracadabra"
    print(f"Trying to encode a nonexisting token {test_token} using token_to_id. This should return None.")
    id1 = t_inst.token_to_id(test_token)
    assert id1 is None, "encoded a nonexisting token {test_token} to a valid id {id1}"

    # test max token ID
    max_id = t_inst.get_max_id()
    tokenizer_vocab_size = t_inst.get_vocab_size()
    print(
        f"tokenizer max id is {max_id}, of which max mapped id is {t_inst._get_max_mapped_id()}, and vocabulary size: {tokenizer_vocab_size}"
    )
    added_vocab = t_inst.get_added_vocab()
    print(f"Found {len(added_vocab)} added tokens")

    # Test special token addition to the tokenizer
    print(
        f"Adding 6 special tokens, of which 2 are already in the tokenizer vocabulary (as special). Expecting the number of special tokens to increase by 4, from {len(added_vocab)} to {len(added_vocab)+4}"
    )
    special_tokens_to_add = [
        "<test_token_1>",
        "<test_token_2>",
        "<test_token_3>",
        "<test_token_4>",
        "<SEP>",
        "<SENTINEL_ID_1>",
    ]
    t_inst.add_special_tokens(special_tokens_to_add)
    max_id = t_inst.get_max_id()
    tokenizer_vocab_size = t_inst.get_vocab_size()
    print(
        f"tokenizer max id is {max_id}, of which max mapped id is {t_inst._get_max_mapped_id()}, and vocabulary size: {tokenizer_vocab_size}"
    )
    added_vocab = t_inst.get_added_vocab()
    print(f"Found {len(added_vocab)} added tokens")
    # a = 1


def create_base_AA_tokenizer(cfg_raw: Dict[str, Any]) -> None:
    def get_training_corpus(dataset: List) -> Generator:
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]

    AA_vocab_data = pd.read_csv(TITAN_AA_PATH, sep="\t", header=None, names=["repr", "ID"])

    # Tokenizer example taken from https://huggingface.co/course/chapter6/8?fw=pt

    ############################## unwrapped AA tokenizer: AAs are treated as letters
    unwrapped_AA_tokenizer = Tokenizer(models.BPE())
    added_tokens = get_additional_tokens(subset=["special", "task"])
    initial_alphabet = get_additional_tokens(subset="AA")
    trainer_AA = trainers.BpeTrainer(vocab_size=100, special_tokens=added_tokens, initial_alphabet=initial_alphabet)
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


@hydra.main(config_path="./configs", config_name="tokenizer_config_personal", version_base=None)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    # create_base_AA_tokenizer(
    #     cfg_raw=cfg_raw
    # )  # uncomment if a new AA tokenizer is needed. Note - be really careful about it as this will override any existing tokenizer
    special_tokens_dict = get_special_tokens_dict()
    added_tokens_list = get_additional_tokens(["task"])
    cfg_tokenizer: Dict[str, Any] = cfg_raw["data"]["tokenizer"]
    t_mult = ModularTokenizer(
        **cfg_tokenizer,
        special_tokens_dict=special_tokens_dict,
        additional_tokens_list=added_tokens_list,
    )
    # t_mult.save_jsons(tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"]) #This is a less preferable way to save a tokenizer

    t_mult.save(path=cfg_raw["data"]["tokenizer"]["out_path"])

    test_tokenizer(t_mult, cfg_raw)

    print("Fin")


if __name__ == "__main__":
    main()
