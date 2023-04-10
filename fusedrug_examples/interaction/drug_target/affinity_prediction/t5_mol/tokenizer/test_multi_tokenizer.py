import hydra
from omegaconf import DictConfig, OmegaConf
from multi_tokenizer import ModularTokenizer
import os
from typing import Dict, Optional, Any

from special_tokens import (
    get_special_tokens,
)
from tokenizers import (
    models,
    trainers,
    Tokenizer,
)
import pandas as pd


TITAN_AA_PATH = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/public/epitopes.csv"
TITAN_SMILES_PATH = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/public/epitopes.smi"


def test_tokenizer(t_inst: ModularTokenizer, cfg_raw: Dict, mode: Optional[str] = "") -> None:
    input_strings = [
        ("AA", "<BINDING>ACDEFGHIJKLMNOPQRSTACDEF"),
        ("SMILES", "CCCHH"),
        ("AA", "EFGH"),
        ("SMILES", "C=H==CC=HCCC"),
    ]
    # Test general encoding: (per-tokenizer truncation works)
    enc = t_inst.encode(
        typed_input_list=input_strings,
        max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
    )
    print(f"encoded tokens: {enc.tokens}")
    # Test overall padding: (global padding works)
    enc_pad = t_inst.encode(
        typed_input_list=input_strings,
        max_len=50,
    )
    assert len(enc_pad.ids) == 50, f"Didn't pad to the expected number of tokens, mode: {mode}"
    # Test overall cropping: (global truncation works)
    enc_trunc = t_inst.encode(
        typed_input_list=input_strings,
        max_len=15,
    )
    assert len(enc_trunc.ids) == 15, f"Didn't truncate to the expected number of tokens, mode: {mode}"

    decoded_tokens = t_inst.decode(id_list=list(enc_pad.ids))
    assert (
        "".join(enc_pad.tokens) == decoded_tokens
    ), f"decoded tokens do not correspond to the original tokens, mode: {mode}"
    decoded_tokens_no_special = t_inst.decode(id_list=list(enc_pad.ids), skip_special_tokens=True)
    print(f"decoded tokens: {decoded_tokens}")
    print(f"decoded tokens, no special tokens: {decoded_tokens_no_special}")
    # a = 1


def create_base_AA_tokenizer(cfg_raw: Dict[str, Any]):
    def get_training_corpus(dataset):
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]

    AA_vocab_data = pd.read_csv(TITAN_AA_PATH, sep="\t", header=None, names=["repr", "ID"])

    # Tokenizer example taken from https://huggingface.co/course/chapter6/8?fw=pt

    ############################## unwrapped AA tokenizer: AAs are treated as letters
    unwrapped_AA_tokenizer = Tokenizer(models.BPE())
    special_tokens = get_special_tokens(subset=["special", "task"])
    trainer_AA = trainers.BpeTrainer(vocab_size=100, special_tokens=special_tokens)
    unwrapped_AA_vocab = list(AA_vocab_data["repr"])
    unwrapped_AA_tokenizer.train_from_iterator(get_training_corpus(dataset=unwrapped_AA_vocab), trainer=trainer_AA)
    if not os.path.exists(os.path.dirname(cfg_raw["data"]["tokenizer"]["tokenizers_info"]["AA"]["json_path"])):
        os.makedirs(os.path.dirname(cfg_raw["data"]["tokenizer"]["tokenizers_info"]["AA"]["json_path"]))
    unwrapped_AA_tokenizer.save(path=cfg_raw["data"]["tokenizer"]["tokenizers_info"]["AA"]["json_path"])
    print("Fin")


@hydra.main(config_path="../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    create_base_AA_tokenizer(cfg_raw=cfg_raw)

    cfg_tokenizer: Dict[str, Any] = cfg_raw["data"]["tokenizer"]
    t_mult = ModularTokenizer(**cfg_tokenizer)
    t_mult.save_jsons(tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"])

    test_tokenizer(t_mult, cfg_raw)

    t_mult_loaded = ModularTokenizer.load_from_jsons(tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"])

    test_tokenizer(t_mult_loaded, cfg_raw=cfg_raw, mode="loaded")

    print("Fin")


if __name__ == "__main__":
    os.environ["TITAN_DATA"] = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/"
    os.environ["TITAN_RESULTS"] = "/dccstor/fmm/users/vadimra/dev/output/TITAN_t5/08-02-2023/"
    main()
