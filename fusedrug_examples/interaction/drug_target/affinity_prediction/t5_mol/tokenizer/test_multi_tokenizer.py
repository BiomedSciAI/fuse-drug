import hydra
from omegaconf import DictConfig, OmegaConf
from multi_tokenizer import ModularTokenizer
import os
from typing import Dict, Optional


def test_tokenizer(
    t_inst: ModularTokenizer, cfg_raw: Dict, mode: Optional[str] = ""
) -> None:
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
    # Test overall padding: (global padding works)
    enc_pad = t_inst.encode(
        typed_input_list=input_strings,
        max_len=50,
    )
    assert (
        len(enc_pad.ids) == 50
    ), f"Didn't pad to the expected number of tokens, mode: {mode}"
    # Test overall cropping: (global truncation works)
    enc_trunc = t_inst.encode(
        typed_input_list=input_strings,
        max_len=15,
    )
    assert (
        len(enc_trunc.ids) == 15
    ), f"Didn't truncate to the expected number of tokens, mode: {mode}"

    decoded_tokens = t_inst.decode(id_list=list(enc_pad.ids))
    assert (
        "".join(enc_pad.tokens) == decoded_tokens
    ), f"decoded tokens do not correspond to the original tokens, mode: {mode}"
    decoded_tokens_no_special = t_inst.decode(
        id_list=list(enc_pad.ids), skip_special_tokens=True
    )
    a = 1


@hydra.main(config_path="../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    cfg_raw = OmegaConf.to_object(cfg)

    t_mult = ModularTokenizer(**cfg_raw["data"]["tokenizer"])
    t_mult.save_jsons(tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"])

    test_tokenizer(t_mult, cfg_raw)

    t_mult_loaded = ModularTokenizer.load_from_jsons(
        tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"]
    )

    test_tokenizer(t_mult_loaded, cfg_raw, mode="loaded")

    a = 1


if __name__ == "__main__":
    os.environ["TITAN_DATA"] = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/"
    os.environ[
        "TITAN_RESULTS"
    ] = "/dccstor/fmm/users/vadimra/dev/output/TITAN_t5/08-02-2023/"
    main()
