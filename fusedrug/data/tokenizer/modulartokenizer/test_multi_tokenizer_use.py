import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict, Any, List
from test_multi_tokenizer_creation import test_tokenizer
from fusedrug.data.tokenizer.modulartokenizer.special_tokens import (
    get_additional_tokens,
)


def update_special_tokens(
    tokenizer_inst: ModularTokenizer, added_tokens: List, path_out: str
) -> ModularTokenizer:
    tokenizer_inst.add_special_tokens(tokens=added_tokens)
    tokenizer_inst.save(path=path_out)
    return tokenizer_inst


@hydra.main(
    config_path="./configs", config_name="tokenizer_config_personal", version_base=None
)
@hydra.main(
    config_path="./configs", config_name="tokenizer_config_personal", version_base=None
)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    ### load_from_jsons example. This is a less preferable way to load a tokenizer
    # t_mult_loaded = ModularTokenizer.load_from_jsons(
    #     tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"]
    # )

    # test_tokenizer(t_mult_loaded, cfg_raw=cfg_raw, mode="loaded")

    t_mult_loaded_path = ModularTokenizer.load(
        path=cfg_raw["data"]["tokenizer"]["out_path"]
    )

    test_tokenizer(t_mult_loaded_path, cfg_raw=cfg_raw, mode="loaded_path")

    # Update tokenizer with special tokens:
    added_tokens = get_additional_tokens(subset=["special", "task"])
    t_mult_updated = update_special_tokens(
        tokenizer_inst=t_mult_loaded_path,
        added_tokens=added_tokens,
        path_out=cfg_raw["data"]["tokenizer"]["out_path"],
    )
    test_tokenizer(t_mult_updated, cfg_raw=cfg_raw, mode="updated_tokenizer")

    print("Fin")


if __name__ == "__main__":
    main()
