import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict, Any
from fusedrug.data.tokenizer.modulartokenizer.create_multi_tokenizer import (
    test_tokenizer,
)
from fusedrug.data.tokenizer.modulartokenizer.special_tokens import (
    get_additional_tokens,
)

# add cell type:
config_name = "tokenizer_config_with_celltype"
# add gene names and extend tokenizer:
# config_name = "extended_tokenizer_config"


@hydra.main(config_path="./configs", config_name=config_name, version_base=None)
def main(cfg: DictConfig) -> None:
    """script to add a tokenizer (and all special tokens from it and special_tokens.py) to an existing tokenizer.
    The old tokenizer is read from the in_path, tokenizer to add is taken from the tokenizer_to_add variable.
    max_possible_token_id will be updated if the new max is larger then the old one.
    Add the tokenizer_info of the new tokenizer, as usual.


    Args:
        cfg (DictConfig): the config file.
    """

    # cfg= OmegaConf.load(config_path /  config_name)
    # print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    new_max_token_id = cfg_raw["data"]["tokenizer"]["max_possible_token_id"]

    t_mult = ModularTokenizer.load(path=cfg_raw["data"]["tokenizer"]["in_path"])

    test_tokenizer(t_mult, mode="loaded_path")

    # Update tokenizer with special tokens:
    added_tokens = get_additional_tokens(subset=["special", "task"])
    t_mult.update_special_tokens(
        added_tokens=added_tokens,
        save_tokenizer_path=cfg_raw["data"]["tokenizer"]["out_path"],
    )
    test_tokenizer(t_mult, mode="updated_tokenizer")

    new_tokenizer_name = cfg_raw["data"]["tokenizer"]["tokenizer_to_add"]
    cfg_tokenizer_info = {
        info["name"]: info for info in cfg_raw["data"]["tokenizer"]["tokenizers_info"]
    }
    new_tokenizer_info = cfg_tokenizer_info[new_tokenizer_name]
    if new_max_token_id > t_mult._max_possible_token_id:
        print(
            f"updating the max possible token ID from {t_mult._max_possible_token_id} to {new_max_token_id}"
        )
        t_mult._max_possible_token_id = new_max_token_id

    t_mult.add_single_tokenizer(new_tokenizer_info)
    t_mult.save(path=cfg_raw["data"]["tokenizer"]["out_path"])
    print("Fin")


if __name__ == "__main__":
    main()
