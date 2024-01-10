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


# this needs to be run on the bmfm_modular_tokenizer and the bmfm_extended_modular_tokenizer
@hydra.main(config_path="./configs", config_name="tokenizer_config", version_base=None)
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
    t_mult_loaded_path.update_special_tokens(
        added_tokens=added_tokens,
        save_tokenizer_path=cfg_raw["data"]["tokenizer"]["out_path"],
    )
    test_tokenizer(t_mult_loaded_path, cfg_raw=cfg_raw, mode="updated_tokenizer")

    print("Fin")


if __name__ == "__main__":
    main()
