import hydra
from omegaconf import DictConfig, OmegaConf
from multi_tokenizer import ModularTokenizer
import os
from typing import Dict, Any
from .test_multi_tokenizer_creation import test_tokenizer


@hydra.main(config_path="./configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    t_mult_loaded = ModularTokenizer.load_from_jsons(tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"])

    test_tokenizer(t_mult_loaded, cfg_raw=cfg_raw, mode="loaded")

    print("Fin")


if __name__ == "__main__":
    main()
