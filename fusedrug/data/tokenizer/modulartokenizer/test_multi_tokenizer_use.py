import hydra
from omegaconf import DictConfig, OmegaConf
from multi_tokenizer import ModularTokenizer
from typing import Dict, Any
from test_multi_tokenizer_creation import test_tokenizer


@hydra.main(config_path="./configs", config_name="tokenizer_config_personal", version_base=None)
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

    t_mult_loaded_path = ModularTokenizer.load(path=cfg_raw["data"]["tokenizer"]["out_path"])

    test_tokenizer(t_mult_loaded_path, cfg_raw=cfg_raw, mode="loaded_path")

    print("Fin")


if __name__ == "__main__":
    main()
