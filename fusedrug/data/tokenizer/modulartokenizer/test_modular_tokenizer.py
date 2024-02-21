import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict, Any
from fusedrug.data.tokenizer.modulartokenizer.create_multi_tokenizer import (
    test_tokenizer,
)
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import TypedInput

# add cell type:
config_name = "tokenizer_config_with_celltype"
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
    input_strings = [
        TypedInput("AA", "<BINDING>ACDEFGHIJKLMNPQRSUVACDEF", 10),
        TypedInput("SMILES", "CCCHHCCCHC", 4),
        TypedInput("AA", "EFGHEFGHEFGH", 5),
        TypedInput("SMILES", "C=H==CC=HCCC", None),
    ]
    test_tokenizer(t_mult_loaded_path, cfg_raw=cfg_raw, mode="loaded_path", input_strings=input_strings)
    
    print("Testing input that sontains characters mapped to <UNK> token, suppressing exception, should raise warning")
    input_strings = [
        TypedInput("AA", "<BINDING>AC11DEFGHIJKLMNPQRSUVACDEF", 10),
        TypedInput("SMILES", "CCCHHCCCHC", 4),
        TypedInput("AA", "EFGHEFGHEFGH", 5),
        TypedInput("SMILES", "C=H==CC=HCCC", None),
    ]
    test_tokenizer(t_mult_loaded_path, cfg_raw=cfg_raw, mode="loaded_path", input_strings=input_strings, on_unknown='warn')
    
    print("Testing input that sontains characters mapped to <UNK> token, should raise exception")
    input_strings = [
        TypedInput("AA", "<BINDING>AC11DEFGHIJKLMNPQRSUVACDEF", 10),
        TypedInput("SMILES", "CCCHHCCCHC", 4),
        TypedInput("AA", "EFGHEFGHEFGH", 5),
        TypedInput("SMILES", "C=H==CC=HCCC", None),
    ]
    test_tokenizer(t_mult_loaded_path, cfg_raw=cfg_raw, mode="loaded_path", input_strings=input_strings, on_unknown='raise')


if __name__ == "__main__":
    main()

