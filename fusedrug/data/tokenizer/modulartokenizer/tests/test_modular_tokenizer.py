import os
from pathlib import Path
import unittest
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


CONFIG_PATH = str((Path(__file__).parents[1] / "configs").absolute())
CONFIG_NAME = "tokenizer_config"


class ConfigHolder:
    def __init__(self, cfg: DictConfig = None) -> None:
        print(f' L23 {os.environ.get("MY_GIT_REPOS")}')
        if cfg is None:
            self._setup_test_env()
            with hydra.initialize_config_dir(CONFIG_PATH):
                cfg = hydra.compose(CONFIG_NAME)
        self.config_obj = cfg

    def get_config(self) -> DictConfig:
        return self.config_obj

    def _setup_test_env(self) -> None:
        REPO_ROOT = "fuse-drug"
        if "MY_GIT_REPOS" not in os.environ:
            for i in range(len(Path(__file__).parents)):
                print(f" L37 dir[{i}] =  {Path(__file__).parents[i].name}")
                if Path(__file__).parents[i].name == REPO_ROOT:
                    os.environ["MY_GIT_REPOS"] = str(Path(__file__).parents[i + 1])
                    break


class TestModularTokenizer(unittest.TestCase):
    def setUp(self, config_holder: ConfigHolder = None) -> None:
        if config_holder is None:
            config_holder = ConfigHolder()
        cfg = config_holder.get_config()
        self.cfg = hydra.utils.instantiate(cfg)
        tmp = OmegaConf.to_object(cfg)
        self.cfg_raw: Dict[str, Any] = tmp

    ### load_from_jsons example. This is a less preferable way to load a tokenizer
    # t_mult_loaded = ModularTokenizer.load_from_jsons(
    #     tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"]
    # )

    # test_tokenizer(t_mult_loaded, cfg_raw=cfg_raw, mode="loaded")

    def common_tokenizer_test(self, on_unknown: str = "warn") -> None:

        t_mult_loaded_path = ModularTokenizer.load(
            path=self.cfg_raw["data"]["tokenizer"]["out_path"]
        )
        input_strings = [
            TypedInput("AA", "<BINDING>ACDEFGHIJKLMNPQRSUVACDEF", 10),
            TypedInput("SMILES", "CCCHHCCCHC", 4),
            TypedInput("AA", "EFGHEFGHEFGH", 5),
            TypedInput("SMILES", "C=H==CC=HCCC", None),
        ]
        test_tokenizer(
            t_mult_loaded_path,
            cfg_raw=self.cfg_raw,
            mode="loaded_path",
            input_strings=input_strings,
        )

        input_strings = [
            TypedInput("AA", "<BINDING>AC11DEFGHIJKLMNPQRSUVACDEF", 10),
            TypedInput("SMILES", "CCCHHCCCHC", 4),
            TypedInput("AA", "EFGHEFGHEFGH", 5),
            TypedInput("SMILES", "C=H==CC=HCCC", None),
        ]
        test_tokenizer(
            t_mult_loaded_path,
            cfg_raw=self.cfg_raw,
            mode="loaded_path",
            input_strings=input_strings,
            on_unknown=on_unknown,
        )

    def test_tokenizer_with_warning(self) -> None:
        print(
            "Testing input that sontains characters mapped to <UNK> token, suppressing exception, should raise warning"
        )
        self.common_tokenizer_test(on_unknown="warn")

    def test_tokenizer_with_exception(self) -> None:
        print(
            "Testing input that contains characters mapped to <UNK> token, should raise exception"
        )
        with self.assertRaises(RuntimeError):
            self.common_tokenizer_test(on_unknown="raise")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    print(str(cfg))
    config_holder = ConfigHolder(cfg)
    tester = TestModularTokenizer()
    tester.setUp(config_holder)
    tester.test_tokenizer_with_warning()
    tester.test_tokenizer_with_exception()  # handles getting the exception


if __name__ == "__main__":
    main()
