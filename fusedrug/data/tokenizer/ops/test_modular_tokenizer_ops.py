import unittest

import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Dict, Optional, Any
import pytorch_lightning as pl
from fuse.utils import NDict
from fusedrug.data.tokenizer.ops import FastModularTokenizer as FastTokenizer
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import TypedInput


class TestModularTokenizerOps(unittest.TestCase):
    def test_main(self) -> None:
        main()


def seed(seed_value: int) -> int:
    pl.seed_everything(seed_value, workers=True)

    return seed_value


def test_tokenizer_op(
    tokenizer_op_inst: FastTokenizer, mode: Optional[str] = ""
) -> None:
    """_summary_

    Args:
        tokenizer_op_inst (FastTokenizer): ModularTokenizer operator instance
        mode (Optional[str], optional): _description_. Defaults to "".
    """
    max_len = tokenizer_op_inst.get_max_len()
    print("Testing limited-in-length sub-tokenizer inputs")
    input_list = [
        TypedInput("AA", "<BINDING>ACDEFGHIJKLMNPQRSUVACDEF", 10),
        TypedInput("SMILES", "CCCHHCCCHC", 4),
        TypedInput("AA", "EFGHEFGHEFGH", 5),
        TypedInput("SMILES", "C=H==CC=HCCC<EOS>", None),
    ]

    test_input = NDict(
        dict_like={"data.query.encoder_input": input_list, "data.sample_id": 1}
    )
    # TODO: named tuples with specific properties, e.g. max_len for every input, not for input type
    # Test general encoding: (per-tokenizer truncation works)
    print(
        f"Testing encoding lengths with upper limits on internal tokenizer lengths (10,4,5,None), overall limit of {max_len}, and no overriding limit:"
    )
    enc = tokenizer_op_inst(
        sample_dict=test_input,
        key_in="data.query.encoder_input",
        key_out_tokens_ids="data.encoder_input_token_ids",
        key_out_attention_mask="data.encoder_attention_mask",
        key_out_tokenized_object="data.encoder_tokenized_object",
        # typed_input_list=input_strings,
        # max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
    )
    print(f"encoded tokens: {enc['data.encoder_input_token_ids']}")
    # Test overall padding: (global padding works)
    if max_len is not None:
        assert (
            len(enc["data.encoder_input_token_ids"]) == max_len
        ), f"Didn't pad/retruncate to the expected number of tokens ({max_len}), resulting length: {len(enc['data.encoder_input_token_ids'])} mode: {mode}"
    max_seq_len = 20
    print(
        f"Testing encoding lengths with upper limits on internal tokenizer lengths (10,4,5,None), overall limit of {max_len}, and overriging limit of {max_seq_len}:"
    )
    enc = tokenizer_op_inst(
        sample_dict=test_input,
        key_in="data.query.encoder_input",
        key_out_tokens_ids="data.encoder_input_token_ids",
        key_out_attention_mask="data.encoder_attention_mask",
        key_out_tokenized_object="data.encoder_tokenized_object",
        max_seq_len=max_seq_len,
        # typed_input_list=input_strings,
        # max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
    )
    print(f"encoded tokens: {enc['data.encoder_input_token_ids']}")
    # Test overall padding: (global padding works)
    assert (
        len(enc["data.encoder_input_token_ids"]) == max_seq_len
    ), f"Didn't pad/retruncate to the expected number of tokens ({max_seq_len}), resulting length: {len(enc['data.encoder_input_token_ids'])} mode: {mode}"

    print("testing unlimited-in-length subtokenizer inputs")
    input_list = [
        TypedInput("AA", "<BINDING>ACDEFGHIJKLMNPQRSUVACDEF", None),
        TypedInput("SMILES", "CCCHHCCCHC", None),
        TypedInput("AA", "EFGHEFGHEFGH", None),
        TypedInput("SMILES", "C=H==CC=HCCC<EOS>", None),
    ]

    test_input = NDict(
        dict_like={"data.query.encoder_input": input_list, "data.sample_id": 1}
    )
    # TODO: named tuples with specific properties, e.g. max_len for every input, not for input type
    # Test general encoding: (per-tokenizer truncation works)
    print(
        f"Testing encoding lengths with no limits on internal tokenizer lengths, overall limit of {max_len}, and no overriding limit:"
    )
    enc = tokenizer_op_inst(
        sample_dict=test_input,
        key_in="data.query.encoder_input",
        key_out_tokens_ids="data.encoder_input_token_ids",
        key_out_attention_mask="data.encoder_attention_mask",
        key_out_tokenized_object="data.encoder_tokenized_object",
        max_seq_len=None,
        # typed_input_list=input_strings,
        # max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
    )
    print(f"encoded tokens: {enc['data.encoder_input_token_ids']}")
    # Test overall padding: (global padding works)
    if max_len is not None:
        assert (
            len(enc["data.encoder_input_token_ids"]) == max_len
        ), f"Didn't pad/retruncate to the expected number of tokens ({max_len}), resulting length: {len(enc['data.encoder_input_token_ids'])} mode: {mode}"

    print(
        f"Testing encoding lengths with no limits on internal tokenizer lengths, overall limit of {max_len}, and overriging limit of {max_seq_len}:"
    )
    enc = tokenizer_op_inst(
        sample_dict=test_input,
        key_in="data.query.encoder_input",
        key_out_tokens_ids="data.encoder_input_token_ids",
        key_out_attention_mask="data.encoder_attention_mask",
        key_out_tokenized_object="data.encoder_tokenized_object",
        max_seq_len=max_seq_len,
        # typed_input_list=input_strings,
        # max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
    )
    print(f"encoded tokens: {enc['data.encoder_input_token_ids']}")
    # Test overall padding: (global padding works)
    assert (
        len(enc["data.encoder_input_token_ids"]) == max_seq_len
    ), f"Didn't pad/retruncate to the expected number of tokens ({max_seq_len}), resulting length: {len(enc['data.encoder_input_token_ids'])} mode: {mode}"


@hydra.main(
    config_path="../modulartokenizer/configs",
    config_name="tokenizer_config_personal",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    global_max_len = 15
    mod_tokenizer_op = FastTokenizer(
        tokenizer_path=cfg_raw["data"]["tokenizer"]["out_path"],
        max_size=global_max_len,
        pad_token="<PAD>",
        validate_ends_with_eos=True,
    )
    test_tokenizer_op(
        tokenizer_op_inst=mod_tokenizer_op,
        mode="truncation",
    )

    global_max_len = None
    mod_tokenizer_op = FastTokenizer(
        tokenizer_path=cfg_raw["data"]["tokenizer"]["out_path"],
        max_size=global_max_len,
        pad_token="<PAD>",
        validate_ends_with_eos=True,
    )
    test_tokenizer_op(
        tokenizer_op_inst=mod_tokenizer_op,
        mode="padding",
    )


if __name__ == "__main__":
    unittest.main()
