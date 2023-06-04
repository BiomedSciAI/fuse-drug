import os
import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Tuple, Dict, Optional, Any
import pytorch_lightning as pl
from fuse.utils import NDict
from fusedrug.data.tokenizer.ops import FastModularTokenizer as FastTokenizer
from fusedrug.data.tokenizer.modulartokenizer import pretrained_tokenizers
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import TypedInput
from 

def seed(seed_value: int) -> int:
    pl.seed_everything(seed_value, workers=True)

    return seed_value


# def tokenizer(
#     tokenizer_path: str, encoder_inputs_max_seq_len: int, labels_inputs_max_seq_len: int
# ) -> Tuple[FastTokenizer, FastTokenizer]:
#     """
#     Create tokenizer instances. One for encoder input  with encoder_inputs_max_seq_len and one for labels with labels_inputs_max_seq_len
#     """
#     tokenizer_path = os.path.join(pretrained_tokenizers.get_dir_path(), tokenizer_path)
#     encoder_inputs_tokenizer_op = FastTokenizer(
#         tokenizer_path=tokenizer_path,
#         max_size=encoder_inputs_max_seq_len,
#         pad_token="<PAD>",
#     )
#     labels_tokenizer_op = FastTokenizer(
#         tokenizer_path=tokenizer_path,
#         max_size=labels_inputs_max_seq_len,
#         pad_token="<PAD>",
#     )

#     return encoder_inputs_tokenizer_op, labels_tokenizer_op


def test_tokenizer_op(
    tokenizer_op_inst: FastTokenizer, max_len: int = None, mode: Optional[str] = ""
) -> None:
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


@hydra.main(
    config_path="../modulartokenizer/configs",
    config_name="tokenizer_config_personal",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    NDict(cfg).print_tree(True)

    if "track_clearml" in cfg:
        from fuse.dl.lightning.pl_funcs import start_clearml_logger

        clearml_task = start_clearml_logger(**cfg.track_clearml)
        if clearml_task is not None:
            clearml_logger = clearml_task.get_logger()
        else:
            clearml_logger = None  # will be None in dist training and rank != 0
    else:
        clearml_logger = None

    # seed
    seed_value = seed(**cfg.seed)

    # tokenizer
    encoder_inputs_tokenizer_op, labels_tokenizer_op = tokenizer(**cfg.tokenizer)
    
    tmp = OmegaConf.to_object(cfg)
    cfg_raw: Dict[str, Any] = tmp

    mod_tokenizer_op = FastTokenizer(
        tokenizer_path=cfg_raw["data"]["tokenizer"]["out_path"],
        max_size=15,
        pad_token="<PAD>",
        validate_ends_with_eos="<EOS>",
    )
    test_tokenizer_op(
        tokenizer_op_inst=mod_tokenizer_op,
        max_len=15,
        mode="truncation",
    )

    mod_tokenizer_op = FastTokenizer(
        tokenizer_path=cfg_raw["data"]["tokenizer"]["out_path"],
        max_size=cfg_raw["data"]["tokenizer"]["overall_max_len"],
        pad_token="<PAD>",
        validate_ends_with_eos="<EOS>",
    )
    test_tokenizer_op(
        tokenizer_op_inst=mod_tokenizer_op,
        max_len=cfg_raw["data"]["tokenizer"]["overall_max_len"],
        mode="padding",
    )


if __name__ == "__main__":
    main()
