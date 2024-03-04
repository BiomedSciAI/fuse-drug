import logging
from pathlib import Path
import unittest

from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer


"""
This is a short test to verify that the regular and extended tokenizers are in sync."""

lgr = logging.getLogger(__file__)


def compare_modular_tokenizers(tokenizer1_name: str, tokenizer2_name: str) -> None:

    pertrained_tokenizers_path = Path(__file__).parents[1] / "pretrained_tokenizers"

    lgr.info(
        f"comparing modular tokenizers {tokenizer1_name} and {tokenizer2_name} in {pertrained_tokenizers_path}"
    )

    modular_tokenizer_1 = ModularTokenizer.load(
        path=pertrained_tokenizers_path / tokenizer1_name
    )
    modular_tokenizer_2 = ModularTokenizer.load(
        path=pertrained_tokenizers_path / tokenizer2_name
    )
    # we go over all the tokenizers in modular_tokenizer_1
    for t_type in modular_tokenizer_1.tokenizers_info:
        if t_type not in modular_tokenizer_2.tokenizers_info:
            # this is fine - we expect some different sub-tokenizers - just report
            lgr.info(f"could not find sub-tokenizer {t_type} in {tokenizer2_name}")
            continue

        # get the tokenizer
        tokenizer_instance_1 = modular_tokenizer_1.tokenizers_info[t_type][
            "tokenizer_inst"
        ]
        # get the corresponding tokenizer from the second modular tokenizer
        tokenizer_instance_2 = modular_tokenizer_2.tokenizers_info[t_type][
            "tokenizer_inst"
        ]
        #  get the vocabularies for each
        tokenizer_1_vocabulary = tokenizer_instance_1.get_vocab()
        tokenizer_2_vocabulary = tokenizer_instance_2.get_vocab()

        # if they vocabularies are identical, all is good.  Otherwise, report the problem
        if not (tokenizer_1_vocabulary == tokenizer_2_vocabulary):
            for k in tokenizer_1_vocabulary:
                if k not in tokenizer_2_vocabulary:
                    lgr.error(f"key {k} missing in voc3")
                else:
                    if tokenizer_1_vocabulary[k] != tokenizer_2_vocabulary[k]:
                        lgr.error(
                            f"key {k} mismatch: {tokenizer_1_vocabulary[k]} != {tokenizer_2_vocabulary[k]}"
                        )
            for k in tokenizer_2_vocabulary:
                if k not in tokenizer_1_vocabulary:
                    lgr.error(f"key {k} missing in voc1")
            raise ValueError(
                f"tokenizers {tokenizer1_name} and {tokenizer2_name} are not compatible"
            )


class TestModularTokenizersCompatibility(unittest.TestCase):
    def test_modular_tokenizers_compatibility(self) -> None:
        old_tokenizer_name = "modular_AA_SMILES_single_path"
        regular_tokenizer_name = "bmfm_modular_tokenizer"
        extended_tokenizer_name = "bmfm_extended_modular_tokenizer"
        # If new modular tokenizers are added, and need to be kept synchronized, add more name and tests

        compare_modular_tokenizers(old_tokenizer_name, regular_tokenizer_name)
        compare_modular_tokenizers(regular_tokenizer_name, extended_tokenizer_name)


if __name__ == "__main__":
    unittest.main()
