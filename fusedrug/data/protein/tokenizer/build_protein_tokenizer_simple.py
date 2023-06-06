from typing import Union, Optional
from tokenizers.models import WordLevel
from tokenizers import Regex
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Split
from tokenizers import normalizers
from pytoda.proteins.processing import IUPAC_VOCAB, UNIREP_VOCAB
from tokenizers import processors
from fusedrug.data.tokenizer.fast_tokenizer_learn import build_tokenizer

# TODO: make this import and related code below optional
# TODO: consider removing the dependency altogether
from pytoda.proteins.processing import IUPAC_VOCAB
import click

# https://github.com/huggingface/tokenizers/issues/547
# custom components: https://github.com/huggingface/tokenizers/blob/master/bindings/python/examples/custom_components.py


def build_simple_vocab_protein_tokenizer(
    vocab: Union[str, dict],
    unknown_token: str,
    save_to_json_file: Optional[str] = None,
    override_normalizer: Optional[normalizers.Normalizer] = None,
    override_pre_tokenizer: Optional[Union[pre_tokenizers.PreTokenizer, str]] = "per_char_split",
    override_post_processor: Optional[processors.PostProcessor] = None,
):
    """
    Builds a simple tokenizer, without any learning aspect (so it doesn't require any iterations on a dataset)

    args:
        vocab:
        unknown_token:
        save_to_json_file:
        override_normalizer: defaults to no normalizers
        override_pre_tokenizer: defaults to 'per_char_split' - which is useful for most simple (non learned) vocabularies.
            set to None to disable
            provide a pre_tokenizers.PreTokenizer instance to override
        override_post_processor:
    """

    if isinstance(vocab, str):
        vocab = _get_raw_vocab_dict(vocab)
        if unknown_token is None:
            raise Exception('"unknown_token" was not provided')
    else:
        assert isinstance(vocab, dict)

    assert unknown_token in vocab
    model = WordLevel(vocab=vocab, unk_token=unknown_token)

    if isinstance(override_pre_tokenizer, str):
        assert "per_char_split" == override_pre_tokenizer
        per_char_regex_split = Split(
            pattern=Regex("\S"), behavior="removed", invert=True
        )  ##.pre_tokenize_str('b  an\nana  \t\r\n')
        override_pre_tokenizer = pre_tokenizers.Sequence([per_char_regex_split])

    tokenizer = build_tokenizer(
        model=model,
        save_to_json_file=save_to_json_file,
        override_normalizer=override_normalizer,
        override_pre_tokenizer=override_pre_tokenizer,
        override_post_processor=override_post_processor,
    )

    return tokenizer


# Split(pattern='.', behavior='isolated').pre_tokenize_str('blah')


def _get_raw_vocab_dict(name):
    if "iupac" == name:
        return IUPAC_VOCAB
    elif "unirep" == name:
        return UNIREP_VOCAB

    raise Exception(f"unfamiliar vocab name {name} - allowed options are 'iupac' or 'unirep'")


# def _process_vocab_dict_def(token_str):
#     if '<' in token_str:
#         return token_str
#     return token_str.lower()


### NOTE: not serializable (tokenizer.save()) - so dropped it in favor of "Lowercase"
### see: https://github.com/huggingface/tokenizers/issues/581

# class UppercaseNormalizer:
#     def normalize(self, normalized: NormalizedString):
#         normalized.uppercase()
# running on whi-3


@click.command()
@click.argument("vocab_name")
@click.argument("output_tokenizer_json_file")
@click.option(
    "--unknown-token",
    default="<UNK>",
    help="allows to override the default unknown token",
)
def main(vocab_name: str, output_tokenizer_json_file: str, unknown_token: str):
    """
    Builds a simple (not learned) vocabulary based tokenizer.
    Args:
        VOCAB_NAME: available options are "iupac" and "unirep"
        OUTPUT_TOKENIZER_JSON_FILE: the tokenize will be serialized into this output file path. It can be then loaded using tokenizers.Tokenizer.from_file
    """
    print(f"vocab_name set to {vocab_name}")
    print(f"unknown_token set to {unknown_token}")
    print(f"output_tokenizer_json_file set to {output_tokenizer_json_file}")

    build_simple_vocab_protein_tokenizer(
        vocab=vocab_name,
        unknown_token=unknown_token,  #'<UNK>',
        save_to_json_file=output_tokenizer_json_file,
    )


if __name__ == "__main__":
    # TODO: add usage example/test
    main()
