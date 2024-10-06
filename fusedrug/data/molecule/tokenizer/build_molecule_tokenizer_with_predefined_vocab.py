from typing import Union, Optional
from tokenizers.models import WordPiece
from tokenizers import pre_tokenizers, normalizers, processors, Tokenizer
import json
from fusedrug.data.tokenizer.fast_tokenizer_learn import build_tokenizer

import click

# https://github.com/huggingface/tokenizers/issues/547
# custom components: https://github.com/huggingface/tokenizers/blob/master/bindings/python/examples/custom_components.py


def build_molecule_tokenizer_with_predefined_vocab(
    vocab: Union[str, dict],
    unknown_token: str,
    save_to_json_file: Optional[str] = None,
    override_normalizer: Optional[normalizers.Normalizer] = None,
    override_pre_tokenizer: Optional[pre_tokenizers.PreTokenizer] = None,
    override_post_processor: Optional[processors.PostProcessor] = None,
) -> Tokenizer:
    """
    Builds a simple tokenizer, without any learning aspect (so it doesn't require any iterations on a dataset)

    args:
        vocab: if string then it is assumed to be a json file containing the vocabulary.
        if it's dict it is assumed to be a python dictionary mapping from token string to token id.
        unknown_token:
        save_to_json_file:
        override_normalizer: defaults to no normalizers
        override_pre_tokenizer: provide a pre_tokenizers.PreTokenizer instance to override
        override_post_processor:
    """

    if isinstance(vocab, str):
        with open(vocab, "r") as f:
            vocab = json.load(f)
    else:
        assert isinstance(vocab, dict)

    assert unknown_token in vocab
    # model = WordLevel(vocab=vocab, unk_token=unknown_token)
    model = WordPiece(vocab=vocab, unk_token=unknown_token)

    tokenizer = build_tokenizer(
        model=model,
        save_to_json_file=save_to_json_file,
        override_normalizer=override_normalizer,
        override_pre_tokenizer=override_pre_tokenizer,
        override_post_processor=override_post_processor,
    )

    return tokenizer

### NOTE: not serializable (tokenizer.save()) - so dropped it in favor of "Lowercase"
### see: https://github.com/huggingface/tokenizers/issues/581

# class UppercaseNormalizer:
#     def normalize(self, normalized: NormalizedString):
#         normalized.uppercase()
# running on whi-3


@click.command()
@click.argument("input_vocab_json_file")
@click.argument("output_tokenizer_json_file")
@click.option(
    "--unknown-token",
    default="<UNK>",
    help="allows to override the default unknown token",
)
def main(
    input_vocab_json_file: str, output_tokenizer_json_file: str, unknown_token: str
) -> None:
    """
    Builds a simple (not learned) vocabulary based tokenizer.
    Args:
        INPUT_VOCAB_JSON_FILE: path to a json file mapping from token string to token id
        OUTPUT_TOKENIZER_JSON_FILE: the tokenize will be serialized into this output file path. It can be then loaded using tokenizers.Tokenizer.from_file
    """
    print(f"input_vocab_json_file set to {input_vocab_json_file}")
    print(f"unknown_token set to {unknown_token}")
    print(f"output_tokenizer_json_file set to {output_tokenizer_json_file}")

    build_molecule_tokenizer_with_predefined_vocab(
        vocab=input_vocab_json_file,
        unknown_token=unknown_token,  # '<UNK>',
        save_to_json_file=output_tokenizer_json_file,
    )


if __name__ == "__main__":
    main()
