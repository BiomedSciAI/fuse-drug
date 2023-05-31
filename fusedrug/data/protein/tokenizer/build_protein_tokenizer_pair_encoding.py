from typing import Union, Optional
from tokenizers.models import WordLevel, BPE
from tokenizers import Regex
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Split
from tokenizers import normalizers
from pytoda.proteins.processing import IUPAC_VOCAB, UNIREP_VOCAB
from tokenizers import processors
from tokenizers.processors import TemplateProcessing
from torch.utils.data import RandomSampler, BatchSampler
import numpy as np
from fusedrug.data.tokenizer.fast_tokenizer_learn import build_tokenizer
from fusedrug.utils.file_formats import IndexedFasta
from fuse.utils.file_io import change_extension
from tokenizers.trainers import BpeTrainer
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
        )  # .pre_tokenize_str('b  an\nana  \t\r\n')
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


# TODO: add support for providing multiple fasta files
@click.command()
@click.argument("train_on_fasta_file")
@click.option(
    "--output-tokenizer-json-file",
    default=None,
    help='output json file that can be loaded with tokenizers.Tokenizer.from_file(). If not provided defaults to output into the same path as INPUT_FASTA_FILE but with modified extension to ".bpe_vocab.json" ',
)
@click.option("--vocab-size", default=3000, help="choose one of full-cycles-num, iterations-num or time-limit-minutes")
@click.option(
    "--augment/--no-augment",
    default=True,
    help="When enabled activates augmentation during training the tokenizer. Currently only order flipping is supported.",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="When enabled activates shuffle of the dataset. This is useful especially in cases that a dataset is sorted, to avoid biased estimation of pairs during the beginning of the training.",
)
@click.option(
    "--full-cycles-num", default=None, help="choose one of full-cycles-num, iterations-num or time-limit-minutes"
)
@click.option(
    "--iterations-num", default=None, help="choose one of full-cycles-num, iterations-num or time-limit-minutes"
)
@click.option(
    "--time-limit-minutes", default=None, help="choose one of full-cycles-num, iterations-num or time-limit-minutes"
)
def main(
    train_on_fasta_file: str,
    output_tokenizer_json_file: Optional[str],
    vocab_size: int,
    augment: bool,
    shuffle: bool,
    full_cycles_num: int,
    iterations_num: int,
    time_limit_minutes: int,
):
    """
    builds a pair-encoding based tokenizer, which is trained on the provided fasta file
    """

    print(f"train_on_fasta_file set to {train_on_fasta_file}")
    print(f"vocab_size set to {vocab_size}")
    print(f"shuffle set to {shuffle}")
    print(f"augment set to {augment}")
    print(f"full_cycles_num set to {full_cycles_num}")
    print(f"iterations_num set to {iterations_num}")
    print(f"time_limit_minutes set to {time_limit_minutes}")

    if 1 != sum([x is not None for x in [full_cycles_num, iterations_num, time_limit_minutes]]):
        raise Exception("You must provide exactly one of full_cycles_num, iterations_num, or time_limit_minutes !")

    if output_tokenizer_json_file is None:
        output_tokenizer_json_file = change_extension(train_on_fasta_file, ".bpe_vocab.json")
    print(f"output_tokenizer_json_file set to {output_tokenizer_json_file}")

    special_tokens = ["<UNK>", "<CLS>", "<SEP>", "<PAD>", "<MASK>"]
    special_tokens_ids = [x for x in range(len(special_tokens))]

    unknown_token = "<UNK>"
    assert "<UNK>" in special_tokens

    model = BPE(unk_token=unknown_token)
    trainer = BpeTrainer(
        special_tokens=special_tokens,  # NOTE:the order here defined the tokens ids !
        min_frequency=2000,
        show_progress=True,
        initial_alphabet=[k for k in IUPAC_VOCAB.keys() if 1 == len(k) and k >= "A" and k <= "Z"],
        vocab_size=vocab_size,
    )

    special_tokens_tuples = list(zip(special_tokens, special_tokens_ids))

    if False:
        # keeping this for future reference. There's no need to retrain for a new post-processor, so no need for this now
        override_post_processor = TemplateProcessing(  # noqa: F841
            single="<CLS> $0 <SEP>",
            pair="<CLS> $A <SEP> $B:1 <SEP>:1",
            ###NOTE!!!! IMPORTANT!!!!! it needs to match the token ids in the trainer special_tokens!!
            # based on "The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth."
            # from https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
            special_tokens=special_tokens_tuples,
        )

    def to_string(x):
        return str(x)

    def to_upper_case(x):
        return x.upper()

    def random_flip_order(x):
        if 0 == np.random.choice(2):
            return x
        return x[::-1]

    indexed_fasta = IndexedFasta(
        train_on_fasta_file, process_funcs_pipeline=[to_string, random_flip_order, to_upper_case]
    )

    if shuffle:
        train_batch_sampler = BatchSampler(RandomSampler(indexed_fasta), batch_size=2, drop_last=False)
    else:
        train_batch_sampler = None

    tokenizer = build_tokenizer(  # noqa: F841
        model,
        trainer=trainer,
        train_dataset=indexed_fasta,
        train_batch_sampler=train_batch_sampler,
        num_workers=0,
        save_to_json_file=output_tokenizer_json_file,
        ###override_post_processor=override_post_processor,
        override_normalizer=None,
        override_pre_tokenizer=None,
        override_post_processor=None,
        full_cycles_num=full_cycles_num,
        iterations_num=iterations_num,
        time_limit_minutes=time_limit_minutes,
        stop_filename=change_extension(output_tokenizer_json_file, ".stop"),
    )


if __name__ == "__main__":
    # TODO: add usage example/test
    main()
