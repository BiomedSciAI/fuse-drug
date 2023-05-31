from typing import Union, Optional
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from fusedrug.data.tokenizer.fast_tokenizer_learn import build_tokenizer
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset, IterableDataset, Sampler
import click

# https://github.com/huggingface/tokenizers/issues/547
# custom components: https://github.com/huggingface/tokenizers/blob/master/bindings/python/examples/custom_components.py
# TODO: since we introduced dataset usage, this function can be made a generic BPE trainer, not specific to molecules


def build_molecule_tokenizer(
    dataset: Union[Dataset, IterableDataset],
    output_tokenizer_json: str,
    batch_sampler: Optional[Sampler],
    num_workers: int = 0,
    full_cycles_num=None,
    iterations_num=None,
    time_limit_minutes=None,
    stop_filename=None,
):
    """
    Trains a byte-pair encoding (BPE) based tokenizer on the provided SMI file.

    Args:
        dataset: a torch Dataset or IterableDataset
        output_tokenizer_json: the generated tokenizer will be serialized into this output json
        batch_sampler: optional.
            If you provided a pytorch Dataset, you can provide a sampler to control the sampling order.
              this is useful, for example, if you want to shuffle the order of samples being read to avoid a bias in the tokenizer learning algorithm if the dataset is sorted
              if you do not provide a sampler the provided Dataset will be processed sequentially.
            If you provided a pytorch IterableDatset, the sampler arg must remain None

        full_cycles_num: number of full cycles on the provided dataset. Note - you must provide exactly one of full_cycles_num, iteations_num,time_limit_minutes
        iterations_num: number of iterations (samples shown to the tokenizer learner). Note - you must provide exactly one of full_cycles_num, iteations_num,time_limit_minutes
        time_limit_minutes: time limit for learning the tokenizer. Note - you must provide exactly one of full_cycles_num, iteations_num,time_limit_minutes
    """

    if not isinstance(dataset, (Dataset, IterableDataset)):
        raise Exception(f"dataset must be a torch Dataset or IterableDataset. Instead got {type(dataset)}")

    assert isinstance(output_tokenizer_json, str)

    special_tokens = ["<UNK>", "<CLS>", "<SEP>", "<PAD>", "<MASK>"]
    special_tokens_ids = [x for x in range(len(special_tokens))]

    unknown_token = "<UNK>"
    assert "<UNK>" in special_tokens

    model = BPE(unk_token=unknown_token)
    trainer = BpeTrainer(
        special_tokens=special_tokens,  # NOTE:the order here defined the tokens ids !
        min_frequency=2000,
        show_progress=True,
        # limit_alphabet=,
        vocab_size=3000,  # remember that this is an important hyperparam
    )

    special_tokens_tuples = list(zip(special_tokens, special_tokens_ids))

    if False:  # keeping this for future reference. There's no need to retrain for a new post-processor, so no need
        override_post_processor = TemplateProcessing(  # noqa: F841
            single="<CLS> $0 <SEP>",
            pair="<CLS> $A <SEP> $B:1 <SEP>:1",
            ###NOTE!!!! IMPORTANT!!!!! it needs to match the token ids in the trainer special_tokens!!
            # based on "The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth."
            # from https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
            special_tokens=special_tokens_tuples,
        )

    tokenizer = build_tokenizer(
        model,
        trainer=trainer,
        train_dataset=dataset,
        train_batch_sampler=batch_sampler,
        num_workers=num_workers,
        save_to_json_file=output_tokenizer_json,
        full_cycles_num=full_cycles_num,
        iterations_num=iterations_num,
        time_limit_minutes=time_limit_minutes,
        stop_filename=stop_filename,
    )

    return tokenizer


@click.command()
@click.argument("input_smi_file")
@click.argument("molecule_id_column_idx")
@click.option(
    "--output-tokenizer-json",
    default=None,
    help='path for output tokenizer json file which can be quickly loaded and used. If not provided, defaults to same as "INPUT_SMI_FILE" but with the extension modified to ".bpe_vocab.json"',
)
@click.option(
    "--augment/--no-augment", default=True, help="When enabled activates augmentation when training the tokenizer."
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="When enabled activates shuffle of the dataset. This is useful especially in cases that a dataset is sorted, to avoid biased estimation of pairs during the beginning of the training.",
)
@click.option(
    "--full-cycles-num",
    default=None,
    help="Sets the numbe of full cycles over the dataset. Useful when augment is activated in case you want to train on each sample multiple times. (mutually exclusive with iterations-num and with limit-time)",
)
@click.option(
    "--iterations-num",
    default=None,
    help="Sets the number of iterations. (mutually exclusive with full-cycles-num and with limit-time)",
)
@click.option(
    "--time-limit-minutes",
    default=None,
    help="limits the training to the defined amount of minutes. (mutually exclusive with full-cycles-num and with iterations-num)",
)
def main(
    input_smi_file: str,
    molecule_id_column_idx: int,
    output_tokenizer_json: Optional[str],
    augment: bool,
    shuffle: bool,
    full_cycles_num,
    iterations_num,
    time_limit_minutes,
):
    """
    Trains a pair-encoding based tokenizer on the provided SMI file.
    NOTE: expects the SMI file to already be (rdkit) sanitized!

    Args:
        INPUT_SMI_FILE: the SMI file to base training of the tokenizer on
        MOLECULE_ID_COLUMN_IDX: the index (integer) of the ID column (usually 0 or 1)
    """


if __name__ == "__main__":
    # TODO: add usage example/test
    main()
