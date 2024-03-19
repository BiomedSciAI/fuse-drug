import click
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from fusedrug.data.tokenizer.modulartokenizer.create_multi_tokenizer import (
    test_tokenizer,
)
from fusedrug.data.tokenizer.modulartokenizer.special_tokens import (
    get_additional_tokens,
)


@click.command()
@click.argument(
    "tokenizer-path",
    # "-p",
    default="pretrained_tokenizers/bmfm_modular_tokenizer",
    # help="the directory containing the modular tokenizer",
)
@click.option(
    "--output-path",
    "-o",
    default=None,
    help="path to write tokenizer in",
)
# # this needs to be run on all the related modular tokenizers
def main(tokenizer_path: str, output_path: str | None) -> None:
    print(f"adding special tokens to {tokenizer_path}")
    if output_path is None:
        output_path = tokenizer_path
    else:
        print(f"output into  {output_path}")

    overall_max_length = None

    ### load_from_jsons example. This is a less preferable way to load a tokenizer
    # t_mult_loaded = ModularTokenizer.load_from_jsons(
    #     tokenizers_info=cfg_raw["data"]["tokenizer"]["tokenizers_info"]
    # )

    # test_tokenizer(t_mult_loaded, cfg_raw=cfg_raw, mode="loaded")

    tokenizer = ModularTokenizer.load(path=tokenizer_path)

    test_tokenizer(
        tokenizer, overall_max_length == overall_max_length, mode="loaded_path"
    )

    # Update tokenizer with special tokens:
    added_tokens = get_additional_tokens(subset=["special", "task"])
    tokenizer.update_special_tokens(
        added_tokens=added_tokens,
        save_tokenizer_path=output_path,
    )
    test_tokenizer(
        tokenizer, overall_max_length == overall_max_length, mode="updated_tokenizer"
    )

    print("Fin")


if __name__ == "__main__":
    main()
