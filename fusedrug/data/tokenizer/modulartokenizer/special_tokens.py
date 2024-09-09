special_token_marker = [
    "<",
    ">",
]


def special_wrap_input(x: str) -> str:
    return special_token_marker[0] + x + special_token_marker[1]


def strip_special_wrap(x: str) -> str:
    for spec_wrap in special_token_marker:
        x = x.replace(spec_wrap, "")
    return x


# keeping for backward compatibility - it's better not to use it.
special_tokens = {
    "unk_token": "UNK",  # Unknown token
    "pad_token": "PAD",  # Padding token
    "cls_token": "CLS",  # Classifier token (probably irrelevant in the T5 setting)
    "sep_token": "SEP",  # Separator token
    "mask_token": "MASK",  # Mask token
    "eos_token": "EOS",  # End of Sentence token
}
