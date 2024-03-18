#!/bin/bash -e

# short script to iterate over all the modular tokenizers and update the special tokens, and then run the tests to verify that the tokenizers are in sync

cd $MY_GIT_REPOS/fuse-drug/fusedrug/data/tokenizer/modulartokenizer
tokenizers_configs=("tokenizer_config" "extended_tokenizer_config" "legacy_tokenizer_config")

for config in "${tokenizers_configs[@]}"; do
    echo "doing "$config
    python ./add_multi_tokenizer_special_tokens.py -cn $config
    done;

echo "testing that all tokenizers still match"
python tests/test_extended_and_regular_tokenizers_match.py && echo "all good";
