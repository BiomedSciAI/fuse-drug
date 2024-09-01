#!/bin/bash -e

# short script to iterate over all the modular tokenizers and update the special tokens, and then run the tests to verify that the tokenizers are in sync

cd $MY_GIT_REPOS/fuse-drug/fusedrug/data/tokenizer/modulartokenizer/pretrained_tokenizers
tokenizers=("bmfm_modular_tokenizer" "bmfm_extended_modular_tokenizer" "modular_AA_SMILES_single_path")

for tokenizer in "${tokenizers[@]}"; do
    echo "doing "$tokenizer
    python ../add_multi_tokenizer_special_tokens.py $tokenizer $@
    done;

echo "testing that all tokenizers still match"
python ../tests/test_extended_and_regular_tokenizers_match.py && echo "all good";
