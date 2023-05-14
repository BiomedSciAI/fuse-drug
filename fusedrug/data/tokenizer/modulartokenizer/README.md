## Modular Tokenizer:
* A modular tokenizer combines multiple pre-trained (huggingface-based) tokenizers and maps their tokens to a single, consistent ID space. It's useful for sequence-to-sequence problems, where different tokenizers should be used for different parts of the input sequence, depending on the context, and straightforward merging of the tokenizers may not be possible due to token overlap (e.g. 'C' in an amino acid sequence, standing for Cysteine, and 'C' in a SMILES sequence, standing for Carbon, should be mapped to different IDs). 
* The modular tokenizer retains most of huggingface tokenizer interface (but not the underlying logic), so it can be plugged into existing code with very few (if any at all) changes.
# Interface:
* __init__(): Creates a modular tokenizer that combines multiple existing tokenizers, adjusting them so that:
        a. They all share the same special tokens (combined special tokens from all the source tokenizers),
        b. Each tokenizer retains its regular tokens, however their IDs are remapped to a single space, with no overlaps.
* save_jsons():
* load_from_jsons(): 
* diagnose():
* encode_list():
* encode():
* decode():
* get_vocab_size(): Returns the size of the vocabulary of the modular tokenizer (i.e. the number of unique IDs, which may be greater than the number of unique tokens)
* 