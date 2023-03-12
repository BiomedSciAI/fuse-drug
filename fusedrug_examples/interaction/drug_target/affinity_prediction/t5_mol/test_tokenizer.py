# Taken from https://huggingface.co/course/chapter6/8?fw=pt
from datasets import load_dataset
from tokenizer.special_tokens import get_special_tokens, special_mark

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

import pandas as pd


from rdkit import Chem

FORBIDDEN = set(['B', 'O', 'U', 'X', 'Z'])


def aas_to_smiles(aas, sanitize=True):
    """
    Taken from pytoda.proteins.utils
    Converts an amino acid sequence (IUPAC) into SMILES.

    Args:
        aas (str): The amino acid sequence to be converted.
            Following IUPAC notation.
        sanitize (bool, optional): [description]. Defaults to True.

    Raises:
        TypeError: If aas is not a string.
        ValueError: If string cannot be converted to mol.

    Returns:
        smiles: SMILES string of the AA sequence.
    """
    if not isinstance(aas, str):
        raise TypeError(f'Provide string not {type(aas)}.')
    if len(set(aas).intersection(FORBIDDEN)) > 0:
        raise ValueError(
            f'Characters from: {FORBIDDEN} cant be parsed. Found one in: {aas}'
        )
    mol = Chem.MolFromFASTA(aas, sanitize=sanitize)
    if mol is None:
        raise ValueError(f'Sequence could not be converted to SMILES: {aas}')
    smiles = Chem.MolToSmiles(mol)
    return smiles


TITAN_AA_PATH = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/public/epitopes.csv"
TITAN_SMILES_PATH = "/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/public/epitopes.smi"

def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]


if __name__ == "__main__":
    
    vocab_data = pd.read_csv(TITAN_SMILES_PATH, sep='\t', header=None, names=['repr', 'ID'])
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    
    special_tokens = get_special_tokens()
    trainer = trainers.BpeTrainer(vocab_size=2500, special_tokens=special_tokens)
    
    tokenizer.model = models.BPE()
    
    tokenizer.train_from_iterator(get_training_corpus(dataset=list(vocab_data['repr'])), trainer=trainer) 
    
    encoding = tokenizer.encode(special_mark("ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT"))
    encoding_smiles = tokenizer.encode(aas_to_smiles("ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT"))
    print(encoding.tokens)
    print(len(encoding.tokens))
    print(encoding_smiles.tokens)
    print(len(encoding_smiles.tokens))


