# Taken from https://huggingface.co/course/chapter6/8?fw=pt
from datasets import load_dataset
from tokenizer.special_tokens import get_special_tokens, special_mark_AA, special_wrap_input
import os

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
import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.interaction.drug_target.datasets.dti_binding_dataset import dti_binding_dataset as dti_dataset


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


@hydra.main(config_path='./configs', config_name='train_config', version_base=None)
def main(cfg: DictConfig) -> None: 
    print(str(cfg))

    cfg = hydra.utils.instantiate(cfg)
    cfg_raw = OmegaConf.to_object(cfg)
    ppi_dataset, pairs_df = dti_dataset(**cfg_raw['data']['TITAN_benchmark']['lightning_data_module'])
    # ppi_dataset, pairs_df = dti_dataset.dti_binding_dataset(                       
        
    #                     pairs_tsv=self.pairs_tsv, 
    #                     ligands_tsv=self.ligands_tsv, 
    #                     targets_tsv=self.targets_tsv,
    #                     splits_tsv=self.splits_tsv,
    #                     use_folds=['train1', 'train2', 'train3', 'train4', 'train5'],                         
    #                     pairs_columns_to_extract=['ligand_id', 'target_id', 'activity_label'], 
    #                     pairs_rename_columns={'activity_label': 'data.label'}, 
    #                     ligands_columns_to_extract=['canonical_smiles', 'canonical_aa_sequence', 'full_canonical_aa_sequence'], 
    #                     ligands_rename_columns={'canonical_smiles': 'ligand_str'}, 
    #                     targets_columns_to_extract=['canonical_smiles', 'canonical_aa_sequence', 'full_canonical_aa_sequence'], 
    #                     targets_rename_columns={'canonical_aa_smiles': 'target_str'}, 
                        
    #                     keep_activity_labels=list(self.class_label_to_idx.keys()),
    #                     cache_dir=Path(self.data_dir, 'PLM_DTI_cache'),
    #                     dynamic_pipeline=[(OpLookup(map=self.class_label_to_idx), dict(key_in='data.label', key_out='data.label')),
    #                                       (OpToTensor(), {'dtype': torch.float32, 'key': 'data.label'})],
    #                     )

    
    vocab_data = pd.read_csv(TITAN_SMILES_PATH, sep='\t', header=None, names=['repr', 'ID'])
    
    # Tokenizer example taken from https://huggingface.co/course/chapter6/8?fw=pt
    tokenizer = Tokenizer(models.BPE())
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # Significantly reduces the resulting vocabulary size
        
    special_tokens = get_special_tokens()
    trainer = trainers.BpeTrainer(
        vocab_size=8000, 
        special_tokens=special_tokens)
    
    tokenizer.model = models.BPE()
    
    tokenizer.train_from_iterator(get_training_corpus(dataset=list(vocab_data['repr'])), trainer=trainer) 
    
    encoding = tokenizer.encode(special_wrap_input('BINDING') + special_mark_AA("ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT") + special_wrap_input('SEP') + special_mark_AA("ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT"))
    encoding_smiles = tokenizer.encode(special_wrap_input('BINDING') + aas_to_smiles("ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT") + special_wrap_input('SEP') + aas_to_smiles("ATGCCTTACGCCCCTGGAGACGAAAAGAAGGGT"))
    print(encoding.tokens)
    print(len(encoding.tokens))
    print(encoding_smiles.tokens)
    print(len(encoding_smiles.tokens))
    print(tokenizer.get_vocab_size())
    print('Fin')
    
if __name__ == "__main__":
    os.environ['TITAN_DATA'] =  '/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/'
    os.environ['TITAN_RESULTS'] =  '/dccstor/fmm/users/vadimra/dev/output/TITAN_t5/08-02-2023/'
    main()
    


