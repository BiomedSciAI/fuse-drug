#%%

import hydra
from omegaconf import DictConfig, OmegaConf
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer
from typing import Dict, Any

os.environ['TITAN_DATA'] = '/your_TITAN_data_path/'
os.environ['MERGED_PPI_DATA'] = '/your_TITAN_data_path/'

from fusedrug.data.tokenizer.modulartokenizer.create_multi_tokenizer import (
    test_tokenizer,
)
from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import TypedInput

# add cell type:
config_name = "tokenizer_config_with_celltype"


#%%
t_mult_loaded_path = ModularTokenizer.load(
        path="/Users/matann/git/fuse-drug/fusedrug/data/tokenizer/modulartokenizer/pretrained_tokenizers/bmfm_modular_tokenizer"
    )
# %%
print (1)
# %%
