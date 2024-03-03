#%%

from fusedrug.data.tokenizer.modulartokenizer.modular_tokenizer import ModularTokenizer


# add cell type:
config_name = "tokenizer_config_with_celltype"


#%%
t_mult_loaded_path = ModularTokenizer.load(
    path="/Users/matann/git/fuse-drug/fusedrug/data/tokenizer/modulartokenizer/pretrained_tokenizers/bmfm_modular_tokenizer"
)
# %%
print(1)
# %%
