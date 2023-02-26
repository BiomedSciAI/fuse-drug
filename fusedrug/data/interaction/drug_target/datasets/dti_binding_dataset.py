from typing import Tuple, Optional, Union, List
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from fuse.data import DatasetDefault
from fusedrug.data.interaction.drug_target.loaders.dti_binding_dataset_loader import DTIBindingDatasetLoader
from fuse.data.ops.caching_tools import run_cached_func
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault

def fix_df_types(df):
    if 'source_dataset_activity_id' in df.columns:
        df.source_dataset_activity_id = df.source_dataset_activity_id.astype('string')

    if 'ligand_id' in df.columns:
        df.ligand_id = df.ligand_id.astype('string')

    if 'target_id' in df.columns:
        df.target_id = df.target_id.astype('string')
    return df

def set_activity_multiindex(df):
    df.set_index(['source_dataset_versioned_name','source_dataset_activity_id'], inplace=True)
    return df


def itemify(x):
    try:
        x.item()
    except:
        pass
    return x

def dti_binding_dataset(pairs_tsv:str, ligands_tsv:str, targets_tsv:str, split_tsv:str=None, 
                                    pairs_columns_to_extract=None, pairs_rename_columns=None, \
                                    ligands_columns_to_extract=None, ligands_rename_columns=None, \
                                    targets_columns_to_extract=None, targets_rename_columns=None, **kwargs) -> DatasetDefault:

    # load tsvs with opional caching:
    _args = [pairs_tsv, ligands_tsv, targets_tsv, split_tsv]

    if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None:
        ans_dict = run_cached_func(kwargs['cache_dir'], _load_dataframes,
            *_args, **kwargs
            )
    else:
        ans_dict = _load_dataframes(*_args)

    pairs_df = ans_dict['pairs']
    ligands_df = ans_dict['ligands']
    targets_df = ans_dict['targets']
    
    dynamic_pipeline = [
        (OpReadDataframe(pairs_df, columns_to_extract=pairs_columns_to_extract, rename_columns=pairs_rename_columns, key_column=None), {}), 
        (OpReadDataframe(ligands_df, columns_to_extract=ligands_columns_to_extract, rename_columns=ligands_rename_columns, key_column=None, key_name="ligand_id"), {}), 
        (OpReadDataframe(targets_df, columns_to_extract=targets_columns_to_extract, rename_columns=targets_rename_columns, key_column=None, key_name="target_id"), {}),
    ]
    dynamic_pipeline = PipelineDefault("DTI dataset", dynamic_pipeline)

    dataset = DatasetDefault(sample_ids=None, dynamic_pipeline=dynamic_pipeline)
    dataset.create()

    return dataset

class DTIBindingDataset(Dataset):
    """
    PyTorch dataset for DTI tasks
    """
    def __init__(self,
        pairs_tsv:str,
        ligands_tsv:str,
        targets_tsv:str,
        splits_tsv:str=None, 
        return_index:bool=False,
        use_folds:Optional[Union[List, str]]=None,
        keep_activity_labels:Optional[List[str]]=None,
        cache_dir:Optional[str]=None,
    ):
        """
            Params:
                pairs_tsv: path to the activity pairs tsv (like csv but tab separated)
                ligands_tsv: path to the ligands tsv
                targets_tsv: path to the targets tsv
                splits_tsv: splits_tsv and use_folds can be used together to select only a subset of pairs_tsv - useful for selecting only training folds, or a validation set, etc.
                Use None for both (default) to use the entirety of the pairs_tsv
                splits_tsv points to a tsv file containing the folds description
                return_index: whether to return the unique index (or multiindex) of each sample from getitem (not necessarily equivalent to the dataset sample ids)
                use_folds: splits_tsv and use_folds can be used together to select only a subset of pairs_tsv - useful for selecting only training folds, or a validation set, etc.
                    Use None for both (default) to use the entirety of the pairs_tsv
                    use_folds provides a list (or a single string) describing the folds to use. 
                    For example: use_folds=['fold0','fold1','fold3','fold4']
                    Another example: use_folds='test_set'

                keep_activity_labels: keep only activity_label from this list
                    provide None (default) to keep all.
                    example usage: keep_activity_labels=['Active','Inactive']



                cache_dir: optional - set a path if you want the constructor calculations to be cached.
                Note - caching takes into consideration the arguments and the *direct* code. If any deeper code 
                changes it will be unnoticed and the cache will be stale!
                USE WITH CAUTION!
        """

        _args = [pairs_tsv, ligands_tsv, targets_tsv, splits_tsv, use_folds, keep_activity_labels,]

        print(f"creating dataset with:\n\tpairs_tsv={pairs_tsv},\n\tligands_tsv={ligands_tsv},\n\ttargets_tsv={targets_tsv},\n\tsplits_tsv={splits_tsv},\n\tuse_folds={use_folds},\n\tkeep_activity_labels={keep_activity_labels}")

        if cache_dir is not None:
            ans_dict = run_cached_func(cache_dir, _load_dataframes,
                *_args
                )
        else:
            ans_dict = _load_dataframes(*_args)

        self._pairs = ans_dict['pairs']
        self._ligands = ans_dict['ligands']
        self._targets = ans_dict['targets']
        self.return_index = return_index

    def __len__(self):
        return len(self._pairs)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        """
        two options are supported:        
        1. if index is an integer, the row with this index will be loaded (using .iloc).
        This is considered unsafe, because the order of the table might change due to operations,
        and it might create a shift which breaks sync with, e.g., a sampler trying to balance things

        2. A tuple ([source_dataset_versioned_name:str], [source_dataset_activity_id:str])
        """        
        
        ### keeping for now  it helps profiling
        # return dict(
        #     ligand_str='c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1',
        #     target_str='MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK',
        #     ground_truth_activity_value=0.7,
        #     ground_truth_activity_label='Active',
        # )
        
        if isinstance(index, (int, np.integer)):            
            row = self._pairs.iloc[index]       
            df_index = row.name
        else:            
            assert isinstance(index, tuple)
            assert 2==len(index)                        
            source_dataset_versioned_name, source_dataset_activity_id = index    
            df_index = index       
            row = self._pairs.loc[source_dataset_versioned_name, source_dataset_activity_id ]            
        if not self.return_index:
            df_index = None
        ground_truth_activity_value = itemify(row['activity_value'])
        if not np.isscalar(ground_truth_activity_value):
            try:
                ground_truth_activity_value = float(ground_truth_activity_value)
                print(f'converted from nonscalar: {ground_truth_activity_value}')
            except:
                raise Exception(f'Could not convert activity value "{ground_truth_activity_value}" to float!')

        ground_truth_activity_label = itemify(row['activity_label'])
                                
        ligand_id = itemify(row['ligand_id'])                                  
        target_id = itemify(row['target_id'])            

        ligand_row = self._ligands.loc[ligand_id]
        
        # #### remove this ! trying with offset to see if training still works well
        # try:
        #     ligand_row = self._ligands.iloc[int(ligand_id)+30]
        # except:
        #     print("DEBUG::had issue accessing self._ligands.iloc[ligand_id+30]")
        #     ligand_row = self._ligands.loc[ligand_id]
            
        ligand_str = ligand_row.canonical_smiles

        if not isinstance(ligand_str, str):
            raise Exception(f'ERROR!!! expected a string for canonical_smiles !!! instead got {type(ligand_str)}for index {index} - ligand row = {ligand_row}')
            
        target_row = self._targets.loc[target_id]
        target_str = target_row.canonical_aa_sequence

        if not isinstance(target_str, str):
            raise Exception(f'ERROR!!! expected a string for canonical_aa_sequence !!! instead got {type(target_str)}for index {index} - target row = {target_row}')
    
        
        #     ligand_str='c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1',
        #     target_str='MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK',
        if not isinstance(ground_truth_activity_label, str):
            raise Exception(f'ERROR!!! expected a string for ground_truth_activity_label !!! instead got {type(ground_truth_activity_label)} for index {index}')
        
        return dict(
            df_index=df_index,
            ligand_str=ligand_str,
            
            #debug - use a constant ligand
            #ligand_str='c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1', 

            target_str=target_str,

            #debug - use a constant target
            #target_str='MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK',
           
            ground_truth_activity_value=ground_truth_activity_value,
            ground_truth_activity_label=ground_truth_activity_label,
        )


def _load_dataframes(pairs_tsv:str,
        ligands_tsv:str,
        targets_tsv:str,
        splits_tsv:str=None, 
        use_folds:Optional[Union[List, str]]=None,
        keep_activity_labels:List[str]=None,
        **kwargs,
    ):
    """
    Loads pairs, ligands and targets, and optionally filters in a subset

    We use it mostly with run_cached_func() to minimize time and memory footfprint
    Args:
        ligands_tsv:
        targets_tsv:
        splits_tsv:
        use_folds: Optionally provide a list of folds to keep, pass None (default) to keep all
        keep_activity_labels: Optionally provide a list of activity labels to keep, pass None (default) to keep all        
    """
                                
    assert isinstance(pairs_tsv, str)       
    print(f'loading {pairs_tsv}')
    _pairs = pd.read_csv(pairs_tsv, sep='\t')
    _pairs = fix_df_types(_pairs)
    #_pairs = concat_full_activity_col(_pairs)
    set_activity_multiindex(_pairs)
    print(f'pairs num: {len(_pairs)}')
    
    if splits_tsv is not None:
        if use_folds is None:
            raise Exception(f'splits_tsv was provided ({splits_tsv}) but no use_folds provided')

    if use_folds is not None:
        if splits_tsv is None:
            raise Exception(f'use_folds was provided ({use_folds}) but no splits_tsv provided')

    if splits_tsv is not None:            
        print(f'loading split file {splits_tsv}')
        _splits = pd.read_csv(splits_tsv, sep='\t')
        _splits = fix_df_types(_splits)
        set_activity_multiindex(_splits)
        #_splits = concat_full_activity_col(_splits)
        print(f'it contains {len(_splits)} rows')

        if len(_splits) != len(_pairs):
            raise Exception(f'split file {splits_tsv} contains {len(_splits)} rows while the pairs file {pairs_tsv} contains {len(_pairs)} rows! they should be identical.')
        
        _pairs_MERGED = _pairs.merge(
            _splits,
            how='inner',
            #on='full_activity_id',
            on=['source_dataset_versioned_name','source_dataset_activity_id']
        )

        _pairs = _pairs_MERGED
        del _pairs_MERGED

        _pairs.reset_index(inplace=True)
        set_activity_multiindex(_pairs)

        _pairs = _pairs[_pairs.split.isin(use_folds)]       
        print(f'use_folds={use_folds} keeps {len(_pairs)} rows')            
        

                    
    assert isinstance(ligands_tsv, str)           
    print(f'loading {ligands_tsv}')        
    _ligands = pd.read_csv(ligands_tsv, sep='\t')        
    _ligands = fix_df_types(_ligands)
    _ligands.set_index('ligand_id', inplace=True)
    print(f'ligands num: {len(_ligands)}')
    _ligands = _ligands[~_ligands.canonical_smiles.isnull()]
    print(f'ligands num after keeping only ligands with non-NaN canonical_smiles: {len(_ligands)}')


    assert isinstance(targets_tsv, str)    
    print(f'loading {targets_tsv}')
    _targets = pd.read_csv(targets_tsv, sep='\t')
    _targets = fix_df_types(_targets)
    _targets.set_index('target_id', inplace=True)
    print(f'tagets num: {len(_targets)}')
    
    
    print(f'pairs num before keeping only pairs with ligands found in the (preprocessed) ligands table: {len(_pairs)}')
    _pairs = _pairs[_pairs.ligand_id.isin(_ligands.index)]
    print(f'pairs num after keeping only pairs with ligands found in the (preprocessed) ligands table: {len(_pairs)}')
    
    _pairs = _pairs[_pairs.target_id.isin(_targets.index)]
    print(f'pairs num after keeping only pairs with target found in the (preprocessed) targets table: {len(_pairs)}')
            
    if keep_activity_labels is not None:
        _pairs = _pairs[_pairs.activity_label.isin(keep_activity_labels)]
        print(f'pairs num after keeping only activity_label in {keep_activity_labels}: {len(_pairs)}')

    return dict(
        pairs=_pairs,
        ligands=_ligands,
        targets=_targets,
    )




def _fill_in_dummy_sample(sample_dict):
    _ligand_size = 696
    sample_dict['data.input.tokenized_ligand'] = np.random.randint(0,3000,size=_ligand_size)
    sample_dict['data.input.tokenized_ligand_attention_mask'] = [True]*_ligand_size

    _target_size = 2536
    sample_dict['data.input.tokenized_target'] = np.random.randint(0,33,size=_target_size)
    sample_dict['data.input.tokenized_target_attention_mask'] = [True]*_target_size

    sample_dict['data.gt.activity_value'] = np.random.rand(1).item()
    sample_dict['data.gt.activity_label_class_idx'] = np.random.randint(0,5)     
    return sample_dict