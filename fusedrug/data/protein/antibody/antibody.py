from typing import List
from fusedrug.data.protein.structure.sabdab import load_sabdab_dataframe
import pandas as pd
from collections import namedtuple
Antibody = namedtuple('Antibody', 'pdb_id index_within_pdb light_chain_id heavy_chain_id antigen_chain_id')

def get_antibodies_info(antibodies_pdb_ids:List[str]) -> List[Antibody]:
    """
    Collects information on all provided antibodies_pdb_ids based on SabDab DB.
    
    """
    sabdab_df = load_sabdab_dataframe()
    antibodies = []
    for pdb_id in antibodies_pdb_ids:
        found = sabdab_df[sabdab_df.pdb==pdb_id]
        for i in range(len(found)):
            row = found.iloc[i]
            if row.model!=0:
                print(f'warning: found an antibody not in model 0, skipping (TODO: add support). For an entry in pdb_id={pdb_id}')
                continue
            
            light_chain = row.Lchain
            heavy_chain = row.Hchain
            antigen_chain = row.antigen_chain
            if pd.isnull(light_chain) or pd.isnull(heavy_chain):
                print(f'warning: at least one of the light chain or heavy chains are missing, skipping it. For an entry in pdb_id={pdb_id}')
                continue

            antibodies.append(Antibody(pdb_id=pdb_id, index_within_pdb=i, light_chain_id=light_chain, heavy_chain_id=heavy_chain, antigen_chain_id=antigen_chain))
    return antibodies

def get_omegafold_preprint_test_antibodies():
    return [        
        '7k7r',
        '7e6p',
        '7kpj',              
        '6xjq',
        '7sjs',
        '7v5n',
        '7phu',
        '7phw',
        '7qji',        
    ]    
