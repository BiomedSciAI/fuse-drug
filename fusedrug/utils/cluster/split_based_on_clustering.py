"""
splits to train/val/test/etc. sets based on cluster_using_mmseqs.py output
"""

from frozendict import frozendict
from typing import Dict
from os.path import join, dirname, basename
import mmap
import numpy as np
from collections import defaultdict

def split(
        cluster_tsv:str, 
        splits_desc:Dict = frozendict(train=0.90, val=0.05, test=0.05),        
        ):
    """
    gets a cluster tsv file (usually the output of cluster_using_mmseqs.py:cluster() call

    at the expected columns structure (no title)
    cluster_center, member

    for example:
    
    """
    files_names = {set_name:join(dirname(cluster_tsv), f'{set_name}@'+basename(cluster_tsv)) for (set_name,_) in splits_desc.items()}
    files_handles = {set_name: open(filename, 'wb') for (set_name, filename) in files_names.items() }
    members_num = defaultdict(int)
    centers_num = defaultdict(int)

    assigned_centers = {}

    with open(cluster_tsv, 'rt') as f:
        mm_read = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) #useful for massive files
        linenum = 0
        line = None
        while True:      
            line = mm_read.readline()      
            if line == b'':
                break    
            
            center, member = line.decode().rstrip().split('\t')
            #print('center=',center,'member=',member)

            if center in assigned_centers:
                use_set = assigned_centers[center]
                members_num[use_set] += 1
            else:
                use_set = _select_set(splits_desc)
                centers_num[use_set] += 1
                members_num[use_set] += 1
                assigned_centers[center] = use_set

            out_fh = files_handles[use_set]            
            out_fh.write(line)

            if not linenum%10**5:
                print(f'line {linenum}: {line}')

            linenum += 1

    #assert False
            
    print('sets summary:')
    for set_name in splits_desc.keys():
        print(f'set {set_name} has {centers_num[set_name]} centers (clusters)')
        print(f'set {set_name} has {members_num[set_name]} members')

    for _, fh in files_handles.items():
        fh.close()

    return files_names


def _select_set(splits_desc:Dict):
    rand_val = np.random.random()

    cumulative_region = 0.0

    for set_name,set_part in splits_desc.items():
        cumulative_region += set_part
        if rand_val<=cumulative_region:
            return set_name
        
    return set_name        


if __name__ == "__main__":
    fn = '/dccstor/fmm/datasets/ddfb/benchmarks/protein/protein_structure_prediction/pdb_structure_pred/0/clustered.tsv'
    split(fn)


