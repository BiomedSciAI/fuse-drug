import pyfastx   #https://pyfastx.readthedocs.io/en/latest/usage.html
import torch
from typing import List, Callable, Optional
from functools import partial
from copy import deepcopy
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.utils.file_formats import IndexedFasta

class FastaLoader(OpBase):
    '''
    Loads an entry from a fasta file. Uses pyfastx library for quick fetching of relevant parts.
    '''
    def __init__(self, fasta_file_loc=None, check_for_duplicate_names=False, **kwargs):
        '''
        :param fasta_file_loc: location of .fasta or .fasta.gz file
        :param check_for_duplicate_names: checks for duplicates (in names, does not check sequences!)
            may take few minutes.
        '''
        super().__init__(**kwargs)
        self._fasta = IndexedFasta(fasta_file_loc=fasta_file_loc, check_for_duplicate_names=check_for_duplicate_names)

    def __call__(self, sample_dict: NDict, key_out='data.gt.seq'):
        '''
        '''
        sid = get_sample_id(sample_dict)
        entry = self._fasta[sid]
        sample_dict[key_out] = entry.seq        

        return sample_dict
