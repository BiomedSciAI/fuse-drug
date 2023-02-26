import numpy as np

from fuse.utils import NDict
from fuse.data import OpBase

class ProteinRandomFlipOrder(OpBase):
    '''
    Randomizes the order of a protein (amino acids) sequence
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, sample_dict: NDict, key_in='data.input.protein_str', key_out='data.input.protein_str'):
        data = sample_dict[key_in]
        assert isinstance(data, str)
        if 1==np.random.choice(2):
            sample_dict[key_out] = data[::-1]            
        return sample_dict


class ProteinIntroduceNoise(OpBase):
    '''
    Introduces noise with possibly different magnitude inside active site amino acids and outside

    expects key_in_aligned_sequence to be the entire amino acids sequence with lowercase representing non-active site and high case representing active site
    
    '''
    def __init__(self, 
        p=0.1,
        **kwargs):
        super().__init__(**kwargs)
        if not isinstance(p, float):
            raise TypeError(f'Please provide float, not {type(p)}.')

        self.p = np.clip(p, 0.0, 1.0)

    def __call__(self, sample_dict: NDict,
        key_in='data.input.protein_str', key_out='data.input.protein_str'):
        data = sample_dict[key_in]
        assert isinstance(data, str)

        amino_acids_num = len(G_AMINO_ACIDS)                                     

        ans = ''
        for c in data:
            if np.random.rand()<self.p:
                #print('mutate inside active site')
                ans += G_AMINO_ACIDS[np.random.randint(amino_acids_num)]
            else:
                ans += c
                        
        sample_dict[key_out] = ans
        return sample_dict


def extract_active_sites_info(aligned_seq:str):
    '''
    processes and extracts useful information from an aligned active site sequence,
    expects low case amino acids to be outside of the active site and high case amino acids to be inside it
    '''
    non_active_sites = ''
    active_sites = ''
    #total_len = len(aligned_seq)
    prev_was_highcase = False
    for c in aligned_seq:            
        next_is_highcase = c<='Z'
        if next_is_highcase ^ prev_was_highcase:
            if next_is_highcase:
                active_sites += '#'
            else:
                non_active_sites += '#'
        
        if next_is_highcase:
            active_sites += c
            prev_was_highcase = True
        else:
            non_active_sites += c
            prev_was_highcase = False

    non_active_sites = [_ for _ in non_active_sites.split('#') if _!='']
    active_sites = [_ for _ in active_sites.split('#') if _!='']

    if aligned_seq[0]<='Z':
        zip_obj = zip(active_sites, non_active_sites)
    else:
        zip_obj = zip(non_active_sites, active_sites)

    all_seqs = [i  for one_tuple in zip_obj for i in one_tuple]

    if len(active_sites) > len(non_active_sites):
        assert len(active_sites) == len(non_active_sites)+ 1
        all_seqs.append(active_sites[-1])
    elif len(active_sites) < len(non_active_sites):
        assert len(active_sites) + 1 == len(non_active_sites)
        all_seqs.append(non_active_sites[-1])
        
    return aligned_seq, non_active_sites, active_sites, all_seqs

class ProteinFlipIndividualActiveSiteSubSequences(OpBase):
    '''
    Randomly flip individual contigeous active site sub-sequences

    expects key_in_aligned_sequence to be the entire amino acids sequence with lowercase representing non-active site and high case representing active site
    
    '''
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(p, float):
            raise TypeError(f'Please pass float, not {type(p)}.')
        self.p = np.clip(p, 0.0, 1.0)

    def __call__(self, sample_dict: NDict,
        key_in_aligned_sequence='data.input.protein_str', key_out='data.input.protein_str'):
        data = sample_dict[key_in_aligned_sequence]
        assert isinstance(data, str)

        aligned_seq, non_active_sites, active_sites, all_seqs = extract_active_sites_info(data)  

        ans = ''
        for substr in all_seqs:
            if substr[0]<='Z':
                if np.random.rand()<self.p:
                    ans += substr[::-1]
                else:
                    ans += substr
            else:
                ans += substr
        
        sample_dict[key_out] = ans
        return sample_dict

G_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

class ProteinIntroduceActiveSiteBasedNoise(OpBase):
    '''
    Introduces noise with possibly different magnitude inside active site amino acids and outside

    expects key_in_aligned_sequence to be the entire amino acids sequence with lowercase representing non-active site and high case representing active site
    
    '''
    def __init__(self, 
        mutate_prob_in_active_site=0.01, 
        mutate_prob_outside_active_site=0.1,
        **kwargs):
        super().__init__(**kwargs)
        if not isinstance(mutate_prob_in_active_site, float):
            raise TypeError(f'Please provide float, not {type(mutate_prob_in_active_site)}.')

        if not isinstance(mutate_prob_outside_active_site, float):
            raise TypeError(f'Please provide float, not {type(mutate_prob_outside_active_site)}.')

        self.mutate_prob_in_active_site = np.clip(mutate_prob_in_active_site, 0.0, 1.0)
        self.mutate_prob_outside_active_site = np.clip(mutate_prob_outside_active_site, 0.0, 1.0)

    def __call__(self, sample_dict: NDict,
        key_in_aligned_sequence='data.input.protein_str', key_out='data.input.protein_str'):
        data = sample_dict[key_in_aligned_sequence]
        assert isinstance(data, str)

        aligned_seq, non_active_sites, active_sites, all_seqs = extract_active_sites_info(data)

        amino_acids_num = len(G_AMINO_ACIDS)                                     
        
        ans = ''
        for curr_sub_seq in all_seqs:
            for c in curr_sub_seq:
                if curr_sub_seq[0]<='Z': #it's uppercase, so it's inside an active site
                    if np.random.rand()<self.mutate_prob_in_active_site:
                        #print('mutate inside active site')
                        ans += G_AMINO_ACIDS[np.random.randint(amino_acids_num)]
                    else:
                        ans += c
                else:
                    if np.random.rand()<self.mutate_prob_outside_active_site:
                        #print('mutate outside active site')
                        ans += G_AMINO_ACIDS[np.random.randint(amino_acids_num)].lower()
                    else:
                        ans += c
        
        sample_dict[key_out] = ans
        return sample_dict

