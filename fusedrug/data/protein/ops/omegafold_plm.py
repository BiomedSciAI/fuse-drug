from fuse.utils import NDict
from fuse.data import OpBase

try:
    from omegafold import standalone_plm
except ImportError:
    from warnings import warn
    warn('Could not import standalone_plm from omegafold')

class OmegaFoldPLM_AASequenceTokenizer(OpBase):
    """
    Tokenize and create pseudo msa in the way that OmegaFoldPLM (Protein Language Model) expects
    """
    def __init__(self, 
        num_pseudo_msa: int = 0, #15, #they used 15
        mask_rate: float = 0.12,
        **kwargs):
        super().__init__(**kwargs)
        self._num_pseudo_msa = num_pseudo_msa
        self._mask_rate = mask_rate
    
    def __call__(self, sample_dict: NDict, 
        key_in='data.input.protein',
        key_out_pmsa='data.input.protein_pmsa',
        key_out_pmsa_mask='data.input.protein_pmsa_mask',
        ):
        """
        key_in:
        key_out_pmsa: pseudo msa
        key_out_pmsa_mask: pseudo msa mask
        """
        data = sample_dict[key_in]
        assert isinstance(data, str)

        p_msa, p_msa_mask = standalone_plm.tokenize_aa_sequence(data,
            num_pseudo_msa=self._num_pseudo_msa,
            mask_rate=self._mask_rate,
        )

        sample_dict[key_out_pmsa] = p_msa
        sample_dict[key_out_pmsa_mask] = p_msa_mask
        return sample_dict