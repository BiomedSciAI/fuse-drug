from fuse.utils import NDict
from fuse.data import OpBase
import numpy as np


class OpAddAttentionMask(OpBase):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)

    def __call__(
        self, sample_dict: NDict, based_on_key: str, key_out: str, value: int = 1
    ) -> NDict:
        """
        Will add an attention mask at the same shape as 'based_on_key', with value 'value'
        :param based_on_key: shape and type will be taken from the sample in this key
        :param key_out: the new attention mask will be written to the sample in this key
        :param value: the value that will be set in the attention mask tensor
        """

        assert isinstance(sample_dict[based_on_key], np.ndarray)

        sample_dict[key_out] = np.full_like(sample_dict[based_on_key], value)

        return sample_dict
