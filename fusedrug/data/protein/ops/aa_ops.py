import numpy as np

from fuse.utils import NDict
from fuse.data import OpBase


class OpToUpperCase(OpBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key_in="data.input.protein_str", key_out="data.input.protein_str"):
        data = sample_dict[key_in]
        assert isinstance(data, str)

        ans = data.upper()
        sample_dict[key_out] = ans
        return sample_dict


class OpKeepOnlyUpperCase(OpBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, sample_dict: NDict, key_in="data.input.protein_str", key_out="data.input.protein_str"):
        data = sample_dict[key_in]
        assert isinstance(data, str)

        ans = "".join([x for x in data if (x >= "A" and x <= "Z")])
        sample_dict[key_out] = ans
        return sample_dict
