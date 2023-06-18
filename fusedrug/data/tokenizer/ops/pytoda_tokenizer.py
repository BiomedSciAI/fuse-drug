from typing import Optional

from fuse.utils import NDict
from fuse.data import OpBase
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.proteins.transforms import SequenceToTokenIndexes
from pytoda.transforms import (
    Compose,
    LeftPadding,
    ToTensor,
)
import torch


class Op_pytoda_SMILESTokenizer(OpBase):
    def __init__(self, SMILES_Tokenizer_kwargs: dict, **kwargs: dict):
        super().__init__(**kwargs)
        self.language = SMILESTokenizer(
            **SMILES_Tokenizer_kwargs,
            device=torch.device("cpu"),  # this is critical for DataLoader multiprocessing to work well !!!
        )

    def __call__(
        self, sample: NDict, key_in: str = None, key_out_tokens_ids: str = None
    ) -> NDict:  # key_out_tokens_ids=None, key_out_tokenized_object=None,
        data_str = sample[key_in]
        if not isinstance(data_str, str):
            raise Exception(f"Expected key_in={key_in} to point to a string, and instead got a {type(data_str)}")

        tokenized = self.language.smiles_to_token_indexes(data_str)
        sample[key_out_tokens_ids] = tokenized
        return sample


class Op_pytoda_ProteinTokenizer(OpBase):
    def __init__(
        self,
        amino_acid_dict: str,
        add_start_and_stop: bool = True,
        padding: bool = False,
        padding_length: Optional[int] = None,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.protein_language = ProteinLanguage(amino_acid_dict=amino_acid_dict, add_start_and_stop=add_start_and_stop)

        transforms = []

        transforms += [SequenceToTokenIndexes(protein_language=self.protein_language)]
        if padding:
            if padding_length is None:
                padding_length = self.protein_language.max_token_sequence_length
            transforms += [
                LeftPadding(padding_length=padding_length, padding_index=self.protein_language.token_to_index["<PAD>"],)
            ]

        if isinstance(self.protein_language, ProteinLanguage):
            # transforms += [ToTensor(device=self.device)]
            transforms += [ToTensor(torch.device("cpu"))]
        else:
            # note: ProteinFeatureLanguage supported wasn't transferred here.
            raise TypeError("Please choose either ProteinLanguage or " "ProteinFeatureLanguage")
        self.transform = Compose(transforms)

    def __call__(
        self, sample: NDict, key_in: str, key_out_tokens_ids: str
    ) -> NDict:  # key_out_tokens_ids=None, key_out_tokenized_object=None,
        data_str = sample[key_in]
        if not isinstance(data_str, str):
            raise Exception(f"Expected key_in={key_in} to point to a string, and instead got a {type(data_str)}")
        tokenized = self.transform(data_str)
        sample[key_out_tokens_ids] = tokenized

        return sample


# For a usage example in an end to end model, see: examples/fuse_examples/interaction/drug_target/affinity_prediction/bimodal_mca
