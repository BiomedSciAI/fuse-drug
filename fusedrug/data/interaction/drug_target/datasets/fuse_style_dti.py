import os
import re
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from fuse.utils.ndict import NDict
from fuse.data import DatasetDefault, OpBase
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from torch.utils.data import DataLoader
from fuse.data.ops.ops_cast import OpToTensor

from fusedrug.data.tokenizer.ops import Op_pytoda_SMILESTokenizer, Op_pytoda_ProteinTokenizer
from fusedrug.data.molecule.tokenizer.pretrained import get_path as get_molecule_pretrained_vocab_path


class DTIDataset:
    """
    off-the-shelf dti dataset (mainly for unit-testing and such?)
    """

    @staticmethod
    def _protein_seq_is_valid(seq: str) -> bool:
        _ALPHABET = "ARNDCQEGHILKMFPSTWYV"
        assert len(_ALPHABET) == 20

        if not isinstance(seq, str):
            return False
        if len(seq) == 0:
            return False
        if re.compile(rf"[^{_ALPHABET}]").search(seq):
            return False
        return True

    @staticmethod
    def sample_ids(data: pd.DataFrame) -> np.ndarray:
        """
        :param data: path to the csv
        """

        return np.arange(len(data))

    @staticmethod
    def dynamic_pipeline(data: pd.DataFrame, drug_fixed_size: int, target_fixed_size: int) -> PipelineDefault:
        """
        :param data:
        :param drug_fixed_size:
        :param target_fixed_size:
        """
        vocab_file = os.path.join(get_molecule_pretrained_vocab_path(), "pytoda_molecules_vocab.json")

        rename_columns = {
            "SMILES": "data.drug.smiles",
            "Target Sequence": "data.target.sequence",
            "Label": "data.label",
        }

        dynamic_pipeline = [
            # read data
            (OpReadDataframe(data=data, rename_columns=rename_columns, key_column=None), dict()),
            # pytoda tokenization
            (
                Op_pytoda_SMILESTokenizer(dict(vocab_file=vocab_file)),
                dict(key_in="data.drug.smiles", key_out_tokens_ids="data.drug.tokenized"),
            ),
            (
                Op_pytoda_ProteinTokenizer(amino_acid_dict="iupac"),
                dict(key_in="data.target.sequence", key_out_tokens_ids="data.target.tokenized"),
            ),
            (OpToTensor(), dict(key=["data.drug.tokenized", "data.target.tokenized"], dtype=torch.int32)),
            (OpToOneSize(size=drug_fixed_size), dict(key="data.drug.tokenized")),
            (OpToOneSize(size=target_fixed_size), dict(key="data.target.tokenized")),
        ]

        return PipelineDefault("dynamic", dynamic_pipeline)

    @staticmethod
    def dataset(data_path: str, drug_fixed_size: int = 40, target_fixed_size: int = 700) -> DatasetDefault:
        """
        :param data_path: path to the csv file
        :param stage: train, val, predict (test?)
        """
        df = pd.read_csv(data_path, usecols=["SMILES", "Target Sequence", "Label"])
        data = df[df["Target Sequence"].apply(DTIDataset._protein_seq_is_valid)].reset_index()

        sample_ids = DTIDataset.sample_ids(data)
        dynamic_pipeline = DTIDataset.dynamic_pipeline(
            data, drug_fixed_size=drug_fixed_size, target_fixed_size=target_fixed_size
        )

        dataset = DatasetDefault(sample_ids=sample_ids, dynamic_pipeline=dynamic_pipeline)
        dataset.create()

        return dataset


class OpToOneSize(OpBase):
    """
    Trim or Pad so all the samples' values for the same key will have the same length
    """

    def __init__(self, size: int):
        """
        :param size: the wanted length of the sequences
        """
        super().__init__()
        self._size = size

    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        """
        :param key: the key for the value to trim/pad
        """

        tns = sample_dict[key]

        if tns.shape[0] < self._size:
            # <PAD> token's value = 0, so we pad with zeros
            tns_pad = torch.zeros((self._size), dtype=tns.dtype)
            tns_pad[: tns.shape[0]] = tns
            tns = tns_pad

        elif tns.shape[0] > self._size:
            tns = tns.narrow(0, 0, self._size)

        sample_dict[key] = tns
        return sample_dict


class DTIDataModule(pl.LightningDataModule):
    """
    Simple implementation of a PyTorch Lightning's datamodule that uses the DTIDataset
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
        batch_size: int,
        num_workers: int = 10,
        drug_fixed_size: int = 40,
        target_fixed_size: int = 700,
    ):
        """
        :param train_data_path: path to the training data
        :param val_data_path: path to the validation data
        :param test_data_path: path to the test data
        :param batch_size: batch size
        :param num_workers: number of multi processes
        :param drug_fixed_size: fixed length size for all the drug sequences
        :param target_fixed_size: fixed length size for all the target sequences
        """
        super().__init__()

        self._train_data_path = train_data_path
        self._val_data_path = val_data_path
        self._predict_data_path = test_data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._drug_fixed_size = drug_fixed_size
        self._target_fixed_size = target_fixed_size

    def setup(self, stage: str) -> None:

        if stage == "fit":
            self._dataset_train = DTIDataset.dataset(
                self._train_data_path, drug_fixed_size=self._drug_fixed_size, target_fixed_size=self._target_fixed_size
            )
            self._dataset_val = DTIDataset.dataset(
                self._val_data_path, drug_fixed_size=self._drug_fixed_size, target_fixed_size=self._target_fixed_size
            )

        if stage == "predict":
            self._dataset_predict = DTIDataset.dataset(
                self._predict_data_path,
                drug_fixed_size=self._drug_fixed_size,
                target_fixed_size=self._target_fixed_size,
            )

    def train_dataloader(self) -> DataLoader:
        batch_sampler = BatchSamplerDefault(
            dataset=self._dataset_train,
            balanced_class_name="data.label",
            num_balanced_classes=2,
            batch_size=self._batch_size,
            workers=self._num_workers,
        )
        dl_train = DataLoader(
            dataset=self._dataset_train,
            batch_sampler=batch_sampler,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
        )
        return dl_train

    def val_dataloader(self) -> DataLoader:
        dl_val = DataLoader(
            dataset=self._dataset_val,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return dl_val

    def predict_dataloader(self) -> DataLoader:
        dl_predict = DataLoader(
            dataset=self._dataset_predict,
            collate_fn=CollateDefault(),
            num_workers=self._num_workers,
            batch_size=self._batch_size,
        )
        return dl_predict


if __name__ == "__main__":

    # use data from "https://github.com/samsledje/ConPLex" repository
    TRAIN_URL = "https://raw.githubusercontent.com/samsledje/ConPLex/main/dataset/BindingDB/train.csv"
    VAL_URL = "https://raw.githubusercontent.com/samsledje/ConPLex/main/dataset/BindingDB/train.csv"
    TEST_URL = "https://raw.githubusercontent.com/samsledje/ConPLex/main/dataset/BindingDB/train.csv"

    ## Dataset sanity check
    dataset = DTIDataset.dataset(data_path=TRAIN_URL)
    sample = dataset[42]

    ## DataModule sanity check
    datamodule = DTIDataModule(
        train_data_path=TRAIN_URL,
        val_data_path=VAL_URL,
        test_data_path=TEST_URL,
        batch_size=42,
    )

    print("Done.")
