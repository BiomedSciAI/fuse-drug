"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----

Dataset for AMP peptide generation task. Built from several publicly available datasets. mixing labeled and unlabeled peptides. Labels are provided for both toxicity and AMP.
  * [DBAASP - Database of Antimicrobial Activity and Structure of Peptides](https://dbaasp.org/)
  * [SATPdb - structurally annotated therapeutic peptides database](http://crdd.osdd.net/raghava/satpdb/)
  * [TOXINPRED](https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php)
  * [UniProt](https://www.uniprot.org/)
The function PeptidesDatasets.dataset() will create the a dataset instance.

Based on Payel Das et al. Accelerated antimicrobial discovery via deep generative models and molecular dynamics simulations.
Some parts of the code (mostly the part that infer the AMP and Toxicity labels in DBAASP dataset) are based on repo https://github.com/IBM/controlled-peptide-generation
"""

from typing import Sequence, Union, List, Tuple, Any
import re
import os
from glob import glob
from collections import OrderedDict

import pandas as pd


from fuse.data import DatasetDefault, PipelineDefault, OpBase
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpCond, OpSet
from fuse.data.utils.split import dataset_balanced_division_to_folds
from fuse.utils import NDict
from Bio import SeqIO


from modlamp.descriptors import *  # noqa


_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
assert len(_ALPHABET) == 20


def _seq_is_valid(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    if len(seq) == 0:
        return False
    if re.compile(rf"[^{_ALPHABET}]").search(seq):
        return False
    return True


def _seq_len_less_50(seq: str) -> bool:
    if len(seq) > 50:
        return False
    return True


class OpProcessTargetActivities(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:

        target_activties = sample_dict["targetActivities"]
        seq = sample_dict["sequence"]
        desc = GlobalDescriptor(seq.strip())
        desc.calculate_MW(amide=True)
        molecular_weight_g_mol = desc.descriptor[0][0]  # g/mol
        activity_list = []

        for target in target_activties:  # one seq has a list of targets

            concentration = self.parse_concentration(target["concentration"])
            if concentration is None:
                continue
            if "unit" not in target:
                continue

            # convert to µg/ml
            if target["unit"]["name"] in ["µg/ml"]:
                concentration_ug_mg = concentration  # µg/ml
            elif target["unit"]["name"] in ["µM"]:  # µmol/L
                concentration_ug_mg = (
                    concentration * molecular_weight_g_mol / 1000
                )  # µg/ml
            elif target["unit"]["name"] in ["nmol/g", "nmol/g "]:
                # TODO: implement - skip for now
                continue
            elif target["unit"]["name"] in ["µg/g"]:
                # TODO: implement - skip for now
                continue
            else:
                print(target["unit"])
                continue

            activity_list.append(concentration_ug_mg)

        if len(activity_list) == 0:
            sample_dict["amp"] = {"min": "N/A", "max": "N/A", "label": -1}
            return sample_dict

        min_mic = min(activity_list)
        sample_dict["amp"] = {"min": min_mic, "max": max(activity_list)}
        if min_mic <= 25:
            sample_dict["amp.label"] = 1
        elif min_mic >= 100:
            sample_dict["amp.label"] = 0
        else:
            sample_dict["amp.label"] = -1  # unknown

        return sample_dict

    @staticmethod
    def parse_concentration(concentration: str) -> float:
        if concentration == "NA":
            return None

        concentration = concentration.replace(">", "")  # '>10' => 10
        concentration = concentration.replace("<", "")  # '<1.25' => 1.25
        concentration = concentration.replace("=", "")  # '=2' => 2
        if concentration == "NA":
            return None
        index = concentration.find("±")
        if index != -1:
            concentration = concentration[:index]  # 10.7±4.6 => 10.7
        index = concentration.find("-")
        if index != -1:
            concentration = concentration[:index]  # 12.5-25.0 => 12.5
        concentration = concentration.strip()
        try:
            return float(concentration)
        except:
            return None


class OpProcessHemoliticCytotoxicActivities(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:

        targets = sample_dict["hemoliticCytotoxicActivities"]
        seq = sample_dict["sequence"]
        desc = GlobalDescriptor(seq.strip())
        desc.calculate_MW(amide=True)
        molecular_weight_g_mol = desc.descriptor[0][0]  # g/mol
        toxcity_list = []

        for target in targets:  # one seq has a list of targets

            concentration = self.parse_concentration(target["concentration"])
            if concentration is None:
                continue
            if "unit" not in target:
                continue

            # convert to µg/ml
            if target["unit"]["name"] in ["µg/ml"]:
                concentration_ug_mg = concentration  # µg/ml
            elif target["unit"]["name"] in ["µM"]:  # µmol/L
                concentration_ug_mg = (
                    concentration * molecular_weight_g_mol / 1000
                )  # µg/ml
            elif target["unit"]["name"] in ["nmol/g", "nmol/g "]:
                # TODO: implement - skip for now
                continue
            elif target["unit"]["name"] in ["µg/g"]:
                # TODO: implement - skip for now
                continue
            else:
                print(target["unit"])
                continue

            toxcity_list.append(concentration_ug_mg)

        if len(toxcity_list) == 0:
            sample_dict["toxicity"] = {"min": "N/A", "max": "N/A", "label": -1}
            return sample_dict

        min_mic = min(toxcity_list)
        sample_dict["toxicity"] = {"min": min_mic, "max": max(toxcity_list)}
        if min_mic <= 200:
            sample_dict["toxicity.label"] = 1
        elif min_mic >= 250:
            sample_dict["toxicity.label"] = 0
        else:
            sample_dict["toxicity.label"] = -1  # unknown

        return sample_dict

    @staticmethod
    def parse_concentration(concentration: str) -> Union[float, None]:
        if concentration == "NA":
            return None

        concentration = concentration.replace(">", "")  # '>10' => 10
        concentration = concentration.replace("<", "")  # '<1.25' => 1.25
        concentration = concentration.replace("=", "")  # '=2' => 2
        if concentration == "NA":
            return None
        index = concentration.find("±")
        if index != -1:
            concentration = concentration[:index]  # 10.7±4.6 => 10.7
        index = concentration.find("-")
        if index != -1:
            concentration = concentration[:index]  # 12.5-25.0 => 12.5
        concentration = concentration.strip()
        try:
            return float(concentration)
        except:
            return None


class Dbaasp:
    @staticmethod
    def load_and_process_df(
        raw_data_path: str,
        fields: Sequence[str] = [
            "dbaaspId",
            "name",
            "sequence",
            "targetActivities",
            "hemoliticCytotoxicActivities",
            "unusualAminoAcids",
        ],
    ) -> pd.DataFrame:
        """
        :param raw_data_path: path to peptides-complete.json as downloaded from: https://dbaasp.org/download-dataset?source=peptides
        """
        # load the raw data
        peptides = pd.read_json(raw_data_path)
        peptides = pd.json_normalize(peptides["peptides"], max_level=0)

        # keep the required fields only
        peptides = peptides[fields]

        # filter invalid sequences
        peptides = peptides[peptides.sequence.apply(_seq_is_valid)]

        # keep only len < 50
        peptides = peptides[peptides.sequence.apply(_seq_len_less_50)]

        # filter duplicates
        peptides = peptides.drop_duplicates(subset=["sequence"], keep="first")

        # reset index to reenumerate the samples
        peptides = peptides.reset_index()

        return peptides

    @staticmethod
    def process_pipeline() -> List[Tuple]:
        return [
            (OpProcessTargetActivities(), dict()),
            (OpProcessHemoliticCytotoxicActivities(), dict()),
        ]


class ToxinPred:
    @staticmethod
    def load_and_process_df(data_path: str) -> pd.DataFrame:
        """
        :param data_path: path to a folder that contains multiple files downloaded directly from "https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php"
        """
        data_files = glob(os.path.join(data_path, "*"))
        sequence_files = {}
        for filename in data_files:
            if "neg" in os.path.basename(filename):
                sequence_files[filename] = 0
            elif "pos" in os.path.basename(filename):
                sequence_files[filename] = 1

        peptides = pd.DataFrame(columns=["sequence", "toxicity.label"])
        for filename, label in sequence_files.items():
            seqs = pd.read_csv(filename, names=["sequence"], header=None)
            seqs["toxicity.label"] = label
            peptides = pd.concat((peptides, seqs))

        # filter invalid sequences
        peptides = peptides[peptides.sequence.apply(_seq_is_valid)]

        # keep only len < 50
        peptides = peptides[peptides.sequence.apply(_seq_len_less_50)]

        # filter duplicates
        peptides = peptides.drop_duplicates(subset=["sequence"], keep="first")

        # reset index to reenumerate the samples
        peptides = peptides.reset_index()

        return peptides

    @staticmethod
    def process_pipeline() -> List[Tuple]:
        return [(OpSet(), dict(key="amp.label", value=-1))]


class AsciiHandle:
    """workaround to remove non-ascii characters from file while iterating"""

    def __init__(self, handle: Sequence):
        self._handle = handle

    def __iter__(self) -> str:
        for line in self._handle:
            yield line.encode("ascii", errors="ignore").decode()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except:
            return self._handle.__getattribute__(name)


class Satpdb:
    @staticmethod
    def load_and_process_df(data_path: str) -> pd.DataFrame:
        """
        :param data_path: path to a folder that contains files 'antimicrobial.fasta' and 'toxic.fasta' downloaded from https://webs.iiitd.edu.in/raghava/satpdb/down.php
        """

        sequence_files = {
            os.path.join(data_path, "antimicrobial.fasta"): {
                "amp.label": 1,
                "toxicity.label": -1,
            },
            os.path.join(data_path, "toxic.fasta"): {
                "amp.label": -1,
                "toxicity.label": 1,
            },
        }

        peptides = pd.DataFrame(columns=["sequence", "toxicity.label", "amp.label"])
        for filename, label_dict in sequence_files.items():
            with open(filename) as handle:
                # workaround for non-asi
                seqs = []
                for record in SeqIO.parse(AsciiHandle(handle), "fasta"):
                    seqs.append(str(record.seq).strip())
                seqs = pd.DataFrame(seqs, columns=["sequence"])

                for label_name, label_value in label_dict.items():
                    seqs[label_name] = label_value
                peptides = pd.concat((peptides, seqs))

        # filter invalid sequences
        peptides = peptides[peptides.sequence.apply(_seq_is_valid)]

        # keep only len < 50
        peptides = peptides[peptides.sequence.apply(_seq_len_less_50)]

        # filter duplicates
        peptides = peptides.drop_duplicates(subset=["sequence"], keep="first")

        # reset index to reenumerate the samples
        peptides = peptides.reset_index()

        return peptides

    @staticmethod
    def process_pipeline() -> list:
        return []


class Axpep:
    @staticmethod
    def load_and_process_df(
        data_path: str, files_prefix: str = "train"
    ) -> pd.DataFrame:
        """
        :param data_path: path to a folder that contains files '*_ne.fasta' and '*_po.fasta' downloaded fromhttps://sourceforge.net/projects/axpep/
        :param files_prefix: either "train" or "test"
        """

        sequence_files = {
            os.path.join(data_path, f"{files_prefix}_po.fasta"): {"amp.label": 1},
            os.path.join(data_path, f"{files_prefix}_ne.fasta"): {"amp.label": 0},
        }

        peptides = pd.DataFrame(columns=["sequence", "amp.label"])
        for filename, label_dict in sequence_files.items():
            with open(filename) as handle:
                seqs = []
                for record in SeqIO.parse(handle, "fasta"):
                    seqs.append(str(record.seq))
                seqs = pd.DataFrame(seqs, columns=["sequence"])

                for label_name, label_value in label_dict.items():
                    seqs[label_name] = label_value
                peptides = pd.concat((peptides, seqs))

        # filter invalid sequences
        peptides = peptides[peptides.sequence.apply(_seq_is_valid)]

        # keep only len < 50
        peptides = peptides[peptides.sequence.apply(_seq_len_less_50)]

        # filter duplicates
        peptides = peptides.drop_duplicates(subset=["sequence"], keep="first")

        # reset index to reenumerate the samples
        peptides = peptides.reset_index()

        return peptides

    @staticmethod
    def process_pipeline() -> List[Tuple]:
        return [(OpSet(), dict(key="toxicity.label", value=-1))]


class Uniprot:
    @staticmethod
    def load_and_process_df(raw_data_path_list: dict) -> pd.DataFrame:
        """
        :param raw_data_path_list: list of files downloaded from uniprot that contains two columns (name and sequence) - uncompressed tsv format.
        Specifically we used it with:
        one file of reviewed peptides downloaded from https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue%2Clength%3A%5B1%20TO%20200%5D&query=%2A
        and second file of not reviewed peptides downloaded from https://www.uniprot.org/uniprotkb?facets=reviewed%3Afalse%2Clength%3A%5B1%20TO%20200%5D&query=%2A
        """
        peptides_df_list = []
        # load the raw data
        for raw_data_path in raw_data_path_list:
            peptides = pd.read_csv(raw_data_path, sep="\t", names=["name", "sequence"])
            peptides["data_path"] = raw_data_path
            peptides_df_list.append(peptides)

        peptides = pd.concat(peptides_df_list)

        # filter invalid sequences
        peptides = peptides[peptides.sequence.apply(_seq_is_valid)]

        # keep only len < 50
        peptides = peptides[peptides.sequence.apply(_seq_len_less_50)]

        # filter duplicates
        peptides = peptides.drop_duplicates(subset=["sequence"], keep="first")

        # reset index to reenumerate the samples
        peptides = peptides.reset_index()

        return peptides

    @staticmethod
    def process_pipeline() -> List[Tuple[OpBase, dict]]:
        return [
            (OpSet(), dict(key="amp.label", value=-1)),
            (OpSet(), dict(key="toxicity.label", value=-1)),
        ]


class PeptidesDatasets:
    """
    Mixing labeled datasets with unlabeled datasets
    Keeping valid peptides (see _seq_is_valid() with length <= 50
    * dbaasp
    * satpdb
    * toxin pred
    * uniprot

    main keys in sample_dict:
    * sequence - trimmed string representation of the chain
    * amp.label - 1 for amp, 0 for non amp, -1 for unlabeled
    * toxicity.label - 1 for toxic, 0 for non toxic, -1 for unlabeled
    """

    @staticmethod
    def load_and_process_df(
        dbaasp_raw_data_path: str,
        uniprot_raw_data_path_list: dict,
        toxin_pred_data_path: str,
        satpdb_data_path: str,
        axpep_data_path: str,
    ) -> pd.DataFrame:
        # Use OrderedDict since the order is important. Currently we drop duplicates and keep the first instance.
        # TODO: merge duplicates instead.
        ds_df = OrderedDict(
            [
                (
                    "dbaasp",
                    (
                        Dbaasp.load_and_process_df(dbaasp_raw_data_path)
                        if dbaasp_raw_data_path is not None
                        else None
                    ),
                ),
                (
                    "axpep",
                    (
                        Axpep.load_and_process_df(axpep_data_path)
                        if axpep_data_path is not None
                        else None
                    ),
                ),
                (
                    "satpdb",
                    (
                        Satpdb.load_and_process_df(satpdb_data_path)
                        if satpdb_data_path is not None
                        else None
                    ),
                ),
                (
                    "toxin_pred",
                    (
                        ToxinPred.load_and_process_df(toxin_pred_data_path)
                        if toxin_pred_data_path is not None
                        else None
                    ),
                ),
                (
                    "uniprot",
                    (
                        Uniprot.load_and_process_df(uniprot_raw_data_path_list)
                        if None not in uniprot_raw_data_path_list
                        else None
                    ),
                ),
            ]
        )

        # add source column and boolean column per dataset name
        peptides = pd.DataFrame()
        for ds_name, df in ds_df.items():
            if df is None:
                continue
            df["source"] = ds_name
            for ds_name_i in ds_df:
                df[ds_name_i] = ds_name_i == ds_name

            peptides = pd.concat((peptides, df))
        # filter duplicates
        peptides = peptides.drop_duplicates(subset=["sequence"], keep="first")
        # reset index to reenumerate the samples
        peptides = peptides.reset_index()

        return peptides

    @staticmethod
    def dataset(
        dbaasp_raw_data_path: str,
        uniprot_raw_data_path_reviewed: str,
        uniprot_raw_data_path_not_reviewed: str,
        toxin_pred_data_path: str,
        satpdb_data_path: str,
        axpep_data_path: str,
        num_folds: int,
        split_filename: str,
        seed: int,
        reset_split: bool,
        train_folds: Sequence[int],
        validation_folds: Sequence[int],
        test_folds: Sequence[int],
    ) -> DatasetDefault:
        """
        :param dbaasp_raw_data_path: path to peptides-complete.json as downloaded from: https://dbaasp.org/download-dataset?source=peptides
        :param uniprot_raw_data_path_reviewed: only two columns file of reviewed peptides downloaded from uniprot - uncompressed tsv format: https://www.uniprot.org/uniprotkb?facets=reviewed%3Atrue%2Clength%3A%5B1%20TO%20200%5D&query=%2A
        :param uniprot_raw_data_path_not_reviewed:only two columns file of unreviewed peptides downloaded from uniprot - uncompressed tsv format: https://www.uniprot.org/uniprotkb?facets=reviewed%3Afalse%2Clength%3A%5B1%20TO%20200%5D&query=%2A
        :param toxin_pred_data_path:path to a folder that contains multiple files downloaded directly from "https://webs.iiitd.edu.in/raghava/toxinpred/dataset.php"
        :param satpdb_data_path: path to a folder that contains files 'antimicrobial.fasta' and 'toxic.fasta' downloaded from https://webs.iiitd.edu.in/raghava/satpdb/down.php
        :param axpep_data_path:  path to a folder that contains files '*_ne.fasta' and '*_po.fasta' downloaded fromhttps://sourceforge.net/projects/axpep/
        """

        df = PeptidesDatasets.load_and_process_df(
            dbaasp_raw_data_path,
            [uniprot_raw_data_path_reviewed, uniprot_raw_data_path_not_reviewed],
            toxin_pred_data_path,
            satpdb_data_path,
            axpep_data_path,
        )
        dynamic_pipeline = [
            (OpReadDataframe(df, key_column=None), {}),
            (
                OpCond(PipelineDefault("uniprot process", Uniprot.process_pipeline())),
                dict(condition="uniprot"),
            ),
            (
                OpCond(PipelineDefault("dbaasp process", Dbaasp.process_pipeline())),
                dict(condition="dbaasp"),
            ),
            (
                OpCond(
                    PipelineDefault("toxin_pred process", ToxinPred.process_pipeline())
                ),
                dict(condition="toxin_pred"),
            ),
            (
                OpCond(PipelineDefault("satpdb process", Satpdb.process_pipeline())),
                dict(condition="satpdb"),
            ),
            (
                OpCond(PipelineDefault("axpep process", Axpep.process_pipeline())),
                dict(condition="axpep"),
            ),
        ]
        dynamic_pipeline = PipelineDefault("peptides dataset", dynamic_pipeline)

        dataset_all = DatasetDefault(len(df), dynamic_pipeline)
        dataset_all.create()

        folds = dataset_balanced_division_to_folds(
            dataset=dataset_all,
            output_split_filename=split_filename,
            keys_to_balance=[],
            nfolds=num_folds,
            seed=seed,
            reset_split=reset_split,
        )

        train_sample_ids = []
        for fold in train_folds:
            train_sample_ids += folds[fold]
        dataset_train = DatasetDefault(train_sample_ids, dynamic_pipeline)
        dataset_train.create()

        validation_sample_ids = []
        for fold in validation_folds:
            validation_sample_ids += folds[fold]
        dataset_validation = DatasetDefault(validation_sample_ids, dynamic_pipeline)
        dataset_validation.create()

        test_sample_ids = []
        for fold in test_folds:
            test_sample_ids += folds[fold]
        dataset_test = DatasetDefault(test_sample_ids, dynamic_pipeline)
        dataset_test.create()

        return dataset_train, dataset_validation, dataset_test


if __name__ == "__main__":
    """used just to test and to play with the data! No need to run it"""
    from fuse.data.utils.export import ExportDataset

    ds_train, ds_valid, ds_test = PeptidesDatasets.dataset(
        os.environ["DBAASP_DATA_PATH"],
        os.environ["UNIPROT_PEPTIDE_REVIEWED_DATA_PATH"],
        os.environ["UNIPROT_PEPTIDE_NOT_REVIEWED_DATA_PATH"],
        os.environ["TOXIN_PRED_DATA_PATH"],
        os.environ["SATPDB_DATA_PATH"],
        os.environ["AXPEP_DATA_PATH"],
        5,
        None,
        2580,
        True,
        [0, 1, 2],
        [3],
        [4],
    )
    df = ExportDataset.export_to_dataframe(
        ds_train, ["toxicity.label", "amp.label"], workers=0
    )
    print(f"Train toxicity stat:\n {df['toxicity.label'].value_counts()}")
    print(f"Train amp stat:\n {df['amp.label'].value_counts()}")
    df = ExportDataset.export_to_dataframe(ds_valid, ["toxicity.label", "amp.label"])
    print(f"Valid toxicity stat:\n {df['toxicity.label'].value_counts()}")
    print(f"Valid amp stat:\n {df['amp.label'].value_counts()}")
    df = ExportDataset.export_to_dataframe(ds_test, ["toxicity.label", "amp.label"])
    print(f"Test toxicity stat:\n {df['toxicity.label'].value_counts()}")
    print(f"Test amp stat:\n {df['amp.label'].value_counts()}")

    ds_train, ds_valid, ds_test = Dbaasp.dataset(
        os.environ["DBAASP_DATA_PATH"], 5, None, 1234, True, [0, 1, 2], [3], [4]
    )
    df = ExportDataset.export_to_dataframe(ds_train, ["toxicity.label"])
    print(f"Train stat:\n {df['toxicity.label'].value_counts()}")
    df = ExportDataset.export_to_dataframe(ds_valid, ["toxicity.label"])
    print(f"Valid stat:\n {df['toxicity.label'].value_counts()}")
    df = ExportDataset.export_to_dataframe(ds_test, ["toxicity.label"])
    print(f"Test stat:\n {df['toxicity.label'].value_counts()}")
