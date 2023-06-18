from typing import List, Optional, Union, Tuple
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.data.interaction.drug_target.datasets.pytoda_style_target_affinity_dataset import (
    PytodaStyleDrugTargetAffinityDataset,
)
from fusedrug.utils.file_formats import IndexedTextTable
import numpy as np
import os


class DrugTargetAffinityLoader(OpBase):
    def __init__(
        self,
        ligands_smi: Union[str, IndexedTextTable],
        ligand_sequence_column_name: str,
        #
        proteins_smi: Union[str, IndexedTextTable],
        protein_sequence_column_name: str,
        #
        affinity_pairs_csv_path: str,
        affinity_pairs_csv_ligand_id_column_name: str,
        affinity_pairs_csv_protein_id_column_name: str,
        affinity_pairs_csv_affinity_value_column_name: str,
        #
        ligands_indexed_text_table_kwargs: Optional[dict] = None,
        proteins_indexed_text_table_kwargs: Optional[dict] = None,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.drug_target_affinity_dataset = PytodaStyleDrugTargetAffinityDataset(
            ligands_smi=ligands_smi,
            ligand_sequence_column_name=ligand_sequence_column_name,
            ligands_indexed_text_table_kwargs=ligands_indexed_text_table_kwargs,
            #
            proteins_smi=proteins_smi,
            protein_sequence_column_name=protein_sequence_column_name,
            proteins_indexed_text_table_kwargs=proteins_indexed_text_table_kwargs,
            #
            affinity_pairs_csv_path=affinity_pairs_csv_path,
            affinity_pairs_csv_ligand_id_column_name=affinity_pairs_csv_ligand_id_column_name,
            affinity_pairs_csv_protein_id_column_name=affinity_pairs_csv_protein_id_column_name,
            affinity_pairs_csv_affinity_value_column_name=affinity_pairs_csv_affinity_value_column_name,
        )

    def __call__(
        self,
        sample_dict: NDict,
        key_out_ligand: str = "data.input.ligand",
        key_out_protein: str = "data.input.protein",
        key_out_ground_truth_affinity: str = "data.gt.affinity_val",
    ) -> NDict:
        """ """
        sid = get_sample_id(sample_dict)
        if isinstance(sid, str) or not np.isscalar(sid):
            raise Exception(f"expected an int sample_id but got {type(sid)}")
        sid = int(sid)
        entry = self.drug_target_affinity_dataset[sid]
        sample_dict[key_out_ground_truth_affinity] = [entry["affinity_val"]]
        sample_dict[key_out_ligand] = entry["ligand_str"]
        sample_dict[key_out_protein] = entry["protein_str"]

        return sample_dict


if __name__ == "__main__":
    from fuse.data import create_initial_sample
    from fuse.data.pipelines.pipeline_default import PipelineDefault
    from fuse.data import OpKeepKeypaths
    from fusedrug.data.molecule.ops import SmilesRandomizeAtomOrder, SmilesToRDKitMol, RDKitMolToSmiles
    from fusedrug.data.protein.ops import ProteinRandomFlipOrder
    from fusedrug.data.tokenizer.ops import FastTokenizer
    from fusedrug.data.molecule.tokenizer.pretrained import get_path as get_molecule_pretrained_tokenizer_path
    from fusedrug.data.protein.tokenizer.pretrained import get_path as get_protein_pretrained_tokenizer_path
    from torch.utils.data import default_collate
    from fuse.data.ops.ops_cast import OpToTensor

    # TODO: provide example files or remove this example use:
    smiles_path = ""
    proteins_path = ""
    affinity_set_path = ""

    ligands_table = IndexedTextTable(
        smiles_path,
        seperator="\t",
        first_row_is_columns_names=False,
        columns_names=["molecule_sequence", "molecule_id"],
        id_column_name="molecule_id",
        allow_access_by_id=True,
    )

    proteins_table = IndexedTextTable(
        proteins_path,
        seperator="\t",
        first_row_is_columns_names=False,
        columns_names=["protein_sequence", "protein_id"],
        id_column_name="protein_id",
        allow_access_by_id=True,
    )

    affinity_loader = DrugTargetAffinityLoader(
        ligands_smi=ligands_table,
        ligand_sequence_column_name="molecule_sequence",
        proteins_smi=proteins_table,
        protein_sequence_column_name="protein_sequence",
        affinity_pairs_csv_path=affinity_set_path,
        affinity_pairs_csv_ligand_id_column_name="ligand_name",
        affinity_pairs_csv_protein_id_column_name="uniprot_accession",
        affinity_pairs_csv_affinity_value_column_name="pIC50",
    )

    _molecule_tokenizer_path = os.path.join(
        get_molecule_pretrained_tokenizer_path(),
        "bpe_tokenizer_trained_on_chembl_zinc_with_aug_4272372_samples_balanced_1_1.json",
    )
    molecule_pad_id = 3

    _protein_tokenizer_path = os.path.join(get_protein_pretrained_tokenizer_path(), "simple_protein_tokenizer.json")
    protein_pad_id = 0

    pipeline_desc = [
        # load affinity sample
        (
            affinity_loader,
            dict(
                key_out_ligand="data.input.ligand",
                key_out_protein="data.input.protein",
                key_out_ground_truth_affinity="data.gt.affinity_val",
            ),
        ),
        # ligand related ops
        (SmilesToRDKitMol(), dict(key_in="data.input.ligand", key_out="data.input.ligand")),
        (SmilesRandomizeAtomOrder(), dict(key="data.input.ligand")),
        (RDKitMolToSmiles(), dict(key_in="data.input.ligand", key_out="data.input.ligand")),
        (
            FastTokenizer(_molecule_tokenizer_path, pad_length=256, pad_id=molecule_pad_id),
            dict(key_in="data.input.ligand", key_out_tokens_ids="data.input.tokenized_ligand"),
        ),
        (OpToTensor(), dict(key="data.input.tokenized_ligand")),
        # protein related ops
        (ProteinRandomFlipOrder(), dict(key_in="data.input.protein", key_out="data.input.protein")),
        (
            FastTokenizer(_protein_tokenizer_path, pad_length=3000, pad_id=molecule_pad_id),
            dict(key_in="data.input.protein", key_out_tokens_ids="data.input.tokenized_protein"),
        ),
        (OpToTensor(), dict(key="data.input.tokenized_protein")),
        # affinity val (pIC50)
        (OpToTensor(), dict(key="data.gt.affinity_val")),
        # keep only the keys we want, to make sure that multiprocessing doesn't need to transfer anything else
        (
            OpKeepKeypaths(),
            dict(keep_keypaths=["data.input.tokenized_ligand", "data.input.tokenized_protein", "data.gt.affinity_val"]),
        ),
    ]

    pipeline = PipelineDefault("test_drug_target_affinity_pipeline", pipeline_desc)

    def get_sample(sid: int) -> NDict:
        sample = create_initial_sample(sid)
        processed_sample = pipeline(sample, "")
        return processed_sample

    s1 = get_sample(100)
    s2 = get_sample(200)
    s3 = get_sample(300)
    s4 = get_sample(400)

    def _split_sample(sample: NDict, default_collate_keys: Optional[List[str]] = None) -> Tuple[NDict, NDict]:
        assert isinstance(sample, NDict)
        if default_collate_keys is None:
            return sample, []

        to_default_collate = NDict()
        append_in_list = NDict()

        for k in sample.flatten().keys():
            if k in default_collate_keys:
                to_default_collate[k] = sample[k]
            else:
                append_in_list[k] = sample[k]

        return to_default_collate, append_in_list

    def my_collate(samples: List[NDict], default_collate_keys: Optional[List[str]] = None) -> Union[list, dict]:
        for_default_collate_minibatch = []
        # for_just_list_minibatch = []

        for s in samples:
            to_default_collate_sample, append_in_list_sample = _split_sample(
                s, default_collate_keys=default_collate_keys
            )
            for_default_collate_minibatch.append(to_default_collate_sample)
            # for_just_list_minibatch.append(append_in_list_sample)

        collated_mb = default_collate(for_default_collate_minibatch)

        return collated_mb  # , for_just_list_minibatch

    collated_mb = my_collate(
        [s1, s2, s3, s4],
        default_collate_keys=["data.input.tokenized_ligand", "data.input.tokenized_protein", "data.gt.affinity_val",],
    )
