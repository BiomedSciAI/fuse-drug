import pytorch_lightning as pl
from typing import Optional, List
from fusedrug.data.molecule.ops import (
    SmilesRandomizeAtomOrder,
    SmilesToRDKitMol,
    RDKitMolToSmiles,
)
import os
from fuse.data import OpBase
from fusedrug.data.protein.ops import ProteinRandomFlipOrder, OpToUpperCase
from fusedrug.data.molecule.tokenizer.pretrained import (
    get_path as get_molecule_pretrained_tokenizer_path,
)
from fusedrug.data.protein.tokenizer.pretrained import (
    get_path as get_protein_pretrained_tokenizer_path,
)
from fusedrug.data.tokenizer.ops import (
    FastTokenizer,
    Op_pytoda_SMILESTokenizer,
    Op_pytoda_ProteinTokenizer,
)
from fuse.data import DatasetDefault, PipelineDefault, OpToTensor, OpKeepKeypaths
from fusedrug.data.interaction.drug_target.loaders import DrugTargetAffinityLoader
from torch.utils.data import DataLoader
import numpy as np


class AffinityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ligand_padding_length: int,
        receptor_padding_length: int,
        molecules_smi: str = None,
        proteins_smi: str = None,
        train_dataset_path: str = None,
        val_dataset_path: str = None,
        test_dataset_path: str = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 512,
        num_workers: int = 0,
        train_shuffle: bool = True,
        train_augment_molecule_shuffle_atoms: bool = False,
        train_augment_protein_flip: bool = False,
        pytoda_wrapped_tokenizer: bool = False,
        pytoda_ligand_tokenizer_json: Optional[str] = None,
        pytoda_target_tokenizer_amino_acid_dict: Optional[str] = None,
        pairs_table_ligand_column: str = "ligand_name",
        pairs_table_sequence_column: str = "uniprot_accession",
        pairs_table_affinity_column: str = "pIC50",
        partial_sample_ids: Optional[List[int]] = None,
    ):
        """
        a ligand vs. target affinity prediction data module

        Args:
            ligand_padding_length: common length of ligand token sequences
            receptor_padding_length: common length of target token sequences
            molecules_smi: path to smi file containing molecules (will be indexed into from the affinity datasets)
            protein_smi: path to smi file containing targets (will be indexed into from the affinity datasets)
            train_dataset_path: path to the training affinity dataset file
            val_dataset_path: path to the validation affinity dataset file
            test_dataset_path: path to the test affinity dataset file
            train_batch_size: self explanatory
            eval_batch_size: self explanatory
            num_workers: number of workers used in dataloader, pass 0 for easy debugging
            train_shuffle: whether to reshuffle the data in every epoch
            train_augment_molecule_shuffle_atoms: randomize the order of a smiles string representation of a molecule (while preserving the molecule structure)
            train_augment_protein_flip: randomize the order of amino sequences in a protein during training
            pytoda_wrapped_tokenizer: if true, PyToda tokenizer is used, otherwise HuggingFace based FastTokenizer
            pytoda_ligand_tokenizer_json: filepath to tokenizer vocab json or directory
            pytoda_target_tokenizer_amino_acid_dict: PyToda tokenization regime for amino acid
                sequence. 'iupac', 'unirep' or 'human-kinase-alignment'.
            pairs_table_ligand_column: name of the ligand column in the smi file
            pairs_table_sequence_column: name of target sequence column in the smi file
            pairs_table_affinity_column: name of affinity measure column
            partial_sample_ids: optional - use partial subset of sample ids for the train, val and test datasets. Useful for testing purposes.
                                If None, uses all sample ids.

        """

        self.ligand_padding_length = ligand_padding_length
        self.receptor_padding_length = receptor_padding_length

        self.molecules_smi = molecules_smi
        self.proteins_smi = proteins_smi
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle

        self.train_augment_molecule_shuffle_atoms = train_augment_molecule_shuffle_atoms
        self.train_augment_protein_flip = train_augment_protein_flip

        self.pytoda_wrapped_tokenizer = pytoda_wrapped_tokenizer
        self.pytoda_ligand_tokenizer_json = pytoda_ligand_tokenizer_json
        self.pytoda_target_tokenizer_amino_acid_dict = (
            pytoda_target_tokenizer_amino_acid_dict
        )

        self.pairs_table_ligand_column = pairs_table_ligand_column
        self.pairs_table_sequence_column = pairs_table_sequence_column
        self.pairs_table_affinity_column = pairs_table_affinity_column

        self.partial_sample_ids = partial_sample_ids

        if self.pytoda_wrapped_tokenizer:
            if self.pytoda_ligand_tokenizer_json is None:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_ligand_tokenizer_json + pytoda_target_tokenizer_amino_acid_dict) or neither"
                )

        if self.pytoda_ligand_tokenizer_json is not None:
            if not self.pytoda_wrapped_tokenizer:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_ligand_tokenizer_json + pytoda_target_tokenizer_amino_acid_dict) or neither"
                )

        if self.pytoda_wrapped_tokenizer:
            if self.pytoda_target_tokenizer_amino_acid_dict is None:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_ligand_tokenizer_json + pytoda_target_tokenizer_amino_acid_dict) or neither"
                )

        if self.pytoda_target_tokenizer_amino_acid_dict is not None:
            if not self.pytoda_wrapped_tokenizer:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_ligand_tokenizer_json + pytoda_target_tokenizer_amino_acid_dict) or neither"
                )

        super().__init__()

        self._shared_ligands_indexed_text_table_kwargs = dict(
            separator="\t",
            first_row_is_columns_names=False,
            columns_names=["molecule_sequence", "molecule_id"],
            id_column_name="molecule_id",
            allow_access_by_id=True,
        )

        self._shared_proteins_indexed_text_table_kwargs = dict(
            separator="\t",
            first_row_is_columns_names=False,
            columns_names=["protein_sequence", "protein_id"],
            id_column_name="protein_id",
            allow_access_by_id=True,
        )

        self._shared_affinity_dataset_loader_kwargs = dict(
            ligand_sequence_column_name="molecule_sequence",
            affinity_pairs_csv_ligand_id_column_name=self.pairs_table_ligand_column,  # 'ligand_name',
            protein_sequence_column_name="protein_sequence",
            affinity_pairs_csv_protein_id_column_name=self.pairs_table_sequence_column,  # 'uniprot_accession',
            affinity_pairs_csv_affinity_value_column_name=self.pairs_table_affinity_column,  # 'pIC50',
        )

    def _create_pipeline_desc(
        self,
        is_training: bool,
        drug_target_affinity_loader_op: DrugTargetAffinityLoader,
    ) -> List[OpBase]:
        """
        Note: in the current implementation, augmentation is activated only if is_training==False
        """

        pipeline_desc = [
            # load affinity sample
            (
                drug_target_affinity_loader_op,
                dict(
                    key_out_ligand="data.input.ligand",
                    key_out_protein="data.input.protein",
                    key_out_ground_truth_affinity="data.gt.affinity_val",
                ),
            ),
        ]

        if is_training and self.train_augment_molecule_shuffle_atoms:
            pipeline_desc += [
                (
                    SmilesToRDKitMol(),
                    dict(key_in="data.input.ligand", key_out="data.input.ligand"),
                ),
                (SmilesRandomizeAtomOrder(), dict(key="data.input.ligand")),
                (
                    RDKitMolToSmiles(),
                    dict(key_in="data.input.ligand", key_out="data.input.ligand"),
                ),
            ]

        if is_training and self.train_augment_protein_flip:
            pipeline_desc += [
                (
                    ProteinRandomFlipOrder(),
                    dict(key_in="data.input.protein", key_out="data.input.protein"),
                ),
            ]

        pipeline_desc += [
            (
                OpToUpperCase(),
                dict(key_in="data.input.protein", key_out="data.input.protein"),
            ),
        ]

        _molecule_tokenizer_path = os.path.join(
            get_molecule_pretrained_tokenizer_path(),
            "bpe_tokenizer_trained_on_chembl_zinc_with_aug_4272372_samples_balanced_1_1.json",
        )
        molecule_pad_id = 3

        _protein_tokenizer_path = os.path.join(
            get_protein_pretrained_tokenizer_path(), "simple_protein_tokenizer.json"
        )
        protein_pad_id = 0

        if self.pytoda_wrapped_tokenizer:
            ligand_tokenizer_op = Op_pytoda_SMILESTokenizer(
                dict(
                    vocab_file=self.pytoda_ligand_tokenizer_json,
                    padding_length=self.ligand_padding_length,
                    randomize=None,
                    add_start_and_stop=True,
                    padding=True,
                    augment=False,
                    canonical=False,
                    kekulize=False,
                    all_bonds_explicit=False,
                    all_hs_explicit=False,
                    remove_bonddir=False,
                    remove_chirality=False,
                    selfies=False,
                    sanitize=False,
                )
            )
        else:
            ligand_tokenizer_op = FastTokenizer(
                _molecule_tokenizer_path,
                pad_length=self.ligand_padding_length,
                pad_id=molecule_pad_id,
            )

        if self.pytoda_wrapped_tokenizer:
            protein_tokenizer_op = Op_pytoda_ProteinTokenizer(
                amino_acid_dict=self.pytoda_target_tokenizer_amino_acid_dict,
                add_start_and_stop=True,
                padding=True,
                padding_length=self.receptor_padding_length,
            )
        else:
            protein_tokenizer_op = FastTokenizer(
                _protein_tokenizer_path,
                pad_length=self.receptor_padding_length,
                pad_id=protein_pad_id,
            )

        pipeline_desc += [
            # molecule
            (
                ligand_tokenizer_op,
                dict(
                    key_in="data.input.ligand",
                    key_out_tokens_ids="data.input.tokenized_ligand",
                ),
            ),
            (OpToTensor(), dict(key="data.input.tokenized_ligand")),
            # proteinligand_
            (
                protein_tokenizer_op,
                dict(
                    key_in="data.input.protein",
                    key_out_tokens_ids="data.input.tokenized_protein",
                ),
            ),
            (OpToTensor(), dict(key="data.input.tokenized_protein")),
            # affinity gt val
            (OpToTensor(), dict(key="data.gt.affinity_val")),
            # keep only the keys we want, which helps in two ways:
            # 1. We make sure that multiprocessing doesn't need to transfer anything beyond what's needed (which is sometimes very slow, if certain elements get pickled and transfered)
            # 2. pytorch's default_collate works as is and there's no need to provide a custom collate
            (
                OpKeepKeypaths(),
                dict(
                    keep_keypaths=[
                        "data.input.tokenized_ligand",
                        "data.input.tokenized_protein",
                        "data.gt.affinity_val",
                    ]
                ),
            ),
        ]

        return pipeline_desc

    def train_dataloader(self) -> DataLoader:

        affinity_loader_op = DrugTargetAffinityLoader(
            ligands_smi=self.molecules_smi,
            proteins_smi=self.proteins_smi,
            affinity_pairs_csv_path=self.train_dataset_path,
            ligands_indexed_text_table_kwargs=self._shared_ligands_indexed_text_table_kwargs,
            proteins_indexed_text_table_kwargs=self._shared_proteins_indexed_text_table_kwargs,
            **self._shared_affinity_dataset_loader_kwargs,
        )

        pipeline_desc = self._create_pipeline_desc(
            is_training=True, drug_target_affinity_loader_op=affinity_loader_op
        )

        all_sample_ids = np.arange(len(affinity_loader_op.drug_target_affinity_dataset))

        train_dataset = DatasetDefault(
            (
                all_sample_ids
                if self.partial_sample_ids is None
                else self.partial_sample_ids
            ),
            static_pipeline=None,
            dynamic_pipeline=PipelineDefault(
                name="train_pipeline_affinity_predictor", ops_and_kwargs=pipeline_desc
            ),
        )
        train_dataset.create()

        dl = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return dl

    def val_dataloader(self) -> DataLoader:

        affinity_loader_op = DrugTargetAffinityLoader(
            ligands_smi=self.molecules_smi,
            proteins_smi=self.proteins_smi,
            affinity_pairs_csv_path=self.val_dataset_path,
            ligands_indexed_text_table_kwargs=self._shared_ligands_indexed_text_table_kwargs,
            proteins_indexed_text_table_kwargs=self._shared_proteins_indexed_text_table_kwargs,
            **self._shared_affinity_dataset_loader_kwargs,
        )

        pipeline_desc = self._create_pipeline_desc(
            is_training=False, drug_target_affinity_loader_op=affinity_loader_op
        )

        all_sample_ids = np.arange(len(affinity_loader_op.drug_target_affinity_dataset))

        val_dataset = DatasetDefault(
            (
                all_sample_ids
                if self.partial_sample_ids is None
                else self.partial_sample_ids
            ),
            static_pipeline=None,
            dynamic_pipeline=PipelineDefault(
                name="val_pipeline_affinity_predictor", ops_and_kwargs=pipeline_desc
            ),
        )
        val_dataset.create()

        dl = DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dl

    def test_dataloader(self) -> DataLoader:
        affinity_loader_op = DrugTargetAffinityLoader(
            ligands_smi=self.molecules_smi,
            proteins_smi=self.proteins_smi,
            affinity_pairs_csv_path=self.test_dataset_path,
            ligands_indexed_text_table_kwargs=self._shared_ligands_indexed_text_table_kwargs,
            proteins_indexed_text_table_kwargs=self._shared_proteins_indexed_text_table_kwargs,
            **self._shared_affinity_dataset_loader_kwargs,
        )

        pipeline_desc = self._create_pipeline_desc(
            is_training=False, drug_target_affinity_loader_op=affinity_loader_op
        )

        all_sample_ids = np.arange(len(affinity_loader_op.drug_target_affinity_dataset))

        test_dataset = DatasetDefault(
            (
                all_sample_ids
                if self.partial_sample_ids is None
                else self.partial_sample_ids
            ),
            static_pipeline=None,
            dynamic_pipeline=PipelineDefault(
                name="test_pipeline_affinity_predictor", ops_and_kwargs=pipeline_desc
            ),
        )
        test_dataset.create()

        dl = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dl
