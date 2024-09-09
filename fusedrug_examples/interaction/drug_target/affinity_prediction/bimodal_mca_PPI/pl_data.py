import os

# import tensorflow as tf #uncomment if getting pl error

import pytorch_lightning as pl
import numpy as np
from fuse.data import (
    DatasetDefault,
    PipelineDefault,
    OpToTensor,
    OpKeepKeypaths,
    OpBase,
)

from fusedrug.data.protein.ops import (
    ProteinRandomFlipOrder,
    ProteinIntroduceNoise,
    ProteinFlipIndividualActiveSiteSubSequences,
    ProteinIntroduceActiveSiteBasedNoise,
    OpToUpperCase,
    OpKeepOnlyUpperCase,
)
from fusedrug.data.tokenizer.ops.tokenizer_op import TokenizerOp
from fusedrug.data.tokenizer.ops.pytoda_tokenizer import (
    Op_pytoda_SMILESTokenizer,
    Op_pytoda_ProteinTokenizer,
)
from fusedrug.data.interaction.drug_target.loaders import (
    DrugTargetAffinityLoader as PeptideTargetAffinityLoader,
)
from torch.utils.data import DataLoader

from fusedrug.data.molecule.tokenizer.pretrained import (
    get_path as get_molecule_pretrained_tokenizer_path,
)
from fusedrug.data.protein.tokenizer.pretrained import (
    get_path as get_protein_pretrained_tokenizer_path,
)
from typing import Optional
from fuse.utils import NDict
from fusedrug.utils.file_formats import IndexedTextTable
import colorama
from typing import Dict, Callable

colorama.init(autoreset=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OpLoadActiveSiteAlignmentInfo(OpBase):
    """
    TODO: Move to fuse-drug ops
    """

    def __init__(self, kinase_alignment_smi: str, **kwargs: Dict) -> None:
        """_summary_

        Args:
            kinase_alignment_smi (_type_): _description_
        """

        super().__init__(**kwargs)
        self.kinase_alignment_smi_name = kinase_alignment_smi
        self.kinase_alignment_smi = IndexedTextTable(
            self.kinase_alignment_smi_name,
            first_row_is_columns_names=True,
            id_column_idx=1,
            columns_num_expectation=3,
            allow_access_by_id=True,
            # num_workers=0, #DEBUGGING! remove this
        )

    def __call__(
        self,
        sample_dict: NDict,
        op_id: Optional[str],
        key_in: str = "data.input.protein_str",
        key_out: str = "data.input.protein_str",
    ) -> NDict:
        """
        params
            key_in:str - expected to contain only the active site, in high case
            key_out:str - will contain the entire sequence, high case for amino acids inside the active site, low case for amino acids outside it
        """
        data = sample_dict[key_in]
        assert isinstance(data, str)

        _, data = self.kinase_alignment_smi[data]
        aligned_seq = data.aligned_protein_seq

        sample_dict[key_out] = aligned_seq

        return sample_dict


class AffinityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ligand_padding_length: int,
        receptor_padding_length: int,
        protein_representation_type: str,
        peptide_representation_type: str,
        peptides_smi: str = None,
        proteins_smi: str = None,
        train_dataset_path: str = None,
        val_dataset_path: str = None,
        test_dataset_path: str = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 512,
        num_workers: int = 0,
        train_shuffle: bool = True,
        train_augment_peptide_shuffle_atoms: bool = False,
        train_augment_protein_flip: bool = False,
        protein_augment_full_sequence_noise: bool = False,
        protein_augment_full_sequence_noise_p: float = 0.1,
        active_site_alignment_info_smi: Optional[str] = None,
        protein_augment_by_reverting_individual_active_site_sub_sequences: bool = False,
        protein_augment_by_reverting_individual_active_site_sub_sequences_p: float = 0.5,
        protein_augment_by_introducing_noise_to_non_active_site_residues: bool = False,
        protein_augment_by_introducing_noise_to_non_active_site_residues_p_inside_active_site: float = 0.01,
        protein_augment_by_introducing_noise_to_non_active_site_residues_p_outside_active_site: float = 0.1,
        pytoda_wrapped_tokenizer: bool = False,
        pytoda_SMILES_tokenizer_json: Optional[str] = None,
        pytoda_target_target_tokenizer_amino_acid_dict: Optional[str] = None,
        pairs_table_ligand_column: str = "ligand_name",
        pairs_table_sequence_column: str = "uniprot_accession",
        pairs_table_affinity_column: str = "pIC50",
        protein_final_keep_only_uppercase: bool = False,
        **kwargs: Dict,
    ) -> None:
        """a ligand vs. target affinity prediction data module

        Args:
            ligand_padding_length (int): _description_
            receptor_padding_length (int): _description_
            protein_representation_type (str): How proteins are represented ('AA' - amino acid sequence, 'SMILES' - SMILES string)
            peptide_representation_type (str): How peptides are represented ('AA' - amino acid sequence, 'SMILES' - SMILES string)
            peptides_smi (str, optional): _description_. Defaults to None.
            proteins_smi (str, optional): _description_. Defaults to None.
            train_dataset_path (str, optional): _description_. Defaults to None.
            val_dataset_path (str, optional): _description_. Defaults to None.
            test_dataset_path (str, optional): _description_. Defaults to None.
            train_batch_size (int, optional): _description_. Defaults to 128.
            eval_batch_size (int, optional): _description_. Defaults to 512.
            num_workers (int, optional): number of workers used in dataloader, pass 0 for easy debugging. Defaults to 0.
            train_shuffle (bool, optional): _description_. Defaults to True.
            train_augment_peptide_shuffle_atoms (bool, optional): _description_. Defaults to False.
            train_augment_protein_flip (bool, optional): _description_. Defaults to False.
            protein_augment_full_sequence_noise (bool, optional): _description_. Defaults to False.
            protein_augment_full_sequence_noise_p (float, optional): _description_. Defaults to 0.1.
            active_site_alignment_info_smi (Optional[str], optional): _description_. Defaults to None.
            protein_augment_by_reverting_individual_active_site_sub_sequences (bool, optional): _description_. Defaults to False.
            protein_augment_by_reverting_individual_active_site_sub_sequences_p (float, optional): _description_. Defaults to 0.5.
            protein_augment_by_introducing_noise_to_non_active_site_residues (bool, optional): _description_. Defaults to False.
            protein_augment_by_introducing_noise_to_non_active_site_residues_p_inside_active_site (float, optional): _description_. Defaults to 0.01.
            protein_augment_by_introducing_noise_to_non_active_site_residues_p_outside_active_site (float, optional): _description_. Defaults to 0.1.
            pytoda_wrapped_tokenizer (bool, optional): _description_. Defaults to False.
            pytoda_SMILES_tokenizer_json (Optional[str], optional): _description_. Defaults to None.
            pytoda_target_target_tokenizer_amino_acid_dict (Optional[str], optional): _description_. Defaults to None.
            pairs_table_ligand_column (str, optional): _description_. Defaults to 'ligand_name'.
            pairs_table_sequence_column (str, optional): _description_. Defaults to 'uniprot_accession'.
            pairs_table_affinity_column (str, optional): _description_. Defaults to 'pIC50'.
            protein_final_keep_only_uppercase (bool, optional): _description_. Defaults to False.

        Raises:
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_
        """

        """

        Args:
            peptide_language_model: SMILESTokenizer instance
            protein_language_model: ProteinLanguage instance
            peptides_smi: path to smi file containing peptides (will be indexed into from the affinity datasets)
            protein_smi: path to smi file containing targets (will be indexed into from the affinity datasets)
            train_dataset_path: path to the training affinity dataset file
            val_dataset_path: path to the validation affinity dataset file
            test_dataset_path: path to the test affinity dataset file
            train_augment_peptide_shuffle_atoms:
            train_augment_protein_flip:
            eval_augment_peptide:
            eval_augment_protein:
            train_batch_size:
            eval_batch_size:
            num_workers: number of workers used in dataloader, pass 0 for easy debugging
            train_shuffle:

        """
        # assert protein_representation_type == peptide_representation_type, "For now proteins and peptides must be represented in the same way, either as AA sequence or SMILES string"
        self.ligand_padding_length = ligand_padding_length
        self.receptor_padding_length = receptor_padding_length
        self.protein_representation_type = protein_representation_type
        self.peptide_representation_type = protein_representation_type

        self.peptides_smi = peptides_smi
        self.proteins_smi = proteins_smi
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle

        self.train_augment_peptide_shuffle_atoms = train_augment_peptide_shuffle_atoms
        self.train_augment_protein_flip = train_augment_protein_flip

        self.protein_augment_full_sequence_noise = protein_augment_full_sequence_noise
        self.protein_augment_full_sequence_noise_p = (
            protein_augment_full_sequence_noise_p
        )

        self.protein_augment_by_reverting_individual_active_site_sub_sequences = (
            protein_augment_by_reverting_individual_active_site_sub_sequences
        )
        self.protein_augment_by_reverting_individual_active_site_sub_sequences_p = (
            protein_augment_by_reverting_individual_active_site_sub_sequences_p
        )
        self.protein_augment_by_introducing_noise_to_non_active_site_residues = (
            protein_augment_by_introducing_noise_to_non_active_site_residues
        )
        self.protein_augment_by_introducing_noise_to_non_active_site_residues_p_inside_active_site = protein_augment_by_introducing_noise_to_non_active_site_residues_p_inside_active_site
        self.protein_augment_by_introducing_noise_to_non_active_site_residues_p_outside_active_site = protein_augment_by_introducing_noise_to_non_active_site_residues_p_outside_active_site

        self.active_site_alignment_info_smi = active_site_alignment_info_smi

        self.pytoda_wrapped_tokenizer = pytoda_wrapped_tokenizer
        self.pytoda_SMILES_tokenizer_json = pytoda_SMILES_tokenizer_json
        self.pytoda_target_target_tokenizer_amino_acid_dict = (
            pytoda_target_target_tokenizer_amino_acid_dict
        )

        self.pairs_table_ligand_column = pairs_table_ligand_column
        self.pairs_table_sequence_column = pairs_table_sequence_column
        self.pairs_table_affinity_column = pairs_table_affinity_column

        self.protein_final_keep_only_uppercase = protein_final_keep_only_uppercase

        if self.pytoda_wrapped_tokenizer:
            if self.pytoda_SMILES_tokenizer_json is None:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_SMILES_tokenizer_json + pytoda_target_target_tokenizer_amino_acid_dict) or neither"
                )

        if self.pytoda_SMILES_tokenizer_json is not None:
            if not self.pytoda_wrapped_tokenizer:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_SMILES_tokenizer_json + pytoda_target_target_tokenizer_amino_acid_dict) or neither"
                )

        if self.pytoda_wrapped_tokenizer:
            if self.pytoda_target_target_tokenizer_amino_acid_dict is None:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_SMILES_tokenizer_json + pytoda_target_target_tokenizer_amino_acid_dict) or neither"
                )

        if self.pytoda_target_target_tokenizer_amino_acid_dict is not None:
            if not self.pytoda_wrapped_tokenizer:
                raise Exception(
                    "you need to set both (pytoda_wrapped_tokenizer + pytoda_SMILES_tokenizer_json + pytoda_target_target_tokenizer_amino_acid_dict) or neither"
                )

        super().__init__()

        self._shared_ligands_indexed_text_table_kwargs = dict(
            separator="\t",
            first_row_is_columns_names=False,
            columns_names=["peptide_sequence", "peptide_id"],
            id_column_name="peptide_id",
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
            ligand_sequence_column_name="peptide_sequence",
            affinity_pairs_csv_ligand_id_column_name=self.pairs_table_ligand_column,  # 'ligand_name',
            protein_sequence_column_name="protein_sequence",
            affinity_pairs_csv_protein_id_column_name=self.pairs_table_sequence_column,  # 'uniprot_accession',
            affinity_pairs_csv_affinity_value_column_name=self.pairs_table_affinity_column,  # 'pIC50',
        )

    def _create_pipeline_desc(
        self, is_training: bool, drug_target_affinity_loader_op: Callable
    ) -> list:
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

        # if is_training and self.train_augment_peptide_shuffle_atoms:
        #     pipeline_desc += [
        #         (SmilesToRDKitMol(), dict(key_in='data.input.ligand', key_out='data.input.ligand')),
        #         (SmilesRandomizeAtomOrder(), dict(key='data.input.ligand')),
        #         (RDKitMolToSmiles(), dict(key_in='data.input.ligand', key_out='data.input.ligand')),
        #     ]

        if self.active_site_alignment_info_smi is not None:
            pipeline_desc += [
                (
                    OpLoadActiveSiteAlignmentInfo(self.active_site_alignment_info_smi),
                    dict(key_in="data.input.protein", key_out="data.input.protein"),
                ),
            ]

            if (
                is_training
                and self.protein_augment_by_reverting_individual_active_site_sub_sequences
            ):
                pipeline_desc += [
                    (
                        ProteinFlipIndividualActiveSiteSubSequences(
                            p=self.protein_augment_by_reverting_individual_active_site_sub_sequences_p
                        ),
                        dict(
                            key_in_aligned_sequence="data.input.protein",
                            key_out="data.input.protein",
                        ),
                    ),
                ]

            if (
                is_training
                and self.protein_augment_by_introducing_noise_to_non_active_site_residues
            ):
                pipeline_desc += [
                    (
                        ProteinIntroduceActiveSiteBasedNoise(
                            mutate_prob_in_active_site=self.protein_augment_by_introducing_noise_to_non_active_site_residues_p_inside_active_site,
                            mutate_prob_outside_active_site=self.protein_augment_by_introducing_noise_to_non_active_site_residues_p_outside_active_site,
                        ),
                        dict(
                            key_in_aligned_sequence="data.input.protein",
                            key_out="data.input.protein",
                        ),
                    ),
                ]

        if (
            is_training and self.train_augment_protein_flip
        ):  # keep this after all active site alignment based operations! otherwise the extraction of aligned kinase info won't work
            pipeline_desc += [
                (
                    ProteinRandomFlipOrder(),
                    dict(key_in="data.input.protein", key_out="data.input.protein"),
                ),
                (
                    ProteinRandomFlipOrder(),
                    dict(key_in="data.input.ligand", key_out="data.input.ligand"),
                ),
            ]

        if is_training and self.protein_augment_full_sequence_noise:
            if self.protein_representation_type == "AA":
                pipeline_desc += [
                    (
                        ProteinIntroduceNoise(
                            p=self.protein_augment_full_sequence_noise_p
                        ),
                        dict(key_in="data.input.protein", key_out="data.input.protein"),
                    ),
                ]
            else:
                raise Exception(
                    "Adding noise to proteins that are not represented as AAs is not implemented. protein_representation_type conflicts with protein_augment_full_sequence_noise"
                )
            if self.peptide_representation_type == "AA":
                pipeline_desc += [
                    (
                        ProteinIntroduceNoise(
                            p=self.protein_augment_full_sequence_noise_p
                        ),
                        dict(key_in="data.input.ligand", key_out="data.input.ligand"),
                    ),
                ]
            else:
                raise Exception(
                    "Adding noise to peptides that are not represented as AAs is not implemented. protein_representation_type conflicts with protein_augment_full_sequence_noise"
                )

        # The two operators below, OpKeepOnlyUpperCase and OpToUpperCase, should only be applied if protein and peptide are representad as Amino Acid
        # sequences, not as SMILES strings (SMILES representation is case-sentitive, unlike AAs)
        if self.protein_final_keep_only_uppercase:
            if self.protein_representation_type == "AA":
                pipeline_desc += [
                    (
                        OpKeepOnlyUpperCase(),
                        dict(key_in="data.input.protein", key_out="data.input.protein"),
                    ),
                ]
            else:
                raise Exception(
                    "Keeping only upper case characters is only relevant to proteins represented as AAs. protein_representation_type conflicts with protein_final_keep_only_uppercase"
                )
            if self.peptide_representation_type == "AA":
                pipeline_desc += [
                    (
                        OpKeepOnlyUpperCase(),
                        dict(key_in="data.input.ligand", key_out="data.input.ligand"),
                    ),
                ]
            else:
                raise Exception(
                    "Keeping only upper case characters is only relevant to peptides represented as AAs. protein_representation_type conflicts with protein_final_keep_only_uppercase"
                )

        if self.protein_representation_type == "AA":
            pipeline_desc += [
                (
                    OpToUpperCase(),
                    dict(key_in="data.input.protein", key_out="data.input.protein"),
                ),
            ]
        if self.peptide_representation_type == "AA":
            pipeline_desc += [
                (
                    OpToUpperCase(),
                    dict(key_in="data.input.ligand", key_out="data.input.ligand"),
                ),
            ]

        # Up to this point we worked only with string representations. From now on we tokenize:
        if self.protein_representation_type == "SMILES":
            # If proteins and peptides are represented as SMILES strings, the tokenizer should work on smiles
            _protein_tokenizer_path = os.path.join(
                get_molecule_pretrained_tokenizer_path(),
                "bpe_tokenizer_trained_on_chembl_zinc_with_aug_4272372_samples_balanced_1_1.json",
            )
            protein_pad_id = 0
        elif self.protein_representation_type == "AA":
            # If proteins and peptides are represented as AA sequences, they should both use protein tokenizer
            _protein_tokenizer_path = os.path.join(
                get_protein_pretrained_tokenizer_path(), "simple_protein_tokenizer.json"
            )
            protein_pad_id = 0
        else:
            raise NotImplementedError(
                f"Unexpected representation type: {self.protein_representation_type}"
            )

        if self.peptide_representation_type == "SMILES":
            # If proteins and peptides are represented as SMILES strings, the tokenizer should work on smiles
            _peptide_tokenizer_path = os.path.join(
                get_molecule_pretrained_tokenizer_path(),
                "bpe_tokenizer_trained_on_chembl_zinc_with_aug_4272372_samples_balanced_1_1.json",
            )
            peptide_pad_id = (
                3  # TODO: check if peptide_pad_id should be the same as protein_pad_id
            )
        elif self.peptide_representation_type == "AA":
            # If proteins and peptides are represented as AA sequences, they should both use protein tokenizer
            _peptide_tokenizer_path = os.path.join(
                get_protein_pretrained_tokenizer_path(), "simple_protein_tokenizer.json"
            )
            peptide_pad_id = (
                3  # TODO: check if peptide_pad_id should be the same as protein_pad_id
            )
        else:
            raise NotImplementedError(
                f"Unexpected representation type: {self.peptide_representation_type}"
            )

        if self.protein_representation_type == "SMILES":
            if self.pytoda_wrapped_tokenizer:
                protein_tokenizer_op = Op_pytoda_SMILESTokenizer(
                    dict(
                        vocab_file=self.pytoda_SMILES_tokenizer_json,
                        padding_length=self.receptor_padding_length,
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
                protein_tokenizer_op = TokenizerOp(
                    _protein_tokenizer_path,
                    pad_length=self.receptor_padding_length,
                    pad_id=protein_pad_id,
                )
        elif self.protein_representation_type == "AA":
            if self.pytoda_wrapped_tokenizer:
                protein_tokenizer_op = Op_pytoda_ProteinTokenizer(
                    amino_acid_dict=self.pytoda_target_target_tokenizer_amino_acid_dict,
                    add_start_and_stop=True,
                    padding=True,
                    padding_length=self.receptor_padding_length,
                )
            else:
                protein_tokenizer_op = TokenizerOp(
                    _protein_tokenizer_path,
                    pad_length=self.receptor_padding_length,
                    pad_id=protein_pad_id,
                )
        else:
            raise NotImplementedError(
                f"Unexpected representation type: {self.protein_representation_type}"
            )

        if self.peptide_representation_type == "SMILES":
            if self.pytoda_wrapped_tokenizer:
                ligand_tokenizer_op = Op_pytoda_SMILESTokenizer(
                    dict(
                        vocab_file=self.pytoda_SMILES_tokenizer_json,
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
                ligand_tokenizer_op = TokenizerOp(
                    _peptide_tokenizer_path,
                    pad_length=self.ligand_padding_length,
                    pad_id=peptide_pad_id,
                )
        elif self.peptide_representation_type == "AA":
            if self.pytoda_wrapped_tokenizer:
                ligand_tokenizer_op = Op_pytoda_ProteinTokenizer(
                    amino_acid_dict=self.pytoda_target_target_tokenizer_amino_acid_dict,
                    add_start_and_stop=True,
                    padding=True,
                    padding_length=self.ligand_padding_length,
                )
            else:
                ligand_tokenizer_op = TokenizerOp(
                    _peptide_tokenizer_path,
                    pad_length=self.ligand_padding_length,
                    pad_id=peptide_pad_id,
                )

        else:
            raise NotImplementedError(
                f"Unexpected representation type: {self.peptide_representation_type}"
            )

        pipeline_desc += [
            # peptide
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
        # affinity_loader_op = PeptideTargetAffinityLoader(
        #     affinity_pairs_csv_path=self.train_dataset_path,
        #     **self._affinity_loader_shared_arguments
        # )

        affinity_loader_op = PeptideTargetAffinityLoader(
            ligands_smi=self.peptides_smi,
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
            all_sample_ids,
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

        affinity_loader_op = PeptideTargetAffinityLoader(
            ligands_smi=self.peptides_smi,
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
            all_sample_ids,
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
        affinity_loader_op = PeptideTargetAffinityLoader(
            ligands_smi=self.peptides_smi,
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
            all_sample_ids,
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
