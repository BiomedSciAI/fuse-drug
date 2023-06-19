from typing import Optional, Union
from torch.utils.data import Dataset
from fusedrug.utils.file_formats import IndexedTextTable
import pandas as pd


class PytodaStyleDrugTargetAffinityDataset(Dataset):
    def __init__(
        self,
        ligands_smi: Union[str, IndexedTextTable],
        ligand_sequence_column_name: str,
        proteins_smi: Union[str, IndexedTextTable],
        protein_sequence_column_name: str,
        affinity_pairs_csv_path: str,
        affinity_pairs_csv_ligand_id_column_name: str,
        affinity_pairs_csv_protein_id_column_name: str,
        affinity_pairs_csv_affinity_value_column_name: str,
        ligands_indexed_text_table_kwargs: Optional[dict] = None,
        proteins_indexed_text_table_kwargs: Optional[dict] = None,
    ):
        assert isinstance(ligands_smi, (str, IndexedTextTable))
        self._ligand_sequence_column_name = ligand_sequence_column_name
        self._protein_sequence_column_name = protein_sequence_column_name

        self._affinity_pairs_csv_ligand_id_column_name = (
            affinity_pairs_csv_ligand_id_column_name
        )
        self._affinity_pairs_csv_protein_id_column_name = (
            affinity_pairs_csv_protein_id_column_name
        )
        self._affinity_pairs_csv_affinity_value_column_name = (
            affinity_pairs_csv_affinity_value_column_name
        )

        self._ligands_smi = ligands_smi
        # _indexed_table_table_kwargs = dict(
        #     #seperator='\t',
        #     #id_column_idx=1,
        #     allow_access_by_id=True
        # )
        if ligands_indexed_text_table_kwargs is None:
            ligands_indexed_text_table_kwargs = {}

        if isinstance(self._ligands_smi, str):
            self._ligands_smi = IndexedTextTable(
                ligands_smi,
                **ligands_indexed_text_table_kwargs,
            )
        elif len(ligands_indexed_text_table_kwargs) > 0:
            raise Exception(
                "the provided ligands_smi is a table, you cannot provide ligands_indexed_text_table_kwargs for it"
            )

        if not self._ligands_smi._allow_access_by_id:
            raise Exception(
                "_allow_access_by_id is required for DrugTargetAffinityDataset ! please add _allow_access_by_id=True to ligands_indexed_text_table_kwargs"
            )

        assert isinstance(proteins_smi, (str, IndexedTextTable))
        self._proteins_smi = proteins_smi

        if proteins_indexed_text_table_kwargs is None:
            proteins_indexed_text_table_kwargs = {}

        if isinstance(self._proteins_smi, str):
            self._proteins_smi = IndexedTextTable(
                proteins_smi,
                **proteins_indexed_text_table_kwargs,
            )
        elif len(proteins_indexed_text_table_kwargs) > 0:
            raise Exception(
                "the provided proteins_smi is a table, you cannot provide proteins_indexed_text_table_kwargs for it"
            )

        if not self._proteins_smi._allow_access_by_id:
            raise Exception(
                "_allow_access_by_id is required for DrugTargetAffinityDataset ! please add _allow_access_by_id=True to proteins_indexed_text_table_kwargs"
            )

        self._affinity_pairs_csv_path = affinity_pairs_csv_path
        self._affinity_df = pd.read_csv(self._affinity_pairs_csv_path)

        self._affinity_df[affinity_pairs_csv_ligand_id_column_name] = self._affinity_df[
            affinity_pairs_csv_ligand_id_column_name
        ].str.rstrip()
        self._affinity_df[
            affinity_pairs_csv_protein_id_column_name
        ] = self._affinity_df[affinity_pairs_csv_protein_id_column_name].str.rstrip()

    def __len__(self) -> int:
        return self._affinity_df.shape[0]

    def __iter__(self) -> dict:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> dict:
        row = self._affinity_df.iloc[index]

        ligand_id = row[self._affinity_pairs_csv_ligand_id_column_name]
        protein_id = row[self._affinity_pairs_csv_protein_id_column_name]

        _ligand_id, ligand_data = self._ligands_smi[ligand_id]

        assert ligand_id == _ligand_id
        ligand_str = ligand_data[self._ligand_sequence_column_name]

        _protein_id, protein_data = self._proteins_smi[protein_id]
        assert protein_id == _protein_id
        protein_str = protein_data[self._protein_sequence_column_name]

        # pIC50 = float(row.pIC50)
        affinity_val = float(row[self._affinity_pairs_csv_affinity_value_column_name])

        # row.ligand_name, row.sequence_id, row.pIC50

        return dict(
            ligand_str=ligand_str,
            protein_str=protein_str,
            affinity_val=affinity_val,
        )
