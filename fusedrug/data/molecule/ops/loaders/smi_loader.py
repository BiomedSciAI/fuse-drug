from copy import deepcopy
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.utils.file_formats import IndexedTextTable


class SmiLoader(OpBase):
    """
    Loads an entry from an smi file (large files are supported!).
    """

    def __init__(
        self,
        smi_file_loc=None,
        molecule_id_column_idx: int = 0,
        seperator: str = "\t",
        allow_access_by_id=True,
        **kwargs
    ):
        """
        :param smi_file_loc: location of .smi file
            the file format is expected to be a text file in which each line is expected to be '\t' separated - two columns,
            the first is the molecule_id and the second is the string representation (usually SMILES)
        """
        super().__init__(**kwargs)
        self._smi_file_loc = smi_file_loc
        self._molecule_id_column_idx = molecule_id_column_idx
        self._seperator = seperator
        self._allow_access_by_id = allow_access_by_id
        self._indexed_text_table = IndexedTextTable(
            smi_file_loc,
            id_column_idx=self._molecule_id_column_idx,
            allow_access_by_id=self._allow_access_by_id,
            columns_num_expectation=2,
        )

    def __call__(self, sample_dict: NDict, key_out_seq="data.gt.seq", key_out_mol_id=None):
        sid = get_sample_id(sample_dict)
        assert isinstance(sid, (int, str))

        entry_id, entry_data = self._indexed_text_table[sid]

        assert 1 == len(entry_data)
        molecule_str = entry_data[0]

        sample_dict[key_out_seq] = molecule_str
        if key_out_mol_id is not None:
            sample_dict[key_out_mol_id] = entry_id
        return sample_dict
