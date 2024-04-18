from typing import Optional, Dict
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.utils.file_formats import IndexedTextTable
import numpy


class IndexedTextPPITableLoader(OpBase):
    """
    Loads an entry from an txt file (large files are supported!).
    """

    def __init__(
        self,
        table_file_loc: Optional[str] = None,
        index_filename: Optional[str] = None,
        id_column_index: int = 0,
        rename_columns: Optional[Dict[str, str]] = None,
        separator: str = " ",
        allow_access_by_id: bool = False,  # best leave it at False for large files
        **kwargs: dict,
    ):
        """
        :param table_file_loc: location of .txt file
            the file format is expected to be a text file in which each line is expected to be ' ' separated,
            containing the columns named
        :param index_filename: index file for the table, if not exist or None, it will recreate the index
        """
        super().__init__(**kwargs)
        self._table_file_loc = table_file_loc
        self._id_column_index = id_column_index
        self._rename_columns = rename_columns
        self._separator = separator
        self._allow_access_by_id = allow_access_by_id
        self._indexed_text_table = IndexedTextTable(
            filename=table_file_loc,
            index_filename=index_filename,
            seperator=self._separator,
            id_column_idx=self._id_column_index,
            allow_access_by_id=self._allow_access_by_id,
        )

    def __call__(self, sample_dict: NDict, prefix: Optional[str] = None) -> NDict:
        sid = get_sample_id(sample_dict)
        assert isinstance(
            sid, (int, numpy.int64, numpy.int32, numpy.uint32, numpy.uint64)
        )

        _, entry_data = self._indexed_text_table[sid]

        for c in entry_data.axes[0]:
            if prefix is None:
                sample_dict[self._rename_columns.get(c, c)] = entry_data[c]
            else:
                sample_dict[f"{prefix}.{self._rename_columns.get(c,c)}"] = entry_data[c]

        return sample_dict
