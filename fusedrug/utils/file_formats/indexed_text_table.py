from typing import Tuple, Optional, List, Union
from torch.utils.data import Dataset
from warnings import warn
from fusedrug.utils.file_formats import IndexedTextFile
import numbers
import pandas as pd
from fuse.utils.multiprocessing import run_multiprocessed, get_from_global_storage, get_chunks_ranges
import os


class IndexedTextTable(Dataset):
    def __init__(
        self,
        filename: str,
        seperator="\t",
        first_row_is_columns_names: bool = True,
        columns_names: Optional[List[str]] = None,
        id_column_idx: Optional[int] = None,
        id_column_name: Optional[str] = None,
        columns_num_expectation: Optional[int] = None,
        allow_access_by_id: bool = True,
        index_filename: Optional[str] = None,
        process_funcs_pipeline=None,
        force_recreate_index=False,
        limit_lines=None,  # useful for debugging
        num_workers: Union[str, int] = "auto",  # will be used when building the in-memory key-search map
    ):
        """
        Args:
            process_funcs_pipeline: a list of functions.
                The first function is expected to be:
                def foo(mol_id:str, mol_data:List[str]) -> Any
                    #...

                and the next functions will be called based on the previous **return_value
        Usage example:
            see tests/test_indexed_text_table.py

        """

        if isinstance(num_workers, str):
            assert num_workers == "auto"
            num_workers = os.cpu_count()

        assert isinstance(num_workers, int)
        assert num_workers >= 0

        self._num_workers = num_workers

        self._filename = filename
        self._id_column_idx = id_column_idx
        self._id_column_name = id_column_name
        self._columns_num_expectation = columns_num_expectation
        if self._columns_num_expectation is not None:
            assert (
                self._id_column_idx < self._columns_num_expectation
            ), f"self._id_column_idx={self._id_column_idx} is outside of range for the provided while self._columns_num_expectation={self._columns_num_expectation}"

        self._seperator = seperator
        self._allow_access_by_id = allow_access_by_id

        if (not first_row_is_columns_names) and (columns_names is None):
            raise Exception("if you pass first_row_is_columns_names=False you must provide columns_names")

        self._columns_names = columns_names

        self._first_row_is_columns_names = first_row_is_columns_names

        self._indexed_text_file = IndexedTextFile(
            filename=filename,
            index_filename=index_filename,
            force_recreate_index=force_recreate_index,
        )

        if limit_lines is not None:
            assert isinstance(limit_lines, int)
            assert limit_lines >= 0
        self._limit_lines = limit_lines

        self._process_funcs_pipeline = process_funcs_pipeline
        if self._process_funcs_pipeline is None:
            self._process_funcs_pipeline = []

        # read columns names
        if self._first_row_is_columns_names:
            first_line = self._indexed_text_file[0]
            print("")
            first_line_splitted = self._split_line_basic(first_line)
            if 1 == len(first_line_splitted):
                warn(f"columns line only contains a single column! maybe a separator issue? first line = {first_line}")
            elif 0 == len(first_line_splitted):
                raise Exception(f"could not find any columns names in first line - {first_line}")

            if self._columns_names is None:
                self._columns_names = first_line_splitted
            else:
                print(
                    f"Since columns_names was provided, using columns names = {self._columns_names} instead of {first_line_splitted}"
                )

        if len(self._columns_names) != len(set(self._columns_names)):
            raise Exception(f"duplicate column names found! columns={self._columns_names}")
        print(
            f"column names used={self._columns_names} - if it does not look like column names make sure that the following args are properly set: first_row_is_columns_names, columns_names"
        )

        if self._columns_num_expectation is not None:
            assert self._columns_num_expectation == len(self._columns_names)

        if self._columns_num_expectation is None:
            self._columns_num_expectation = len(self._columns_names)

        if self._allow_access_by_id:

            if not ((id_column_idx is not None) ^ (id_column_name is not None)):
                raise Exception(
                    "allow_access_by_id is enabled, so you must provide exactly one of id_column_idx, id_column_name"
                )

        self._id_column_idx = id_column_idx
        self._id_column_name = id_column_name

        if self._id_column_name is None:
            assert self._id_column_idx >= 0
            self._id_column_name = self._columns_names[self._id_column_idx]

        if self._id_column_idx is None:
            self._id_column_idx = self._columns_names.index(self._id_column_name)

        if self._allow_access_by_id:
            self._offsets_map = dict()  # maps from id to line index
            print(f"allow_access_by_id is enabled, building in-memory offset map (num_workers={self._num_workers})")

            total = len(self._indexed_text_file)
            chunks_defs = get_chunks_ranges(total, chunk_size=1000)

            all_offsets_maps = run_multiprocessed(
                worker_func=_key_map_build_worker,
                args_list=chunks_defs,
                workers=self._num_workers,
                verbose=1,
                keep_results_order=False,
                copy_to_global_storage=dict(itf=self),
            )

            self._offsets_map = {}
            for m in all_offsets_maps:
                self._offsets_map.update(m)

    def _split_line_basic(self, line_str: str) -> Tuple:
        splitted = line_str.split(self._seperator)

        if self._columns_num_expectation is not None:
            if self._columns_num_expectation != len(splitted):
                raise Exception(
                    f"Expected each line to be {repr(self._seperator)} separated and contain exactly {self._columns_num_expectation} values, but instead got {len(splitted)} values : {splitted} Make sure that your smi file {self._filename} is in correct format."
                )

        splitted[-1] = splitted[-1].rstrip()  # remove trailing endline etc.
        return splitted

    def _process_line(self, line_str: str) -> Tuple[str, Tuple]:
        """
        returns (id, tuple of the other elements in the parsed text row)
        """
        splitted = self._split_line_basic(line_str)
        ser = pd.Series(dict(zip(self._columns_names, splitted)), index=None)
        ans_id = ser[self._id_column_idx]
        return ans_id, ser

    def __len__(self) -> int:
        # since the first line might be columns description, we must consider it
        valid_text_file_lines = len(self._indexed_text_file)
        if self._first_row_is_columns_names:
            valid_text_file_lines -= 1

        if self._limit_lines is not None:
            return min(self._limit_lines, valid_text_file_lines)
        return valid_text_file_lines

    def __iter__(self):
        for i in range(len(self)):
            if self._limit_lines is not None:
                if i >= self._limit_lines:
                    break
            yield self.__getitem__(i)

    def __getitem__(self, index: int):
        if isinstance(index, numbers.Number):
            index = int(index)

        if isinstance(index, int):
            line_index = int(index)
            if self._first_row_is_columns_names:
                line_index += 1
        elif isinstance(index, str):
            if not self._allow_access_by_id:
                raise Exception(
                    "allow_access_by_id was set to False, so you can not access by string id. pass allow_access_by_id=True if you are interested in this."
                )
            line_index = self._offsets_map[index]
        else:
            raise Exception(f"supported index types are int or str, instead got {type(index)}")

        index = None  # just clearing this var to make sure no one uses it in the func

        line = self._indexed_text_file[line_index]

        ans = self._process_line(line)

        for func in self._process_funcs_pipeline:
            assert callable(func)
            ans = func(*ans)

        return ans


def _key_map_build_worker(args: Tuple[int, int]) -> dict:
    start_index, end_index = args
    itf = get_from_global_storage("itf")
    ans = {}
    for line_idx in range(start_index, end_index):
        if itf._first_row_is_columns_names and (0 == line_idx):
            continue
        if (itf._limit_lines is not None) and (line_idx >= itf._limit_lines):
            break
        line = itf._indexed_text_file[line_idx]
        mol_id, _ = itf._process_line(line)
        ans[mol_id] = line_idx
    return ans
