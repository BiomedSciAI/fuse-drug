import os
from fuse.utils.cpu_profiling import Timer
from fuse.utils.file_io import save_hdf5_safe, load_hdf5
import numpy as np

from contextlib import nullcontext
from torch.utils.data import Dataset
from copy import deepcopy
from fuse.utils.multiprocessing import (
    run_multiprocessed,
    get_from_global_storage,
    get_chunks_ranges,
)
from typing import Tuple, Union, Sequence, Callable

# TODO: consider using SQLite for files with too big number of identifiers to fit in memory
# maybe use "raw mode" in https://github.com/Congyuwang/RocksDict ? (or maybe not, as it just uses a DB in the backend, so maybe it's better to use a DB ourselves)


def _default_identifier_extractor(
    identifier_line: str, also_return_comment: bool = True, verbose: int = 0
) -> Union[Tuple[str, str], str]:
    # return identifier_line.split('|')[1]
    pos = identifier_line.find(" ")
    if pos < 0:
        if verbose > 0:
            print("Warning: no space found in identifier line:", identifier_line)
        return (identifier_line, None) if also_return_comment else identifier_line

    identifier = identifier_line[:pos].rstrip()
    if also_return_comment:
        comment = identifier_line[pos + 1 :]
        return identifier, comment

    return identifier


class IndexedFastaCustom(Dataset):
    """
    Handles large fasta files - builds a file offset index for each line and stores it in hdf5 format.
    allows random (index) access, and also a lightweight iterator (which doesn't hold all of the data in mem)

    The expected loaded (potentially large) text file is in fasta format, so each line is a sequence of:

    1. Identifier line starting with ">" and ending with "\n"
    2. Any number of lines ending with "\n"

    Usage example: see tests/indexed_fasta_custom.py
    """

    def __init__(
        self,
        filename: str,
        index_filename: str = None,
        process_identifier_pipeline: Sequence[Callable] = (
            _default_identifier_extractor,
        ),
        force_recreate_index: bool = False,
        allow_access_by_id: bool = False,
        num_workers: Union[int, str] = "auto",
        verbose: int = 1,
    ):
        """
        args:
            filename: for example, 'some_massive_file.fasta'
            index_filename (optional): custom index_filename. If not provided, defaults to using [filename] with a modified extension - '.index.hdf5'
            process_identifier_pipeline:
            force_recreate_index: recreates (and overrides!) index file even if already found

            for example:
                itf = IndexedTextFile('bigfile.fasta')
                s = itf['sp|Q6GZX3|002L_FRG3G Uncharacterized protein 002L OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-002L PE=4 SV=1']

            NOTE: it increases loading time and also memory usage, so only set if really needed

            verbose: verbosity level, how much information is printed. pass 0 to silence

        """
        self.filename = filename

        self.process_identifier_pipeline = process_identifier_pipeline
        if self.process_identifier_pipeline is None:
            self.process_identifier_pipeline = []

        timer = Timer("Process") if verbose > 0 else nullcontext()
        with timer:
            if index_filename is None:
                # index_filename = change_extension(filename, ".fastaindex.hdf5")
                index_filename = filename + ".fastaindex.hdf5"
                if verbose > 0:
                    print(f"IndexedTextFile:: index_filename={index_filename}")

            self.index_filename = index_filename

            already_found = False
            if (not force_recreate_index) and (os.path.isfile(index_filename)):
                if verbose > 0:
                    print(f"index file already found: {index_filename}")
                    already_found = True

            if not already_found:
                lines_offsets = []
                # names = []

                line_num = 0
                offset = 0
                # important - using 'rb' will not remove things line '\r' from the line, making the offset ok! (as opposed to using 'r' !!)
                if verbose > 0:
                    print("building fasta index ... ")
                with open(filename, "rb") as read_file:
                    for line in read_file:

                        # note - only decoding the first character to avoid wasting time on decoding the whole line
                        if (
                            not line[:1].decode() == ">"
                        ):  # skip all non-identifier lines
                            offset += len(line)
                            continue
                        lines_offsets.append(offset)
                        line_num += 1
                        offset += len(line)

                        if (verbose > 0) and (0 == line_num % 1e7):
                            print("line_num=", line_num)

                lines_offsets = np.array(lines_offsets, dtype=np.int64)

                timer = Timer("Store") if verbose > 0 else nullcontext()
                with timer:
                    save_hdf5_safe(
                        index_filename,
                        use_blosc=False,
                        lines_offsets=lines_offsets,
                    )
                lines_offsets = None

        timer = Timer("Process") if verbose > 0 else nullcontext()
        with timer:
            loaded_hdf5 = load_hdf5(
                self.index_filename
            )  # reloading even in the creation time (intentional)
            self.offsets = loaded_hdf5["lines_offsets"]

        if isinstance(num_workers, str):
            assert num_workers == "auto"
            num_workers = os.cpu_count()
        self._num_workers = num_workers

        self._allow_access_by_id = allow_access_by_id
        if self._allow_access_by_id:
            self._offsets_map = self._build_offsets_map()

    def _build_offsets_map(self) -> dict:
        self._offsets_map = dict()  # maps from id to line index
        print(
            f"allow_access_by_id is enabled, building in-memory offset map (num_workers={self._num_workers})"
        )

        total = len(self)
        chunks_defs = get_chunks_ranges(total, chunk_size=1000)

        all_offsets_maps = run_multiprocessed(
            worker_func=_key_map_build_worker,
            args_list=chunks_defs,
            workers=self._num_workers,
            verbose=1,
            keep_results_order=False,
            copy_to_global_storage=dict(ifc=self),
        )

        offsets_map = {}
        for m in all_offsets_maps:
            offsets_map.update(m)

        return offsets_map

    def __len__(self) -> int:
        return self.offsets.shape[0]

    def __iter__(self) -> Tuple[Union[Tuple[str], Tuple[str, str]], str, str]:
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitem__(
        self, index: Union[str, int]
    ) -> Tuple[Union[Tuple[str], Tuple[str, str]], str, str]:
        """
        returns a single FASTA entry in as a tuple with 3 elements:
            element 0: a tuple with either a. a single element, the identifier of the entry or b. a tuple with 2 elements, the identifier of the entry followed by the comment of the entry
            element 1: the sequence data
            element 2: the full identifier raw line
        """

        if isinstance(index, str):
            if not self._allow_access_by_id:
                raise Exception(
                    "IndexedFastaCustom must be initialized with allowed_access_by_id to access by string identifier"
                )

            # convert it to integer value
            if index in self._offsets_map:
                index = self._offsets_map[index]
            else:
                raise KeyError(f"Could not find identifier {index}")
        elif not isinstance(index, (int, np.int32, np.int64, np.uint32)):
            raise Exception(
                f"indexing is supported by integer index, or by string identifier if allow_access_by_id=True was set, got {type(index)}"
            )
        offset = self.offsets[index].item()

        data = []
        with open(self.filename, "rb") as f:
            f.seek(offset, 0)

            identifier_line = f.readline().decode().rstrip()
            assert identifier_line.startswith(">")
            identifier_line = identifier_line[1:]  # removing '>'

            while True:
                curr_line = f.readline().decode()
                if ("" == curr_line) or (curr_line.startswith(">")):
                    break
                data.append(curr_line)

        data = [x.rstrip() for x in data]
        data = "".join(data)

        processed_identifier = deepcopy(identifier_line)

        for curr_func in self.process_identifier_pipeline:
            processed_identifier = curr_func(processed_identifier)

        return processed_identifier, data, identifier_line


def uniprot_identifier_extractor(identifier_line: str, verbose: int = 0) -> str:
    """
    returns the first part of the identifier line, which is the uniprot accession number
    """
    return identifier_line.split("|")[1]


def _key_map_build_worker(args: Tuple[int, int]) -> dict:
    start_index, end_index = args
    ifc = get_from_global_storage("ifc")
    ans = {}
    for line_idx in range(start_index, end_index):
        entry = ifc[line_idx]
        identifier, _, _ = entry
        ans[identifier] = line_idx
    return ans
