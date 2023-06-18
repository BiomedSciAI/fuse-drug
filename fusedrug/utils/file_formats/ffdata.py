# format description
# https://github.com/soedinglab/ffindex_soedinglab
# https://github.com/ahcm/ffindex/blob/master/python/ffindex.py
# https://bfd.mmseqs.com/
# https://github.com/ahcm/ffindex

import os
from fuse.utils.file_io import save_hdf5_safe, load_hdf5, change_extension
import numpy as np
from tqdm import tqdm
from fuse.utils.cpu_profiling import Timer
import mmap
from typing import Optional, List


class FFData:
    def __init__(
        self, filename: str, ffindex_filename: Optional[str] = None, force_recreate_binary_index: bool = False,
    ):
        """
        args:
            filename: path to ffdata file
            ffindex: path to ffindex file, if not provided, will use the same path as [filename] which modified extension to ".ffindex"
        """
        if ffindex_filename is None:
            ffindex_filename = change_extension(filename, ".ffindex")

        self.filename = filename
        self.ffindex_filename = ffindex_filename
        self.index_filename = change_extension(ffindex_filename, ".multiline_index.hdf5")

        if force_recreate_binary_index or (not os.path.isfile(self.index_filename)):
            with Timer("creating hdf5 index"):
                names = []
                offsets = []
                lengths = []

                with open(self.ffindex_filename, "r") as f_index:
                    for line in tqdm(f_index.readlines()):
                        name, offset, length = line.split()
                        names.append(int(name))
                        offsets.append(int(offset))
                        lengths.append(int(length))

            with Timer("creating np array"):
                index_data = np.array([names, offsets, lengths], dtype=np.int64).transpose()
            with Timer("storing np array"):
                save_hdf5_safe(
                    self.index_filename, use_blosc=True, index_data=index_data,
                )

        names = offsets = lengths = None
        with Timer("loading hdf5 index"):
            loaded_hdf5 = load_hdf5(self.index_filename)  # reloading even in the creation time (intentional)
        self.index_data = loaded_hdf5["index_data"]

        # load the index file
        data_filehandle = open(self.filename, "rb")
        self.mmap_data = mmap.mmap(data_filehandle.fileno(), 0, prot=mmap.PROT_READ)
        data_filehandle.close()

    def __getitem__(self, index: int) -> List[str]:
        info = self.index_data[index]
        name, offset, length = info[0], info[1], info[2]  # noqa: F841
        lines = self.mmap_data[offset : offset + length - 1].decode("utf-8").split("\n")
        return lines

    def __del__(self) -> None:
        self.mmap_data.close()


# TODO: add usage example/test
