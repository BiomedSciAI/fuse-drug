import os
from fuse.utils.cpu_profiling import Timer
from fuse.utils.file_io import save_hdf5_safe, load_hdf5, change_extension
import numpy as np
from fuse.utils.misc.context import DummyContext
import click
from torch.utils.data import Dataset

# TODO: add ignore column line (default to False)
class IndexedTextFile(Dataset):
    """
    Handles large text files - builds a file offset index for each line and stores it in hdf5 format.
    allows random (index) access, and also a lightweight iterator (which doesn't hold all of the data in mem)

    The expected loaded (potentially large) text file is:
    * One line per entry
    * each line is some delim seperated tokens

    Example usage: smi files which store many smiles strings
                   see tests/test_indexed_text_file.py
    """

    def __init__(
        self, filename: str, index_filename=None, process_funcs_pipeline=None, force_recreate_index=False, verbose=1
    ):
        """
        args:
            filename: for example, 'some_massive_file.smi'
            index_filename (optional): custom index_filename. If not provided, defaults to using [filename] with a modified extension - '.index.hdf5'
            process_line_func:
            force_recreate_index: recreates (and overrides!) index file even if already found

            for example:
                itf = IndexedTextFile('bigfile.smi')
                s = itf['ZINC00345034503450']

            NOTE: it increases loading time and also memory usage, so only set if really needed

            verbose: verbosity level, how much information is printed. pass 0 to silence

        """
        self.filename = filename

        self.process_funcs_pipeline = process_funcs_pipeline
        if self.process_funcs_pipeline is None:
            self.process_funcs_pipeline = []

        timer = Timer("Process") if verbose > 0 else DummyContext()
        with timer:
            if index_filename is None:
                index_filename = filename+'.index.hdf5'
                if verbose>0:
                    print(f'IndexedTextFile:: index_filename={index_filename}')
            

            self.index_filename = index_filename

            already_found = False
            if (not force_recreate_index) and (os.path.isfile(index_filename)):
                if verbose > 0:
                    print(f"index file already found: {index_filename}")
                    already_found = True

            if not already_found:
                lines_offsets = []

                line_num = 0
                offset = 0
                # important - using 'rb' will not remove things line '\r' from the line, making the offset ok! (as opposed to using 'r' !!)
                with open(filename, "rb") as read_file:
                    for line in read_file.readlines():
                        lines_offsets.append(offset)
                        line_num += 1
                        offset += len(line)

                        if (verbose > 0) and (0 == line_num % 1e7):
                            print("line_num=", line_num)

                lines_offsets = np.array(lines_offsets, dtype=np.int64)

                timer = Timer("Store") if verbose > 0 else DummyContext()
                with timer:
                    save_hdf5_safe(
                        index_filename,
                        use_blosc=False,
                        lines_offsets=lines_offsets,
                    )
                lines_offsets = None
        loaded_hdf5 = load_hdf5(self.index_filename)  # reloading even in the creation time (intentional)
        self.offsets = loaded_hdf5["lines_offsets"]

    def __len__(self):
        return self.offsets.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        assert isinstance(index, int)  # not supporting named access yet
        offset = self.offsets[index].item()
        with open(self.filename, "rb") as f:
            f.seek(offset, 0)
            curr_line = f.readline()
            ans = curr_line.decode()
            for curr_func in self.process_funcs_pipeline:
                ans = curr_func(ans)
            return ans
