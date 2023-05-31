import pyfastx  # https://pyfastx.readthedocs.io/en/latest/usage.html
from torch.utils.data import Dataset
from typing import Optional
# pyfast - access sequence data: https://pyfastx.readthedocs.io/en/latest/usage.html#get-a-sequence-from-fasta


class IndexedFasta(Dataset):
    """
    Loads an entry from a fasta file. Uses pyfastx library for quick fetching of relevant parts.
    Usage example: see tests/indexed_fasta.py
    """

    def __init__(self, fasta_file_loc: Optional[str]=None, check_for_duplicate_names=False, process_funcs_pipeline=None, **kwargs):
        """
        :param fasta_file_loc: location of .fasta or .fasta.gz file
        :param check_for_duplicate_names: checks for duplicates (in names, does not check sequences!)
            may take few minutes.
        """
        super().__init__(**kwargs)

        self.process_funcs_pipeline = process_funcs_pipeline
        if self.process_funcs_pipeline is None:
            self.process_funcs_pipeline = []

        for curr_func in self.process_funcs_pipeline:
            if not callable(curr_func):
                raise Exception(
                    f"process_funcs_pipeline is expected to be a either None or a list of callables, but {curr_func} is not callable."
                )

        print("WARNING: DUPLICATES ARE NOT SEARCHED FOR YET!!!")
        self._fasta_file_loc = fasta_file_loc
        print(
            "loading fasta file - note, if it's the first time you are loading this file, index building (using pyfastx) may take time ..."
        )
        self._fasta = pyfastx.Fasta(self._fasta_file_loc)
        print("Done loading fasta file.")
        self._check_for_duplicate_names = check_for_duplicate_names

        if self._check_for_duplicate_names:
            print(
                "checking for duplicates (in sequences names! does not check the sequences content!!). May take few minutes..."
            )
            keys = self._fasta.keys()
            if len(set(keys)) != len(keys):
                raise Exception("Found duplicates!")

        self._length = len(self._fasta)

    def __len__(self):
        return self._length

    def __iter__(self):
        for i in range(self._length):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        """ """
        entry = self._fasta[index]

        for curr_func in self.process_funcs_pipeline:
            entry = curr_func(entry)

        return entry
