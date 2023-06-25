from typing import Dict, Optional
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from fusedrug.utils.file_formats import IndexedFasta, IndexedFastaCustom

"""
Note: I'm getting crashes in pyfastx library code when using FastaLoader on very large FASTA files (400M+ entries)
In such case you can use the alternative IndexedFastaCustom class.
"""


class FastaLoaderCustom(OpBase):
    """
    Allows random access to a large FASTA file
    """

    def __init__(self, fasta_filename: str, **indexed_fasta_custom_kwargs: Dict):
        """
        Args:
            fasta_filename: the name of the FASTA file to load
            indexed_fasta_custom_kwargs: optional kwargs that will be redirected to IndexedFastaCustom constructor
        """
        super().__init__()
        self._fasta = IndexedFastaCustom(fasta_filename, **indexed_fasta_custom_kwargs)

    def __call__(
        self,
        sample_dict: NDict,
        key_out: str = "data.seq",
        key_out_metadata: Optional[str] = None,
    ) -> NDict:
        """ """
        sid = get_sample_id(sample_dict)
        entry = self._fasta[sid]
        sample_dict[key_out] = entry[1]
        if key_out_metadata is not None:
            sample_dict[key_out_metadata] = entry[0]

        return sample_dict


class FastaLoader(OpBase):
    """
    Loads an entry from a fasta file. uses IndexedFasta which uses pyfastx under the hood
    """

    def __init__(
        self,
        fasta_file_loc: Optional[str] = None,
        check_for_duplicate_names: bool = False,
    ):
        """
        :param fasta_file_loc: location of .fasta or .fasta.gz file
        :param check_for_duplicate_names: checks for duplicates (in names, does not check sequences!)
            may take few minutes.
        """
        super().__init__()
        self._fasta = IndexedFasta(
            fasta_file_loc=fasta_file_loc,
            check_for_duplicate_names=check_for_duplicate_names,
        )

    def __call__(self, sample_dict: NDict, key_out: str = "data.seq") -> NDict:
        """ """
        sid = get_sample_id(sample_dict)
        entry = self._fasta[sid]
        sample_dict[key_out] = entry.seq

        return sample_dict
