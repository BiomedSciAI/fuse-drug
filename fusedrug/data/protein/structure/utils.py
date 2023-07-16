# TODO: temp until openfold will be added to the dependency list
from typing import List

try:
    from openfold.np import residue_constants as rc
except ImportError:
    print("Warning: import openfold failed - some functions might fail")


def aa_sequence_from_aa_integers(aatype: List[int]) -> str:
    """
    converts a list of amino acid integers to string of amino acids
    """
    ans = "".join(["X" if x == 20 else rc.restypes[x] for x in aatype])
    return ans


def aa_integers_from_aa_sequence(aaseq: str) -> str:
    """
    converts a string of amino acids  to list of amino acid integers
    """
    ans = [rc.restypes.index(x) for x in aaseq]
    return ans


def get_structure_file_type(filename: str) -> str:
    if (
        filename.endswith(".pdb")
        or filename.endswith(".pdb.gz")
        or filename.endswith(".ent.gz")
    ):
        return "pdb"
    if filename.endswith(".cif") or filename.endswith(".cif.gz"):
        return "cif"
    raise Exception(f"Could not detect structure file format for {filename}")


# from OmegaFold version of residue_constants.py
# Compute a mask whether the group exists.
# (N, 8)
def residx_to_3(idx: int) -> str:
    return rc.restype_1to3[rc.restypes[idx]]
