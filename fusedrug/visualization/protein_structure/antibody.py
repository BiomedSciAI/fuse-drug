from typing import Optional, Dict, List
import abnumber
import openfold.np.protein as protein_utils
from openfold.np import residue_constants as rc

_default_regions_colors = dict(
    FR1="yellow",
    CDR1="red",
    FR2="yellow",
    CDR2="green",
    FR3="yellow",
    CDR3="blue",
    FR4="yellow",
)


def get_indices_colors_scheme_from_chain(
    openfold_prot: protein_utils.Protein,
    regions_colors: Optional[Dict[str, str]] = None,
    scheme: str="chothia",
    verbose: int = 1,
) -> List:
    """
    A helper function that helps to color different regions in an antibody chain

    Usage example: (also see highlight_antibody_regions.ipynb)

    from nglview.color import ColormakerRegistry
    cm = ColormakerRegistry

    heavy_color_scheme = get_indices_colors_scheme_from_chain(native_prot_heavy, regions_colors=dict(
        FR1='purple', CDR1='red', FR2='purple', CDR2='green', FR3='purple', CDR3='blue', FR4='purple',
    ))
    cm.add_selection_scheme('heavy', heavy_color_scheme)

    light_color_scheme = get_indices_colors_scheme_from_chain(native_prot_light, regions_colors=dict(
        FR1='yellow', CDR1='red', FR2='yellow', CDR2='green', FR3='yellow', CDR3='blue', FR4='yellow',
    ))
    cm.add_selection_scheme('light', light_color_scheme)
    """
    if regions_colors is None:
        regions_colors = _default_regions_colors

    aa_seq = "".join([rc.restypes[r] if r < 20 else "X" for r in openfold_prot.aatype])
    chain = abnumber.Chain(aa_seq, scheme=scheme)
    if verbose:
        print("is heavy chain? ", chain.is_heavy_chain())
    per_residue_region_data = [(aa, pos.get_region(), pos) for (pos, aa) in chain]
    color_scheme_desc = []
    for index, (aa, region, pos) in enumerate(per_residue_region_data):
        # region_indices[region].append(index)
        color_scheme_desc.append([regions_colors[region], str(index + 1)])

    return color_scheme_desc
