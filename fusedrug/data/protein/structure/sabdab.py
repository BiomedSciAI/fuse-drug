import os
from typing import Optional
import pandas as pd


def load_sabdab_dataframe(path: Optional[str] = None) -> pd.DataFrame:
    """
    loads a dataframe containing sabdab data

    To download 'sabdab_summary_all.tsv' you need to download it from https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/
    Or you can manually reach it by browsing to: https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/
        click on "Search Structures"
        -> "Get All Structures" -> "Get All Structures" -> "Downloads" -> click on "summary file" in the sentence:
            "Download the summary file for the structures selected by this search. See help for more details on file formats."

    """
    if path is None:
        if "SABDAB_DIR" not in os.environ:
            raise Exception("Could not find env. var SABDAB_DIR")
        path = os.path.join(os.environ["SABDAB_DIR"], "sabdab_summary_all.tsv")
    df = pd.read_csv(path, sep="\t")
    return df


class SAbDAb:
    def __init__(self, main_dataframe_path: str = None):
        self.df = load_sabdab_dataframe()

    def get_entry(self, pdb_id: str, heavy_chain_id: Optional[str] = None) -> pd.Series:
        if heavy_chain_id is not None:
            found = self.df.loc[
                (self.df.pdb == pdb_id) & (self.df.Hchain == heavy_chain_id)
            ]
        else:
            found = self.df.loc[self.df.pdb == pdb_id]

        if found.shape[0] == 0:
            raise Exception(
                f"could not find an entry for pdb_id={pdb_id} heavy_chain_id={heavy_chain_id}"
            )
        elif found.shape[0] > 1:
            raise Exception(
                f"found multiple entries for pdb_id={pdb_id} heavy_chain_id={heavy_chain_id}"
            )

        found = found.iloc[0]

        return found


if __name__ == "__main__":
    inst = SAbDAb()
    inst.get_entry(pdb_id="7vux", heavy_chain_id="H")
    inst.get_entry(pdb_id="7vux")
