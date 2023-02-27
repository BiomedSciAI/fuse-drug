import os
from typing import Optional
import pandas as pd

def load_sabdab_dataframe(path:Optional[str]=None):
    """
    loads a dataframe containing sabdab data

    To download 'sabdab_summary_all.tsv' you need to download it from https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/
    Or you can manually reach it by browsing to: https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/
        click on "Search Structures"
        -> "Get All Structures" -> "Get All Structures" -> "Downloads" -> click on "summary file" in the sentence: 
            "Download the summary file for the structures selected by this search. See help for more details on file formats."
        
    """
    if path is None:
        if 'SABDAB_DIR' not in os.environ:
            raise Exception('Could not find env. var SABDAB_DIR')
        path = os.path.join(os.environ['SABDAB_DIR'],'sabdab_summary_all.tsv')
    df = pd.read_csv(path, sep='\t')
    return df

