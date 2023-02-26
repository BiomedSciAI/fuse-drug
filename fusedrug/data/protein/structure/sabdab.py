import os
from typing import Optional
import pandas as pd

def load_sabdab_dataframe(path:Optional[str]=None):
    if path is None:
        if 'SABDAB_DIR' not in os.environ:
            raise Exception('Could not find env. var SABDAB_DIR')
        path = os.path.join(os.environ['SABDAB_DIR'],'sabdab_summary_all.tsv')
    df = pd.read_csv(path, sep='\t')
    return df

