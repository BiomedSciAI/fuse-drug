from typing import List, Dict, Optional
from fusedrug.data.protein.structure.sabdab import load_sabdab_dataframe
import pandas as pd
from collections import namedtuple

try:
    import abnumber
except ImportError:
    print(
        "ERROR: had a problem importing abnumber, please install using 'conda install -c bioconda abnumber'"
    )
    raise

Antibody = namedtuple(
    "Antibody", "pdb_id index_within_pdb light_chain_id heavy_chain_id antigen_chain_id"
)


def get_antibody_regions(sequence: str, scheme: str = "chothia") -> Dict[str, str]:
    chain = abnumber.Chain(sequence, scheme=scheme)
    ans = {
        region: getattr(chain, region)
        for region in [
            "fr1_seq",
            "cdr1_seq",
            "fr2_seq",
            "cdr2_seq",
            "fr3_seq",
            "cdr3_seq",
            "fr4_seq",
        ]
    }
    return ans


def get_antibodies_info_from_sabdab(antibodies_pdb_ids: Optional[List[str]] = None) -> List[Antibody]:
    """
    Collects information on all provided antibodies_pdb_ids based on SabDab DB.

    """
    sabdab_df = load_sabdab_dataframe()
    if antibodies_pdb_ids is None:
        antibodies_pdb_ids = sabdab_df.pdb.unique().tolist()
    antibodies = []
    for pdb_id in antibodies_pdb_ids:
        found = sabdab_df[sabdab_df.pdb == pdb_id]
        for i in range(len(found)):
            row = found.iloc[i]
            if row.model != 0:
                print(
                    f"warning: found an antibody not in model 0, skipping (TODO: add support). For an entry in pdb_id={pdb_id}"
                )
                continue

            light_chain = row.Lchain
            heavy_chain = row.Hchain
            antigen_chain = row.antigen_chain
            if pd.isnull(light_chain) or pd.isnull(heavy_chain):
                print(
                    f"warning: at least one of the light chain or heavy chains are missing, skipping it. For an entry in pdb_id={pdb_id}"
                )
                continue

            antibodies.append(
                Antibody(
                    pdb_id=pdb_id,
                    index_within_pdb=i,
                    light_chain_id=light_chain,
                    heavy_chain_id=heavy_chain,
                    antigen_chain_id=antigen_chain,
                )
            )
    return antibodies


def get_omegafold_preprint_test_antibodies() -> List[str]:
    """
    https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1
    Figure 2
    """
    return [
        "7k7r",
        "7e6p",
        "7kpj",
        "6xjq",
        "7sjs",
        "7v5n",
        "7phu",
        "7phw",
        "7qji",
    ]
