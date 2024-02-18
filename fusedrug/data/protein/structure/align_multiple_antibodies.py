from os.path import join, dirname
from fusedrug.data.protein.structure.flexible_align_chains_structure import (
    flexible_align_chains_structure,
)
from jsonargparse import CLI
import pandas as pd
from typing import Optional
import numpy as np


def main(
    input_excel_filename: str,
    unique_id_column: str,
    reference_heavy_chain_pdb_filename_column: str,
    reference_heavy_chain_id_column: str,
    heavy_chain_pdb_filename_column: str,
    heavy_chain_id_column: str,
    light_chain_pdb_filename_column: str,
    light_chain_id_column: str,
    aligned_using_only_heavy_chain: bool = True,
    output_structure_file_prefix: str = "aligned_antibody_",
    output_excel_filename: Optional[str] = None,
    output_excel_aligned_heavy_chain_pdb_filename_column: str = "aligned_heavy_chain_pdb_filename",
    output_excel_aligned_heavy_chain_id_column: str = None,
    output_excel_aligned_light_chain_pdb_filename_column: str = "aligned_light_chain_pdb_filename",
    output_excel_aligned_light_chain_id_column: str = None,
) -> pd.DataFrame:

    assert (
        aligned_using_only_heavy_chain
    ), "only supporting aligned_using_only_heavy_chain=True for now. Note that flexible_align_chains_structure is indeed flexible enough to support this, if needed."

    df = pd.read_excel(input_excel_filename, index_col=unique_id_column)

    # base = '/dccstor/dsa-ab-cli-val-0/2024_feb_delivery/top_100_with_indels/antibody_dimers_af2_predicted_structure'
    # reference_heavy_chain = '/dccstor/dsa-ab-cli-val-0/targets/PD-1/7VUX/relaxed_complex/PD1_7VUX_H_eq.pdb'

    df[output_excel_aligned_heavy_chain_pdb_filename_column] = np.nan
    df[output_excel_aligned_heavy_chain_id_column] = np.nan
    df[output_excel_aligned_light_chain_pdb_filename_column] = np.nan
    df[output_excel_aligned_light_chain_id_column] = np.nan

    for index, row in df.iterrows():
        reference_heavy_chain_pdb_filename = row[
            reference_heavy_chain_pdb_filename_column
        ]
        reference_heavy_chain_id = row[reference_heavy_chain_id_column]
        # reference_light_chain_id = row[reference_light_chain_id_column]

        # heavy chain
        heavy_chain_pdb_filename = row[heavy_chain_pdb_filename_column]
        heavy_chain_id = row[heavy_chain_id_column]  # 'A'
        # light chain
        light_chain_pdb_filename = row[light_chain_pdb_filename_column]
        light_chain_id = row[light_chain_id_column]  # 'B'

        output_aligned_fn = join(
            dirname(heavy_chain_pdb_filename_column), output_structure_file_prefix
        )

        flexible_align_chains_structure(
            dynamic_ordered_chains=[(heavy_chain_pdb_filename, heavy_chain_id)],
            apply_rigid_transformation_to_dynamic_chain_ids=[
                (heavy_chain_pdb_filename, heavy_chain_id),
                (light_chain_pdb_filename, light_chain_id),
            ],
            static_ordered_chains=[
                (reference_heavy_chain_pdb_filename, reference_heavy_chain_id)
            ],
            output_pdb_filename_extentionless=output_aligned_fn,
        )

    if output_excel_filename is not None:
        df.to_excel(output_excel_filename)
        print("saved ", output_excel_filename)

    return df


if __name__ == "__main__":
    CLI(main)
