from typing import Tuple, Optional, Union, List
import pandas as pd
from fuse.data import DatasetDefault
from fuse.data.ops.caching_tools import run_cached_func
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fusedrug.data.molecule.ops.featurizer_ops import FeaturizeDrug
from fusedrug.data.protein.ops.featurizer_ops import FeaturizeTarget


def fix_df_types(df):
    if "source_dataset_activity_id" in df.columns:
        df.source_dataset_activity_id = df.source_dataset_activity_id.astype("string")

    if "ligand_id" in df.columns:
        df.ligand_id = df.ligand_id.astype("string")

    if "target_id" in df.columns:
        df.target_id = df.target_id.astype("string")
    return df


def set_activity_multiindex(df):
    df.set_index(
        ["source_dataset_versioned_name", "source_dataset_activity_id"], inplace=True
    )
    return df


def itemify(x):
    try:
        x.item()
    except:
        pass
    return x


def dti_binding_dataset_with_featurizers(
    pairs_tsv: str,
    ligands_tsv: str,
    targets_tsv: str,
    split_tsv: str = None,
    pairs_columns_to_extract=None,
    pairs_rename_columns=None,
    ligands_columns_to_extract=None,
    ligands_rename_columns=None,
    targets_columns_to_extract=None,
    targets_rename_columns=None,
    **kwargs,
) -> DatasetDefault:

    # custom imlpementation based on fuse.data.interaction.drug_target.datasets.dti_binding_datasets
    # to allow featurizer ops that require the ligand and target strings during initialization

    # load tsvs with opional caching:
    _args = [pairs_tsv, ligands_tsv, targets_tsv]

    if "cache_dir" in kwargs and kwargs["cache_dir"] is not None:
        cache_dir = kwargs["cache_dir"]
        del kwargs["cache_dir"]
        ans_dict = run_cached_func(cache_dir, _load_dataframes, *_args, **kwargs)
    else:
        ans_dict = _load_dataframes(*_args, **kwargs)

    pairs_df = ans_dict["pairs"]
    ligands_df = ans_dict["ligands"]
    targets_df = ans_dict["targets"]

    if pairs_df.index.duplicated().sum() > 0:
        print(
            f"There are {pairs_df.index.duplicated().sum()} elements with duplicate indices in pairs_df. removing them..."
        )
        pairs_df = pairs_df[~pairs_df.index.duplicated(keep="first")]

    if ligands_df.index.duplicated().sum() > 0:
        print(
            f"There are {ligands_df.index.duplicated().sum()} elements with duplicate indices in ligands_df. removing them..."
        )
        ligands_df = ligands_df[~ligands_df.index.duplicated(keep="first")]

    if targets_df.index.duplicated().sum() > 0:
        print(
            f"There are {targets_df.index.duplicated().sum()} elements with duplicate indices in targets_df. removing them..."
        )
        targets_df = targets_df[~targets_df.index.duplicated(keep="first")]

    dynamic_pipeline = [
        (
            OpReadDataframe(
                pairs_df,
                columns_to_extract=pairs_columns_to_extract,
                rename_columns=pairs_rename_columns,
                key_column=None,
            ),
            {},
        ),
        (
            OpReadDataframe(
                ligands_df,
                columns_to_extract=ligands_columns_to_extract,
                rename_columns=ligands_rename_columns,
                key_column=None,
                key_name="ligand_id",
            ),
            {},
        ),
        (
            OpReadDataframe(
                targets_df,
                columns_to_extract=targets_columns_to_extract,
                rename_columns=targets_rename_columns,
                key_column=None,
                key_name="target_id",
            ),
            {},
        ),
    ]

    all_drugs = list(
        set(
            [
                dynamic_pipeline[1][0]._data[item]["data.drug"]
                for item in dynamic_pipeline[1][0]._data
            ]
        )
    )
    all_targets = list(
        set(
            [
                dynamic_pipeline[2][0]._data[item]["data.target"]
                for item in dynamic_pipeline[2][0]._data
            ]
        )
    )
    dynamic_pipeline += [
        (
            FeaturizeDrug(
                dataset=None,
                all_drugs=all_drugs,
                featurizer=kwargs["drug_featurizer"],
                debug=kwargs["featurizer_debug_mode"],
            ),
            dict(key_out_ligand="data.drug"),
        ),
        (
            FeaturizeTarget(
                dataset=None,
                all_targets=all_targets,
                featurizer=kwargs["target_featurizer"],
                debug=kwargs["featurizer_debug_mode"],
            ),
            dict(key_out_target="data.target"),
        ),
    ]
    # append custom pipeline:
    if "dynamic_pipeline" in kwargs and kwargs["dynamic_pipeline"] is not None:
        dynamic_pipeline += kwargs["dynamic_pipeline"]

    dynamic_pipeline = PipelineDefault("DTI dataset", dynamic_pipeline)

    dataset = DatasetDefault(
        sample_ids=None, dynamic_pipeline=dynamic_pipeline
    )  # list(dynamic_pipeline._ops_and_kwargs[0][0]._data.keys())
    dataset.create()

    return dataset, pairs_df


def _load_dataframes(
    pairs_tsv: str,
    ligands_tsv: str,
    targets_tsv: str,
    splits_tsv: str = None,
    use_folds: Optional[Union[List, str]] = None,
    keep_activity_labels: List[str] = None,
    **kwargs,
):
    """
    Loads pairs, ligands and targets, and optionally filters in a subset

    We use it mostly with run_cached_func() to minimize time and memory footfprint
    Args:
        ligands_tsv:
        targets_tsv:
        splits_tsv:
        use_folds: Optionally provide a list of folds to keep, pass None (default) to keep all
        keep_activity_labels: Optionally provide a list of activity labels to keep, pass None (default) to keep all
    """

    assert isinstance(pairs_tsv, str)
    print(f"loading {pairs_tsv}")
    _pairs = pd.read_csv(pairs_tsv, sep="\t")
    _pairs = fix_df_types(_pairs)
    # _pairs = concat_full_activity_col(_pairs)
    set_activity_multiindex(_pairs)
    print(f"pairs num: {len(_pairs)}")

    if splits_tsv is not None:
        if use_folds is None:
            raise Exception(
                f"splits_tsv was provided ({splits_tsv}) but no use_folds provided"
            )

    if use_folds is not None:
        if splits_tsv is None:
            raise Exception(
                f"use_folds was provided ({use_folds}) but no splits_tsv provided"
            )

    if splits_tsv is not None:
        print(f"loading split file {splits_tsv}")
        _splits = pd.read_csv(splits_tsv, sep="\t")
        _splits = fix_df_types(_splits)
        set_activity_multiindex(_splits)
        # _splits = concat_full_activity_col(_splits)
        print(f"it contains {len(_splits)} rows")

        if len(_splits) != len(_pairs):
            raise Exception(
                f"split file {splits_tsv} contains {len(_splits)} rows while the pairs file {pairs_tsv} contains {len(_pairs)} rows! they should be identical."
            )

        _pairs_MERGED = _pairs.merge(
            _splits,
            how="inner",
            # on='full_activity_id',
            on=["source_dataset_versioned_name", "source_dataset_activity_id"],
        )

        _pairs = _pairs_MERGED
        del _pairs_MERGED

        _pairs.reset_index(inplace=True)
        set_activity_multiindex(_pairs)

        _pairs = _pairs[_pairs.split.isin(use_folds)]
        print(f"use_folds={use_folds} keeps {len(_pairs)} rows")

    assert isinstance(ligands_tsv, str)
    print(f"loading {ligands_tsv}")
    _ligands = pd.read_csv(ligands_tsv, sep="\t")
    _ligands = fix_df_types(_ligands)
    _ligands.set_index("ligand_id", inplace=True)
    print(f"ligands num: {len(_ligands)}")
    _ligands = _ligands[~_ligands.canonical_smiles.isnull()]
    print(
        f"ligands num after keeping only ligands with non-NaN canonical_smiles: {len(_ligands)}"
    )

    assert isinstance(targets_tsv, str)
    print(f"loading {targets_tsv}")
    _targets = pd.read_csv(targets_tsv, sep="\t")
    _targets = fix_df_types(_targets)
    _targets.set_index("target_id", inplace=True)
    print(f"tagets num: {len(_targets)}")

    print(
        f"pairs num before keeping only pairs with ligands found in the (preprocessed) ligands table: {len(_pairs)}"
    )
    _pairs = _pairs[_pairs.ligand_id.isin(_ligands.index)]
    print(
        f"pairs num after keeping only pairs with ligands found in the (preprocessed) ligands table: {len(_pairs)}"
    )

    _pairs = _pairs[_pairs.target_id.isin(_targets.index)]
    print(
        f"pairs num after keeping only pairs with target found in the (preprocessed) targets table: {len(_pairs)}"
    )

    if keep_activity_labels is not None:
        _pairs = _pairs[_pairs.activity_label.isin(keep_activity_labels)]
        print(
            f"pairs num after keeping only activity_label in {keep_activity_labels}: {len(_pairs)}"
        )

    return dict(
        pairs=_pairs,
        ligands=_ligands,
        targets=_targets,
    )
