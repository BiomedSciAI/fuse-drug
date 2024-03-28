from typing import Any, Optional, Union, List, Dict, Tuple
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from fuse.utils import NDict
from fuse.data import DatasetDefault
from fuse.data.ops.caching_tools import run_cached_func
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault


def fix_df_types(df: pd.DataFrame) -> pd.DataFrame:
    if "source_dataset_activity_id" in df.columns:
        df.source_dataset_activity_id = df.source_dataset_activity_id.astype("string")

    if "ligand_id" in df.columns:
        df.ligand_id = df.ligand_id.astype("string")

    if "target_id" in df.columns:
        df.target_id = df.target_id.astype("string")
    return df


def set_activity_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index(
        ["source_dataset_versioned_name", "source_dataset_activity_id"],
        drop=False,
        inplace=True,
    )
    return df


def itemify(x: Any) -> Any:
    try:
        x.item()
    except:
        pass
    return x


def dti_binding_dataset(
    pairs_tsv: str,
    ligands_tsv: str,
    targets_tsv: str,
    split_tsv: str = None,
    use_folds: Optional[Union[List, str]] = None,
    pairs_columns_to_extract: Optional[List[str]] = None,
    pairs_rename_columns: Optional[Dict[str, str]] = None,
    ligands_columns_to_extract: Optional[List[str]] = None,
    ligands_rename_columns: Optional[Dict[str, str]] = None,
    targets_columns_to_extract: Optional[List[str]] = None,
    targets_rename_columns: Optional[Dict[str, str]] = None,
    get_indices_per_class: bool = False,
    **kwargs: Any,
) -> DatasetDefault:
    """_summary_

    Args:
        pairs_tsv (str): path to tab-separated pairs csv (tsv) file
        ligands_tsv (str): path to tab-separated ligands csv (tsv) file
        targets_tsv (str): path to tab-separated targets csv (tsv) file
        split_tsv (str, optional): _description_. Defaults to None.
        use_folds (Union[List, str], optional): A list of folds (as defined in split_tsv) to use. Defaults to None: use all folds.
        pairs_columns_to_extract (_type_, optional): _description_. Defaults to None.
        pairs_rename_columns (_type_, optional): _description_. Defaults to None.
        ligands_columns_to_extract (_type_, optional): _description_. Defaults to None.
        ligands_rename_columns (_type_, optional): _description_. Defaults to None.
        targets_columns_to_extract (_type_, optional): _description_. Defaults to None.
        targets_rename_columns (_type_, optional): _description_. Defaults to None.

    Returns:
        DatasetDefault: _description_
    """

    # load tsvs with opional caching:
    _args = [pairs_tsv, ligands_tsv, targets_tsv, split_tsv, use_folds]

    if "cache_dir" in kwargs and kwargs["cache_dir"] is not None:
        ans_dict = run_cached_func(
            kwargs["cache_dir"], _load_dataframes, *_args, **kwargs
        )
    else:
        ans_dict = _load_dataframes(*_args, **kwargs)

    pairs_df = ans_dict["pairs"]
    ligands_df = ans_dict["ligands"]
    targets_df = ans_dict["targets"]

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

    # append custom pipeline:
    if "dynamic_pipeline" in kwargs and kwargs["dynamic_pipeline"] is not None:
        dynamic_pipeline += kwargs["dynamic_pipeline"]

    dynamic_pipeline = PipelineDefault("DTI dataset", dynamic_pipeline)

    dataset = DatasetDefault(
        # sample_ids=None,
        sample_ids=pairs_df.index,
<<<<<<< HEAD
        dynamic_pipeline=dynamic_pipeline
=======
        dynamic_pipeline=dynamic_pipeline,
>>>>>>> 6f5d6f69dda5c67a06cee5257dec4179774ba1ab
    )  # TODO: sample_ids here should be either pairs_df.index, or len(pairs_df)
    dataset.create()
    if get_indices_per_class:
        indices_per_class = {
            label: [
                pairs_df.index.get_loc(key)
                for key in pairs_df[pairs_df.activity_label == label].index
            ]
            for label in pairs_df.activity_label.unique()
        }
        return dataset, indices_per_class
    else:
        return dataset


def dti_binding_dataset_combined(
    pairs_tsv: str,
    ligands_tsv: str,
    targets_tsv: str,
    split_tsv: str = None,
    use_folds: Optional[Union[List, str]] = None,
    pairs_columns_to_extract: Optional[List[str]] = None,
    pairs_rename_columns: Optional[Dict[str, str]] = None,
    ligands_columns_to_extract: Optional[List[str]] = None,
    ligands_rename_columns: Optional[Dict[str, str]] = None,
    targets_columns_to_extract: Optional[List[str]] = None,
    targets_rename_columns: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> DatasetDefault:
    """returns a combined dataset, where pairs, targets, ligands and split information is found in a single dataframe

    Args:
        pairs_tsv (str): path to tab-separated pairs csv (tsv) file
        ligands_tsv (str): path to tab-separated ligands csv (tsv) file
        targets_tsv (str): path to tab-separated targets csv (tsv) file
        split_tsv (str, optional): _description_. Defaults to None.
        use_folds (Union[List, str], optional): A list of folds (as defined in split_tsv) to use. Defaults to None: use all folds.
        pairs_columns_to_extract (_type_, optional): _description_. Defaults to None.
        pairs_rename_columns (_type_, optional): _description_. Defaults to None.
        ligands_columns_to_extract (_type_, optional): _description_. Defaults to None.
        ligands_rename_columns (_type_, optional): _description_. Defaults to None.
        targets_columns_to_extract (_type_, optional): _description_. Defaults to None.
        targets_rename_columns (_type_, optional): _description_. Defaults to None.

    Returns:
        DatasetDefault: _description_
    """
    ligand_suffix = "_ligands"
    target_suffix = "_targets"
    suffixes = [ligand_suffix, target_suffix]
    # load tsvs with opional caching:
    _args = [pairs_tsv, ligands_tsv, targets_tsv, split_tsv, use_folds]

    if "cache_dir" in kwargs and kwargs["cache_dir"] is not None:
        ans_dict = run_cached_func(
            kwargs["cache_dir"], _load_dataframes, *_args, **kwargs
        )
    else:
        ans_dict = _load_dataframes(*_args, combine=True, suffixes=suffixes, **kwargs)

    pairs_df = ans_dict["pairs"]

    # Since _load_dataframes with combine == True may change some (overlapping) column names, we need to correct the following:
    ligands_columns_to_extract = [
        c if c in pairs_df.columns else c + ligand_suffix
        for c in ligands_columns_to_extract
    ]
    ligands_rename_columns = {
        (k if k in pairs_df.columns else k + ligand_suffix): v
        for k, v in ligands_rename_columns.items()
    }
    targets_columns_to_extract = [
        c if c in pairs_df.columns else c + target_suffix
        for c in targets_columns_to_extract
    ]
    targets_rename_columns = {
        (k if k in pairs_df.columns else k + target_suffix): v
        for k, v in targets_rename_columns.items()
    }

    columns_to_extract = (
        pairs_columns_to_extract
        + ligands_columns_to_extract
        + targets_columns_to_extract
    )
    rename_columns = {
        **pairs_rename_columns,
        **ligands_rename_columns,
        **targets_rename_columns,
    }

    dynamic_pipeline = [
        (
            OpReadDataframe(
                pairs_df,
                columns_to_extract=columns_to_extract,
                rename_columns=rename_columns,
                key_column=None,
            ),
            dict(prefix="data"),
        ),
    ]
    dynamic_pipeline = PipelineDefault("DTI dataset", dynamic_pipeline)

    dataset = DatasetDefault(
        sample_ids=pairs_df.index, dynamic_pipeline=dynamic_pipeline
    )
    dataset.create()

    return dataset


class DTIBindingDataset(Dataset):
    """
    PyTorch dataset for DTI tasks
    """

    def __init__(
        self,
        pairs_tsv: str,
        ligands_tsv: str,
        targets_tsv: str,
        splits_tsv: str = None,
        use_folds: Optional[Union[List, str]] = None,
        keep_activity_labels: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Params:
            pairs_tsv: path to the activity pairs tsv (like csv but tab separated)
            ligands_tsv: path to the ligands tsv
            targets_tsv: path to the targets tsv
            splits_tsv: splits_tsv and use_folds can be used together to select only a subset of pairs_tsv - useful for selecting only training folds, or a validation set, etc.
            Use None for both (default) to use the entirety of the pairs_tsv
            splits_tsv points to a tsv file containing the folds description

            use_folds: splits_tsv and use_folds can be used together to select only a subset of pairs_tsv - useful for selecting only training folds, or a validation set, etc.
                Use None for both (default) to use the entirety of the pairs_tsv
                use_folds provides a list (or a single string) describing the folds to use.
                For example: use_folds=['fold0','fold1','fold3','fold4']
                Another example: use_folds='test_set'

            keep_activity_labels: keep only activity_label from this list
                provide None (default) to keep all.
                example usage: keep_activity_labels=['Active','Inactive']



            cache_dir: optional - set a path if you want the constructor calculations to be cached.
            Note - caching takes into consideration the arguments and the *direct* code. If any deeper code
            changes it will be unnoticed and the cache will be stale!
            USE WITH CAUTION!
        """

        _args = [
            pairs_tsv,
            ligands_tsv,
            targets_tsv,
            splits_tsv,
            use_folds,
            keep_activity_labels,
        ]

        print(
            f"creating dataset with:\n\tpairs_tsv={pairs_tsv},\n\tligands_tsv={ligands_tsv},\n\ttargets_tsv={targets_tsv},\n\tsplits_tsv={splits_tsv},\n\tuse_folds={use_folds},\n\tkeep_activity_labels={keep_activity_labels}"
        )

        if cache_dir is not None:
            ans_dict = run_cached_func(cache_dir, _load_dataframes, *_args)
        else:
            ans_dict = _load_dataframes(*_args)

        self._pairs = ans_dict["pairs"]
        self._ligands = ans_dict["ligands"]
        self._targets = ans_dict["targets"]

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> dict:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: Union[int, Tuple[str, str]]) -> dict:
        """
        two options are supported:
        1. if index is an integer, the row with this index will be loaded (using .iloc).
        This is considered unsafe, because the order of the table might change due to operations,
        and it might create a shift which breaks sync with, e.g., a sampler trying to balance things

        2. A tuple ([source_dataset_versioned_name:str], [source_dataset_activity_id:str])
        """

        ### keeping for now  it helps profiling
        # return dict(
        #     ligand_str='c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1',
        #     target_str='MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK',
        #     ground_truth_activity_value=0.7,
        #     ground_truth_activity_label='Active',
        # )

        if isinstance(index, (int, np.integer)):
            row = self._pairs.iloc[index]
        else:
            assert isinstance(index, tuple)
            assert 2 == len(index)
            source_dataset_versioned_name, source_dataset_activity_id = index
            row = self._pairs.loc[
                source_dataset_versioned_name, source_dataset_activity_id
            ]

        ground_truth_activity_value = itemify(row["activity_value"])
        if not np.isscalar(ground_truth_activity_value):
            try:
                ground_truth_activity_value = float(ground_truth_activity_value)
                print(f"converted from nonscalar: {ground_truth_activity_value}")
            except:
                raise Exception(
                    f'Could not convert activity value "{ground_truth_activity_value}" to float!'
                )

        ground_truth_activity_label = itemify(row["activity_label"])

        ligand_id = itemify(row["ligand_id"])
        target_id = itemify(row["target_id"])

        ligand_row = self._ligands.loc[ligand_id]

        # #### remove this ! trying with offset to see if training still works well
        # try:
        #     ligand_row = self._ligands.iloc[int(ligand_id)+30]
        # except:
        #     print("DEBUG::had issue accessing self._ligands.iloc[ligand_id+30]")
        #     ligand_row = self._ligands.loc[ligand_id]

        ligand_str = ligand_row.canonical_smiles

        if not isinstance(ligand_str, str):
            raise Exception(
                f"ERROR!!! expected a string for canonical_smiles !!! instead got {type(ligand_str)}for index {index} - ligand row = {ligand_row}"
            )

        target_row = self._targets.loc[target_id]
        target_str = target_row.canonical_aa_sequence

        if not isinstance(target_str, str):
            raise Exception(
                f"ERROR!!! expected a string for canonical_aa_sequence !!! instead got {type(target_str)}for index {index} - target row = {target_row}"
            )

        #     ligand_str='c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1',
        #     target_str='MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK',
        if not isinstance(ground_truth_activity_label, str):
            raise Exception(
                f"ERROR!!! expected a string for ground_truth_activity_label !!! instead got {type(ground_truth_activity_label)} for index {index}"
            )

        return dict(
            ligand_str=ligand_str,
            # debug - use a constant ligand
            # ligand_str='c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1',
            target_str=target_str,
            # debug - use a constant target
            # target_str='MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK',
            ground_truth_activity_value=ground_truth_activity_value,
            ground_truth_activity_label=ground_truth_activity_label,
        )


def _load_dataframes(
    pairs_tsv: str,
    ligands_tsv: str,
    targets_tsv: str,
    splits_tsv: str = None,
    use_folds: Optional[Union[List, str]] = None,
    keep_activity_labels: List[str] = None,
    combine: Optional[bool] = False,
    suffixes: Optional[List[str]] = ["_ligands", "_targets"],
    require_non_null_ligand_columns: Optional[List[str]] = [
        "canonical_smiles",
    ],
    **kwargs: Any,
) -> dict:
    """
    Loads pairs, ligands and targets, and optionally filters in a subset

    We use it mostly with run_cached_func() to minimize time and memory footfprint
    Args:
        ligands_tsv:
        targets_tsv:
        splits_tsv:
        use_folds: Optionally provide a list of folds to keep, pass None (default) to keep all
        keep_activity_labels: Optionally provide a list of activity labels to keep, pass None (default) to keep all
        combine (Optional[bool], optional): If True, all dataframes are combined into return["pairs"]. Defaults to False
        suffixes (Optional[List[str]], optional): Suffixes to be assigned to overlapping ligand and target columns respectively.
            Defaults to ['_ligands', '_targets'].
        require_non_null_ligand_columns (Optional[List[str]]): remove rows from the ligands dataframe for which the values in these columns (if exist) are null
    returns: The following dictionary:
        {
            "pairs": pairs df,
            "ligands": ligands df,
            "targets": targets df
    """

    assert isinstance(pairs_tsv, str)
    print(f"loading {pairs_tsv}")
    _pairs = pd.read_csv(pairs_tsv, sep="\t")
    _pairs = fix_df_types(_pairs)
    # _pairs = concat_full_activity_col(_pairs)
    # set_activity_multiindex(_pairs)
    # A.G note: disabled this so that merge happens on columns.
    # this is important where one of the index columns needs to be extracted. then,
    # if the merge is on index, we have to drop the columns (otherwise there will be ambiguity)
    # so they cannot be extracted.
    print(f"pairs num: {len(_pairs)}")

    if splits_tsv is not None and use_folds is None:
        raise Exception(
            f"splits_tsv was provided ({splits_tsv}) but no use_folds provided"
        )

    if use_folds is not None and splits_tsv is None:
        raise Exception(
            f"use_folds was provided ({use_folds}) but no splits_tsv provided"
        )

    if splits_tsv is not None:
        print(f"loading split file {splits_tsv}")
        _splits = pd.read_csv(splits_tsv, sep="\t")
        _splits = fix_df_types(_splits)
        # set_activity_multiindex(_splits)
        # _splits = concat_full_activity_col(_splits)
        print(f"split contains {len(_splits)} rows")

        if len(_splits) != len(_pairs):
            raise Exception(
                f"split file {splits_tsv} contains {len(_splits)} rows while the pairs file {pairs_tsv} contains {len(_pairs)} rows! they should be identical."
            )

        _pairs_MERGED = _pairs.merge(
            _splits,
            how="inner",
            # on='full_activity_id',
            on=["source_dataset_versioned_name", "source_dataset_activity_id"],
            suffixes=[None, "_split_duplicate"],
        )

        _pairs = _pairs_MERGED
        del _pairs_MERGED

        _pairs.reset_index(inplace=True)
        set_activity_multiindex(_pairs)
        set_activity_multiindex(_splits)

        _pairs = _pairs[_pairs.split.isin(use_folds)]
        print(f"use_folds={use_folds} keeps {len(_pairs)} rows")

    set_activity_multiindex(_pairs)
    assert isinstance(ligands_tsv, str)
    print(f"loading {ligands_tsv}")
    _ligands = pd.read_csv(ligands_tsv, sep="\t")
    _ligands = fix_df_types(_ligands)
    _ligands.set_index("ligand_id", inplace=True)
    print(f"ligands num: {len(_ligands)}")

    for ligand_col in require_non_null_ligand_columns:
        if ligand_col in _ligands.columns:
            _ligands = _ligands[~_ligands[ligand_col].isnull()]
            print(
                f"ligands num after keeping only ligands with non-NaN {ligand_col}: {len(_ligands)}"
            )

    assert isinstance(targets_tsv, str)
    print(f"loading {targets_tsv}")
    _targets = pd.read_csv(targets_tsv, sep="\t")
    _targets = fix_df_types(_targets)
    _targets.set_index("target_id", inplace=True)
    print(f"tagets num: {len(_targets)}")
    _targets = _targets[_targets.index.to_series().notna()]
    print(f"tagets num after keeping only non-NaN ids: {len(_targets)}")
<<<<<<< HEAD

=======
>>>>>>> 6f5d6f69dda5c67a06cee5257dec4179774ba1ab

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

    if combine:
        ligand_id_key = _ligands.index.name
        if len(_ligands.index.unique()) != len(_ligands):
            raise Exception("ligands dataframe must have unique index values")
        affinities_with_ligands = _pairs.merge(_ligands, on=ligand_id_key)
        target_id_key = _targets.index.name
        if len(_targets.index.unique()) != len(_targets):
            raise Exception("targets dataframe must have unique index values")
        _pairs = affinities_with_ligands.merge(
            _targets, on=target_id_key, suffixes=suffixes
        )
        print(f"pairs num after merging with ligands and targets: {len(_pairs)}")

    return dict(
        pairs=_pairs,
        ligands=_ligands,
        targets=_targets,
    )


def _fill_in_dummy_sample(sample_dict: NDict) -> NDict:
    _ligand_size = 696
    sample_dict["data.input.tokenized_ligand"] = np.random.randint(
        0, 3000, size=_ligand_size
    )
    sample_dict["data.input.tokenized_ligand_attention_mask"] = [True] * _ligand_size

    _target_size = 2536
    sample_dict["data.input.tokenized_target"] = np.random.randint(
        0, 33, size=_target_size
    )
    sample_dict["data.input.tokenized_target_attention_mask"] = [True] * _target_size

    sample_dict["data.gt.activity_value"] = np.random.rand(1).item()
    sample_dict["data.gt.activity_label_class_idx"] = np.random.randint(0, 5)
    return sample_dict
