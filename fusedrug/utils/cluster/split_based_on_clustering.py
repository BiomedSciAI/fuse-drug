"""
splits to train/val/test/etc. sets based on cluster_using_mmseqs.py output
"""

from frozendict import frozendict
from typing import Dict, List
from os.path import join, dirname, basename
import mmap
import numpy as np
from collections import defaultdict


def split(
    cluster_tsv: str,
    cluster_center_column_name: str,
    columns_names: List[str] = None,
    splits_desc: Dict = frozendict(train=0.90, val=0.05, test=0.05),
) -> Dict:
    """
    Gets a cluster tsv file (usually the output of cluster_using_mmseqs.py:cluster() call
    and distributes randomly the clusters into sets in the requested (statistical, not necessarily exact) proportions
    this function will generate one file per requested set, for example, if the input was /a/b/c/d/clustered.tsv
    you may get
    /a/b/c/d/train@clustered.tsv
    /a/b/c/d/val@clustered.tsv
    /a/b/c/d/test@clustered.tsv

    Args:
        cluster_tsv - a TSV (like csv but tab separated) at the expected columns structure (no title line)
        cluster_center_column_name - the name of the column that contains the cluster center representative
        columns_names - the name of the columns. If None, assumes that the first line contains the columns names

        for example:
            6usf_B  6usf_B
            6vkl_G  6vkl_G
            6wed_B  6wed_B
            6wed_B  6wee_A
            6wed_B  3lzo_A
            6wed_B  4nwn_H
            6wed_B  6wef_C
            6xds_A  6xds_A
            6y6x_LW 6y6x_LW
        splits_desc - a dictionary representing the names and proportions of the sets

        returns: a dictionary that maps the requested sets names to the generates files paths
    """
    files_names = {
        set_name: join(dirname(cluster_tsv), f"{set_name}@" + basename(cluster_tsv))
        for (set_name, _) in splits_desc.items()
    }
    files_handles = {set_name: open(filename, "wb") for (set_name, filename) in files_names.items()}
    members_num = defaultdict(int)
    centers_num = defaultdict(int)

    assigned_centers = {}

    with open(cluster_tsv, "rt") as f:
        mm_read = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # useful for massive files
        linenum = 0

        if columns_names is None:
            columns_names = mm_read.readline().decode().rstrip().split("\t")

        if cluster_center_column_name not in columns_names:
            raise Exception(f"Could not find {cluster_center_column_name} in columns: {columns_names}")
        column_index = columns_names.index(cluster_center_column_name)

        # write the columns names in all sets outputs
        for _, fh in files_handles.items():
            fh.write(("\t".join(columns_names) + "\n").encode())

        line = None
        while True:
            line = mm_read.readline()
            if line == b"":
                break

            center = line.decode().rstrip().split("\t")[column_index]
            # print('center=',center,'member=',member)

            if center in assigned_centers:
                use_set = assigned_centers[center]
                members_num[use_set] += 1
            else:
                use_set = _select_set(splits_desc)
                centers_num[use_set] += 1
                members_num[use_set] += 1
                assigned_centers[center] = use_set

            out_fh = files_handles[use_set]
            out_fh.write(line)

            if not linenum % 10 ** 5:
                print(linenum, line)

            linenum += 1

    print("sets summary:")
    for set_name in splits_desc.keys():
        print(f"set {set_name} has {centers_num[set_name]} centers (clusters)")
        print(f"set {set_name} has {members_num[set_name]} members")

    for _, fh in files_handles.items():
        fh.close()

    return files_names


def _select_set(splits_desc: Dict) -> str:
    rand_val = np.random.random()

    cumulative_region = 0.0

    for set_name, set_part in splits_desc.items():
        cumulative_region += set_part
        if rand_val <= cumulative_region:
            return set_name

    return set_name
