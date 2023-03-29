import mmap
from collections import defaultdict
import os
from typing import List


def create_balanced_sampling_tsv(
        cluster_tsv: str, 
        output_balanced_tsv: str,
        cluster_center_column_name: str,
        
        columns_names: List[str] = None,    
        ) -> None:
    """
    processes cluster_tsv (see args for expected format) and generates a new tsv with an added column that
        represents what chance should it have for sampling, if you want to remove the bias of sampling too frequently from large clusters.

    Args:
        cluster_tsv - a TSV (like csv but tab separated) at the expected columns structure (no title line)
            

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

        output_balanced_tsv - a TSV format, with an added balanced_sampling column, for example:
            6usf_B  6usf_B 0.01
            6vkl_G  6vkl_G 0.01
            6wed_B  6wed_B 0.004
            6wed_B  6wee_A 0.004
            6wed_B  3lzo_A 0.004
            6wed_B  4nwn_H 0.004
            6wed_B  6wef_C 0.004
            6xds_A  6xds_A 0.01
            6y6x_LW 6y6x_LW 0.01

        cluster_center_column_name - the name of the column that contains the cluster center representative
        
        columns_names - the name of the columns. If None, assumes that the first line contains the columns names

    """

    os.makedirs(os.path.dirname(output_balanced_tsv), exist_ok=True)

    with open(cluster_tsv, "rt") as f:
        mm_read = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # useful for massive files
        cluster_sizes = defaultdict(int)

        if columns_names is None:
            columns_names = mm_read.readline().decode().rstrip().split('\t')

        if not cluster_center_column_name in columns_names:
            raise Exception(f'Could not find center column: {cluster_center_column_name} in columns: {columns_names}')
        center_index = columns_names.index(cluster_center_column_name)

        
        
        




        linenum = 0
        print("going over the cluster info to calculate total count and cluster sizes")
        while True:
            line = mm_read.readline()
            if line == b"":
                break

            parts = line.decode().rstrip().split("\t")
            
            center = parts[center_index]
            
            

            cluster_sizes[center] += 1

            if not linenum % 10**5:
                print(linenum, line)

            linenum += 1

    total_seen = linenum

    print(f"total seen samples: {total_seen}")
    print(f"total clusters seen: {len(cluster_sizes)}")

    cluster_sizes = {k: total_seen / d for (k, d) in cluster_sizes.items()}

    new_total = sum(cluster_sizes.values())

    cluster_sizes = {k: d / new_total for (k, d) in cluster_sizes.items()}

    with open(cluster_tsv, "rt") as f:
        mm_read = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # useful for massive files
        with open(output_balanced_tsv, "wt") as outfh:
            print(f"writing {output_balanced_tsv}")
            outfh.write('\t'.join(columns_names+['inverse_balance_proportion'])+'\n')
            linenum = 0
            while True:
                line = mm_read.readline()
                if line == b"":
                    break
                parts = line.decode().rstrip().split("\t") ##make this and next line more efficient, just copy fulle line as pure text, rstrip() and add '\t' + value + '\n'
                outfh.write("\t".join( parts + [f"{cluster_sizes[center]}"]) + "\n")

                if not linenum % 10**5:
                    print(linenum, line)

                linenum += 1
