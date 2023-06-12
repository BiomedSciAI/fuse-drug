from fuse.data.ops import get_function_call_str
import shutil
import hashlib
import subprocess
from fuse.utils.file_io import save_text_file_safe, read_text_file
import os
from typing import Dict, Optional
from os.path import join
import yaml
import pandas as pd


def cached_cluster(output_dir: str, force_rebuild: bool = False, **kwargs: dict) -> Dict[str, str]:
    """
    Uses mmseqs to:

    1. Remove 100% sequence identity duplicates
    2. Cluster the remaining unique sequences into multiple clusters (e.g. with 70% sequence similarity threshold )
        This is useful for multiple purposes:
        a. Creating cross validation/test splits that evaluate and demonstrate generalizability
        b. Balanced sampling during training, sampling with inverse proportion to cluster size. (Similar to Class Balancing)

    Note: depends on availability of mmseqs binary
        Please install mmseqs2 . See install instructions here: https://github.com/soedinglab/MMseqs2
        Also note that you can download prebuilt static binaries such as: https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz - extract and add the binary to your system PATH.
        The version we used was MMseqs2 Version: 13.45111
        Note: we've experienced "segmentation fault" when not using -c 1.0 (and keeping it at default value)

    Args:
        force_rebuild: rebuilds the data even if the hash indicates that it was already cached.
        input_fasta_filename: a fasta with an entry per molecular entity
        output_dir: where the output will be generated
        cluster_min_sequence_identity: the minimal sequence identity for member within a cluster
        threads: number of threads (for multithreading). if None -> by default it will use as many as the system has
        cluster_method: any of 'cluster', 'linclust':
            'cluster' is the "vanilla" one
            'linclust' is faster (claims linear runtime) but less accurate. Might be suitable for massive data.
                NOTE: I've compared cluster and linclust results for the deduplication phase, and results aren't identical, which means it probably misses few identical cases.
        deduplicate: if False, deduplication step will be skipped and clustering will be done directly on the input
        kmer_per_seq: Sets the number of k-mers selected per sequence in a "linclust" cluster_method. More k-mers per sequences results in a higher sensitivity.
        split_memory_limit: (optional), limit the memory usage. should be max 70% of system's available RAM

    For more information visit here -> https://mmseqs.com/latest/userguide.pdf
    Note - this function wraps cluster_impl() to allow caching
    """
    os.makedirs(output_dir, exist_ok=True)

    str_repr = get_function_call_str(cluster_impl, _ignore_kwargs_names=["num_workers"], _include_code=False, **kwargs)

    hash_value = hashlib.md5(str_repr.encode()).hexdigest()
    already_created_hash_filename = join(output_dir, "created_hash")
    already_created_description_filename = join(output_dir, "created_desc")
    cached_answer_yaml = join(output_dir, "cached_cluster_answer.yaml")

    rebuild_needed = False
    if os.path.isfile(already_created_hash_filename):
        existing_hash_value = read_text_file(already_created_hash_filename)
        if hash_value == existing_hash_value:
            if force_rebuild:
                print(
                    f"cluster::Hash value on disk matches the calculated one ({hash_value}), but force_rebuild was requested so rebuilding anyway."
                )
                rebuild_needed = True
        else:
            raise Exception("hash value on disk is a mismatch! please delete the directory manually and rerun.")

        existing_desc = read_text_file(already_created_description_filename)
    else:
        rebuild_needed = True

    if (not force_rebuild) and (not rebuild_needed):
        print(f"Hash value indicates that it was already built, not rebuilding. See description: {existing_desc}")
        with open(cached_answer_yaml, "rt") as f:
            ans = yaml.safe_load(f)
        return ans

    ans = cluster_impl(output_dir=output_dir, **kwargs)

    # store the answer in json (note - we limit the answer to be simple python objects that can be easily stored in json, we don't want pickling as it is hard to maintain!)
    with open(cached_answer_yaml, "wt") as f:
        yaml.dump(ans, f, default_flow_style=False)

    save_text_file_safe(already_created_hash_filename, hash_value)
    save_text_file_safe(already_created_description_filename, str_repr)

    return ans


def cluster(
    *,
    input_fasta_filename: str,
    output_dir: str,
    cluster_min_sequence_identity: float = 0.4,
    threads: Optional[int] = None,
    cluster_method: str = "cluster",
    deduplicate: bool = True,
    override: bool = False,
    kmer_per_seq: Optional[int] = None,
    split_memory_limit: Optional[str] = None,
) -> Dict[str, str]:
    """
    see cached_cluster() doc
    """
    which_mmseqs = shutil.which("mmseqs")

    if which_mmseqs is None:
        raise Exception(
            "Please install mmseqs2 . See install instructions here: https://github.com/soedinglab/MMseqs2 "
            "Also note that you can download prebuilt static binaries such as: https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz - extract and add the binary to your system PATH."
        )
    else:
        print(f"identified mmseqs binary at: {which_mmseqs}")

    if not os.path.isfile(input_fasta_filename):
        if os.path.isfile(input_fasta_filename + ".gz"):
            cmd = f"gunzip {input_fasta_filename}.gz"
            _run_system_cmd(cmd)

    # Create workspace (supports override)
    workspace_dir = join(output_dir, "mmseqs_workspace")
    if override and os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)

    os.makedirs(workspace_dir)
    mmseqs_db_path = join(output_dir, "mmseqs_workspace", "mmseqs_DB")

    print("cluster_method=", cluster_method)

    ans = {}
    ########### Major step A - remove all redundancies
    if deduplicate:

        print(
            "A.1 - creating mmseqs DB. It converts the input fasta into mmseqs DB internal format (made of multiple files)"
        )  #
        cmd = f"mmseqs createdb {input_fasta_filename} {mmseqs_db_path}"
        _run_system_cmd(cmd)

        # mmseqs cluster DB DB_clu tmp --min-seq-id 1.0 --threads 32
        mmseqs_cluster_full_identity = join(output_dir, "mmseqs_workspace", "mmseqs_DB_cluster_full_identity")
        mmseqs_tmp_for_clustering = join(output_dir, "mmseqs_workspace", "mmseqs_DB_tmp")
        print(
            r"A.2 - clustering with 100% identity to remove duplicates. The generated DB does not contain (directly) the sequences data, it only maps clusters centers to members."
        )
        cmd = f"mmseqs {cluster_method} {mmseqs_db_path} {mmseqs_cluster_full_identity} {mmseqs_tmp_for_clustering} -c 1.0"
        cmd = _handle_cli_arguments(cmd, threads=threads, split_memory_limit=split_memory_limit, min_seq_id=1.0)
        _run_system_cmd(cmd)

        mmseqs_only_representatives = join(output_dir, "mmseqs_workspace", "mmseqs_DB_full_identity_representatives")
        print(r"A.3 - keeping only cluster centers")
        cmd = f"mmseqs createsubdb {mmseqs_cluster_full_identity} {mmseqs_db_path}  {mmseqs_only_representatives}"
        _run_system_cmd(cmd)

        mmseqs_only_unique_sequences_representatives_fasta = join(output_dir, "unique_representatives.fasta")
        print(r"A.4 - creating a fasta file that contains only the representatives, including their sequence data.")
        cmd = f"mmseqs convert2fasta {mmseqs_only_representatives} {mmseqs_only_unique_sequences_representatives_fasta}"
        _run_system_cmd(cmd)

    # TODO: I can probably avoid converting to fasta in the end of major step A, and do major step B still in mmseqs DB format, which might speed things.

    ########### Major step B - create clusters

    # description on how to read the cluster files format: https://mmseqs.com/latest/userguide.pdf - search for "Internal cluster format",
    # also describes how to convert it to TSV for convenience

    print("B.1 - creating mmseqs DB for our unique DB")
    step_B_initial_db = join(output_dir, "mmseqs_workspace", "step_B_initial_DB")

    if deduplicate:
        cmd = f"mmseqs createdb {mmseqs_only_unique_sequences_representatives_fasta} {step_B_initial_db}"
    else:
        cmd = f"mmseqs createdb {input_fasta_filename} {step_B_initial_db}"

    _run_system_cmd(cmd)

    print(
        "B.2 - cluster the remaining (unique representatives if deduplication was used) into different clusters based on the requested sequence identity threshold"
    )
    mmseqs_tmp_2_for_clustering = join(output_dir, "mmseqs_workspace", "mmseqs_DB_tmp_2")
    clustered_db = join(output_dir, "mmseqs_workspace", "mmseqs_DB_clustered")
    cmd = f"mmseqs {cluster_method} {step_B_initial_db} {clustered_db} {mmseqs_tmp_2_for_clustering} -c 1.0"
    cmd = _handle_cli_arguments(
        cmd,
        threads=threads,
        kmer_per_seq=kmer_per_seq,
        split_memory_limit=split_memory_limit,
        min_seq_id=cluster_min_sequence_identity,
    )
    _run_system_cmd(cmd)

    print(
        "B.3 - generate cluster TSV for convenience"
    )  # for massive datasets, we might skip this and use mmseqs output format directly (possibly worth checking if there's already a python lib that handles this)
    clustered_tsv = join(output_dir, "clustered.tsv")
    cmd = f"mmseqs createtsv {step_B_initial_db} {step_B_initial_db} {clustered_db} {clustered_tsv}"
    _run_system_cmd(cmd)

    # sort it to avoid issues with people assuming it's sorted

    clustered_tsv_df = pd.read_csv(filepath_or_buffer=clustered_tsv, sep="\t", header=None, names=["center", "id"])
    clustered_tsv_df.sort_values(by="id", inplace=True)
    clustered_tsv_df.set_index("id", inplace=True)
    clustered_tsv_df.to_csv(clustered_tsv, sep="\t")

    ####
    final_clusters_mmseqs_only_representatives = join(
        output_dir, "mmseqs_workspace", "mmseqs_DB_final_clusters_representatives"
    )
    print(r"B.4 - create a sub DB only with the clusters centers (representatives)")
    cmd = f"mmseqs createsubdb {clustered_db} {step_B_initial_db}  {final_clusters_mmseqs_only_representatives}"
    _run_system_cmd(cmd)

    final_clusters_centers_fasta = join(output_dir, "clusters_representatives.fasta")
    print(
        r"B.5 - creating a fasta file that contains only the representatives (clusters centers), including their sequence data."
    )
    cmd = f"mmseqs convert2fasta {final_clusters_mmseqs_only_representatives} {final_clusters_centers_fasta}"
    _run_system_cmd(cmd)

    print("--------------------------------------")
    print("Final generated key files summary:")
    print("--------------------------------------")

    if deduplicate:
        print(f"a deduplicated FASTA file: {mmseqs_only_unique_sequences_representatives_fasta}")
        ans["deduplicated_fasta"] = mmseqs_only_unique_sequences_representatives_fasta

    print(f"a TSV file containing the clusters (no sequence information): {clustered_tsv}")
    ans["cluster_tsv"] = clustered_tsv

    print(
        f"a FASTA file containing the clusters centers (representatives) including sequence information: {final_clusters_centers_fasta}"
    )
    ans["clusters_centers_fasta"] = final_clusters_centers_fasta

    return ans


cluster_impl = cluster  # for backward compatibility


def _run_system_cmd(cmd: str, capture_output: bool = False) -> None:
    print(f"about to run: {cmd}")
    res = subprocess.run(cmd, shell=True, check=False, capture_output=capture_output)
    if res.stdout and len(res.stdout) > 0:
        print("stdout=")
        print(res.stdout.decode())
    if res.stderr and len(res.stderr) > 0:
        print("stderr=")
        print(res.stderr.decode())
    if res.returncode and res.returncode != 0:
        raise Exception(f"ERROR: failed when trying to run {cmd}, got return val={res.returncode}")


def _handle_cli_arguments(
    cmd: str,
    threads: Optional[int] = None,
    kmer_per_seq: Optional[int] = None,
    split_memory_limit: Optional[str] = None,
    min_seq_id: Optional[float] = None,
) -> str:
    """
    Handles optional command line arguments mmseqs calls.
    """
    cmd = f"{cmd} --threads {threads}" if threads else cmd
    cmd = f"{cmd} --kmer-per-seq {kmer_per_seq}" if kmer_per_seq else cmd
    cmd = f"{cmd} --split-memory-limit {split_memory_limit}" if split_memory_limit else cmd
    cmd = f"{cmd} --min-seq-id {min_seq_id}"
    return cmd
