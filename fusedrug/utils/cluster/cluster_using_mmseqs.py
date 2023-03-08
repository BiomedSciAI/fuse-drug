from fuse.data.ops import get_function_call_str
import shutil
import hashlib
import subprocess
from fuse.utils.file_io import save_text_file_safe, read_text_file
import os

def cluster(force_rebuild:bool = False, **kwargs):
    """
    A wrapped around cluster_impl() to allow caching 
    see cluster_impl doc for details
    
    Args:
        force_rebuild: rebuilds the data even if the hash indicates that it was already cached.
    """

    if 'output_dir' not in kwargs:
        raise Exception('cluster() expects "output_dir" ')

    output_dir = kwargs['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    str_repr = get_function_call_str(
        cluter_impl, 
        _ignore_kwargs_names=['num_workers'], 
        _include_code = False,
        **kwargs
        )

    hash_value = hashlib.md5(str_repr.encode()).hexdigest()
    already_created_hash_filename = os.path.join(output_dir, 'created_hash')
    already_created_description_filename = os.path.join(output_dir, 'created_desc')

    _rebuild_needed = False
    if os.path.isfile(already_created_hash_filename):
        existing_hash_value = read_text_file(already_created_hash_filename)
        if hash_value == existing_hash_value:
            if force_rebuild:
                print(f'cluster::Hash value on disk matches the calculated one ({hash_value}), but force_rebuild was requested so rebuilding anyway.')
                _rebuild_needed = True
        else:
            raise Exception('hash value on disk is a mismatch! please delete the directory manually and rerun.')
            _rebuild_needed = True

        existing_desc = read_text_file(already_created_description_filename)
    else:
        _rebuild_needed = True
            
    if _rebuild_needed and (not _rebuild_needed):
        print(f'Hash value indicates that it was already built, not rebuilding. See description: {existing_desc}')
        return

    cluter_impl(**kwargs)

    save_text_file_safe(already_created_description_filename, hash_value)
    save_text_file_safe(already_created_hash_filename, str_repr)

def cluter_impl(*, all_sequences_fasta:str, output_dir:str, cluster_min_seqeunce_identity:float=0.4, cluster_method:str='cluster'):
    """

    Uses mmseqs to:

    1. Remove 100% sequence identity duplicates
    2. Cluster the remaining unique sequences into multiple clusters (e.g. with 70% sequence similary threshold )
        This is useful for multiple purposes:
        a. Creating cross validation/test splits that evaluate and demonstrate generalizability
        b. Balanced sampling during training, sampling with inverse proportion to cluster size. (Similar to Class Balancing)        

    Args:
        all_sequences_fasta: a fasta with an entry per molecular entity
        output_dir: where the output will be generated
        cluster_min_seqeunce_identity: the minimal sequence identity for member within a cluster
        cluster_method: any of 'cluster', 'linclust':
            'cluster' is the "vanilla" one
            'linclust' is faster (claims linear runtime) but less accurate. Might be suitable for massive data.
                NOTE: I've compared cluster and linclust results for the deduplication phase, and results aren't identical, which means it probably misses few identical cases.

        
    """    

    which_mmseqs = shutil.which('mmseqs')

    if which_mmseqs is None:
        raise Exception('Please install mmseqs2 . See install instructions here: https://github.com/soedinglab/MMseqs2 '
        'Also note that you can download prebuilt static binaries such as: https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz - extract and add the binary to your system PATH.')
    else:
        print(f'identified mmseqs binary at: {which_mmseqs}')
    
    if not os.path.isfile(all_sequences_fasta):
        if os.path.isfile(all_sequences_fasta+'.gz'):
            cmd = f'gunzip {all_sequences_fasta}.gz'
            _run_system_cmd(cmd)            
    
    mmseqs_db_path = os.path.join(output_dir, 'mmseqs_DB')

    print('cluster_method=', cluster_method)

    ########### Major step A - remove all redundancies

    print('A.1 - creating mmseqs DB. It converts the input fasta into mmseqs DB internal format (made of multiple files)') #
    cmd = f'mmseqs createdb {all_sequences_fasta} {mmseqs_db_path}'
    _run_system_cmd(cmd)            

    #mmseqs cluster DB DB_clu tmp --min-seq-id 1.0 --threads 32 
    mmseqs_cluster_full_identity = os.path.join(output_dir, 'mmseqs_DB_cluster_full_identity')
    mmseqs_tmp_for_clustering = os.path.join(output_dir, 'mmseqs_DB_tmp')
    print(r'A.2 - clustering with 100% identity to remove duplicates. The generated DB does not contain (directly) the sequences data, it only maps clusters centers to members.')
    cmd = f'mmseqs {cluster_method} {mmseqs_db_path} {mmseqs_cluster_full_identity} {mmseqs_tmp_for_clustering} --min-seq-id 1.0 --threads 32'
    _run_system_cmd(cmd)            

    mmseqs_only_representatives = os.path.join(output_dir, 'mmseqs_DB_full_identity_representitives')
    print(r'A.3 - keeping only cluster centers')
    cmd = f'mmseqs createsubdb {mmseqs_cluster_full_identity} {mmseqs_db_path}  {mmseqs_only_representatives}'
    _run_system_cmd(cmd)            

    mmseqs_only_unique_sequences_representatives_fasta = os.path.join(output_dir, 'unique_representatives.fasta')
    print(r'A.4 - creating a fasta file that contains only the representatives, including their sequence data.')
    cmd = f'mmseqs convert2fasta {mmseqs_only_representatives} {mmseqs_only_unique_sequences_representatives_fasta}'
    _run_system_cmd(cmd)            

    #TODO: I can probably avoid converting to fasta in the end of major step A, and do major step B still in mmseqs DB format, which might speed things.

    ########### Major step B - create clusters

    #description on how to read the cluster files format: https://mmseqs.com/latest/userguide.pdf - search for "Internal cluster format",
    #also describes how to convert it to TSV for convinience    

    print('B.1 - creating mmseqs DB for our unique DB') #
    mmseqs_all_unique_DB = os.path.join(output_dir, 'all_unique_DB')
    cmd = f'mmseqs createdb {mmseqs_only_unique_sequences_representatives_fasta} {mmseqs_all_unique_DB}'
    _run_system_cmd(cmd)

    print('B.2 - cluster the remaining unique representatives into different clusters based on the requested sequence identity threshold') #
    mmseqs_tmp_2_for_clustering = os.path.join(output_dir, 'mmseqs_DB_tmp_2')
    clustered_db = os.path.join(output_dir, 'mmseqs_DB_clustered')
    cmd = f'mmseqs {cluster_method} {mmseqs_all_unique_DB} {clustered_db} {mmseqs_tmp_2_for_clustering} --min-seq-id {cluster_min_seqeunce_identity} --threads 32'
    _run_system_cmd(cmd)    

    print('B.3 - generate cluster TSV for convinience') #for massive datasets, we might skip this and use mmseqs output format directly (possibly worth checking if there's already a python lib that handles this)    
    clustered_tsv = os.path.join(output_dir, 'clustered.tsv')
    cmd = f'mmseqs createtsv {mmseqs_all_unique_DB} {mmseqs_all_unique_DB} {clustered_db} {clustered_tsv}'
    _run_system_cmd(cmd) 

def _run_system_cmd(cmd:str):
    print('about to run: ', cmd)
    res = subprocess.run(cmd, shell=True, check=False, capture_output=True)
    if len(res.stdout)>0:
        print('stdout=')
        print(res.stdout.decode())
    if len(res.stderr)>0:
        print('stderr=')
        print(res.stderr.decode())
    if res.returncode != 0:
        raise Exception(f'ERROR: failed when trying to run {cmd}, get return val={res.returncode}')

