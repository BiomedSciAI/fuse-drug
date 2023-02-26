from fusedrug.utils.file_formats import IndexedTextFile
from fuse.utils.multiprocessing import run_multiprocessed, get_from_global_storage
from typing import Union
import numpy as np
import os
import click
from collections import Counter
 
def worker_func(arg):
    input_smi_path, start_index, chunk_size, read_delim, read_molecule_sequence_column_idx, verbose = arg
    #itf = IndexedTextFile(input_smi_path, process_line_func = lambda x:x)
    itf = get_from_global_storage('indexed_text_file')
    
    characters_counter = Counter()
    for index in range(start_index, start_index+chunk_size):
        line = itf[index]
        splitted = line.split(read_delim)
        smiles_seq = splitted[read_molecule_sequence_column_idx]
        for c in smiles_seq.rstrip():
            characters_counter[c] += 1        

    return characters_counter

def smi_file_character_histogram_multiprocessed(input_smi_path, 
    read_delim:str='\t', 
    read_molecule_sequence_column_idx=0,
    chunk_size:int=1000,
    num_workers:Union[int,str]='auto', #either int or 'auto'
    verbose:int=1):
    
    try:
        num_workers_int = int(num_workers)
        num_workers = num_workers_int
    except:        
        if isinstance(num_workers, str):
            assert 'auto' == num_workers
            num_workers = os.cpu_count()
        else:
            raise Exception('expected "num_workers" to be int or "auto"')
        
    itf = IndexedTextFile(input_smi_path)
    molecules_num = len(itf)
    print('total molecules num = ', molecules_num)           
    
    #input_smi_path, start_index, chunk_size, verbose = arg
    args = []
    for start_pos in np.arange(0, molecules_num, chunk_size):
        end_pos = min(start_pos+chunk_size, molecules_num)
        use_chunk_size = end_pos-start_pos
        args.append( (input_smi_path, start_pos, use_chunk_size, read_delim, read_molecule_sequence_column_idx, True))
                
    total_characters_counter = Counter()
    
    for curr_counter in run_multiprocessed(
        worker_func, args, workers=num_workers, verbose=1, 
        as_iterator=True, copy_to_global_storage=dict(indexed_text_file=itf)):
        total_characters_counter.update(curr_counter)
    
    for k,d in total_characters_counter.items():
        print(f'{k} -> {d}')
    print('done.')

@click.command()
@click.argument('input_smi')
@click.option('--read-molecule-sequence-column-idx', default=0, help='the column index in which the sequence (usually SMILES) is found')
@click.option('--chunk', default=100000, help='amount of molecules for processing by each multiprocessing worker')
@click.option('--workers', default='auto', help='number of multliprocessing workers. Pass "auto" to decide based on available cpu cores, pass an integer to specificy a specific number, pass 0 to disable multiprocessing')
def main(input_smi, read_molecule_sequence_column_idx, chunk, workers):
    """
    Outputs a rdkit sanitized version of an smi file (usually containing molecules)

    INPUT_SMI - path to input smi file  \n
    OUTPUT_SANITIZED_SMI - path to the output rdkit sanitized smi file \n
    """
    smi_file_character_histogram_multiprocessed(input_smi, 
        read_molecule_sequence_column_idx=read_molecule_sequence_column_idx,
        chunk_size=chunk,        
        num_workers=workers,
    )

if __name__=='__main__':
    main()
    #input_smi_path = '/gpfs/haifa/projects/m/msieve/MedicalSieve/mol_bio_datasets/zinc/10kzinc.smi'
    #output_smi_path = '/gpfs/haifa/projects/m/msieve/MedicalSieve/mol_bio_datasets/zinc/sanitized_10kzinc.smi'

    #input_smi_path = '/gpfs/haifa/projects/m/msieve/MedicalSieve/mol_bio_datasets/zinc/tiny_debug_zinc.smi'
    #output_smi_path = '/gpfs/haifa/projects/m/msieve/MedicalSieve/mol_bio_datasets/zinc/sanitized_tiny_debug_zinc.smi'

    #input_smi_path = '/gpfs/haifa/projects/m/msieve/MedicalSieve/mol_bio_datasets/zinc/zinc.smi'
    #output_smi_path = '/gpfs/haifa/projects/m/msieve/MedicalSieve/mol_bio_datasets/zinc/sanitized_zinc.smi'

    #sanitize_smi_file(input_smi_path, output_smi_path)
    # sanitize_smi_file_multiprocessed(input_smi_path, output_smi_path, 
    #     chunk_size=100000,
    #     num_workers=0,
    # )
    # banana=123


###  python /gpfs/usr/yoels/dev/repos_mol_bio/fuse-drug/fusedrug/data/molecule/smi_sanitizer.py ./chembl_30.smi ./sanitized_chembl_30.smi  --read-molecule-sequence-column-idx=1 
