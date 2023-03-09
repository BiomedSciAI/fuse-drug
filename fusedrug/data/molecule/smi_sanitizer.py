from fusedrug.utils.file_formats import IndexedTextFile
from rdkit import Chem
from fuse.utils.multiprocessing import run_multiprocessed, get_from_global_storage
from typing import Union
import numpy as np
import os
from tqdm import tqdm
import click

# def sanitize_smi_file(input_smi_path,
#     output_smi_path,
#     write_delim='\t',
#     verbose=1):
#     itf = IndexedTextFile(input_smi_path,
#         process_line_func = lambda x:x,
#         )

#     with open(output_smi_path,'wt') as f:
#         line_num = -1
#         for line in iter(itf):
#             line_num += 1
#             if line_num == 15:
#                 banana=123
#             if (verbose>0) and (0==line_num%1e7):
#                 print(line_num)
#             split_line = line.split()
#             smiles_name, smiles_seq = split_line
#             sanitized_smiles_seq = sanitize_smiles(smiles_seq, verbose=verbose)
#             if sanitized_smiles_seq is None:
#                 continue
#             output_line = write_delim.join(split_line)+'\n'
#             f.write(output_line)

#     if verbose>0:
#         print(f'done writing sanitized smi file {output_smi_path}')


def _sanitize_smiles_default(smiles, verbose=1):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        # sanitized_mol = Chem.SanitizeMol(mol) #https://www.rdkit.org/docs/Cookbook.html
        sanitized_smiles = Chem.MolToSmiles(mol)
    except:
        if verbose > 0:
            print(f"ERROR: could not sanitize smiles {smiles}")
        return None

    return sanitized_smiles


def _sanitize_smiles_choose_flags(smiles, sanitize_flags=Chem.rdmolops.SANITIZE_NONE, verbose=1):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # note - it modifies the mol inplace
        Chem.SanitizeMol(mol, sanitize_flags)
        # sanitized_mol = Chem.SanitizeMol(mol) #https://www.rdkit.org/docs/Cookbook.html
        sanitized_smiles = Chem.MolToSmiles(mol)
    except:
        if verbose > 0:
            print(f"ERROR: could not sanitize smiles {smiles}")
        return None

    return sanitized_smiles


def worker_func(arg):
    input_smi_path, start_index, chunk_size, read_delim, read_molecule_sequence_column_idx, verbose = arg
    # itf = IndexedTextFile(input_smi_path, process_line_func = lambda x:x)
    itf = get_from_global_storage("indexed_text_file")

    ans = []
    for index in range(start_index, start_index + chunk_size):
        line = itf[index]
        splitted = line.split(read_delim)
        smiles_seq = splitted[read_molecule_sequence_column_idx]
        sanitized_smiles_seq = _sanitize_smiles_default(smiles_seq)
        if sanitized_smiles_seq is None:
            continue
        ans.append(line)

    return ans


def sanitize_smi_file_multiprocessed(
    input_smi_path,
    output_smi_path: str,
    read_delim: str = "\t",
    read_molecule_sequence_column_idx=0,
    chunk_size: int = 1000,
    num_workers: Union[int, str] = "auto",  # either int or 'auto'
    verbose: int = 1,
):

    try:
        num_workers_int = int(num_workers)
        num_workers = num_workers_int
    except:
        if isinstance(num_workers, str):
            assert "auto" == num_workers
            num_workers = os.cpu_count()
        else:
            raise Exception('expected "num_workers" to be int or "auto"')

    itf = IndexedTextFile(input_smi_path)
    molecules_num = len(itf)
    print("total molecules num (pre sanitize) = ", molecules_num)

    # input_smi_path, start_index, chunk_size, verbose = arg
    args = []
    for start_pos in np.arange(0, molecules_num, chunk_size):
        end_pos = min(start_pos + chunk_size, molecules_num)
        use_chunk_size = end_pos - start_pos
        args.append((input_smi_path, start_pos, use_chunk_size, read_delim, read_molecule_sequence_column_idx, True))

    with open(output_smi_path, "wt") as f:
        for processed_ans in run_multiprocessed(
            worker_func,
            args,
            workers=num_workers,
            verbose=1,
            as_iterator=True,
            copy_to_global_storage=dict(indexed_text_file=itf),
        ):
            for line in processed_ans:
                f.write(line)

    if verbose > 0:
        print(f"done writing sanitized smi file {output_smi_path}")


@click.command()
@click.argument("input_smi")
@click.argument("output_sanitized_smi")
@click.option(
    "--read-molecule-sequence-column-idx",
    default=0,
    help="the column index in which the sequence (usually SMILES) is found",
)
@click.option("--chunk", default=100000, help="amount of molecules for processing by each multiprocessing worker")
@click.option(
    "--workers",
    default="auto",
    help='number of multliprocessing workers. Pass "auto" to decide based on available cpu cores, pass an integer to specificy a specific number, pass 0 to disable multiprocessing',
)
def main(input_smi, output_sanitized_smi, read_molecule_sequence_column_idx, chunk, workers):
    """
    Outputs a rdkit sanitized version of an smi file (usually containing molecules)

    INPUT_SMI - path to input smi file  \n
    OUTPUT_SANITIZED_SMI - path to the output rdkit sanitized smi file \n
    """
    sanitize_smi_file_multiprocessed(
        input_smi,
        output_sanitized_smi,
        read_molecule_sequence_column_idx=read_molecule_sequence_column_idx,
        chunk_size=chunk,
        num_workers=workers,
    )


if __name__ == "__main__":
    main()
