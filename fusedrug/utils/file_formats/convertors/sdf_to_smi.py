from typing import Optional
from fusedrug.utils.file_formats.sdf_iterable_dataset import SDFIterableDataset
from rdkit import Chem
from tqdm import tqdm
import click

@click.command()
@click.argument('input_sdf_file')
@click.argument('output_smi_file')
@click.option('--id-prefix', default='SDF_', help='the prefix that will be added to each ID in the generated smi file. If you do not change it it will be SDF_0, SDF_1 etc.')
def main(input_sdf_file:str, output_smi_file:str, id_prefix:Optional[str]):
    '''
    Converts sdf file which contains molecules into a smi files containing their SMILES representation

    Args:  
        INPUT_SDF_FILE: the input file to process
        OUTPUT_SMI_FILE: the output smi file to generate
    '''

    sdf_ds = SDFIterableDataset(input_sdf_file)    
    with open(output_smi_file, 'w') as f_out:
        mol_index=-1
        for mol in tqdm(iter(sdf_ds)):
            mol_index+=1
            try:
                smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True, kekuleSmiles=False)
            except:
                print('error when converting mol to smiles')
                continue

            line = f'{id_prefix}{mol_index}\t{smiles}\n'

            f_out.write(line)

if __name__=='__main__':
    main()


#python /gpfs/usr/yoels/dev/repos_mol_bio/fuse-drug/fusedrug/utils/file_formats/convertors/sdf_to_smi.py ./chembl_30.sdf ./chembl_30.smi --id-prefix=CHEMBL30_