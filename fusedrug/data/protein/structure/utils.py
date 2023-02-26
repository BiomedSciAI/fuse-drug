
def aa_sequence_from_aa_integers(aatype):
    ans = ''.join(['X' if x==20 else rc.restypes[x] for x in aatype])
    return ans
    
def get_structure_file_type(filename:str) -> str:
    if filename.endswith('.pdb') or filename.endswith('.pdb.gz') or filename.endswith('.ent.gz'):
        return 'pdb'
    if filename.endswith('.cif') or filename.endswith('.cif.gz'):
        return 'cif'
    raise Exception(f'Could not detect structure file format for {filename}')

    