import pytoda
from fuse.data import OpBase, OpFunc, OpLambda

def get_pyotda_chem_transforms_ON_STRING_SEQUENCE(**kwargs):
    '''
    Note - the input and output is a string representation (SMILES or SELFIES)
    but internally during the transforms different representations are used

    To see kwargs documentation see:
    https://paccmann.github.io/paccmann_datasets/api/pytoda.smiles.transforms.html#pytoda.smiles.transforms.compose_smiles_transforms
    '''
    #note:compose_smiles_transform returns a callable 
    func = pytoda.smiles.transforms.compose_smiles_transforms(**kwargs)
    return OpFunc(func)

def get_pytoda_chem_transforms_ON_TOKEN_INDEXES(**kwargs):
    '''
    Note - the input and output are token indexes
    a numpy 1d array of integers or a list of integers

    To see kwargs documentation see:
    https://paccmann.github.io/paccmann_datasets/api/pytoda.smiles.transforms.html#pytoda.smiles.transforms.compose_encoding_transforms
    '''
    #note:compose_encoding_transforms returns a callable 
    func = pytoda.smiles.transforms.compose_encoding_transforms(**kwargs)
    return OpFunc(func)


