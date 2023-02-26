import selfies as sf
#https://github.com/aspuru-guzik-group/selfies
#note - there are several examples there - for example:
# 1. Getting "attribution" list to map from selfies element into the smiles element https://github.com/aspuru-guzik-group/selfies#explaining-translation
# 2. Customizing SELFIES - modifying applied chemical/physical rules - https://github.com/aspuru-guzik-group/selfies#customizing-selfies
# 3. Building a vocabulary based on selfies elements - https://github.com/aspuru-guzik-group/selfies#integer-and-one-hot-encoding-selfies
#     there are multiple important things there, including the [nop] operation that is used for padding


def smiles_to_selfies(smiles_str):
    ans = sf.encoder(smiles_str)
    return ans

def selfies_to_smiles(selfies_str):
    ans = sf.decoder(selfies_str)
    return ans

if __name__=='__main__':    
    benzene_smiles_str = "c1ccccc1"

    print('smiles: ', benzene_smiles_str)
    selfies_str = smiles_to_selfies(benzene_smiles_str)
    print('selfies: ', selfies_str)
    splitted = list(sf.split_selfies(selfies_str))
    print('splitted selfies: ', splitted)
    back_to_smiles = selfies_to_smiles(selfies_str)
    print('back to smiles: ', back_to_smiles)
    

