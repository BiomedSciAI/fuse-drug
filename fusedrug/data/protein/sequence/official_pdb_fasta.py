from io import StringIO
from Bio import SeqIO
from urllib.request import urlopen
from typing import Dict, List


def get_fasta_from_rcsb(
    pdb_id: str,
    author_chain_id: bool = True,
    verbose: bool = False,
    return_duplicates_if_found: bool = True,
) -> Dict:  # TODO: consider adding caching
    """
    Given some pdb_id, (like "7vux"), we will retrieve its fasta file from rcsb database and return it as a dict {chain: sequence}.
    This version handles symmetry, so if there are multiple identical chains in the same complex, it will return them as separate keys
    """
    fasta_data = (
        urlopen(f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}")
        .read()
        .decode("utf-8")
    )

    ans = {}

    io_data = StringIO(fasta_data)
    for record in SeqIO.parse(io_data, "fasta"):
        print(f"{record.description=}")
        chains = get_chain_ids(
            description=record.description, author_chain_id=author_chain_id
        )
        print(f"{chains=}")

        for chain in chains:
            ans[chain] = str(record.seq)
            if not return_duplicates_if_found:
                break

    return ans


def get_chain_ids(*, description: str, author_chain_id: bool) -> List[str]:
    """
    author_chain_id: if True will return the author chain id, if False will return the pdb_chain_id
    TODO: add support for chain ids that have more than one character
    """
    loc1 = description.find(" ")
    assert loc1 >= 0

    description = description[loc1 + 1 :]

    loc2 = description.find("|")
    assert loc2 >= 0
    description = description[:loc2].strip()

    ans = []

    for part in description.split(","):
        part = part.strip()

        curr_pdb_chain_id = part[0]
        curr_author_chain_id = curr_pdb_chain_id
        if author_chain_id:
            if " " in part:
                curr_author_chain_id = part[part.find(" ") + 1]

        if author_chain_id:
            ans.append(curr_author_chain_id)
        else:
            ans.append(curr_pdb_chain_id)

    return ans


# examples for reference:

# >6N25_1|Chains A, B, C, D, E|Bestrophin homolog|Gallus gallus (9031)
#

# seq.description='9DH2_1|Chains A, C[auth G], E[auth K], H[auth P]|Fab heavy chain|Homo sapiens (9606)'
# chains=['A', 'C', 'E', 'H']
# seq.description='9DH2_2|Chains B[auth D], D[auth H], F[auth L], I[auth Q]|Fab light chain|Homo sapiens (9606)'
# chains=['B', 'D', 'F', 'I']
# seq.description='9DH2_3|Chains G[auth M], J[auth R], K[auth S], L[auth T]|NKG2-D type II integral membrane protein|Homo sapiens (9606)'
# chains=['G', 'J', 'K', 'L']
# 9lsh
# seq.description='9LSH_1|Chains A, B|Citrate/sodium symporter|Klebsiella pneumoniae (573)'
# chains=['A', 'B']
# seq.description='9LSH_2|Chain C|Toll-like receptor 3|Homo sapiens (9606)'
# chains=['C']
# seq.description='9LSH_3|Chain D|Diabody (CitS VH-TLR3 VL)|synthetic construct (32630)'
# chains=['D']
# seq.description='9LSH_4|Chain E|Diabody (TLR3 VH-CitS VL)|synthetic construct (32630)'
# chains=['E']
# 9lsj
# seq.description='9LSJ_1|Chains A, B|Citrate/sodium symporter|Klebsiella pneumoniae (573)'
# chains=['A', 'B']
# seq.description='9LSJ_2|Chain C|Toll-like receptor 3|Homo sapiens (9606)'
# chains=['C']
# seq.description='9LSJ_3|Chain D|Diabody (CitS VH-TLR3 VL)|synthetic construct (32630)'
# chains=['D']
# seq.description='9LSJ_4|Chain E|Diabody (TLR3 VH-CitS VL)|synthetic construct (32630)'
# chains=['E']
# 9lsi
# seq.description='9LSI_1|Chains A, B|Citrate/sodium symporter|Klebsiella pneumoniae (573)'
# chains=['A', 'B']
# seq.description='9LSI_2|Chain C|Diabody (CitS VH-TLR3 VL)|synthetic construct (32630)'
# chains=['C']
# seq.description='9LSI_3|Chain D|Diabody (TLR3 VH-CitS VL)|synthetic construct (32630)'
# chains=['D']
# seq.description='9LSI_4|Chain E|Toll-like receptor 3|Homo sapiens (9606)'
# chains=['E']
# 8vqf
# seq.description='8VQF_1|Chain A|IgE 1J11 Light chain|Homo sapiens (9606)'
# chains=['A']
# seq.description='8VQF_2|Chain B|IgE 1J11 Heavy chain|Homo sapiens (9606)'
# chains=['B']
# seq.description='8VQF_3|Chain C|Major allergen Can f 1|Canis lupus familiaris (9615)'
# chains=['C']
# 9h4r
# seq.description='9H4R_1|Chains A, D[auth B]|Zona pellucida sperm-binding protein 2|Mus musculus (10090)'
# chains=['A', 'D']
# seq.description='9H4R_2|Chains B[auth H], E[auth X]|Heavy chain variable (VH) domain of anti-ZP2 monoclonal antibody IE-3|Rattus norvegicus (10116)'
# chains=['B', 'E']
# seq.description='9H4R_3|Chains C[auth L], F[auth Y]|Light chain variable (VL) domain of anti-ZP2 monoclonal antibody IE-3|Rattus norvegicus (10116)'
# chains=['C', 'F']
# 9h4s
# seq.description='9H4S_1|Chains A, D[auth B]|Zona pellucida sperm-binding protein 2|Mus musculus (10090)'
# chains=['A', 'D']
# seq.description='9H4S_2|Chains B[auth H], E[auth X]|Heavy chain variable (VH) domain of anti-ZP2 monoclonal antibody IE-3|Rattus norvegicus (10116)'
# chains=['B', 'E']
# seq.description='9H4S_3|Chains C[auth L], F[auth Y]|Light chain variable (VL) domain of anti-ZP2 monoclonal antibody IE-3|Rattus norvegicus (10116)'
# chains=['C', 'F']
# 8vba
# seq.description='8VBA_1|Chain A|EtpA|Escherichia coli ETEC H10407 (316401)'
# chains=['A']
# seq.description='8VBA_2|Chain B[auth H]|mAb 1C08 Heavy Chain|Mus musculus (10090)'
# chains=['B']
# seq.description='8VBA_3|Chain C[auth L]|mAb 1C08 Light Chain|Mus musculus (10090)'
# chains=['C']
# 8vbb
# seq.description='8VBB_1|Chain A|EtpA|Escherichia coli ETEC H10407 (316401)'
