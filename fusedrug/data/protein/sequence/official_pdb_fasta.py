from io import StringIO
from Bio import SeqIO
from urllib.request import urlopen
from typing import Dict


def get_fasta_from_rcsb(pdb_id: str) -> Dict:  # TODO: consider adding caching
    """
    Given some pdb_id, (like "7vux"), we will retrieve its fasta file from rcsb database and return it as a dict {chain: sequence}.
    """
    fasta_data = (
        urlopen(f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}")
        .read()
        .decode("utf-8")
    )
    fasta_file_handle = StringIO(fasta_data)
    chains_full_seq = SeqIO.to_dict(
        SeqIO.parse(fasta_file_handle, "fasta"),
        key_function=lambda rec: _description_to_author_chain_id(rec.description),
    )
    chains_full_seq = {k: str(d.seq) for (k, d) in chains_full_seq.items()}
    return chains_full_seq


def _description_to_author_chain_id(description: str) -> str:
    loc = description.find(" ")
    assert loc >= 0
    description = description[loc + 1 :]
    loc = description.find(",")
    if loc >= 0:
        description = description[:loc]

    token = "auth "
    loc = description.find(token)
    if loc >= 0:
        return description[loc + len(token)]

    return description[0]
