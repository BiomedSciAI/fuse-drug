import unittest
import os

from fuse.utils.ndict import NDict
from fuse.data.pipelines.pipeline_default import PipelineDefault

from fusedrug.data.tokenizer.ops import (
    Op_pytoda_SMILESTokenizer,
    Op_pytoda_ProteinTokenizer,
)
from fusedrug.data.molecule.tokenizer.pretrained import (
    get_path as get_molecule_pretrained_vocab_path,
)


class TestPytodaTokenizers(unittest.TestCase):
    """
    Basic tests and runnable example of pytoda's molecules (smiles) and protein sequences tokenizers as ops
    """

    def test_pytoda_smiles_tokenizer(self) -> None:

        # set sample
        sample = NDict()
        sample["data.ligand.smiles"] = "COc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1"

        # initiate tokenizer op and data pipeline
        vocab_file = os.path.join(
            get_molecule_pretrained_vocab_path(), "pytoda_molecules_vocab.json"
        )
        tokenizer_op = Op_pytoda_SMILESTokenizer(dict(vocab_file=vocab_file))
        pipeline = PipelineDefault(
            "test_pipeline",
            [
                (
                    tokenizer_op,
                    dict(
                        key_in="data.ligand.smiles",
                        key_out_tokens_ids="data.ligand.tokenized_smiles",
                    ),
                )
            ],
        )

        # process the sample
        sample = pipeline(sample)

        tokenized_smiles = sample["data.ligand.tokenized_smiles"]
        self.assertTrue(len(tokenized_smiles))

    def test_pytoda_protein_tokenizer(self) -> None:

        # set sample
        sample = NDict()
        sample[
            "data.protein.sequence"
        ] = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"

        # initiate tokenizer and pipeline
        tokenizer_op = Op_pytoda_ProteinTokenizer(amino_acid_dict="iupac")
        pipeline = PipelineDefault(
            "test_pipeline",
            [
                (
                    tokenizer_op,
                    dict(
                        key_in="data.protein.sequence",
                        key_out_tokens_ids="data.protein.tokenized_seq",
                    ),
                )
            ],
        )

        # process the sample
        sample = pipeline(sample)

        tokenized_seq = sample["data.protein.tokenized_seq"]
        self.assertTrue(len(tokenized_seq))


if __name__ == "__main__":
    unittest.main()
