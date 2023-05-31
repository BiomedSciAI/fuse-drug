import unittest
from fusedrug.data.protein.ops.loaders.fasta_loader import FastaLoader
from fusedrug.data.protein.ops.aa_ops import (
    OpToUpperCase,
    OpKeepOnlyUpperCase,
)  # OpAddSeperator, OpStrToTokenIds, OpTokenIdsToStr, OpMaskRandom, OpCropRandom

from fuse.data import PipelineDefault
import os
from fusedrug import get_tests_data_dir
from fuse.data import create_initial_sample

# from transformers import BertTokenizer
# from fusedrug.data.tokenizer.huggingface_based_tokenizer import HuggingFaceBasedTokenizer
from fuse.data import OpRepeat


class TestAAOps(unittest.TestCase):
    def test_aa_ops(self) -> None:

        # hf_dir = os.path.join(get_tests_data_dir(), 'prot_bert_bfd')
        # hf_tokenizer = BertTokenizer.from_pretrained(hf_dir)
        # tokenizer = HuggingFaceBasedTokenizer(hf_tokenizer)

        # mask_token_id = tokenizer.tokens_to_ids['[MASK]']

        fasta_file_loc = os.path.join(get_tests_data_dir(), "example_viral_proteins.fasta")

        # on_both = dict(inputs={'data.gt.seq':'seq'}

        fasta_loader = FastaLoader(fasta_file_loc=fasta_file_loc)

        # dict(tokenizer=tokenizer, inputs={'data.gt.seq':'seq'}, outputs='data.gt.seq'),
        # dict(tokenizer=tokenizer, inputs={'data.input.seq':'seq'}, outputs='data.input.seq'),

        pipeline_desc = [
            (OpRepeat(fasta_loader, [dict(key_out="data.gt.seq"), dict(key_out="data.input.seq")]), {}),
            # (OpRepeat(OpCropRandom, [dict(key_out='data.gt.seq'), dict(key_out='data.input.seq')]), {} ),
            # (OpRepeat(OpAddSeperator, [
            #     dict(inputs={'data.gt.seq':'seq'}, outputs='data.gt.seq'),
            #     dict(inputs={'data.input.seq':'seq'}, outputs='data.input.seq'),
            #     ]), {}),
            # #maybe we need to rename inputs to sample_inputs ? to make it clear that it's not ALL of the inputs to the function call ?
            # (OpRepeat(OpStrToTokenIds, [
            #     dict(tokenizer=tokenizer, inputs={'data.gt.seq':'seq'}, outputs='data.gt.seq'),
            #     dict(tokenizer=tokenizer, inputs={'data.input.seq':'seq'}, outputs='data.input.seq'),
            #     ]), {}),
            # (OpMaskRandom, dict(mask_count=4, mask_id=mask_token_id, probabilities=None,
            #     inputs={'data.input.seq':'ids'}, outputs='data.input.seq')),
            #
            (OpToUpperCase(), dict(key_in="data.gt.seq", key_out="data.gt.seq")),
            # (OpToUpperCase, dict(key_in='data.gt.seq', key_out='data.gt.seq')),
            # (1234, dict(key_in='data.gt.seq', key_out='data.gt.seq')),
            (OpKeepOnlyUpperCase(), dict(key_in="data.gt.seq", key_out="data.gt.seq")),
        ]

        pl = PipelineDefault("test_aa_ops_pipeline", pipeline_desc)

        sample_1 = create_initial_sample(100)
        sample_1 = pl(sample_1)

        sample_2 = create_initial_sample("YP_009047135.1")
        sample_2 = pl(sample_2)

        self.assertEqual(sample_1["data.gt.seq"], sample_2["data.gt.seq"])


if __name__ == "__main__":
    unittest.main()
