import unittest
from fusedrug.utils.file_formats import IndexedFastaCustom, uniprot_identifier_extractor
from fusedrug.tests_data import get_tests_data_dir
import os
from typing import Union, Tuple


class TestIndexedFastaCustom(unittest.TestCase):
    def test_basic(self) -> None:
        fasta_path = os.path.join(get_tests_data_dir(), "mini_uniprot_sprot.fasta")
        ifc = IndexedFastaCustom(fasta_path, force_recreate_index=True)

        identifier, data, full_description = ifc[2]
        self.assertEqual(identifier[0], "sp|Q197F8|002R_IIV3")
        self.assertEqual(
            data,
            "MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYKDNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEEIKAILETTPNVTRLVIRRIDDMNYNTHGDLGLDDLEFLTHLMVEDACGFTDFWAPSLTHLTIKNLDMHPRWFGPVMDGIKSMQSTLKYLYIFETYGVNKPFVQWCTDNIETFYCTNSYRYENVPRPIYVWVLFQEDEWHGYRVEDNKFHRRYMYSTILHKRDTDWVENNPLKTPAQVEMYKFLLRISQLNRDGTGYESDSDPENEHFDDESFSSGEEDSSDEDDPTWAPDSDDSDWETETEEEPSVAARILEKGKLTITNLMKSLGFKPKPKKIQSIDRYFCSLDSNYNSEDEDFEYDSDSEDDDSDSEDDC",
        )
        self.assertEqual(
            full_description,
            "sp|Q197F8|002R_IIV3 Uncharacterized protein 002R OS=Invertebrate iridescent virus 3 OX=345201 GN=IIV3-002R PE=4 SV=1",
        )

        ifc = IndexedFastaCustom(
            fasta_path,
            force_recreate_index=False,
            process_identifier_pipeline=[
                uniprot_identifier_extractor,
            ],
        )

        identifier, data, full_description = ifc[2]
        self.assertEqual(identifier, "Q197F8")
        self.assertEqual(
            data,
            "MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYKDNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEEIKAILETTPNVTRLVIRRIDDMNYNTHGDLGLDDLEFLTHLMVEDACGFTDFWAPSLTHLTIKNLDMHPRWFGPVMDGIKSMQSTLKYLYIFETYGVNKPFVQWCTDNIETFYCTNSYRYENVPRPIYVWVLFQEDEWHGYRVEDNKFHRRYMYSTILHKRDTDWVENNPLKTPAQVEMYKFLLRISQLNRDGTGYESDSDPENEHFDDESFSSGEEDSSDEDDPTWAPDSDDSDWETETEEEPSVAARILEKGKLTITNLMKSLGFKPKPKKIQSIDRYFCSLDSNYNSEDEDFEYDSDSEDDDSDSEDDC",
        )
        self.assertEqual(
            full_description,
            "sp|Q197F8|002R_IIV3 Uncharacterized protein 002R OS=Invertebrate iridescent virus 3 OX=345201 GN=IIV3-002R PE=4 SV=1",
        )

    def test_identifier_based_access(self) -> None:
        fasta_path = os.path.join(get_tests_data_dir(), "mini_uniprot_sprot.fasta")

        ifc = IndexedFastaCustom(
            fasta_path,
            force_recreate_index=True,
            process_identifier_pipeline=[
                uniprot_identifier_extractor,
            ],
            allow_access_by_id=True,
        )

        ans_1 = ifc[2]
        ans_2 = ifc["Q197F8"]

        self.assertTupleEqual(ans_1, ans_2)


if __name__ == "__main__":
    unittest.main()


# In [13]: curr = fa[2]

# In [14]: curr.name
# Out[14]: 'sp|Q197F8|002R_IIV3'

# In [15]: curr.description
# Out[15]: 'sp|Q197F8|002R_IIV3 Uncharacterized protein 002R OS=Invertebrate iridescent virus 3 OX=345201 GN=IIV3-002R PE=4 SV=1'

# In [16]: curr.seq
# Out[16]: 'MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQALAERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYKDNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEEIKAILETTPNVTRLVIRRIDDMNYNTHGDLGLDDLEFLTHLMVEDACGFTDFWAPSLTHLTIKNLDMHPRWFGPVMDGIKSMQSTLKYLYIFETYGVNKPFVQWCTDNIETFYCTNSYRYENVPRPIYVWVLFQEDEWHGYRVEDNKFHRRYMYSTILHKRDTDWVENNPLKTPAQVEMYKFLLRISQLNRDGTGYESDSDPENEHFDDESFSSGEEDSSDEDDPTWAPDSDDSDWETETEEEPSVAARILEKGKLTITNLMKSLGFKPKPKKIQSIDRYFCSLDSNYNSEDEDFEYDSDSEDDDSDSEDDC'
