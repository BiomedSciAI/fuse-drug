from collections import OrderedDict
from typing import List, Union, Optional, Dict

special_token_marker = [
    "<",
    ">",
]
# We mark every special token with special token markers (e.g. 'SEP' -> '<SEP>').
# The markers are configurable, in case we come upon a representation that already uses similar characters (such as '[]' in SMILES).
# The special tokens defined below are without the markers. The function get_special_tokens() adds the markers and returns all the
# tokens. The function special_mark(input) takes a sequence, and marks every character in it (e.g. 'ABC' -> '<A><B><C>')
special_tokens = {
    "unk_token": "UNK",  # Unknown token
    "pad_token": "PAD",  # Padding token
    "cls_token": "CLS",  # Classifier token (probably irrelevant in the T5 setting)
    "sep_token": "SEP",  # Separator token
    "mask_token": "MASK",  # Mask token
    "eos_token": "EOS",  # End of Sentence token
}
# Remember: do not use the below tokens as is. They must be wrapped by token markers ('<', '>') first, using special_wrap_input()
task_tokens = [  # pairwise tasks
    "GLOBAL_INTERACTION_ATTRIBUTES",  # the token following this marks which global attribute type is encoded next
    "BINDING_AFFINITY_CLASS",  # Physical binding of proteins
    "GENERAL_AFFINITY_CLASS",  # General affinity - proteins participating in the same process, gene co-expression, etc
    "MOLECULAR_ENTITY",  # the token following this marks which specific type of molecular entity comes next. All tokens
    # after this (and before the next MOLECULAR_ENTITY token) relate to the same molecule
    "MOLECULAR_ENTITY_GENERAL_PROTEIN",
    "MOLECULAR_ENTITY_PROTEIN_CHAIN",
    "MOLECULAR_ENTITY_MUTATED_PROTEIN_CHAIN",
    "MOLECULAR_ENTITY_ANTIGEN",
    "MOLECULAR_ENTITY_EPITOPE",
    "MOLECULAR_ENTITY_ANTIBODY_HEAVY_CHAIN",
    "MOLECULAR_ENTITY_ANTIBODY_LIGHT_CHAIN",
    "MOLECULAR_ENTITY_ANTIBODY_LIGHT_CHAIN_CDR1",
    "MOLECULAR_ENTITY_ANTIBODY_LIGHT_CHAIN_CDR2",
    "MOLECULAR_ENTITY_ANTIBODY_LIGHT_CHAIN_CDR3",
    "MOLECULAR_ENTITY_ANTIBODY_HEAVY_CHAIN_CDR1",
    "MOLECULAR_ENTITY_ANTIBODY_HEAVY_CHAIN_CDR2",
    "MOLECULAR_ENTITY_ANTIBODY_HEAVY_CHAIN_CDR3",
    "MOLECULAR_ENTITY_TCR_ALPHA_CHAIN",  # TCR "light" chain - only V, J and C segments (variable region)
    "MOLECULAR_ENTITY_TCR_BETA_VDJ",  # TCR "heavy" chain - V(ariable), D(iversity), and J(oining) segments, as well as the C(onstant) segment
    "MOLECULAR_ENTITY_TCR_BETA_CDR3",  # TCR beta chain CDR3 region
    "MOLECULAR_ENTITY_TCR_GAMMA_VAR",  # TCR gamma chain variable region
    "MOLECULAR_ENTITY_TCR_DELTA_VAR",  # TCR delta chain variable region
    "MOLECULAR_ENTITY_TCR_ALPHA_CDR3",  # TCR alpha chain CDR3 region
    "MOLECULAR_ENTITY_TCR_GAMMA_CDR3",  # TCR gamma chain CDR3 region
    "MOLECULAR_ENTITY_TCR_DELTA_CDR3",  # TCR delta chain CDR3 region
    "TARGETED_ANTIBODY_DESIGN_ENCODER_ONLY_MODE",  # A prefix to our T5 model to inform it that it will run in "encoder only" mode (so only
    # the encoder-stack is used, plus the encoder-output-tokens-classification-head)
    "DECODER_START",
    "AMINO_ACID_SEQUENCE",  # The next tokens depict a sequence of amino acids
    "SEQUENCE_NATURAL_START",  # To be placed at the beginning of an uncropped sequence, meaning that in nature, the sequence starts with the next token
    "SEQUENCE_NATURAL_END",  # To be placed at the end of an uncropped sequence, meaning that in nature, the sequence ends with the previous token
    "SMILES_SEQUENCE",  # The next tokens depict a SMILES molecule representation
    "SELFIES_SEQUENCE",  # The next tokens depict a SELFIES molecule representation
    "NOOP",  # An empty token. Unlike "PAD", it is a viable network output, and should not be ignored
    "BACKSPACE",  # Deleted the previous token
    "BINDING",  # Binding affinity prediction task
    # Masked fill in tasks
    "FILLIN",  # Fill in masked inputs task
    # Reorder tasks
    "REORDER",  # Given a mixeded up molecule, reorder it (mixing should be over relatively long subsequences, not single AAs)
    # Generation tasks
    "TOAA",  # Translate SMILES to AA sequence task
    "ACTIVE",  # Task to derive an active site
    "GENESEQ",  # Genetic sequence from AAs (SMILES)
    "INCREASE",  # Given a molecule and a property, generate a new molecule with increased property
    "DECREASE",  # Given a molecule and a property, generate a new molecule with decreased property
    "DIFFUSION",  # Discrete diffusion task
    "TIMESTEP",  # Timestep tkn for the discrete diffusion task
    # Structure prediction
    "STRUCTURE",  # predict 3D structure from sequence
    "DISTANCE",  # Given a protein and two groups of AAs in it, predicts the distance between the AA groups within a folded protein.
    # Property predictions:
    "SOLUBILITY",  # Molecule water solubility
    "TOXICITY",  # Binary (quantitative?) toxicity prediction task
    "FDA_APPR",  # Predict drugs approved by the FDA
    "BBBP",  # Predict blood-brain barrier penetration (BBBP)
    "HIV_ACTIVITY",  # Predict HIV activity
    "AB",  # Binary (quantitative?) antibacterial activity predictor
    "ISACTIVE",  # Binary task to predict if first input is an active site of the second input
    "ISSYNTHETIC",  # Binary task to predict if the input protein is natural, or was generated (or filled in) by a model
    "PENETR",  # Task of membrane penetration prediction
    "ABSORPTION",  # Absorption measure predictor
    "DISTRIBUTION",  # Distribution measure
    "METABOLISM",  # Metabolism rate prediction
    "EXCRETION",  # Excretion rate prediction
    "FLUORESCENCE",  #
    "STABILITY",  #
    "DISORDER",
    "DISEASE",  # Prediction of a disease from an antibody sequence. Requires a group of tokens for the different diseases
    # property modifier tokens:
    "BINARY",  # make property prediction binary - yes/no
    "REGRESSION",  # Predict a numerical value of the property
    "ORGANISM",  # Predict the source organism for the protein/antibody
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ".",
    "YES",  # Affirmative answer to the task question
    "NO",  # Negative answer to the task question
    "SENTINEL_ID_0",
    "SENTINEL_ID_1",
    "SENTINEL_ID_2",
    "SENTINEL_ID_3",
    "SENTINEL_ID_4",
    "SENTINEL_ID_5",
    "SENTINEL_ID_6",
    "SENTINEL_ID_7",
    "SENTINEL_ID_8",
    "SENTINEL_ID_9",
    "SENTINEL_ID_10",
    "SENTINEL_ID_11",
    "SENTINEL_ID_12",
    "SENTINEL_ID_13",
    "SENTINEL_ID_14",
    "SENTINEL_ID_15",
    "SENTINEL_ID_16",
    "SENTINEL_ID_17",
    "SENTINEL_ID_18",
    "SENTINEL_ID_19",
    "SENTINEL_ID_20",
    "SENTINEL_ID_21",
    "SENTINEL_ID_22",
    "SENTINEL_ID_23",
    "SENTINEL_ID_24",
    "SENTINEL_ID_25",
    "SENTINEL_ID_26",
    "SENTINEL_ID_27",
    "SENTINEL_ID_28",
    "SENTINEL_ID_29",
    "SENTINEL_ID_30",
    "SENTINEL_ID_31",
    "SENTINEL_ID_32",
    "SENTINEL_ID_33",
    "SENTINEL_ID_34",
    "SENTINEL_ID_35",
    "SENTINEL_ID_36",
    "SENTINEL_ID_37",
    "SENTINEL_ID_38",
    "SENTINEL_ID_39",
    "SENTINEL_ID_40",
    "SENTINEL_ID_41",
    "SENTINEL_ID_42",
    "SENTINEL_ID_43",
    "SENTINEL_ID_44",
    "SENTINEL_ID_45",
    "SENTINEL_ID_46",
    "SENTINEL_ID_47",
    "SENTINEL_ID_48",
    "SENTINEL_ID_49",
    "SENTINEL_ID_50",
    "SENTINEL_ID_51",
    "SENTINEL_ID_52",
    "SENTINEL_ID_53",
    "SENTINEL_ID_54",
    "SENTINEL_ID_55",
    "SENTINEL_ID_56",
    "SENTINEL_ID_57",
    "SENTINEL_ID_58",
    "SENTINEL_ID_59",
    "SENTINEL_ID_60",
    "SENTINEL_ID_61",
    "SENTINEL_ID_62",
    "SENTINEL_ID_63",
    "SENTINEL_ID_64",
    "SENTINEL_ID_65",
    "SENTINEL_ID_66",
    "SENTINEL_ID_67",
    "SENTINEL_ID_68",
    "SENTINEL_ID_69",
    "SENTINEL_ID_70",
    "SENTINEL_ID_71",
    "SENTINEL_ID_72",
    "SENTINEL_ID_73",
    "SENTINEL_ID_74",
    "SENTINEL_ID_75",
    "SENTINEL_ID_76",
    "SENTINEL_ID_77",
    "SENTINEL_ID_78",
    "SENTINEL_ID_79",
    "SENTINEL_ID_80",
    "SENTINEL_ID_81",
    "SENTINEL_ID_82",
    "SENTINEL_ID_83",
    "SENTINEL_ID_84",
    "SENTINEL_ID_85",
    "SENTINEL_ID_86",
    "SENTINEL_ID_87",
    "SENTINEL_ID_88",
    "SENTINEL_ID_89",
    "SENTINEL_ID_90",
    "SENTINEL_ID_91",
    "SENTINEL_ID_92",
    "SENTINEL_ID_93",
    "SENTINEL_ID_94",
    "SENTINEL_ID_95",
    "SENTINEL_ID_96",
    "SENTINEL_ID_97",
    "SENTINEL_ID_98",
    "SENTINEL_ID_99",
    "SENTINEL_ID_100",
    "SENTINEL_ID_101",
    "SENTINEL_ID_102",
    "SENTINEL_ID_103",
    "SENTINEL_ID_104",
    "SENTINEL_ID_105",
    "SENTINEL_ID_106",
    "SENTINEL_ID_107",
    "SENTINEL_ID_108",
    "SENTINEL_ID_109",
    "SENTINEL_ID_110",
    "SENTINEL_ID_111",
    "SENTINEL_ID_112",
    "SENTINEL_ID_113",
    "SENTINEL_ID_114",
    "SENTINEL_ID_115",
    "SENTINEL_ID_116",
    "SENTINEL_ID_117",
    "SENTINEL_ID_118",
    "SENTINEL_ID_119",
    "SENTINEL_ID_120",
    "SENTINEL_ID_121",
    "SENTINEL_ID_122",
    "SENTINEL_ID_123",
    "SENTINEL_ID_124",
    "SENTINEL_ID_125",
    "SENTINEL_ID_126",
    "SENTINEL_ID_127",
    "SENTINEL_ID_128",
    "SENTINEL_ID_129",
    "SENTINEL_ID_130",
    "SENTINEL_ID_131",
    "SENTINEL_ID_132",
    "SENTINEL_ID_133",
    "SENTINEL_ID_134",
    "SENTINEL_ID_135",
    "SENTINEL_ID_136",
    "SENTINEL_ID_137",
    "SENTINEL_ID_138",
    "SENTINEL_ID_139",
    "SENTINEL_ID_140",
    "SENTINEL_ID_141",
    "SENTINEL_ID_142",
    "SENTINEL_ID_143",
    "SENTINEL_ID_144",
    "SENTINEL_ID_145",
    "SENTINEL_ID_146",
    "SENTINEL_ID_147",
    "SENTINEL_ID_148",
    "SENTINEL_ID_149",
    "SENTINEL_ID_150",
    "SENTINEL_ID_151",
    "SENTINEL_ID_152",
    "SENTINEL_ID_153",
    "SENTINEL_ID_154",
    "SENTINEL_ID_155",
    "SENTINEL_ID_156",
    "SENTINEL_ID_157",
    "SENTINEL_ID_158",
    "SENTINEL_ID_159",
    "SENTINEL_ID_160",
    "SENTINEL_ID_161",
    "SENTINEL_ID_162",
    "SENTINEL_ID_163",
    "SENTINEL_ID_164",
    "SENTINEL_ID_165",
    "SENTINEL_ID_166",
    "SENTINEL_ID_167",
    "SENTINEL_ID_168",
    "SENTINEL_ID_169",
    "SENTINEL_ID_170",
    "SENTINEL_ID_171",
    "SENTINEL_ID_172",
    "SENTINEL_ID_173",
    "SENTINEL_ID_174",
    "SENTINEL_ID_175",
    "SENTINEL_ID_176",
    "SENTINEL_ID_177",
    "SENTINEL_ID_178",
    "SENTINEL_ID_179",
    "SENTINEL_ID_180",
    "SENTINEL_ID_181",
    "SENTINEL_ID_182",
    "SENTINEL_ID_183",
    "SENTINEL_ID_184",
    "SENTINEL_ID_185",
    "SENTINEL_ID_186",
    "SENTINEL_ID_187",
    "SENTINEL_ID_188",
    "SENTINEL_ID_189",
    "SENTINEL_ID_190",
    "SENTINEL_ID_191",
    "SENTINEL_ID_192",
    "SENTINEL_ID_193",
    "SENTINEL_ID_194",
    "SENTINEL_ID_195",
    "SENTINEL_ID_196",
    "SENTINEL_ID_197",
    "SENTINEL_ID_198",
    "SENTINEL_ID_199",
    "MOLECULAR_ENTITY_TYPE_ANTIGEN",
    "MOLECULAR_ENTITY_TYPE_ANTIBODY_LIGHT_CHAIN",
    "MOLECULAR_ENTITY_TYPE_ANTIBODY_HEAVY_CHAIN",
    "MOLECULAR_ENTITY_SMALL_MOLECULE",
    "ATTRIBUTE_ORGANISM",
    "ATTRIBUTE_ORGANISM_HUMAN",
    "ATTRIBUTE_ORGANISM_RABBIT",
    "ATTRIBUTE_ORGANISM_RAT",
    "ATTRIBUTE_ORGANISM_MOUSE",
    "ATTRIBUTE_ORGANISM_MONKEY",
    "ATTRIBUTE_ORGANISM_CAMEL",
    "EPITOPE_PARATOPE_PREDICTION",
    "CELL_TYPE_CLASS",
    "TISSUE_TYPE_CLASS",
    "MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED",
    "CORRUPTED_AREA_START",  # Indicates where the corruption area in the diffusion task starts
    "CORRUPTED_AREA_END",  # Indicates where the corruption area in the diffusion task ends
    "COMPLEX_ENTITY",
    "SUBMOLECULAR_ENTITY",
    "ALTERNATIVE",
    "GENERAL_CHAIN",
    "CDR3_REGION",
    "MUTATED",
    "SCALAR",
    "VECTOR",
    "MASKED_SCALAR",
    "MASKED_VECTOR",
    "AUTOENCODER_TASK",
    "DECODED_FROM_LATENT",
    "AUTOENCODER_LATENT_LOG_VARIANCE",
    "AUTOENCODER_LATENT_MEAN",
    "AUTOENCODER_LATENT_SAMPLED_Z",
    "BIOT5_TASK_ID",
]

AA_tokens = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def special_wrap_input(x: str) -> str:
    return special_token_marker[0] + x + special_token_marker[1]


def strip_special_wrap(x: str) -> str:
    for spec_wrap in special_token_marker:
        x = x.replace(spec_wrap, "")
    return x


def special_mark_AA(in_str: str) -> str:
    """wraps every character of the input with special markers

    Args:
        in_str (str): input string

    Returns:
        str: input with every character wrapped by special token markers
    """
    return "".join([special_wrap_input(x) for x in in_str])


def get_special_tokens_dict() -> Dict:
    """Returns a special token dict in huggingface format, with keys in [bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token]

    Returns:
        Dict: _description_
    """
    return {x: special_wrap_input(special_tokens[x]) for x in special_tokens}


def get_additional_tokens(subset: Optional[Union[str, List]] = None) -> List:
    """Wraps all tokens from special_tokens, task_tokens and AA_tokens with special_token_marker and returns them in a single list
    TODO: add a selection argument (by default - None, i.e. selects everything): a list containing the names of special token groups
    {'special_tokens', 'task_tokens', 'AA_tokens'} to return.


    Args:
        subset ([Union[str, List]]): a subset of ['special', 'task', 'AA'], marking which groups of tokens to include. Defaults to None (include all).

    Returns:
        List: _description_
    """
    if isinstance(subset, str):
        subset = [subset]
    if subset is not None:
        for x in subset:
            assert x in ["special", "task", "AA"]

    tokens_wrapped = []
    if subset is None or "special" in subset:
        tokens_wrapped += [special_wrap_input(x) for x in special_tokens.values()]
    if subset is None or "task" in subset:
        tokens_wrapped += [special_wrap_input(x) for x in task_tokens]
    if subset is None or "AA" in subset:
        tokens_wrapped += [special_wrap_input(x) for x in AA_tokens]
    return tokens_wrapped


IUPAC_CODES = OrderedDict(
    [
        ("Ala", "A"),
        ("Asx", "B"),  # Aspartate or Asparagine
        ("Cys", "C"),
        ("Asp", "D"),
        ("Glu", "E"),
        ("Phe", "F"),
        ("Gly", "G"),
        ("His", "H"),
        ("Ile", "I"),
        ("Lys", "K"),
        ("Leu", "L"),
        ("Met", "M"),
        ("Asn", "N"),
        ("Pyl", "O"),  # Pyrrolysin
        ("Pro", "P"),
        ("Gln", "Q"),
        ("Arg", "R"),
        ("Ser", "S"),
        ("Thr", "T"),
        ("Sec", "U"),  # Selenocysteine
        ("Val", "V"),
        ("Trp", "W"),
        ("Xaa", "X"),  # Any AA
        ("Tyr", "Y"),
        ("Glx", "Z"),  # Glutamate or Glutamine
    ]
)

IUPAC_VOCAB = OrderedDict(
    [
        ("<PAD>", 0),
        ("<MASK>", 1),
        ("<CLS>", 2),
        ("<SEP>", 3),
        ("<UNK>", 4),
        ("A", 5),
        ("B", 6),
        ("C", 7),
        ("D", 8),
        ("E", 9),
        ("F", 10),
        ("G", 11),
        ("H", 12),
        ("I", 13),
        ("K", 14),
        ("L", 15),
        ("M", 16),
        ("N", 17),
        ("O", 18),
        ("P", 19),
        ("Q", 20),
        ("R", 21),
        ("S", 22),
        ("T", 23),
        ("U", 24),
        ("V", 25),
        ("W", 26),
        ("X", 27),
        ("Y", 28),
        ("Z", 29),
        ("<START>", 30),
        ("<STOP>", 31),
    ]
)

UNIREP_VOCAB = OrderedDict(
    [
        ("<PAD>", 0),
        ("M", 1),
        ("R", 2),
        ("H", 3),
        ("K", 4),
        ("D", 5),
        ("E", 6),
        ("S", 7),
        ("T", 8),
        ("N", 9),
        ("Q", 10),
        ("C", 11),
        ("U", 12),
        ("G", 13),
        ("P", 14),
        ("A", 15),
        ("V", 16),
        ("I", 17),
        ("F", 18),
        ("Y", 19),
        ("W", 20),
        ("L", 21),
        ("O", 22),
        ("X", 23),
        ("Z", 23),
        ("B", 23),
        ("J", 23),
        ("<CLS>", 24),
        ("<SEP>", 25),
        ("<START>", 26),
        ("<STOP>", 27),
        ("<UNK>", 28),
        ("<MASK>", 29),
    ]
)

HUMAN_KINASE_ALIGNMENT_VOCAB = OrderedDict(
    [
        ("<PAD>", 0),
        ("<MASK>", 1),
        ("<CLS>", 2),
        ("<SEP>", 3),
        ("<UNK>", 4),
        ("-", 5),
        ("A", 6),
        ("C", 7),
        ("D", 8),
        ("E", 9),
        ("F", 10),
        ("G", 11),
        ("H", 12),
        ("I", 13),
        ("K", 14),
        ("L", 15),
        ("M", 16),
        ("N", 17),
        ("P", 18),
        ("Q", 19),
        ("R", 20),
        ("S", 21),
        ("T", 22),
        ("V", 23),
        ("W", 24),
        ("Y", 25),
        ("<START>", 26),
        ("<STOP>", 27),
    ]
)

"""
Polarity, Charge, Hydophobicity, Aromaticity, Ionizability, StartStop
Nonpolar = -1, Polar = 1
Negative = -1, Neutral = 0, Positive = 1
Hydrophobic = -1, Hydrophilic = 1
NonAromatic = -1, Aromatic = 1
NonIonizable = -1, Ionizable = 1
Stop = -1, Start = 1, Neither = 0
"""
AA_PROPERTIES_NUM = OrderedDict(
    [
        ("A", (-1, 0, -1, -1, -1, 0)),
        ("B", (1, -0.5, 1, -1, 0, 0)),  # mean of D, N
        ("C", (1, 0, 1, -1, 1, 0)),
        ("D", (1, -1, 1, -1, 1, 0)),
        ("E", (1, -1, 1, -1, 1, 0)),
        ("F", (-1, 0, -1, 1, -1, 0)),
        ("G", (-1, 0, -1, -1, -1, 0)),
        ("H", (1, 1, 1, -1, 1, 0)),
        ("I", (-1, 0, -1, -1, -1, 0)),
        ("K", (1, 1, 1, -1, 1, 0)),
        ("L", (-1, 0, -1, -1, -1, 0)),
        ("M", (-1, 0, -1, -1, -1, 0)),
        ("N", (1, 0, 1, -1, -1, 0)),
        ("O", (1, 0, 1, 1, 1, 0)),  # Pyrrolysine
        ("P", (-1, 0, -1, -1, -1, 0)),
        ("Q", (1, 0, 1, -1, -1, 0)),
        ("R", (1, 1, 1, -1, 1, 0)),
        ("S", (1, 0, 1, -1, -1, 0)),
        ("T", (1, 0, 1, -1, -1, 0)),
        ("U", (-1, 0, -1, -1, 1, 0)),  # Selenocyteine
        ("V", (-1, 0, -1, -1, -1, 0)),
        ("W", (-1, 0, -1, 1, -1, 0)),
        ("X", (0.2, 0, 0.1, -0.7, -0.2, 0)),  # mean AA (Unknown)
        ("Y", (1, 0, -1, 1, 1, 0)),
        ("Z", (1, -0.5, 1, -1, 0, 0)),  # mean of E, Q
        ("<PAD>", (0, 0, 0, 0, 0, 0)),
        ("<START>", (0, 0, 0, 0, 0, 1)),
        ("<STOP>", (0, 0, 0, 0, 0, -1)),
    ]
)
"""
Molecular Weight, Residue Weight, pKa, pKb, pKx, pI, Hydrophobicity at pH2
Taken from: https://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html
"""
AA_FEAT = OrderedDict(
    [
        ("A", (89.1, 71.08, 2.34, 9.69, 0, 6, 47, 0)),
        ("B", (132.615, 114.6, 1.95, 9.2, 1.825, 4.09, -29.5, 0)),  # D/N mean
        ("C", (121.16, 103.15, 1.96, 10.28, 8.18, 5.07, 52, 0)),
        ("D", (133.11, 115.09, 1.88, 9.6, 3.65, 2.77, -18, 0)),
        ("E", (147.13, 129.12, 2.19, 9.67, 4.25, 3.22, 8, 0)),
        ("F", (165.19, 147.18, 1.83, 9.13, 0, 5.48, 92, 0)),
        ("G", (75.07, 57.05, 2.34, 9.6, 0, 5.97, 0, 0)),
        ("H", (155.16, 137.14, 1.82, 9.17, 6, 7.59, -42, 0)),
        ("I", (131.18, 113.16, 2.36, 9.6, 0, 6.02, 100, 0)),
        ("K", (146.19, 128.18, 2.18, 8.95, 10.53, 9.74, -37, 0)),
        ("L", (131.18, 113.16, 2.36, 9.6, 0, 5.98, 100, 0)),
        ("M", (149.21, 131.2, 2.28, 9.21, 0, 5.74, 74, 0)),
        ("N", (132.12, 114.11, 2.02, 8.8, 0, 5.41, -41, 0)),
        ("O", (131.13, 113.11, 1.82, 9.65, 0, 0, 0, 0)),  # Pyrrolysine
        ("P", (115.13, 97.12, 1.99, 10.6, 0, 6.3, -46, 0)),
        ("Q", (146.15, 128.13, 2.17, 9.13, 0, 5.65, -18, 0)),
        ("R", (174.2, 156.19, 2.17, 9.04, 12.48, 10.76, -26, 0)),
        ("S", (105.09, 87.08, 2.21, 9.15, 0, 5.68, -7, 0)),
        ("T", (119.12, 101.11, 2.09, 9.1, 0, 5.6, 13, 0)),
        ("U", (168.07, 150.05, 5.47, 10.28, 0, 3.9, 52, 0)),  # Selenocyteine
        ("V", (117.15, 99.13, 2.32, 9.62, 0, 5.96, 97, 0)),
        ("W", (204.23, 186.22, 2.83, 9.39, 0, 5.89, 84, 0)),
        (
            "X",
            (136.74, 118.73, 2.06, 9.00, 2.51, 5.74, 21.86, 0),
        ),  # mean AA (Unknown)
        ("Y", (181.19, 163.18, 2.2, 9.11, 10.07, 5.66, 49, 0)),
        (
            "Z",
            (146.64, 128.625, 2.18, 9.4, 2.125, 4.435, -5, 0),
        ),  # mean of E, Q
        ("<PAD>", (0, 0, 0, 0, 0, 0, 0, 0)),
        ("<START>", (0, 0, 0, 0, 0, 0, 0, 1)),
        ("<STOP>", (0, 0, 0, 0, 0, 0, 0, -1)),
    ]
)
"""
Taken from: https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
"""
# yapf: disable
BLOSUM62 = OrderedDict(
    [

        (
            'A', (
                4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0,
                -3, -2, 0, -2, -1, 0, -4, -4, -4
            )
        ),
        (
            'B', (
                -2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1,
                -4, -3, -3, 4, 1, -1, -4, -4, -4
            )
        ),
        (
            'C', (
                0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1,
                -1, -2, -2, -1, -3, -3, -2, -4, -4, -4
            )
        ),
        (
            'D', (
                -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1,
                -4, -3, -3, 4, 1, -1, -4, -4, -4
            )
        ),
        (
            'E', (
                -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3,
                -2, -2, 1, 4, -1, -4, -4, -4
            )
        ),
        (
            'F', (
                -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2,
                1, 3, -1, -3, -3, -1, -4, -4, -4
            )
        ),
        (
            'G', (
                0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2,
                -2, -3, -3, -1, -2, -1, -4, -4, -4
            )
        ),
        (
            'H', (
                -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2,
                -2, 2, -3, 0, 0, -1, -4, -4, -4
            )
        ),
        (
            'I', (
                -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1,
                -3, -1, 3, -3, -3, -1, -4, -4, -4
            )
        ),
        (
            'K', (
                -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1,
                -3, -2, -2, 0, 1, -1, -4, -4, -4
            )
        ),
        (
            'L', (
                -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1,
                -2, -1, 1, -4, -3, -1, -4, -4, -4
            )
        ),
        (
            'M', (
                -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1,
                -1, -1, 1, -3, -1, -1, -4, -4, -4
            )
        ),
        (
            'N', (
                -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4,
                -2, -3, 3, 0, -1, -4, -4, -4
            )
        ),
        (
            'O', (
                0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0,
                0, -2, -1, -1, -1, -1, -1, -4, -4, -4
            )
        ),  # Pyrrolysine encoded as unknown
        (
            'P', (
                -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1,
                -1, -4, -3, -2, -2, -1, -2, -4, -4, -4
            )
        ),
        (
            'Q', (
                -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2,
                -1, -2, 0, 3, -1, -4, -4, -4
            )
        ),
        (
            'R', (
                -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1,
                -3, -2, -3, -1, 0, -1, -4, -4, -4
            )
        ),
        (
            'S', (
                1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3,
                -2, -2, 0, 0, 0, -4, -4, -4
            )
        ),
        (
            'T', (
                0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5,
                -2, -2, 0, -1, -1, 0, -4, -4, -4
            )
        ),
        (
            'U', (
                0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0,
                0, -2, -1, -1, -1, -1, -1, -4, -4, -4
            )
        ),  # Selenocysteine encoded as unknown
        (
            'V', (
                0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0,
                -3, -1, 4, -3, -2, -1, -4, -4, -4
            )
        ),
        (
            'W', (
                -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3,
                -2, 11, 2, -3, -4, -3, -2, -4, -4, -4
            )
        ),
        (
            'X', (
                0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0,
                0, -2, -1, -1, -1, -1, -1, -4, -4, -4
            )
        ),
        (
            'Y', (
                -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2,
                -2, 2, 7, -1, -3, -2, -1, -4, -4, -4
            )
        ),
        (
            'Z', (
                -1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3,
                -2, -2, 1, 4, -1, -4, -4, -4
            )
        ),
        (
            '<PAD>', (
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, -4, -4, -4, -4, -4, 10, -4, -4
            )
        ),
        (
            '<START>', (
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, -4, -4, -4, -4, -4, -4, 10, -4
            )
        ),
        (
            '<STOP>', (
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, -4, -4, -4, -4, -4, -4, -4, 10
            )
        )
    ]
)

BLOSUM62_NORM = OrderedDict(
    [
        (
            'A', (
                0.2901, 0.0310, 0.0256, 0.0297, 0.0216, 0.0256, 0.0405, 0.0783,
                0.0148, 0.0432, 0.0594, 0.0445, 0.0175, 0.0216, 0.0297, 0.0850,
                0.0499, 0.0054, 0.0175, 0.0688, 0, 0
            )
        ),
        (
            'B', (
                0.0419, 0.0374, 0.1930, 0.2403, 0.0083, 0.0318, 0.0704, 0.0559,
                0.0251, 0.0225, 0.0298, 0.0494, 0.0103, 0.0165, 0.0213, 0.0610,
                0.0424, 0.0041, 0.0135, 0.0257, 0, 0
            )
        ),
        (
            'C', (
                0.0650, 0.0163, 0.0163, 0.0163, 0.4837, 0.0122, 0.0163, 0.0325,
                0.0081, 0.0447, 0.0650, 0.0203, 0.0163, 0.0203, 0.0163, 0.0407,
                0.0366, 0.0041, 0.0122, 0.0569, 0, 0
            )
        ),
        (
            'D', (
                0.0410, 0.0299, 0.0690, 0.3974, 0.0075, 0.0299, 0.0914, 0.0466,
                0.0187, 0.0224, 0.0280, 0.0448, 0.0093, 0.0149, 0.0224, 0.0522,
                0.0354, 0.0037, 0.0112, 0.0243, 0, 0
            )
        ),
        (
            'E', (
                0.0552, 0.0497, 0.0405, 0.0902, 0.0074, 0.0645, 0.2965, 0.0350,
                0.0258, 0.0221, 0.0368, 0.0755, 0.0129, 0.0166, 0.0258, 0.0552,
                0.0368, 0.0055, 0.0166, 0.0313, 0, 0
            )
        ),
        (
            'F', (
                0.0338, 0.0190, 0.0169, 0.0169, 0.0106, 0.0106, 0.0190, 0.0254,
                0.0169, 0.0634, 0.1142, 0.0190, 0.0254, 0.3869, 0.0106, 0.0254,
                0.0254, 0.0169, 0.0888, 0.0550, 0, 0
            )
        ),
        (
            'G', (
                0.0783, 0.0229, 0.0391, 0.0337, 0.0108, 0.0189, 0.0256, 0.5101,
                0.0135, 0.0189, 0.0283, 0.0337, 0.0094, 0.0162, 0.0189, 0.0513,
                0.0297, 0.0054, 0.0108, 0.0243, 0, 0
            )
        ),
        (
            'H', (
                0.0420, 0.0458, 0.0534, 0.0382, 0.0076, 0.0382, 0.0534, 0.0382,
                0.3550, 0.0229, 0.0382, 0.0458, 0.0153, 0.0305, 0.0191, 0.0420,
                0.0267, 0.0076, 0.0573, 0.0229, 0, 0
            )
        ),
        (
            'I', (
                0.0471, 0.0177, 0.0147, 0.0177, 0.0162, 0.0133, 0.0177, 0.0206,
                0.0088, 0.2710, 0.1679, 0.0236, 0.0368, 0.0442, 0.0147, 0.0250,
                0.0398, 0.0059, 0.0206, 0.1767, 0, 0
            )
        ),
        (
            'K', (
                0.0570, 0.1071, 0.0415, 0.0415, 0.0086, 0.0535, 0.0708, 0.0432,
                0.0207, 0.0276, 0.0432, 0.2781, 0.0155, 0.0155, 0.0276, 0.0535,
                0.0397, 0.0052, 0.0173, 0.0328, 0, 0
            )
        ),
        (
            'L', (
                0.0445, 0.0243, 0.0142, 0.0152, 0.0162, 0.0162, 0.0202, 0.0213,
                0.0101, 0.1154, 0.3755, 0.0253, 0.0496, 0.0547, 0.0142, 0.0243,
                0.0334, 0.0071, 0.0223, 0.0962, 0, 0
            )
        ),
        (
            'M', (
                0.0522, 0.0321, 0.0201, 0.0201, 0.0161, 0.0281, 0.0281, 0.0281,
                0.0161, 0.1004, 0.1968, 0.0361, 0.1606, 0.0482, 0.0161, 0.0361,
                0.0402, 0.0080, 0.0241, 0.0924, 0, 0
            )
        ),
        (
            'N', (
                0.0427, 0.0449, 0.3169, 0.0831, 0.0090, 0.0337, 0.0494, 0.0652,
                0.0315, 0.0225, 0.0315, 0.0539, 0.0112, 0.0180, 0.0202, 0.0697,
                0.0494, 0.0045, 0.0157, 0.0270, 0, 0
            )
        ),
        (
            'O', (
                0.0649, 0.0515, 0.0528, 0.0616, 0.0332, 0.0416, 0.0616, 0.0608,
                0.0346, 0.0554, 0.0793, 0.0575, 0.0252, 0.0471, 0.0416, 0.0555,
                0.0484, 0.0291, 0.0381, 0.0602, 0, 0
            )
        ),  # Pyrrolysin encoded as unknown
        (
            'P', (
                0.0568, 0.0258, 0.0233, 0.0310, 0.0103, 0.0207, 0.0362, 0.0362,
                0.0129, 0.0258, 0.0362, 0.0413, 0.0103, 0.0129, 0.4935, 0.0439,
                0.0362, 0.0026, 0.0129, 0.0310, 0, 0
            )
        ),
        (
            'Q', (
                0.0559, 0.0735, 0.0441, 0.0471, 0.0088, 0.2147, 0.1029, 0.0412,
                0.0294, 0.0265, 0.0471, 0.0912, 0.0206, 0.0147, 0.0235, 0.0559,
                0.0412, 0.0059, 0.0206, 0.0353, 0, 0
            )
        ),
        (
            'R', (
                0.0446, 0.3450, 0.0388, 0.0310, 0.0078, 0.0484, 0.0523, 0.0329,
                0.0233, 0.0233, 0.0465, 0.1202, 0.0155, 0.0174, 0.0194, 0.0446,
                0.0349, 0.0058, 0.0174, 0.0310, 0, 0
            )
        ),
        (
            'S', (
                0.1099, 0.0401, 0.0541, 0.0489, 0.0175, 0.0332, 0.0524, 0.0663,
                0.0192, 0.0297, 0.0419, 0.0541, 0.0157, 0.0209, 0.0297, 0.2199,
                0.0820, 0.0052, 0.0175, 0.0419, 0, 0
            )
        ),
        (
            'T', (
                0.0730, 0.0355, 0.0434, 0.0375, 0.0178, 0.0276, 0.0394, 0.0434,
                0.0138, 0.0533, 0.0651, 0.0454, 0.0197, 0.0237, 0.0276, 0.0927,
                0.2465, 0.0059, 0.0178, 0.0710, 0, 0
            )
        ),
        (
            'U', (
                0.0649, 0.0515, 0.0528, 0.0616, 0.0332, 0.0416, 0.0616, 0.0608,
                0.0346, 0.0554, 0.0793, 0.0575, 0.0252, 0.0471, 0.0416, 0.0555,
                0.0484, 0.0291, 0.0381, 0.0602, 0, 0
            )
        ),  # Selenocysteine encoded as unknown
        (
            'V', (
                0.0700, 0.0219, 0.0165, 0.0178, 0.0192, 0.0165, 0.0233, 0.0247,
                0.0082, 0.1646, 0.1303, 0.0261, 0.0316, 0.0357, 0.0165, 0.0329,
                0.0494, 0.0055, 0.0206, 0.2689, 0, 0
            )
        ),
        (
            'W', (
                0.0303, 0.0227, 0.0152, 0.0152, 0.0076, 0.0152, 0.0227, 0.0303,
                0.0152, 0.0303, 0.0530, 0.0227, 0.0152, 0.0606, 0.0076, 0.0227,
                0.0227, 0.4924, 0.0682, 0.0303, 0, 0
            )
        ),
        (
            'X', (
                0.0649, 0.0515, 0.0528, 0.0616, 0.0332, 0.0416, 0.0616, 0.0608,
                0.0346, 0.0554, 0.0793, 0.0575, 0.0252, 0.0471, 0.0416, 0.0555,
                0.0484, 0.0291, 0.0381, 0.0602, 0, 0
            )
        ),
        (
            'Y', (
                0.0405, 0.0280, 0.0218, 0.0187, 0.0093, 0.0218, 0.0280, 0.0249,
                0.0467, 0.0436, 0.0685, 0.0312, 0.0187, 0.1308, 0.0156, 0.0312,
                0.0280, 0.0280, 0.3178, 0.0467, 0, 0
            )
        ),
        (
            'Z', (
                0.0556, 0.0616, 0.0423, 0.0687, 0.0081, 0.1396, 0.1997, 0.0381,
                0.0276, 0.0243, 0.0420, 0.0834, 0.0168, 0.0157, 0.0247, 0.0556,
                0.0390, 0.0057, 0.0186, 0.0333, 0, 0
            )
        ),
        (
            '<PAD>', (
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0
            )
        ),
        (
            '<START>', (
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0
            )
        ),
        (
            '<STOP>', (
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1
            )
        )
    ]
)
