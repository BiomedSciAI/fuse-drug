try:
    from biophi.humanization.methods.humanization import (
        humanize_chain,
        SapiensHumanizationParams,
        HumanizationParams,
    )
except ImportError:
    print(
        "WARNING: could not import biophi.humanization. If biophi is not install please run the following within your conda env: conda install biophi -c bioconda -c conda-forge --override-channels"
    )
    raise

try:
    from abnumber import (
        Chain,
    )  # , ChainParseError, SUPPORTED_CDR_DEFINITIONS, SUPPORTED_SCHEMES

except ImportError:
    print(
        "ERROR: had a problem importing abnumber, please install using 'conda install -c bioconda abnumber'"
    )
    raise
from typing import Tuple


def calculate_sapiens_humanness_mean_score(sequence: str) -> Tuple[float, str]:
    """
    Calculates the mean (accross all of the sequence residues) of humanness score
    """
    if not isinstance(sequence, str) or (len(sequence) == 0):
        print(
            f"calculate_sapiens_humanness_mean_score::ERROR: sequence is not string! sequence={sequence}"
        )
        return 0.0, "illegal_sequence"
    humanization_params = SapiensHumanizationParams(
        model_version="latest",
        humanize_cdrs=True,
        scheme=HumanizationParams.cdr_definition,
        cdr_definition=HumanizationParams.cdr_definition,
        iterations=1,
    )

    # they seem to be using 'kabat' scheme by default
    chain = Chain(
        sequence,
        name="dummy",
        scheme=humanization_params.scheme,
        cdr_definition=humanization_params.cdr_definition,
    )

    humanization = humanize_chain(chain, params=humanization_params)
    scores = humanization.to_score_dataframe()
    seq = chain.seq

    # chain_label = 'H' if chain.is_heavy_chain() else 'L'

    scores_by_pos_and_aa = scores.melt(ignore_index=False).set_index(
        "variable", append=True
    )["value"]
    seq_scores = scores_by_pos_and_aa.loc[list(enumerate(seq))]
    mean_score = seq_scores.mean()
    return mean_score, "heavy" if chain.is_heavy_chain() else "light"
