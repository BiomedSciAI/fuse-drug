from transformers import AutoModel, AutoTokenizer
import abnumber
from typing import Any

"""
    https://github.com/oxpig/AbLang

    Res-codings: These encodings are 768 values for each residue, useful for residue specific predictions.

    Seq-codings: These encodings are 768 values for each sequence, useful for sequence specific predictions. The same length of encodings for each sequence, means these encodings also removes the need to align antibody sequences.

    Res-likelihoods: These encodings are the likelihoods of each amino acid at each position in a given antibody sequence, useful for exploring possible mutations.



    https://github.com/TobiasHeOl/AbLang/blob/main/examples/example-ablang-usecases.ipynb

    An additional feature, is the ability to align the rescodings. This can be done by setting the parameter align to "True".


    ##### LORA on it!
    https://huggingface.co/qilowoq/AbLang_light

    nice lora usage example here:
    https://huggingface.co/qilowoq/AbLang_heavy

"""


class AbLang:
    def __init__(self) -> None:
        self.tokenizer_light = AutoTokenizer.from_pretrained("qilowoq/AbLang_light")
        self.model_light = AutoModel.from_pretrained(
            "qilowoq/AbLang_light", trust_remote_code=True
        )

        self.tokenizer_heavy = AutoTokenizer.from_pretrained("qilowoq/AbLang_heavy")
        self.model_heavy = AutoModel.from_pretrained(
            "qilowoq/AbLang_heavy", trust_remote_code=True
        )

    def get_embeddings(
        self,
        seq: str,
        chain_type: str = "auto",
        remove_cls_token: bool = True,
        scheme: str = "chothia",
    ) -> Any:
        assert chain_type in ["light", "heavy", "auto"]
        orig_seq_len = len(seq)
        assert len(seq.strip()) == len(
            seq
        ), "expecting high case AA sequence WITHOUT spaces"

        seq = " ".join(seq)

        if chain_type == "auto":
            chain = abnumber.Chain(seq, scheme=scheme)
            chain_type = "heavy" if chain.is_heavy_chain() else "light"

        if chain_type == "light":
            use_tokenizer = self.tokenizer_light
            use_model = self.model_light
        elif chain_type == "heavy":
            use_tokenizer = self.tokenizer_heavy
            use_model = self.model_heavy
        else:
            assert False

        encoded_input = use_tokenizer(seq, return_tensors="pt")
        model_output = use_model(**encoded_input)

        assert model_output.last_hidden_state.shape[0] == 1

        last_hidden_state = model_output.last_hidden_state[0, 1:-1, ...]
        assert orig_seq_len == last_hidden_state.shape[0]
        return last_hidden_state


# kept for reference
def alternative_ablang_usage_approach() -> None:
    import ablang

    heavy_ablang = ablang.pretrained(
        "heavy"
    )  # Use "light" if you are working with light chains
    heavy_ablang.freeze()

    seqs = [
        "EVKLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
        "EV*LVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
        "*************PGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNK*YADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTL*****",
    ]

    encodings = heavy_ablang(seqs, mode="rescoding")  # 'restore')
    print(encodings)


if __name__ == "__main__":
    ablang = AbLang()

    emb = ablang.get_embeddings(
        seq="EVQLQESGPGLVKPSETLSLTCTVSGGPINNAYWTWIRQPPGKGLEYLGYVYHTGVTNYNPSLKSRLTITIDTSRKQLSLSLKFVTAADSAVYYCAREWAEDGDFGNAFHVWGQGTMVAVSSASTKGPSVFPLAPSSKSTSGGTAALGCL",
    )

    banana = 123
