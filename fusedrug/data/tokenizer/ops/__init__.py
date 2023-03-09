from .fast_tokenizer_ops import FastTokenizer

try:
    from .pytoda_tokenizer import Op_pytoda_SMILESTokenizer, Op_pytoda_ProteinTokenizer
except ImportError:
    pass
