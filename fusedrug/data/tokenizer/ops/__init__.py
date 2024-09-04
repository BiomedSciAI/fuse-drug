# keeping name for backward compatibility
from fusedrug.data.tokenizer.ops.tokenizer_op import TokenizerOp as FastTokenizer
from fusedrug.data.tokenizer.ops.modular_tokenizer_op import (
    ModularTokenizerWithoutInjectOp as FastModularTokenizer,
)
from fusedrug.data.tokenizer.ops.modular_tokenizer_op import (
    ModularTokenizerOp as InjectorTokenizerOp,
)
