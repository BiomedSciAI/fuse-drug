from pytorch_lightning.callbacks import Callback
import os
import sys
import pytoda
from pathlib import Path
from typing import Optional, Dict


def _check_stopfile(stop_filename: str) -> None:
    if os.path.isfile(stop_filename):
        print(f"detected request stop file: [{stop_filename}]. Exiting from process.")
        sys.stdout.flush()
        sys.exit()


class ExitOnStopFileCallback(Callback):
    def __init__(self, stop_filename: Optional[str] = None):
        super().__init__()
        if not isinstance(stop_filename, str):
            raise Exception("stop_filename must be str")
        self.stop_filename = stop_filename
        print(
            f"ExitOnStopFileCallback: To stop the session (even if it is detached) create a file named: {self.stop_filename}"
        )

    def on_predict_batch_start(self, **kwargs: Dict) -> None:
        _check_stopfile(self.stop_filename)

    def on_train_batch_start(self, **kwargs: Dict) -> None:
        # RuntimeError: The `Callback.on_batch_start` hook was removed in v1.8. Please use `Callback.on_train_batch_start` instead.
        _check_stopfile(self.stop_filename)

    def on_test_batch_start(self, **kwargs: Dict) -> None:
        _check_stopfile(self.stop_filename)


def pytoda_ligand_tokenizer_path() -> str:
    path = Path(pytoda.__file__)
    path = os.path.join(path.parent.absolute(), "smiles", "metadata", "tokenizer", "vocab.json")
    return path
