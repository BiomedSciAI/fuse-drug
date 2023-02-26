from pytorch_lightning.callbacks import Callback
import os, sys
import pytoda
from pathlib import Path

def _check_stopfile(stop_filename):
    if os.path.isfile(stop_filename):
        print(f'detected request stop file: [{stop_filename}]. Exiting from process.')
        sys.stdout.flush()
        sys.exit()
class ExitOnStopFileCallback(Callback):
    def __init__(self, stop_filename=None):
        super().__init__()
        if not isinstance(stop_filename, str):
            raise Exception('stop_filename must be str')
        self.stop_filename = stop_filename
        print(f'ExitOnStopFileCallback: To stop the session (even if it is detached) create a file named: {self.stop_filename}')
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        _check_stopfile(self.stop_filename)

    def on_batch_start(self, trainer, pl_module):
        _check_stopfile(self.stop_filename)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        _check_stopfile(self.stop_filename)

def pytoda_ligand_tokenizer_path():
    path = Path(pytoda.__file__)
    path = os.path.join(path.parent.absolute(), 'smiles', 'metadata', 'tokenizer', 'vocab.json')
    return path