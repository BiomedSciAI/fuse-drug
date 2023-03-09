from torch.utils.data import IterableDataset
from rdkit import Chem


class SDFIterableDataset(IterableDataset):
    """
    Usage example: see tests/test_sdf_iterable_dataset.py
    """

    def __init__(self, sdf_filename):
        super().__init__()
        self._sdf_filename = sdf_filename

    def __iter__(self):
        with Chem.MultithreadedSDMolSupplier(self._sdf_filename) as sdSupl:
            for mol in sdSupl:
                if mol is not None:
                    yield mol
