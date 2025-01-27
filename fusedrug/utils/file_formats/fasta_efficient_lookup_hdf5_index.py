import h5py
from Bio import SeqIO
from typing import List


class LargeFASTAConverter:
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
        self.total_seen = 0

    def convert_large_fasta_to_hdf5(self, input_fasta: str, output_hdf5: str) -> None:
        """
        Convert FASTA to HDF5 using sequence IDs as keys.

        Args:
            input_fasta (str): Path to input FASTA file
            output_hdf5 (str): Path to output HDF5 file
        """
        with h5py.File(output_hdf5, "w") as h5file:
            # Create a group to store sequences
            sequences_group = h5file.create_group("sequences")

            # Process file in chunks
            for record_batch in self._batch_iterator(input_fasta):
                for record in record_batch:
                    # Use sequence ID as the key
                    sequences_group.create_dataset(
                        record.id,
                        data=str(record.seq),
                        dtype=h5py.special_dtype(vlen=str),
                    )

        print("Converted FASTA to HDF5 with string keys")

    def lookup_sequence_by_id(self, hdf5_path: str, sequence_id: str) -> str:
        """
        Lookup sequence using string ID.

        Args:
            hdf5_path (str): Path to HDF5 file
            sequence_id (str): ID to lookup

        Returns:
            str: Sequence corresponding to the ID, or None if not found
        """
        with h5py.File(hdf5_path, "r") as h5file:
            sequences_group = h5file["sequences"]

            if sequence_id in sequences_group:
                return sequences_group[sequence_id][()]

            return None

    def _batch_iterator(self, fasta_path: str) -> List:
        """
        Memory-efficient batch iterator for large FASTA files.
        """
        with open(fasta_path, "r") as handle:
            batch = []
            for record in SeqIO.parse(handle, "fasta"):
                batch.append(record)
                self.total_seen += 1
                if not (self.total_seen % 100_000):
                    print(f"self.total_seen={self.total_seen}")

                if len(batch) == self.chunk_size:
                    yield batch
                    batch = []

            # Yield any remaining records
            if batch:
                yield batch
