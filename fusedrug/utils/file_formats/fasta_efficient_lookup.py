import h5py
#import numpy as np
from Bio import SeqIO
#import mmap

class LargeFASTAConverter:
    def __init__(self, chunk_size=100000):
        self.chunk_size = chunk_size
        self.total_seen = 0
    
    def convert_large_fasta(self, input_fasta, output_hdf5):
        """
        Convert FASTA to HDF5 using sequence IDs as keys.
        
        Args:
            input_fasta (str): Path to input FASTA file
            output_hdf5 (str): Path to output HDF5 file
        """
        with h5py.File(output_hdf5, 'w') as h5file:
            # Create a group to store sequences
            sequences_group = h5file.create_group('sequences')
            
            # Process file in chunks
            for record_batch in self._batch_iterator(input_fasta):
                for record in record_batch:
                    # Use sequence ID as the key
                    sequences_group.create_dataset(
                        record.id, 
                        data=str(record.seq),
                        dtype=h5py.special_dtype(vlen=str)
                    )
        
        print(f"Converted FASTA to HDF5 with string keys")
    
    def lookup_sequence_by_id(self, hdf5_path, sequence_id):
        """
        Lookup sequence using string ID.
        
        Args:
            hdf5_path (str): Path to HDF5 file
            sequence_id (str): ID to lookup
        
        Returns:
            str: Sequence corresponding to the ID, or None if not found
        """
        with h5py.File(hdf5_path, 'r') as h5file:
            sequences_group = h5file['sequences']
            
            if sequence_id in sequences_group:
                return sequences_group[sequence_id][()]
            
            return None
    
    def _batch_iterator(self, fasta_path):
        """
        Memory-efficient batch iterator for large FASTA files.
        """
        with open(fasta_path, 'r') as handle:
            batch = []
            for record in SeqIO.parse(handle, 'fasta'):
                batch.append(record)
                self.total_seen += 1
                if ( not (self.total_seen%100_000)):
                    print(f"self.total_seen={self.total_seen}")
                
                if len(batch) == self.chunk_size:
                    yield batch
                    batch = []
            
            # Yield any remaining records
            if batch:
                yield batch

if __name__ == "__main__":
    converter = LargeFASTAConverter(chunk_size=500000)

    is_sample = "" #"_SAMPLE"
    
    converter.convert_large_fasta(f'/proj/bmfm/datasets/uniprot/uniprot_trembl{is_sample}.fasta', f'/proj/bmfm/datasets/uniprot/uniprot_trembll{is_sample}.h5')
    
    # Lookup a specific sequence
    seq = converter.lookup_sequence_by_id(f'/proj/bmfm/datasets/uniprot/uniprot_trembl{is_sample}.h5', 'tr|A0A7C1X5W1|A0A7C1X5W1_9CREN')
    print(seq)