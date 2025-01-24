from typing import Optional, Callable
import mmap
import os
import sqlite3
#from Bio import IndexedFastaIO

class FastFASTAReader:
    def __init__(self, fasta_path:str, index_file_path:Optional[str]=None, entry_id_process_func:Optional[Callable[[str],str]]=None):
        """
        Create an optimized index for rapid FASTA sequence lookup.
        
        Args:
            fasta_path (str): Path to FASTA file
            index_file_path (str): Optional. The file path of the index file. If not provided, will defauylt to {fasta_path}.sqlite
        """
        self.fasta_path = fasta_path
        self.index_file_path = index_file_path
        if (self.index_file_path is None):
            self.index_file_path = f"{self.fasta_path}.sqlite"

        self.entry_id_process_func = entry_id_process_func

        self._create_sqlite_index()
    
    def _create_sqlite_index(self):
        """
        Create SQLite index for fast sequence lookup.
        """
        # SQLite database to store file offsets

        already_found = os.path.isfile(self.index_file_path)
        self.conn = sqlite3.connect(self.index_file_path)
        cursor = self.conn.cursor()

        if not already_found:
            seen_so_far = 0

            # Create index table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fasta_index (
                    seq_id TEXT PRIMARY KEY,
                    offset INTEGER
                )
            ''')
                            
            with open(self.fasta_path, 'rb') as f:
                mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Track file position
                current_pos = 0
                while current_pos < len(mmapped_file):
                    # Find next '>' marker
                    next_marker = mmapped_file.find(b'>', current_pos)
                    if next_marker == -1:
                        break
                    
                    # Extract sequence ID
                    end_id = mmapped_file.find(b'\n', next_marker)
                    seq_id = mmapped_file[next_marker+1:end_id].decode('utf-8').split()[0]

                    if self.entry_id_process_func is not None:
                        seq_id = self.entry_id_process_func(seq_id)
                        assert seq_id!="", "Found an empty id! make sure that your FASTA file is fine and that the provided entry_id_process_func is fine"
                    
                    # Store offset
                    cursor.execute('INSERT OR IGNORE INTO fasta_index VALUES (?, ?)', 
                                (seq_id, next_marker))
                    
                    # Move to next sequence
                    current_pos = end_id + 1

                    seen_so_far+= 1

                    if (not (seen_so_far%1_000_000)):
                        print(f"seen_so_far={seen_so_far}")

                    if not (seen_so_far%1000):
                        self.conn.commit()
            
            self.conn.commit()

        # #count table size (takes too long on 300M+ database, commented out for now)
        # res = cursor.execute('''
        #     SELECT COUNT(*) FROM fasta_index
        # ''')        
        # res.fetchall()
        # print(res)
    
    def get_sequence(self, seq_id):
        """
        Retrieve sequence by ID with O(1) lookup.
        
        Args:
            seq_id (str): Sequence identifier
        
        Returns:
            str: DNA/protein sequence
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT offset FROM fasta_index WHERE seq_id = ?', (seq_id,))
        result = cursor.fetchone()
        
        if not result:
            return None
        
        offset = result[0]
        
        # Memory-mapped file reading
        with open(self.fasta_path, 'rb') as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Find sequence start and end
            seq_start = mmapped_file.find(b'\n', offset) + 1
            seq_end = mmapped_file.find(b'>', seq_start)
            
            if seq_end == -1:
                seq_end = len(mmapped_file)
            
            # Extract and clean sequence
            sequence = mmapped_file[seq_start:seq_end].replace(b'\n', b'').decode('utf-8')
            
            return sequence

