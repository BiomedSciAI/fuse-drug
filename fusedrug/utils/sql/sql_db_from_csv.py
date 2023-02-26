from typing import Optional, List
import sqlite3
import pandas as pd
import numpy as np
import time
from ast import literal_eval
import psycopg2
import os

class SQLfromCSV():
    """
    Create an SQL database from a (potentially) very large CSV/TSV file.
    
    Usage example:
    ==============
    Create a database with two tables (or add two tables to an existing database):
    
    ```
    sql_creator = SQLfromCSV(db_filepath="/path/to/local/db.sqlite3")
    sql_creator.add_table(input_filepath="/path/to/csv/or/tsv/file1.tsv", 
                            table_name='table1', chunk_size=100000)
    sql_creator.add_table(input_filepath="/path/to/csv/or/tsv/file2.tsv", 
                            table_name='table2', chunk_size=100000)
    ```

    """
    def __init__(self,  db_dir:str,
                        db_name: str,
                        db_type:str='sqlite3'):
        """
        :param db_dir: path to output database directory. For sqlite the file extension is ".sqlite3"
        :param db_name: name for the database
        :param db_type: type of SQL database. currently only sqlite is implemented.
        """
        self.db_dir = db_dir
        self.db_name = db_name

        if db_type.lower() in ('sqlite3', 'sqlite'):
            # Currently only sqlite is implemented. But probably only this 
            # basic code block will need to be changed for other databases.

            # see: https://stackoverflow.com/a/49507642/8019724
            sqlite3.register_adapter(np.int64, lambda val: int(val))
            sqlite3.register_adapter(np.int32, lambda val: int(val))

            db_filepath = os.path.join(db_dir, db_name + '.sqlite3')
            self.conn = sqlite3.connect(db_filepath)
            self.db_filepath = db_filepath
        elif db_type.lower() in ('postgres', 'postgresql'):
            self.conn = psycopg2.connect("user=postgress password=password")
            curs = self.conn.cursor()
            # TODO: implement it
            
        else:
            ValueError("db_type currently supports only sqlite3 and postgresql")       

    def add_table(self, input_filepath:str, 
                        table_name: str, 
                        separator='\t',   
                        chunk_size=100000,     
                        first_row_is_columns_names=True,
                        columns_names:Optional[List[str]]=None,
                        force_override=True):
        """
        :param input_filepath: path to input CSV/TSV file
        :param table_name: a name for the table to be created for the data in the file
        :param separator: type of separator. usually ',' (comma) or '\t' (tab)
        :param chunk_size: size of chunks to read from the file by Pandas.
            Can be larger than the total number of rows in the file. In this case 
            the file will be read in a single chunk.
        :param first_row_is_columns_names: whether the first row in the file contains the column names
        :param columns_names: column names. if it is set, then these names will be used, regardless of the value of first_row_is_columns_names  
        :param force_override: whether to remove existing tables in the local database before writing, if they exist
        """
        self.conn = sqlite3.connect(self.db_filepath)
        csv_reader = pd.read_csv(input_filepath, sep=separator, chunksize=chunk_size)

        # read a data row to find out data type
        data_row = pd.read_csv(input_filepath,sep=separator, skiprows=1, nrows=1) # 2nd row should contain data in all cases
        data_types = []
        for val in data_row:
            if not val.isnumeric():
                try:
                    float(val)
                    data_types.append('REAL')
                except:
                    data_types.append('TEXT')
            elif isinstance(literal_eval(val), int):
                data_types.append('INT')

        if columns_names is None and not first_row_is_columns_names:
            # read one row to find out the number of columns
            num_columns = len(data_row.columns)
            # make numbered column names
            columns_names = [str(i) for i in range(num_columns)]
        elif columns_names is None and first_row_is_columns_names:
            firstrow = pd.read_csv(input_filepath,sep=separator, skiprows=0, nrows=1)
            columns_names = firstrow.columns.tolist()
        
        # wrap column names that contain spaces with []:
        columns_names = [c.replace(c, '[' + c + ']') if ' ' in c else c for c in columns_names]
        curs = self.conn.cursor()
        create_table_str = 'CREATE TABLE IF NOT EXISTS ' + table_name + '(\n' + \
            '\n'.join([a + ' ' + b + ',' for a,b in zip(columns_names, data_types)]) + \
            '\n)'
        # remove last comma:
        create_table_str = create_table_str[:-3] + create_table_str[-2:]

        # create table:
        curs.execute(create_table_str)

        # delete existing data from table:
        if force_override:
            curs.execute("DELETE FROM " + table_name)

        # save changes:
        self.conn.commit()

        chunk_count = 0
        offset = 0
        t1=time.time()
        for tab in csv_reader:
            chunk_count+=1

            # insert data in chunks:
            tab.to_sql(table_name, self.conn, if_exists='append', index=False, method=None)
            
            offset+=chunk_size
            print(chunk_count)
            t2=time.time()
            print(t2-t1)

        self.conn.commit()
        self.conn.close()
    
    def query_to_dataframe(self, query):
        """
        Run a query on the database and obtain result in a Pandas DataFrame
        :params query: SQL query string
        """
        self.conn = sqlite3.connect(self.db_filepath)
        return pd.read_sql(query, self.conn)
    
    def run_query(self, query):
        """
        Run a query on the database
        :params query: SQL query string
        """
        print(f"Executing query: \n{query}")
        self.conn = sqlite3.connect(self.db_filepath)
        cur = self.conn.cursor()
        cur.execute(query)
        print('\nDone.')




