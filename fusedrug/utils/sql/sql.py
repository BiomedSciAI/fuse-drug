from sqlalchemy import create_engine
import pandas as pd
from typing import Optional, Union
import os
from fuse.utils.file_io import save_text_file_safe
from fuse.utils.file_io.path import change_extension
from fuse.utils.cpu_profiling import Timer
from .sql_db_from_csv import SQLfromCSV

# NOTE: NOT multi process/threading safe!
class SQL:
    def __init__(self, connection_url: str):
        """
        A basic connection and SQL query class

        Args:
            connection_url: a postgresql connection url of the form:
                'postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
                OR a path to sqlite3 DB file
            Note:
            As the URL is like any other URL, special characters such as those that may be used in
            the password need to be URL encoded to be parsed correctly.. Below is an example of a URL that
            includes the password "kx%jj5/g", where the percent sign and slash characters are represented
            as %25 and %2F, respectively: postgresql+pg8000://dbuser:kx%25jj5%2Fg@pghost10/appdb
            see: https://docs.sqlalchemy.org/en/14/core/engines.html
        """

        self._connection_url = connection_url
        if not connection_url.lower().startswith("postgresql") and connection_url.lower().endswith(".sqlite3"):
            db_dir = os.path.dirname(connection_url)
            db_name = os.path.splitext(os.path.basename(connection_url))[0]
            self.sqlite_obj = SQLfromCSV(db_dir=db_dir, db_name=db_name)
            self._engine = self.sqlite_obj.conn
        else:
            self._engine = create_engine(connection_url)

    def __del__(self):
        self._engine.dispose()

    def query(self, query: str, chunksize: Optional[int] = None):
        return pd.read_sql_query(query, con=self._engine, chunksize=chunksize)


def download_dataframe(
    sql_object: SQL,
    output_tsv: str,
    query: str,
    chunksize: Optional[int] = None,
    merge_chunked_to_single_file: Optional[bool] = True,
) -> None:
    """
    a convinient function to download a DB table as a tsv (like csv but with tab separator)

    Args:
        sql_object:SQL - you can create an instance by calling fusedrug.utils.sql.SQL(...)
        output_tsv:str the output tab separate text table file
        query:str the sql query
        chunksize:Optional[int]: when set to a positive integer will download the table in chunks.
            This is useful if the tables are too big to fit in memory.

            Note - each downloaded chunk will contain a fully loadable tsv table including columns names,
            so don't use a naive concat if you want to join them into a single file.

            if output_tsv = '/a/b/c/some_name.tsv' then the chunks will be:

                '/a/b/c/some_name@_chunk_0000@.tsv',
                '/a/b/c/some_name@_chunk_0001@.tsv',
                '/a/b/c/some_name@_chunk_0002@.tsv',
                ...
        merge_chunked_to_single_file: IF chunksize is set to an integer, if merge_chunked_to_single_file is True (which is the default),
            the downloaded chunks will be concatanated into a single tsv

    """

    done_download = output_tsv + ".DONE"
    if os.path.isfile(done_download):
        print(f"already found {output_tsv}")
        return

    print(f"for {output_tsv} running query:\n{query}")
    if chunksize is None:
        df = sql_object.query(query)
        print(f"saving to tsv {output_tsv} ...")
        df.to_csv(output_tsv, index=False, sep="\t")
        save_text_file_safe(done_download, "")
        return

    assert isinstance(chunksize, int), "chunksize must be None or an integer"
    assert chunksize > 0, "chunksize must be a positive integer"

    it = sql_object.query(query, chunksize=chunksize)
    original_ext = os.path.splitext(output_tsv)[1][1:]
    # original_ext = get_extension(output_tsv)
    parts_num = 0
    for i, df_chunk in enumerate(it):
        current_chunk_output_tsv = change_extension(output_tsv, f"@_chunk_{i:04d}@{original_ext}")
        print(f"writing {current_chunk_output_tsv}")
        df_chunk.to_csv(current_chunk_output_tsv, index=False, sep="\t")
        parts_num += 1

    if merge_chunked_to_single_file:
        print("merging chunk parts into a single file...")
        with Timer("Merging chunks:"):
            with open(output_tsv, "w") as f_write:
                for i in range(parts_num):
                    current_chunk_output_tsv = change_extension(output_tsv, f"@_chunk_{i:04d}@{original_ext}")
                    line_num = -1
                    with open(current_chunk_output_tsv, "r") as f_read:
                        while True:
                            line_num += 1
                            line = f_read.readline()
                            if not line:
                                break
                            if i > 0 and 0 == line_num:
                                # skip columns names row for
                                continue
                            f_write.write(line)
        print(f"done merging chunks into {output_tsv}")

    save_text_file_safe(done_download, "")
