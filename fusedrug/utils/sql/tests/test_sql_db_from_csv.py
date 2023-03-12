import unittest
from fusedrug.utils.sql.sql_db_from_csv import SQLfromCSV
from fusedrug.tests_data import get_tests_data_dir
import os
import pandas as pd
import numpy as np


class TestSQLfromCSV(unittest.TestCase):
    def test_single_chunk(self):

        small_csv_filepath = os.path.join(get_tests_data_dir(), "sql_from_csv", "small_input.tsv")
        db_dir = os.path.join(get_tests_data_dir(), "sql_from_csv")
        db_name = "small_output_db"
        sql_creator = SQLfromCSV(db_dir=db_dir, db_name=db_name)
        sql_creator.add_table(
            input_filepath=small_csv_filepath,
            table_name="my_table",
            chunk_size=100000,  # some chunk size way larger than number of samples
            first_row_is_columns_names=True,
        )
        query = "select * from my_table"
        df = sql_creator.query_to_dataframe(query)
        self.assertEquals(len(df), 3)
        self.assertEquals(df["Int Column"].tolist(), [1, 2, 3])
        self.assertEquals(df["Float Column"].tolist(), [1.3, 2.2, 3.5])

    def test_many_chunks(self):
        large_csv_filepath = os.path.join(get_tests_data_dir(), "sql_from_csv", "large_input.tsv")
        db_dir = os.path.join(get_tests_data_dir(), "sql_from_csv")
        db_name = "large_output_db"
        sql_creator = SQLfromCSV(db_dir=db_dir, db_name=db_name)
        sql_creator.add_table(
            input_filepath=large_csv_filepath, table_name="my_table", chunk_size=10, first_row_is_columns_names=True
        )
        query = "select * from my_table"
        df = sql_creator.query_to_dataframe(query)
        self.assertEquals(len(df), 104)
        self.assertEquals(df["Int Column"].tolist(), list(range(1, 105)))
        self.assertTrue(np.isclose(np.array(df["Float Column"]), np.linspace(1.1, 11.4, 104)).all())

    def test_multiple_files(self):
        # create a database with two tables, each coming from a different TSV file
        small_csv_filepath = os.path.join(get_tests_data_dir(), "sql_from_csv", "small_input.tsv")
        large_csv_filepath = os.path.join(get_tests_data_dir(), "sql_from_csv", "large_input.tsv")
        file_to_table_dict = {small_csv_filepath: "small_table", large_csv_filepath: "large_table"}
        chunk_size_dict = {small_csv_filepath: 100000, large_csv_filepath: 10}
        first_row_is_columns_dict = {small_csv_filepath: True, large_csv_filepath: True}

        db_dir = os.path.join(get_tests_data_dir(), "sql_from_csv")
        db_name = "multiple_tables_db"
        sql_creator = SQLfromCSV(db_dir=db_dir, db_name=db_name)
        for f in file_to_table_dict:
            sql_creator.add_table(
                input_filepath=f,
                table_name=file_to_table_dict[f],
                chunk_size=chunk_size_dict[f],
                first_row_is_columns_names=first_row_is_columns_dict[f],
            )
        query1 = "select * from small_table"
        query2 = "select * from large_table"
        df1 = sql_creator.query_to_dataframe(query1)
        df2 = sql_creator.query_to_dataframe(query2)
        self.assertEquals(len(df1), 3)
        self.assertEquals(df1["Int Column"].tolist(), [1, 2, 3])
        self.assertEquals(df1["Float Column"].tolist(), [1.3, 2.2, 3.5])
        self.assertEquals(len(df2), 104)
        self.assertEquals(df2["Int Column"].tolist(), list(range(1, 105)))
        self.assertTrue(np.isclose(np.array(df2["Float Column"]), np.linspace(1.1, 11.4, 104)).all())


if __name__ == "__main__":
    unittest.main()
