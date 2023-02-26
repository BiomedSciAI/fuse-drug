import wget
import os
from zipfile import ZipFile
import pickle
import pandas as pd
from fuse.utils.file_io.file_io import create_dir


class BindingDB:

    @staticmethod
    def download(data_dir: str) -> None:
        """
        Downloads BindingDB data
        """
        
        create_dir(data_dir)
        zip_path = os.path.join(data_dir, "BindingDB_All.tsv.zip")

        if not os.path.isfile(zip_path): 
            # Download zip
            print("Downloading zip file:")
            url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_2022m4.tsv.zip"
            wget.download(url, zip_path)
            print("Downloading zip file: DONE")

        raw_file_path = os.path.join(data_dir, "BindingDB_All.tsv")
        if not os.path.isfile(raw_file_path):
            # Extract zip
            print("Extracting zip:")
            with ZipFile(zip_path, "r") as zipObj:
                zipObj.extractall(path=data_dir)
            print("Extracting zip: DONE")

        pickle_file_path = os.path.join(data_dir, "BindingDB_All.pkl")
        if not os.path.isfile(pickle_file_path):
            # Load tsv file and pickle it (sort of caching for quicker future use)
            print("Pickling data:")
            df = pd.read_csv(raw_file_path, sep='\t', on_bad_lines='skip')
            df.to_pickle(pickle_file_path)
            print("Pickling data: DONE")


if __name__ == "__main__":

    BindingDB.download("./tutorials/data")