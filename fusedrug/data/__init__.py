import os

MOL_BIO_DATASETS_ENV_VAR = "MOL_BIO_DATASETS"


def get_datasets_dir() -> str:
    assert MOL_BIO_DATASETS_ENV_VAR in os.environ, "could not find MOL_BIO_DATASETS env var!"
    return os.environ[MOL_BIO_DATASETS_ENV_VAR]
