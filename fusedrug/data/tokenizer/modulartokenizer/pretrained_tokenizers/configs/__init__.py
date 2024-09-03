from pathlib import Path

# The directory containing this file
CONFIG_DIRPATH = Path(__file__).parent


def get_modular_tokenizer_config_dirpath() -> str:
    return str(CONFIG_DIRPATH.resolve())
