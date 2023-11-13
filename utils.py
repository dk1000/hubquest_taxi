from pathlib import Path


def get_root_path():
    return Path(__file__).cwd()


def get_file_path():
    return get_root_path() / "src"


def get_data_path():
    return get_root_path() / "data"


def get_model_path():
    return get_root_path() / "models"
