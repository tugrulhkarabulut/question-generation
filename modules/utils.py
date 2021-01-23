import ntpath
from pathlib import Path

def get_file_name(path):
    return ntpath.basename(path).split('.')[0]

def get_file_format(path):
    return ntpath.basename(path).split('.')[1]

def build_file_path(path, file_name, format_):
    file_path = Path(path)
    file_path = file_path / (file_name + '.' + format_)
    return file_path