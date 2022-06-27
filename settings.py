import pathlib

import platform

# required?
plt = platform.system()
if plt == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

SEP = ';'

ROOT_PATH = pathlib.Path(__file__).parent

DATA_PATH = ROOT_PATH / "data"
PROCESSED_PATH = DATA_PATH / "processed"
RAW_PATH = DATA_PATH / "raw"
LOGGING_PATH = ROOT_PATH / "logs"

directories = [DATA_PATH, PROCESSED_PATH, LOGGING_PATH, ]
for directory in directories:
    if not directory.is_dir():
        directory.mkdir()
