import json
import pickle
from itertools import chain
from pathlib import Path
from typing import Any, Union, AnyStr, List, Dict


def write_to_file(filename: Union[str, Path], content: Any, encoding='utf-8') -> None:
    with open(filename, mode='w', encoding=encoding) as file:
        if str(filename).endswith('jsonl') or str(filename).endswith('json'):
            json.dump(content, file, indent=4)
        else:
            file.write(str(content))


def dump(filename: Union[str, Path], content: Any) -> None:
    with open(filename.as_posix(), mode="wb") as file:
        pickle.dump(content, file, pickle.HIGHEST_PROTOCOL)


def loads(filename: Union[str, Path]) -> Any:
    with open(filename.as_posix(), mode="rb") as file:
        return pickle.load(file)


def read_file(filename: Union[str, Path], encoding='utf-8') -> AnyStr:
    with open(filename, mode='r', encoding=encoding) as file:
        return file.read()


def unflatten(nested_list: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(nested_list))

def identity(w):
    return w