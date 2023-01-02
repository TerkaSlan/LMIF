import torch
import io
import numpy as np
from typing import Tuple
import yaml
from pathlib import Path
from datetime import datetime
import pickle
import sys


def get_device() -> torch.device:
    """ Gets the `device` to be used by torch.
    This arugment is needed to operate with the PyTorch model instance.

    Returns
    ------
    torch.device
        Device
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    return device


def data_X_to_torch(data) -> torch.FloatTensor:
    """ Creates torch training data."""
    data_X = torch.from_numpy(np.array(data).astype(np.float32))
    return data_X


def data_to_torch(data, labels) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """ Creates torch training data and labels."""
    data_X = data_X_to_torch(data)
    data_y = torch.as_tensor(torch.from_numpy(labels), dtype=torch.long)
    return data_X, data_y


def load_yaml(path):
    with open(path, 'r') as stream:
        loaded = yaml.safe_load(stream)
    return loaded


def save_yaml(file, path):
    with open(path, 'w+') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def write_to_file(filename: str, row: str) -> None:
    """
    Writes a single row to a file.
    Expects that the directory with the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to write to.
    row: str
        The string to write.
    """
    writing_mode = 'a'
    if not Path.is_file(Path(filename)):
        writing_mode = 'w'
    with open(filename, writing_mode) as f:
        f.write(row + '\n')


def save_as_pickle(filename: str, obj):
    """
    Saves an object as a pickle file.
    Expects that the directory with the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to write to.
    obj: object
        The object to save.
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename: str):
    """
    Loads an object from a pickle file.

    Parameters
    ----------
    filename : str
        Path to the file to load from.

    Returns
    -------
    object
        The loaded object.
    """
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_size_of_file(filename: str) -> int:
    """
    Gets the size of a file in megabytes.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    int
        Size of the file in megabytes.
    """
    return Path(filename).stat().st_size / 1024 / 1024


def create_dir(directory: str) -> None:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def remove_dir(directory: str) -> None:
    """
    Removes a directory if it exists.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    if Path(directory).exists():
        Path(directory).rmdir()


def get_current_datetime() -> str:
    """
    Formats current datetime into a string.

    Returns
    ----------
    str
        Created datetime.
    """
    return datetime.now().strftime('%Y-%m-%d--%H-%M-%S')


def get_hostname() -> str:
    import socket
    return socket.gethostname()


class CPU_Unpickler(pickle.Unpickler):
    # https://stackoverflow.com/a/70741020
    # EXAMPLE USE -- index = CPU_Unpickler(open('../experiments/index-23.pkl',"rb")).load()
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def get_environment_variable(name, default=None):
    import os
    return os.environ.get(name, default)


def increase_max_recursion_limit():
    import resource
    # https://stackoverflow.com/a/16248113
    resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
    sys.setrecursionlimit(10**6)
