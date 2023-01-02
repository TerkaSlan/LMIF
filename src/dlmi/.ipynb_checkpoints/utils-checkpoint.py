import torch
import numpy as np
from typing import Tuple
import yaml
from pathlib import Path
import logging


def get_logger_config() -> str:
    return '[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s'


def remove_logger_handlers():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


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