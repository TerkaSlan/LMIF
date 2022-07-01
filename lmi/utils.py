from os.path import isfile
import pandas as pd
import re
from pathlib import Path
import random
from typing import Tuple, Dict
import yaml
import json
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_logger_config() -> str:
    return '[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s'


def get_current_datetime() -> str:
    """
    Formats current datetime into a string.

    Returns
    ----------
    str
        Created datetime.
    """
    return datetime.now().strftime('%Y-%m-%d--%H-%M-%S')


def load_model_config(model_config_path, n_levels=None):
    return load_yaml(model_config_path)


def intersect_mocap_dataset(
    descriptors: pd.DataFrame,
    labels: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates an intersection of descriptors and labels
    based on common object IDs.
    Few labels (< 1000) IDs in the MoCaP do not correspond to the
    descriptors and vice versa. This function guarantees integrity
    between the two.

    Parameters
    ----------
    descriptors : pd.DataFrame
    labels : pd.DataFrame

    Returns
    ----------
    Tuple[pd.DataFrame, pd.DataFrame]
        Intersected descriptors and labels
    """
    index_intersection = descriptors.index.intersection(labels.index)
    return (
        descriptors.loc[index_intersection],
        labels.loc[index_intersection]
    )


def create_dir(directory: str) -> None:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


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
    if not file_exists(filename):
        writing_mode = 'w'
    with open(filename, writing_mode) as f:
        f.write(row + '\n')


def crop_job_number(pbs_job_id):
    """
    Crops the metacentrum's job number from the whole job ID.

    Parameters
    ----------
    pbs_job_id : str
        Job ID

    Returns
    ----------
    pbs_job_id : str
        Numeric portion of the job ID.
    """
    match = re.findall(r'[0-9]*', pbs_job_id)
    if match[0] != '':
        pbs_job_id = match[0]
    return pbs_job_id


def load_yaml(path):
    with open(path, 'r') as stream:
        loaded = yaml.safe_load(stream)
    return loaded


def load_json(path):
    assert isfile(path), f'{path} does not exist.'
    with open(path, 'r') as stream:
        loaded = json.load(stream)
    return loaded


def save_json(dict, path):
    with open(path, 'w') as f:
        json.dump(dict, f, indent=4)


def remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


def save_yaml(file: Dict, target_path: str) -> None:
    """
    Saves the config file.

    Parameters
    ----------
    config : Dict
        Loaded yaml config file.
    target_path: str
        File location for config to be saved as.

    """
    with open(target_path, 'w+') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)


def file_exists(filename: str) -> bool:
    """
    Checks if a file exists.

    Parameters
    ----------
    filename : str
        Path to the file.

    """
    return Path.is_file(Path(filename))


def sample(df: pd.DataFrame, labels: pd.DataFrame, n: int, to_keep=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Ensures that the same indexes (i.e., object_id) are kept
    """
    Samples a dataset subset of size `n` from the original datasets.
    Ensures that the same indexes (i.e., object_id) are kept.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with descriptors.
    labels : pd.DataFrame
        Dataset with labels.
    n : int
        Number of rows to select.
    """
    assert n > 0 and n < df.shape[0], '`n` needs to be > 0 and < size of the datasets.'
    if to_keep is not None:
        to_keep = list(set(df.index.to_list()).intersection(set(to_keep)))
        if len(to_keep) > n:
            print(f'`to_keep` list cannot be greater than `n`: {len(to_keep)} > {n}. Changing `n` to `to_keep`')
            n = len(to_keep)
        rows = to_keep + random.sample(list(set(df.index.to_list()) - set(to_keep)), n-len(to_keep))
    else:
        rows = random.sample(df.index.to_list(), n)
    df_sampled = df.loc[rows, ]
    df_sampled = df_sampled[~df_sampled.index.duplicated(keep='first')]
    while df_sampled.shape[0] != n:
        df_sampled = pd.concat([df_sampled, df.loc[random.sample(df.index.to_list(), n-df_sampled.shape[0])]])
        df_sampled = df_sampled[~df_sampled.index.duplicated(keep='first')]
    if labels is not None:
        labels = labels.reindex(index=df.index)
        labels_sampled = labels.loc[df_sampled.index, ]
        return df_sampled, labels_sampled
    else:
        return df_sampled, None


def encode_input_labels(
    y: np.ndarray,
    encoder: LabelEncoder
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Performs basic encoding on input labels.
    Takes into account potential nans and 1-dimensional arrays.

    Parameters
    ----------
    y : np.ndarray
        Array with the labels
    encoder : LabelEncoder
        Encoder to perform the encoding with
    """
    y_shape = 0
    if pd.isnull(y).all():
        y_shape = y.shape
        y = [np.nan]
    y = encoder.fit_transform(y)
    if y_shape != 0:
        y = np.zeros(y_shape, dtype=int)
    return y, encoder


def one_hot_frequency_encode(labels: np.ndarray, n_cats: int) -> np.ndarray:
    """
    One-hot-frequency-encodes an array of list of labels.
    Is identical to one-hot encoding with the exception of taking into account the
    number of times a label appeared in the list of labels corresponding to
    one training example.

    E.g. [1,3,5,4,3,1,2,3] -> [0,2,1,3,1,1] -- there are 0 zeros, 2 ones, etc.
    Used in encoding of multilabel labels.

    Parameters
    ----------
    labels : np.ndarray[List]
        Array of the list of labels
    n_cats : int
        Number of unique classes occuring in `labels`
    """
    frequency_labels = []
    for label in labels:
        labels, counts = np.unique(label, return_counts=True)
        curr = np.zeros(n_cats)
        for inner_label, c in zip(labels, counts):
            curr[int(inner_label)] = c
        frequency_labels.append(curr)

    frequency_labels = np.vstack((frequency_labels))
    if frequency_labels.shape[1] == 1:
        frequency_labels = np.hstack((frequency_labels, np.zeros(frequency_labels.shape)))
    return frequency_labels
