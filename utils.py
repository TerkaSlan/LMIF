import psutil
import gc
import os
import pandas as pd
from typing import Dict


def get_exp_name(config_file: Dict) -> str:
    """ Extracts the name of the experiment based on the ML model (separated by '-'):
    'LR', 'NN', 'multilabel-NN', ...

    Parameters:
        config_file (Dict): The configuration file

    Returns:
        str: Name of the experiment
    """
    multilabel = 'multilabel-NN'
    exp_name = config_file.split('-')[-1].split('.')[0]
    if multilabel in config_file:
        exp_name = multilabel
    elif exp_name == '10perc' or exp_name == 'ood':
        exp_name = config_file.split('-')[-2]
    return exp_name


def free_memory(evaluator, index, config, pivot_df):
    """ Deletes the unused variables from the main memory.

    Parameters:
        evaluator: Evaluator instance
        index: LMI, Mtree or Mindex instance
        config (Dict): confguration file
        pivot_df (pd.DataFrame): the pivots file

    """
    del evaluator
    del index
    del config
    if isinstance(pivot_df, pd.DataFrame):
        del pivot_df
    gc.collect()


def get_current_mem() -> float:
    """ Finds the current memory usage of the process. """
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
