from lmi.utils import load_yaml, save_yaml, load_model_config, write_to_file, create_dir,\
                  get_current_datetime, intersect_mocap_dataset, get_logger_config
from lmi.data.DataLoader import ProfisetDataLoader, CoPhIRDataLoader, MocapDataLoader
from lmi.indexes.LearnedMetricIndex import LMI
from lmi.data.enums import Dataset
from lmi.indexes.Mindex import Mindex
from lmi.indexes.Mtree import Mtree
from lmi.Experiment import Evaluator
from utils import get_exp_name, free_memory, get_current_mem

import time
import sys
import logging
import gc
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import ctypes

# Testing
from pathlib import Path

SAVE_DIR='outputs/test-outputs/'

def get_loader(config):
    if config['data']['dataset'] == 'PROFISET':
        loader = ProfisetDataLoader(config)
    if config['data']['dataset'] == 'MOCAP':
        loader = MocapDataLoader(config)
    elif config['data']['dataset'] == 'COPHIR':
        loader = CoPhIRDataLoader(config)
    return loader

# flake8: noqa: C901
def run_test_experiment(config_file):
    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)
    exp_name = get_exp_name(config_file)
    config = load_yaml(config_file)
    current_configuration = '-'.join(config_file.split('/')[-1].split('-')[:-1])
    LOG.info(f'Running an experiment with {exp_name} using {config_file}')

    mem_beginning = get_current_mem()
    start_loading = time.time()
    loader = get_loader(config)
    if 'original' in config['data']:
        labels_orig = loader.load_labels()
        try:
            labels = labels_orig.loc[df_shared.index]
        except AttributeError:
            labels = None
        if config['data']['dataset'] == 'MOCAP':
            df, labels = intersect_mocap_dataset(df, labels)
    else:
        labels = None

    loading_time = time.time() - start_loading
    mem_data_load = max(get_current_mem() - mem_beginning, 0)
    LOG.info(f'Consumed memory [data loading] (MB): {mem_data_load}')

    job_id = config_file.split('/')[-1].split('.')[0] + '--' + get_current_datetime()
    create_dir(f'{SAVE_DIR}/{job_id}')
    save_yaml(config, f'{SAVE_DIR}/{job_id}/used-config.yml')

    # -------------------- INDEX BUILDING -------------------- #
    if exp_name == 'Mindex':
        training_time = 0
        pivot_df = loader.load_mindex_pivots()
        primary_descriptors = df_shared_orig if config['data']['dataset'] == 'COPHIR' else df_shared
        if config['data']['dataset'] == 'MOCAP':
            input_df, labels_orig = intersect_mocap_dataset(pivot_df[:50], labels_orig)
        else:
            input_df = pivot_df[:50]
        index = Mindex(input_df, labels_orig[:50], pivot_df, config['data']['dataset'])
        mem_train = 0
        config['experiment']['search-stop-conditions'] = [0.0001, 0.0005, 0.001, 0.005, 0.1]
    elif exp_name == 'Mtree':
        training_time = 0
        pivot_df = loader.load_mtree_pivots()
        pivot_df = pivot_df[~pivot_df.index.duplicated(keep='first')]
        
        primary_descriptors = df_shared_orig if config['data']['dataset'] == 'COPHIR' else df_shared
        descriptors = primary_descriptors.copy().iloc[:pivot_df.shape[0]]

        slice = 5_000 if 5_000 < pivot_df.shape[0] else pivot_df.shape[0]

        descriptors = descriptors.set_index(pivot_df.index[:slice])
        labels_orig = labels_orig.iloc[:slice]

        labels_orig = labels_orig.set_index(pivot_df.index[:slice])
        index = Mtree(descriptors, labels_orig, pivot_df.iloc[:slice], config['data']['dataset'])
        mem_train = 0
        config['experiment']['search-stop-conditions'] = [0.0001]
    else:
        pivot_df = []
        if labels is not None:
            labels_input = labels.head(10)
        else:
            labels_input = None
        index = LMI(config['LMI'], df_shared.head(10), labels_input, reindex=False)
        model_config = load_model_config(config['LMI']['model-config'], index.n_levels)
        start_training = time.time()
        index.train(model_config)
        training_time = time.time() - start_training
        mem_train = max(get_current_mem() - mem_data_load - mem_beginning, 0)
        save_yaml(model_config, f'{SAVE_DIR}/{job_id}/used-model-config.yml')

    # -------------------- SEARCHING -------------------- #
    evaluator = Evaluator(
        index,
        {str(df_shared.iloc[0].name): {str(df_shared.iloc[0].name): 0.0}},
        df_shared.iloc[:1],
        config,
        job_id=job_id,
        output_dir=SAVE_DIR
    )
    evaluator.bucket_occupancy()
    evaluator.run_evaluate()
    mem_finish = max(get_current_mem() - mem_train - mem_data_load - mem_beginning, 0)
    evaluator.generate_summary(mem_data_load, mem_train, mem_finish)

    execution_time = time.time() - start_loading

    write_to_file(f'{SAVE_DIR}/{job_id}/times.csv', 'loading, training, execution')
    write_to_file(f'{SAVE_DIR}/{job_id}/times.csv', f'{loading_time}, {training_time}, {execution_time}')

    previous_dataset = config['data']['dataset']
    previous_configuration = current_configuration
    free_memory(evaluator, index, config, pivot_df)

    if exp_name == 'multilabel-NN':
        labels = None
    elif labels is not None:
        labels.drop([col for col in labels.columns.to_list() if '_pred' in col], inplace=True, axis=1)

    LOG.info(f'Finished the experiment run: {get_current_datetime()}')


def make_dataset_shared(df):
    # ---------- map df to shared memory
    # the origingal dataframe is df, store the columns/dtypes pairs
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))

    # declare a shared Array with data from df
    mparr = mp.Array(ctypes.c_double, df.values.reshape(-1))

    # create a new df based on the shared array
    new_df = pd.DataFrame(
        np.frombuffer(mparr.get_obj()).reshape(df.shape),
        columns=df.columns,
        index=df.index
        ).astype(df_dtypes_dict)
    
    return new_df


if __name__ == '__main__':
    assert len(sys.argv) > 1, \
        "Please provide the experiment setup you'd like to run as an arugment"\
        "(.yml config files found in experiment-setups/)"
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)
    config_files = sys.argv[1:]
    config_files.sort(key=lambda x: x.split('/')[-1].split('-')[0])

    config_files_cophir = [c for c in config_files if Dataset['COPHIR'].value in c]
    config_files_profi = [c for c in config_files if Dataset['PROFISET'].value in c]
    config_files_mocap = [c for c in config_files if Dataset['MOCAP'].value in c]
    
    if len(config_files_cophir) != 0:
        loader = CoPhIRDataLoader(load_yaml(config_files_cophir[0]))
        config_file = config_files_cophir[0]
        df = loader.load_tiny_descriptors(5000)
        df_orig = df
        if isinstance(df, tuple):
            df_orig = df[1]
            df = df[0]
        else:
            df_orig = df
        
        df_shared = make_dataset_shared(df)
        df_shared_orig = make_dataset_shared(df_orig)

        pool = mp.Pool()
        for i in range(len(config_files_cophir)):
            p=pool.apply_async(run_test_experiment, args=(config_files_cophir[i],))
            p.get()
        pool.close()
        pool.join()
        del df_shared
        del df_shared_orig

    if len(config_files_profi) != 0:
        loader = ProfisetDataLoader(load_yaml(config_files_profi[0]))
        config_file = config_files_profi[0]
        df = loader.load_tiny_descriptors(5000)
        df_orig = df
        if isinstance(df, tuple):
            df_orig = df[1]
            df = df[0]
        else:
            df_orig = df
        df_shared = make_dataset_shared(df)
        df_shared_orig = make_dataset_shared(df_orig)

        pool = mp.Pool()
        for i in range(len(config_files_profi)):
            p=pool.apply_async(run_test_experiment, args=(config_files_profi[i],))
            p.get()
        pool.close()
        pool.join()
        del df_shared
        del df_shared_orig

    if len(config_files_mocap) != 0:
        loader = MocapDataLoader(load_yaml(config_files_mocap[0]))
        df = loader.load_tiny_descriptors(5000)
        df_shared = make_dataset_shared(df)
        pool = mp.Pool()
        for i in range(len(config_files_mocap)):
            p=pool.apply_async(run_test_experiment, args=(config_files_mocap[i],))
            p.get()
        pool.close()
        pool.join()
        del df_shared
