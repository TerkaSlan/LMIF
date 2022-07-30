from lmi.utils import load_yaml, save_yaml, load_model_config, write_to_file, create_dir,\
                  get_current_datetime, intersect_mocap_dataset, get_logger_config
from lmi.data.DataLoader import ProfisetDataLoader, CoPhIRDataLoader, MocapDataLoader
from lmi.data.enums import Dataset
from lmi.indexes.LearnedMetricIndex import LMI
from lmi.indexes.Mindex import Mindex
from lmi.indexes.Mtree import Mtree
from lmi.Experiment import Evaluator
from utils import get_exp_name, free_memory, get_current_mem
import concurrent

import time
import sys
import logging
import gc
import pandas as pd
import numpy as np
import os

import multiprocessing as mp
import ctypes


def get_loader(config):
    if config['data']['dataset'] == 'PROFISET':
        loader = ProfisetDataLoader(config)
    if config['data']['dataset'] == 'MOCAP':
        loader = MocapDataLoader(config)
    elif config['data']['dataset'] == 'COPHIR':
        loader = CoPhIRDataLoader(config)
    return loader



def run_experiment(config_file):
    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)
    exp_name = get_exp_name(config_file)
    config = load_yaml(config_file)
    current_configuration = '-'.join(config_file.split('/')[-1].split('-')[:-1])
    LOG.info(f'Running an experiment with {exp_name} using {config_file}')

    mem_beginning = get_current_mem()
    start_loading = time.time()
    loader = get_loader(config)
    if 'original' in config['data'] and config['data']['dataset'] != 'MOCAP':
        labels_orig = loader.load_labels()
        try:
            labels = labels_orig.loc[df_shared.index]
        except AttributeError:	
            labels = None
    elif config['data']['dataset'] == 'MOCAP':
        labels_orig = loader.load_labels()
        index_intersection = df_shared.index.intersection(labels_orig.index)
        labels = labels_orig.loc[index_intersection]
    else:
        labels = None
    
    loading_time = time.time() - start_loading
    mem_data_load = max(get_current_mem() - mem_beginning, 0)
    LOG.info(f'Consumed memory [data loading] (MB): {mem_data_load}')

    job_id = config_file.split('/')[-1].split('.')[0] + '--' + get_current_datetime()
    create_dir(f'outputs/{job_id}')
    save_yaml(config, f'outputs/{job_id}/used-config.yml')

    # -------------------- INDEX BUILDING -------------------- #
    if exp_name == 'Mindex':
        training_time = 0
        pivot_df = loader.load_mindex_pivots()
        primary_descriptors = df_shared_orig if config['data']['dataset'] == 'COPHIR' else df_shared
        index = Mindex(primary_descriptors, labels, pivot_df, config['data']['dataset'])
        mem_train = 0
    elif exp_name == 'Mtree':
        training_time = 0
        pivot_df = loader.load_mtree_pivots()
        primary_descriptors = df_shared_orig if config['data']['dataset'] == 'COPHIR' else df_shared
        index = Mtree(primary_descriptors, labels, pivot_df, config['data']['dataset'])
        mem_train = 0
    else:
        pivot_df = []
        index = LMI(config['LMI'], df_shared, labels)
        model_config = load_model_config(config['LMI']['model-config'], index.n_levels)
        start_training = time.time()
        index.train(model_config)
        training_time = time.time() - start_training
        mem_train = max(get_current_mem() - mem_data_load - mem_beginning, 0)
        save_yaml(model_config, f'outputs/{job_id}/used-model-config.yml')

    # -------------------- SEARCHING -------------------- #
    knns = loader.get_knn_ground_truth()
    queries = loader.get_queries()
    evaluator = Evaluator(index, knns, queries, config, job_id=job_id)
    evaluator.bucket_occupancy()
    evaluator.run_evaluate()

    # -------------------- LOGGING -------------------- #
    mem_finish = max(get_current_mem() - mem_train - mem_data_load - mem_beginning, 0)
    evaluator.generate_summary(mem_data_load, mem_train, mem_finish)

    execution_time = time.time() - start_loading

    if exp_name == 'Mindex' or exp_name == 'Mtree':
        write_to_file(f'outputs/{job_id}/times.csv', 'loading, execution')
        write_to_file(f'outputs/{job_id}/times.csv', f'{loading_time}, {execution_time}')
    else:
        write_to_file(f'outputs/{job_id}/times.csv', 'loading, training, execution')
        write_to_file(f'outputs/{job_id}/times.csv', f'{loading_time}, {training_time}, {execution_time}')

    previous_original = config['data']['original'] if 'original' in config['data'] else None
    previous_dataset = config['data']['dataset']
    previous_configuration = current_configuration

    free_memory(evaluator, index, config, pivot_df)

def make_dataset_shared(df):
    # ---------- map df to shared memory
    index = df.index
    # the origingal dataframe is df, store the columns/dtypes pairs
    df_dtypes_dict = dict(list(zip(df.columns, df.dtypes)))

    # declare a shared Array with data from df
    mparr = mp.Array(ctypes.c_double, df.values.reshape(-1))

    # create a new df based on the shared array
    new_df = pd.DataFrame(
        np.frombuffer(mparr.get_obj()).reshape(df.shape),
        columns=df.columns,
        index=index
        ).astype(df_dtypes_dict)
    
    return new_df


# flake8: noqa: C901
if __name__ == '__main__':

    assert len(sys.argv) > 1, \
        "Please provide the experiment setup you'd like to run as an arugment"\
        "(.yml config files found in experiment-setups/)"

    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)

    config_files = sys.argv[1:]
    config_files.sort(key=lambda x: x.split('/')[-1].split('-')[0])

    config_files_cophir = [c for c in config_files if Dataset['COPHIR'].value in c]
    config_files_profi = [c for c in config_files if Dataset['PROFISET'].value in c]
    config_files_mocap = [c for c in config_files if Dataset['MOCAP'].value.lower() in c.lower()]
    
    if len(config_files_cophir) != 0:
        loader = CoPhIRDataLoader(load_yaml(config_files_cophir[0]))
        df = loader.load_descriptors()
        df_orig = df[1]
        df = df[0]
        df_shared = make_dataset_shared(df)
        df_shared_orig = make_dataset_shared(df_orig)
        
        for i in range(len(config_files_cophir)):
            p = mp.Process(target=run_experiment, args=(config_files_cophir[i], ))
            p.start()
            p.join()
        del df_shared
        del df_shared_orig

    if len(config_files_profi) != 0:
        loader = ProfisetDataLoader(load_yaml(config_files_profi[0]))
        df = loader.load_descriptors()
        df_shared = make_dataset_shared(df)

        for i in range(len(config_files_profi)):
            p = mp.Process(target=run_experiment, args=(config_files_profi[i], ))
            p.start()
            p.join()
        del df_shared

    if len(config_files_mocap) != 0:
        loader = MocapDataLoader(load_yaml(config_files_mocap[0]))
        df = loader.load_descriptors()
        labels_orig = loader.load_labels()
        df,_ = intersect_mocap_dataset(df, labels_orig)
        df_shared = make_dataset_shared(df)

        for i in range(len(config_files_mocap)):
            p = mp.Process(target=run_experiment, args=(config_files_mocap[i], ))
            p.start()
            p.join()
        del df_shared

    LOG.info(f'Finished the experiment run: {get_current_datetime()}')
