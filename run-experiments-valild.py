from lmi.utils import load_yaml, save_yaml, load_model_config, write_to_file, create_dir,\
                  get_current_datetime, intersect_mocap_dataset, get_logger_config
from lmi.data.DataLoader import ProfisetDataLoader, CoPhIRDataLoader, MocapDataLoader
from lmi.indexes.LearnedMetricIndex import LMI
from lmi.indexes.Mindex import Mindex
from lmi.indexes.Mtree import Mtree
from lmi.Experiment import Evaluator
from utils import get_exp_name, free_memory, get_current_mem

import time
import sys
import logging
import gc
import pandas as pd
import os

# Testing
from pathlib import Path


def get_loader(config):
    if config['data']['dataset'] == 'PROFISET':
        loader = ProfisetDataLoader(config)
    if config['data']['dataset'] == 'MOCAP':
        loader = MocapDataLoader(config)
    elif config['data']['dataset'] == 'COPHIR':
        loader = CoPhIRDataLoader(config)
    return loader

# flake8: noqa: C901
def run_test_experiments(config_files):
    loader = None
    previous_dataset = None
    previous_configuration = None
    labels = None
    queries = None
    knns = None
    pivot_df = None
    df = None
    labels_orig = None
    training_time = 0

    config_files.sort(key=lambda x: x.split('/')[-1].split('-')[0])

    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)

    for exp_counter, config_file in enumerate(config_files):
        exp_name = get_exp_name(config_file)
        if exp_name != 'Mtree':
            config = load_yaml(config_file)       
            current_configuration = '-'.join(config_file.split('/')[-1].split('-')[:-1])
            LOG.info(f'Running an experiment with {exp_name} using {config_file} -- {exp_counter}/{len(sys.argv[1:])}')

            start_loading = time.time()
            
            # Delete any previous `knns` and `queries` if we're running experiment with a different set of them
            if config['experiment']['queries-out-of-dataset'] and knns is not None and queries is not None:
                del knns
                del queries
                gc.collect()
                knns = None
                queries = None

            mem_beginning = get_current_mem()

            if labels is None or previous_configuration != current_configuration:
                loader = get_loader(config)
                if 'original' in config['data']:
                    labels_orig = loader.load_labels()
                else:
                    labels = None
                if config['data']['dataset'] == previous_dataset:
                    try:
                        labels = labels_orig.loc[df.index]
                    except AttributeError:
                        labels = None

            if config['data']['dataset'] != previous_dataset or exp_name == 'GMM':
                if config['data']['dataset'] != previous_dataset and previous_dataset is not None:
                    del df
                    df = None
                    gc.collect()

                if df is None and exp_name != 'Mtree':
                    df = loader.load_tiny_descriptors()

                if exp_name == 'Mtree':
                    df = pd.read_csv(f"test-data/{config_file.split('/')[-1].split('.')[0]}-top.csv")	
                    df_orig = df

                if isinstance(df, tuple):
                    df_orig = df[1]
                    df = df[0]
                else:
                    df_orig = df
                try:
                    labels = labels_orig.loc[df.index]
                except AttributeError:
                    labels = None

                if config['data']['dataset'] == 'MOCAP':
                    df, labels = intersect_mocap_dataset(df, labels)
            
            loading_time = time.time() - start_loading
            mem_data_load = max(get_current_mem() - mem_beginning, 0)
            LOG.info(f'Consumed memory [data loading] (MB): {mem_data_load}')

            job_id = config_file.split('/')[-1].split('.')[0] + '--' + get_current_datetime()
            create_dir(f'test-outputs/{job_id}')
            save_yaml(config, f'test-outputs/{job_id}/used-config.yml')

            # -------------------- INDEX BUILDING -------------------- #
            if exp_name == 'Mindex':
                pivot_df = loader.load_mindex_pivots()
                primary_descriptors = df_orig if config['data']['dataset'] == 'COPHIR' else df
                if config['data']['dataset'] == 'MOCAP':
                    input_df, labels_orig = intersect_mocap_dataset(pivot_df[:50], labels_orig)
                else:
                    input_df = pivot_df[:50]
                index = Mindex(input_df, labels_orig[:50], pivot_df, config['data']['dataset'])
                mem_train = 0
                config['experiment']['search-stop-conditions'] = [0.0001, 0.0005, 0.001, 0.005, 0.1]
            elif exp_name == 'Mtree':
                pivot_df = loader.load_mtree_pivots()
                primary_descriptors = df_orig if config['data']['dataset'] == 'COPHIR' else df
                index = Mtree(primary_descriptors[:10], labels_orig[:10], pivot_df[:10], config['data']['dataset'])
                mem_train = 0
                config['experiment']['search-stop-conditions'] = [0.0001, 0.0005, 0.001, 0.005, 0.1]
            else:
                if labels is not None:
                    labels_input = labels.head(10)
                else:
                    labels_input = None
                index = LMI(config['LMI'], df.head(10), labels_input, reindex=False)
                model_config = load_model_config(config['LMI']['model-config'], index.n_levels)
                start_training = time.time()
                index.train(model_config)
                training_time = time.time() - start_training
                mem_train = max(get_current_mem() - mem_data_load - mem_beginning, 0)
                save_yaml(model_config, f'test-outputs/{job_id}/used-model-config.yml')

            # -------------------- SEARCHING -------------------- #
            evaluator = Evaluator(
                index,
                {str(df.iloc[0].name): {str(df.iloc[0].name): 0.0}},
                df.iloc[:1],
                config,
                job_id=job_id,
                output_dir='test-outputs'
            )
            evaluator.bucket_occupancy()
            evaluator.run_evaluate()
            mem_finish = max(get_current_mem() - mem_train - mem_data_load - mem_beginning, 0)
            evaluator.generate_summary(mem_data_load, mem_train, mem_finish)

            execution_time = time.time() - start_loading

            write_to_file(f'test-outputs/{job_id}/times.csv', 'loading, training, execution')
            write_to_file(f'test-outputs/{job_id}/times.csv', f'{loading_time}, {training_time}, {execution_time}')

            previous_dataset = config['data']['dataset']
            previous_configuration = current_configuration
            free_memory(evaluator, index, config, pivot_df)

            if exp_name == 'multilabel-NN':
                labels = None
            elif labels is not None:
                labels.drop([col for col in labels.columns.to_list() if '_pred' in col], inplace=True, axis=1)

    LOG.info(f'Finished the experiment run: {get_current_datetime()}')

if __name__ == '__main__':
    assert len(sys.argv) > 1, \
        "Please provide the experiment setup you'd like to run as an arugment"\
        "(.yml config files found in experiment-setups/)"
    run_test_experiments(sys.argv[1:])

    # Check
    test_dir = 'test-outputs'
    assert Path.is_dir(Path(test_dir))

    files_to_check_for = [
        'bucket-summary.json',
        'search.csv',
        'summary.json',
        'used-config.yml'
    ]

    for exp_dir in os.listdir(test_dir):
        for file_to_check_for in files_to_check_for:
            assert Path.is_file(Path(f'{test_dir}/{exp_dir}/{file_to_check_for}')), f'Expected to find {test_dir}/{exp_dir}/{file_to_check_for}'
