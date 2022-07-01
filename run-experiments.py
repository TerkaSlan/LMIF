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

loader = None
previous_original = None
previous_dataset = None
previous_configuration = None
labels = None
queries = None
knns = None
pivot_df = None
df = None


def get_loader(config):
    if config['data']['dataset'] == 'PROFISET':
        loader = ProfisetDataLoader(config)
    if config['data']['dataset'] == 'MOCAP':
        loader = MocapDataLoader(config)
    elif config['data']['dataset'] == 'COPHIR':
        loader = CoPhIRDataLoader(config)
    return loader

# flake8: noqa: C901
if __name__ == '__main__':

    assert len(sys.argv) > 1, \
        "Please provide the experiment setup you'd like to run as an arugment"\
        "(.yml config files found in experiment-setups/)"

    experiment_dirs = []
    exp_names = []
    config_files = sys.argv[1:]
    config_files.sort(key=lambda x: x.split('/')[-1].split('-')[0])
    
    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)

    for exp_counter, config_file in enumerate(config_files):       
        exp_name = get_exp_name(config_file)
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
            if df is not None:
                try:
                    labels = labels_orig.loc[df.index]
                except AttributeError:
                    labels = None

        if loader is None or labels is None or config['data']['dataset'] != previous_dataset or exp_name == 'GMM':
            # Free the previously loaded dataset from the memory
            if config['data']['dataset'] != previous_dataset and previous_dataset is not None:
                del df
                df = None
                gc.collect()

            loader = get_loader(config)
            if df is None:
                df = loader.load_descriptors()

            # In case of CoPhIR dataset, the loaded `df` is tuple of normalized and not normalized version
            if isinstance(df, tuple):
                df_orig = df[1]
                df = df[0]
            else:
                df_orig = df

            if 'original' in config['data']:
                labels = loader.load_labels()
                if config['data']['dataset'] == 'MOCAP':
                    df, labels = intersect_mocap_dataset(df, labels)
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
            pivot_df = loader.load_mindex_pivots()
            primary_descriptors = df_orig if config['data']['dataset'] == 'COPHIR' else df
            index = Mindex(primary_descriptors, labels, pivot_df, config['data']['dataset'])
            mem_train = 0
        elif exp_name == 'Mtree':
            pivot_df = loader.load_mtree_pivots()
            primary_descriptors = df_orig if config['data']['dataset'] == 'COPHIR' else df
            index = Mtree(primary_descriptors, labels, pivot_df, config['data']['dataset'])
            mem_train = 0
        else:
            index = LMI(config['LMI'], df, labels)
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

        if exp_name == 'multilabel-NN':
            del labels
            labels = None
        elif labels is not None:
            labels.drop([col for col in labels.columns.to_list() if '_pred' in col], inplace=True, axis=1)

    LOG.info(f'Finished the experiment run: {get_current_datetime()}')
