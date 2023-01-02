import pickle
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import sys

from dlmi.utils import *
from dlmi.Logger import logging, get_logger_config
from dlmi.BulkLMI import BulkLMI
from dlmi.search_utils import get_knn_perf, get_1nn_perf
from dlmi.visualization import plot_build, plot_search, plot_knn_distribution_in_leaf_nodes


def get_data_path(dataset_in_config):
    if dataset_in_config == 'SIFT':
        data_path = '../data/sift-128-euclidean.hdf5'
    elif dataset_in_config == 'GIST':
        data_path = '../data/gist-960-euclidean.hdf5'
    elif dataset_in_config == 'GLOVE':
        data_path = '../data/glove-25-angular.hdf5'
    elif dataset_in_config == 'DEEP1B':
        data_path = '../data/deep-image-96-angular.hdf5'
    else:
        raise Exception('Unknown dataset')
    return data_path


def create_directories(dir_path):
    create_dir(dir_path)
    create_dir(f'{dir_path}/build/csv')
    create_dir(f'{dir_path}/build/png')
    create_dir(f'{dir_path}/search/leaf/csv')
    create_dir(f'{dir_path}/search/leaf/png')
    create_dir(f'{dir_path}/search/leaf-absolute/csv')
    create_dir(f'{dir_path}/search/leaf-absolute/png')
    create_dir(f'{dir_path}/search/time/csv')
    create_dir(f'{dir_path}/search/time/png')
    create_dir(f'{dir_path}/search/knn-distr/csv')
    create_dir(f'{dir_path}/search/knn-distr/png')
    create_dir(f'{dir_path}/struct')
    create_dir(f'{dir_path}/inconsistencies')
    create_dir(f'{dir_path}/index')


if __name__ == '__main__':
    assert len(sys.argv) > 1, \
        "Please provide the config file you'd like to run as an arugment"

    logging.basicConfig(level=logging.INFO, format=get_logger_config())
    LOG = logging.getLogger(__name__)

    increase_max_recursion_limit()

    config_file = sys.argv[1]
    config = load_yaml(config_file)
    data_path = get_data_path(config['Data']['dataset'])
    f = h5py.File(data_path, 'r')

    data = pd.DataFrame(list(f['train']))
    data = data.iloc[:int(config['Data']['subset'])]
    data = data.sample(data.shape[0], random_state=config['Data']['seed'])
    queries = pd.DataFrame(list(f['test'])).sample(
        config['Data']['queries_subset'], random_state=config['Data']['seed']
    )
    queries_s = queries.sample(10, random_state=config['Data']['seed'])

    job_id = get_environment_variable('PBS_JOBID').split('.')[0]
    exp_name = f"{get_current_datetime()}-{job_id}-{config['Data']['dataset']}-bulk-MLP"
    exp_name += f"-{config['Data']['subset']}-{get_hostname()}"
    dir_path = f'../experiments/{exp_name}'
    create_directories(dir_path)
    save_yaml(config, f'{dir_path}/used-config.yaml')

    logging.basicConfig(level=logging.DEBUG, format=get_logger_config())
    logging.debug('Initialized logger')
    distance_function = config['Data']['distance_function']

    blmi = BulkLMI()
    distance_function = config['Data']['distance_function']
    info_df = pd.DataFrame([], columns=['op', 'time-taken', 'size', '#-objects'])
    blmi.insert(data)
    info_df = blmi.deepen(blmi.nodes[(0,)], 100, info_df)

    last_i = 0
    if 'DEEP1B' in config['Data']['dataset'] or 'GLOVE' in config['Data']['dataset']:
        for i, leaf in enumerate(blmi.get_leaf_nodes_pos()):
            info_df = blmi.deepen(blmi.nodes[leaf], 16, info_df)
            #info_df.to_csv(f'{dir_path}/build/csv/build-{last_i}.csv', index=False)
            #plot_build(info_df, f'{dir_path}/build/png/build-{last_i}.png', last_i)
            blmi.dump_structure().to_csv(f'{dir_path}/struct/str-{i}.csv')
            last_i = last_i + 1
    else:
        for i, leaf in enumerate(blmi.get_leaf_nodes_pos()):
            info_df = blmi.deepen(blmi.nodes[leaf], 120, info_df)
            #info_df.to_csv(f'{dir_path}/build/csv/build-{i}.csv', index=False)
            #plot_build(info_df, f'{dir_path}/build/png/build-{i}.png', i)

            blmi.dump_structure().to_csv(f'{dir_path}/struct/str-{i}.csv')
            last_i = i + 1

    blmi.n_leaf_nodes = blmi.get_n_leaf_nodes()
    save_as_pickle(f'{dir_path}/index/index-{i}.pkl', blmi)
    info_df.to_csv(f'{dir_path}/build/csv/build-{last_i}.csv', index=False)

    i = last_i

    stop_condition_leafs = list(set([
        int(blmi.get_n_leaf_nodes()*sc) for sc in config['Experiment']['stop_conditions'] \
        if int(blmi.get_n_leaf_nodes()*sc) > 0
    ]))

    search_df = pd.DataFrame([])
    for j, stop_condition_leaf in enumerate(stop_condition_leafs):
        results_knn, _ = get_knn_perf(
            blmi,
            data,
            queries,
            metric=distance_function,
            k=100,
            stop_condition_leaf=stop_condition_leaf
        )
        results_knn['n-objects-total'] = blmi.get_n_of_objects()
        results_knn['sc-type'] = 'leaf'
        results_knn['sc'] = stop_condition_leaf
        search_df = pd.concat([search_df, results_knn])
        results_knn.to_csv(
            f'{dir_path}/search/leaf/csv/search-knn-leaf-{i}-{j}-{stop_condition_leaf}.csv',
            index=False
        )
        
    if len(stop_condition_leafs) != 0:
        plot_search(
            search_df,
            n_leaf_nodes=blmi.get_n_leaf_nodes(),
            filename=f'{dir_path}/search/leaf/png/search-knn-leaf-{i}.png',
            step=i,
            search_type='leaf'
        )

    search_df = pd.DataFrame([])
    stop_condition_times = config['Experiment']['stop_conditions_time']
    for stop_condition_time in stop_condition_times:
        results_knn, _ = get_knn_perf(
            blmi,
            data,
            queries,
            metric=distance_function,
            k=100,
            stop_condition_time=stop_condition_time
        )
        results_knn['n-objects-total'] = blmi.get_n_of_objects()
        results_knn['sc-type'] = 'time'
        results_knn['sc'] = stop_condition_time
        search_df = pd.concat([search_df, results_knn])
        results_knn.to_csv(
            f'{dir_path}/search/time/csv/search-knn-time-{i}-{stop_condition_time}.csv',
            index=False
        )

    plot_search(
        search_df,
        n_leaf_nodes=None,
        filename=f'{dir_path}/search/time/png/search-knn-time-{i}.png',
        step=i,
        search_type='time'
    )

    ### LEAF -- absoulte
    search_df = pd.DataFrame([])
    stop_condition_leafs = config['Experiment']['stop_conditions_absolute']
    for j, stop_condition_leaf in enumerate(stop_condition_leafs):
        if blmi.get_n_leaf_nodes() >= stop_condition_leaf:
            results_knn, knn_distributions = get_knn_perf(
                blmi,
                data,
                queries,
                metric=distance_function,
                k=100,
                stop_condition_leaf=stop_condition_leaf
            )
            results_knn['n-objects-total'] = blmi.get_n_of_objects()
            results_knn['sc-type'] = 'leaf'
            results_knn['sc'] = stop_condition_leaf
            search_df = pd.concat([search_df, results_knn])
            results_knn.to_csv(
                f'{dir_path}/search/leaf-absolute/csv/search-knn-leaf-{i}-{stop_condition_leaf}.csv',
                index=False
            )

        if i > 3 and stop_condition_leaf == 3:
            create_dir(f'{dir_path}/search/knn-distr/csv/{i}-{j}')
            create_dir(f'{dir_path}/search/knn-distr/png/{i}-{j}')
            for idx, (knn_distribution, q_id) in enumerate(zip(knn_distributions, queries_s.index)):
                found_pos_among_nn = set(
                    knn_distribution.keys()
                ).intersection(set(results_knn.iloc[idx]['query-predicted-pos']))

                knn_distribution.to_csv(
                    f'{dir_path}/search/knn-distr/csv/{i}-{j}/knn-distr-{q_id}.csv',
                )
                plot_knn_distribution_in_leaf_nodes(
                    knn_distribution,
                    q_id,
                    found_pos_among_nn,
                    f'{dir_path}/search/knn-distr/png/{i}-{j}/knn-distr-{q_id}.png'
                )
        
    plot_search(
        search_df,
        n_leaf_nodes=blmi.get_n_leaf_nodes(),
        filename=f'{dir_path}/search/leaf-absolute/png/search-knn-leaf-{i}.png',
        step=i,
        search_type='leaf'
    )
