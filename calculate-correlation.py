import pandas as pd
import numpy as np
import pickle
from os import listdir, remove
from scipy.stats import pearsonr, spearmanr

load_path = 'results-from-primary-paper/'
repro_path = 'outputs/'

def save_pickle(arr, path):
    with open(path, 'wb') as f:
        pickle.dump(arr, f)


def open_pickle(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_to_file(path, row):
    with open(path, 'a') as f:
        f.write(f'{row}\n')


def aggregate(arr):
    assert arr.shape[1] == 9
    return np.array([np.mean(arr[:, i]) for i in range(arr.shape[1])])


def aggregate_mocap(arr):
    arr = arr[:, :8]
    assert arr.shape[1] == 8
    return np.array([np.mean(arr[:, i]) for i in range(arr.shape[1])])


def calculate_statistical_tests(vec_1, vec_2):
    if vec_1.shape != vec_2.shape:
        if vec_1.shape[0] < vec_2.shape[0]:
            vec_2 = vec_2[:vec_1.shape[0]]
        else:
            vec_1 = vec_1[:vec_2.shape[0]]
    pearson_corr, _ = pearsonr(vec_1, vec_2)
    spear_res = spearmanr(vec_1, vec_2)
    spearman_corr = spear_res.correlation
    return pearson_corr, spearman_corr


if __name__ == '__main__':

    all_orig_exps = sorted([f.split('.pickle')[0] for f in listdir(f'{load_path}') if f.endswith('.pickle') and 'time' not in f])
    try:
        remove('outputs/correlation-results.csv')
    except OSError:
        pass

    write_to_file(
        f'outputs/correlation-results.csv',
        'experiment,pearson-corr-recall,spearman-corr-recall'
    )
    
    for exp_name in all_orig_exps:
        # load the experiment files
        recall_orig = open_pickle(f'{load_path}/{exp_name}.pickle')
        if 'ood' not in exp_name and '10perc' not in exp_name:
            if 'MoCaP' in exp_name:
                recall_orig_agg = aggregate_mocap(recall_orig)
            else:
                recall_orig_agg = aggregate(recall_orig)
        else:
            recall_orig_agg = np.sort(recall_orig[:, 5])

        all_repro_exps = listdir(f'{repro_path}/')
        relevant_repro_exps = [repro_exps for repro_exps in all_repro_exps if repro_exps.startswith(f'{exp_name}--')]
        if len(relevant_repro_exps) != 0:
            if 'MoCaP' in exp_name:
                n_stop_conds = 8
            else:
                n_stop_conds = 9
            
            # preprocess the loaded repro DF to fit the original data  
            results_repro = pd.read_csv(f'{repro_path}/{relevant_repro_exps[0]}/search.csv')
            if 'ood' not in exp_name and '10perc' not in exp_name:
                results_repro = results_repro[results_repro['condition'] != 500_000]
                results_correlation = results_repro.set_index(['query'])
                if 'MoCaP' in exp_name:
                    recall_repro_agg = aggregate_mocap(results_repro['knn_score'].values.reshape(-1, n_stop_conds))
                else:
                    recall_repro_agg = aggregate(results_repro['knn_score'].values.reshape(-1, n_stop_conds))
            else:
                recall_repro_agg = np.sort(results_repro[results_repro['condition'] == 50_000]['knn_score'].values)

            recall_p_corr, recall_s_corr = calculate_statistical_tests(recall_orig_agg, recall_repro_agg)


            write_to_file(
                f'outputs/correlation-results.csv', ','.join(
                    [exp_name, str(recall_p_corr), str(recall_s_corr)]
                )
            )

    df = pd.read_csv('outputs/correlation-results.csv')
    df.to_html('outputs/correlation-results.html')
