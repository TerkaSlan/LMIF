import os
import pandas as pd
import numpy as np
from lmi.enums import BasicMtree, BasicMindex, MODELS, MODELS_TABLES, MTREE, MINDEX, GMMMtree, \
                      GMMMindex, GMMCoPhIR, GMMProfiset, Mindex10perc, Mtree10perc, Mocap, \
                      MtreeOOD, MindexOOD, Table1, IndexTable1, Table3, IndexTable3
import itertools
from lmi.visualization import Plot, plot_boxplots
import re
from os.path import isfile
from typing import List, Tuple
from enum import Enum
from lmi.utils import load_json, get_logger_config
import logging
import sys

DIR = './outputs/'
MODEL_NAMES = MODELS + MTREE + MINDEX + ['GMM']
logging.basicConfig(level=logging.INFO, format=get_logger_config())
LOG = logging.getLogger(__name__)


def deduplicate_runs(completed_runs: List[str]) -> List[str]:
    """ Selects only 1 instance out of each experiment run
        (in case there are multiple with different timestamps).
    """
    unique_runs = []
    runs_to_return = []
    for run in completed_runs:
        if run.split('--')[0] not in unique_runs:
            unique_runs.append(run.split('--')[0])
            runs_to_return.append(run)
    return runs_to_return


def get_files_in_dir() -> List[str]:
    """ Finds directories of experiment runs in `DIR`, deduplicates them.
        If there are runs that did not finish, summary.json won't be generated
        and its information can't be used -> disregard such runs.
    """
    completed_runs = [run for run in os.listdir(DIR) if isfile(f'{DIR}/{run}/summary.json')]
    return deduplicate_runs(completed_runs)


def regex_model_names(model_names: List[str]) -> str:
    """ Creates a regex for `model_names`."""
    model_names_re = '('
    for name in model_names:
        model_names_re += f'{name}|'
    model_names_re = model_names_re[:-1]
    model_names_re += ')'
    return model_names_re


def get_combinations(target_enum: Enum) -> Tuple[str]:
    """ Identifies all the possible dataset-index-leafnodecap-mlmodel combinations
        based on the input enum definition."""
    return tuple(['-'.join(p) for p in itertools.product(*[e.value for e in target_enum])])


def get_model_names(experiment_files: List[str], target_model_names=MODEL_NAMES) -> List[str]:
    """ Extracts the model identifiers from experiment setups."""
    model_names = []
    for exp_file in experiment_files:
        for model_name in target_model_names:
            if model_name != 'NN' and model_name in exp_file or \
               model_name == 'NN' and model_name in exp_file and 'multilabel-NN' not in exp_file:
                model_names.append(model_name)
                break
    return model_names


def get_relevant_experiment_files_basic_figures(combination: str) -> List[str]:
    """ Applies regex to match all the experiments of the current experiment combination (e.g. CoPhIR-1M-Mtree-200)."""
    models = regex_model_names(MODELS + MTREE + MINDEX)
    return [exp_file for exp_file in get_files_in_dir() if re.match(f'^{combination}-{models}--.*$', exp_file)]


def get_relevant_experiment_files_gmm(combination: str):
    """ Applies regex to match all the experiments of the current experiment combination."""
    return [exp_file for exp_file in get_files_in_dir() if re.match(f'^{combination}--.*$', exp_file)]


def log_used_exp_files(combination, experiment_dirs_to_plot):
    """ Informs the user which experiments files are used in a given plot."""
    exp_dirs_rel = [e.split('/')[-1] for e in experiment_dirs_to_plot]
    LOG.info(
        f"For {combination} creating plots from the following experiment files: {exp_dirs_rel}"
    )


def plot_basic_exp_fig(fig_enum: Enum):
    """ Creates prediction DataFrame from collected predictions.

    Parameters
    -------
    fig_enum : Enum
        Definition for the possible combinations of dataset-index-leafnodecap-mlmodel

    Saves
    -------
    outputs/`combination`.png : for basic experiment setups (Figures 5, 6 in [1])
    """
    for combination in get_combinations(fig_enum):
        experiment_dirs_to_plot = [f'{DIR}{e}' for e in get_relevant_experiment_files_basic_figures(combination)]
        model_names = get_model_names(experiment_dirs_to_plot)
        log_used_exp_files(combination, experiment_dirs_to_plot)
        plot = Plot(
            experiment_dirs_to_plot
        )
        plot.get_experiment_infos(model_names)
        plot.plot_experiments(save=True, filename=combination, dir_to_save_to='outputs/')


def plot_mocap():
    """ Creates plots for Mocap experiments (Figure 10 in [1]).

    Parameters
    -------
    fig_enum : Enum
        Definition for the possible combinations of dataset-index-leafnodecap-mlmodel

    Saves
    -------
    outputs/`combination`.png : for basic experiment setups
    """
    experiment_dirs_to_plot = []
    for combination in get_combinations(Mocap):
        experiment_dirs_to_plot.extend(
            [f'{DIR}/{exp_file}' for exp_file in get_files_in_dir() if re.match(f'^{combination}--.*$', exp_file)]
        )
    model_names = get_model_names(experiment_dirs_to_plot)
    log_used_exp_files(Mocap.__name__, experiment_dirs_to_plot)
    plot = Plot(
        experiment_dirs_to_plot
    )
    plot.get_experiment_infos(model_names)
    plot.plot_experiments(
        save=True,
        filename=Mocap.__name__,
        dir_to_save_to='outputs/',
        x_ticks=[500, 10_000, 30_000, 50_000, 100_000]
    )


def plot_gmm_fig():
    """ Creates plots for GMM experiments (Figure 7 in [1]).

    Saves
    -------
    outputs/`combination`.png : for basic experiment setups
    """
    for combinations in [
        get_combinations(GMMMtree) + get_combinations(GMMCoPhIR),
        get_combinations(GMMMindex) + get_combinations(GMMProfiset)
    ]:
        experiment_dirs_to_plot = []
        for combination in combinations:
            experiment_dirs_to_plot.extend([f'{DIR}{e}' for e in get_relevant_experiment_files_gmm(combination)])
            log_used_exp_files(combination, experiment_dirs_to_plot)
        model_names = get_model_names(experiment_dirs_to_plot)
        plot = Plot(
            experiment_dirs_to_plot
        )
        plot.get_experiment_infos(model_names)
        plot.plot_experiments(save=True, filename=combination, dir_to_save_to='outputs/')


def reorder_matched_files(results: List[str]) -> List[str]:
    """ Reorders match experiments to keed the ordering of the original Figures in [1] (Figures 8, 9).
    """
    reorder_d = {'LR': 0, 'RF': 1, '00-NN': 2, 'multilabel': 3, '00-Mtree': 4, '00-Mindex': 4}
    final_results = []
    for model_name, target_position in reorder_d.items():
        for result in results:
            if model_name in result:
                final_results.append(result)

    return final_results


def plot_boxplot_figures(
    relevant_enums: List[Enum],
    group_identifier: str,
    labels: List[str],
    legend_loc=['lower right', 'lower right']
):
    """ Creates plots for 10%-100% and out-of-dataset experiments (Figures 8,9 in [1]).

    Saves
    -------
    outputs/`combination`.png : for basic experiment setups
    """
    for exp_enum, l_loc in zip(relevant_enums, legend_loc):
        combinations = get_combinations(exp_enum)
        models = regex_model_names([c[:-1] if c.endswith('-') else c for c in combinations])
        matched_files = [exp_file for exp_file in get_files_in_dir() if re.match(f'^{models}--.*$', exp_file)]
        matched_files = reorder_matched_files(matched_files)
        log_used_exp_files(exp_enum.__name__, matched_files)
        final_candidates_a = [test_file for test_file in matched_files if group_identifier not in test_file]
        final_candidates_b = [test_file for test_file in matched_files if group_identifier in test_file]
        if len(final_candidates_a) != len(final_candidates_b):
            LOG.warn(
                f'Unequal number of experiments in one group {final_candidates_a}\
                vs. in the second group {final_candidates_b}'
            )
        ticks = [final_candidate.split('-')[4] for final_candidate in final_candidates_a]

        scores_a = []
        for t in final_candidates_a:
            search = pd.read_csv(f'{DIR}/{t}/search.csv')
            scores_a.append(search[search['condition'] == 50_000]['knn_score'].values)
        scores_b = []
        for t in final_candidates_b:
            search = pd.read_csv(f'{DIR}/{t}/search.csv')
            scores_b.append(search[search['condition'] == 50_000]['knn_score'].values)

        plot_boxplots(
            scores_a,
            scores_b,
            ticks,
            labels=labels,
            legend_loc=l_loc,
            save=True,
            filename=f'{group_identifier}-{exp_enum.__name__}',
            dir_to_save_to='outputs/'
        )


def table1():
    """ Creates Table 1 from [1].

    Saves
    -------
    outputs/table-1.html
    """
    col_names = get_combinations(Table1)
    index_names = get_combinations(IndexTable1)
    df = pd.DataFrame(np.random.uniform(low=-1, high=0, size=(4, 16)), index=index_names, columns=col_names)

    for basic_enum in [BasicMtree, BasicMindex]:
        for combination in get_combinations(basic_enum):
            models = regex_model_names(MODELS_TABLES)
            res = [
                f'{DIR}{exp_file}' for exp_file in get_files_in_dir()
                if re.match(f'^{combination}-{models}--.*$', exp_file)
            ]
            model_names = get_model_names(res, target_model_names=MODELS_TABLES)
            for model_name, filename in zip(model_names, res):
                hw_info = load_json(f'{filename}/summary.json')['hw_info']
                times = pd.read_csv(f'{filename}/times.csv')
                table_model_name = '-'.join(combination.split('-')[2:] + [model_name])
                df.loc[
                    'build t. (h)-'+combination.split('-')[0], table_model_name
                ] = round(times.iloc[0][' training'] / 60 / 60, 2)
                df.loc[
                    'memory (gb)-'+combination.split('-')[0], table_model_name
                ] = round(hw_info['mem_train'] / 1024, 2)

    df.to_html(os.path.join(DIR, 'table-1.html'))


def table3(save_to=DIR):
    """ Creates Table 3 from [1].

    Saves
    -------
    outputs/table-3.html
    """
    col_names = get_combinations(Table3)
    index_names = get_combinations(IndexTable3)
    df = pd.DataFrame(np.random.uniform(low=-1, high=0, size=(4, 8)), index=index_names, columns=col_names)

    for combination in ['CoPhIR-1M-Mindex-2000', 'Profiset-1M-Mtree-200']:
        models = regex_model_names(MODELS)
        res = [
            f'{DIR}{exp_file}' for exp_file in get_files_in_dir()
            if re.match(f'^{combination}-{models}(-10perc|)--.*$', exp_file)
        ]
        model_names = get_model_names(res)
        for model_name, filename in zip(model_names, res):
            suffix = '10%' if '10perc' in filename else '100%'
            hw_info = load_json(f'{filename}/summary.json')['hw_info']
            times = pd.read_csv(f'{filename}/times.csv')
            table_model_name = f"{combination.replace('-1M', '')}-{model_name.strip(' ')}"
            df.loc['build t. (h)-'+suffix, table_model_name] = round(times.iloc[0][' training'] / 60 / 60, 2)
            df.loc['memory (gb)-'+suffix, table_model_name] = round(hw_info['mem_train'] / 1024, 2)

    df.to_html(os.path.join(save_to, 'table-3.html'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DIR = sys.argv[1]

    # table 1
    LOG.info("Creating 'Table 1'")
    table1()
    # table 3
    LOG.info("Creating 'Table 3'")
    table3()
    # figure 5
    plot_basic_exp_fig(BasicMtree)
    # figure 6
    LOG.info("Creating 'Figure 6'")
    plot_basic_exp_fig(BasicMindex)
    # figure 7
    LOG.info("Creating 'Figure 7'")
    plot_gmm_fig()
    # figure 8
    LOG.info("Creating 'Figure 8'")
    plot_boxplot_figures([Mindex10perc, Mtree10perc], '10perc', ['100% data', '10% data'])
    # figure 9
    LOG.info("Creating 'Figure 9'")
    plot_boxplot_figures(
        [MindexOOD, MtreeOOD],
        'ood',
        ['from dataset', 'out-of-dataset'],
        legend_loc=['lower right', 'upper right']
    )
    # figure 10
    LOG.info("Creating 'Figure 10'")
    plot_mocap()

    if isfile(f'{DIR}/report-template.html'):
        with open(f'{DIR}/table-1.html', 'r') as fd_tab_1:
            table_1_html = fd_tab_1.read()

        with open(f'{DIR}/table-3.html', 'r') as fd_tab_3:
            table_3_html = fd_tab_3.read()

        with open(f'{DIR}/report-template.html', 'r') as f:
            report_html = f.read()

        report_html = report_html.replace('{{ table-1 }}', table_1_html)
        report_html = report_html.replace('{{ table-3 }}', table_3_html)

        with open(f'{DIR}/report.html', 'w') as f:
            f.write(report_html)

    else:
        LOG.warn(f"Did not find 'report.html' in '{DIR}', could not generate the html report.")

    LOG.info('Finished')
