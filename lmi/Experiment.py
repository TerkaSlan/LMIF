from lmi.utils import write_to_file, isfile, get_current_datetime, create_dir,\
    file_exists, save_json
import numpy as np
from lmi.indexes import BaseIndex
import pandas as pd
from typing import Dict, Tuple, List
import json
from os import path
import logging
import cpuinfo


class Evaluator:
    """ Runs an LMI experiment, evaluates it and dumps experiment info."""
    def __init__(
        self,
        index: BaseIndex,
        knns: Dict[str, Dict],
        queries: List,
        config: Dict[str, str],
        n_knns=None,
        job_id=None,
        output_dir=None
    ):
        """
        Arguments:
            index (BaseIndex):
                Trained index - LMI, or instantiated - M-index or M-tree
            job_id (str):
                Identifier of the experiment run
            config (Dict[str, str]):
                Experiment's configuration
        """
        self.index = index
        self.index_data = self.index.data.y.index.values

        assert len(set(knns.keys())) >= len(queries), \
            "`queries` file is larger than the `knns` file, hence not all queries can be evaluated."
        self.gt_knns = knns
        self.queries = queries

        self.stop_conditions_perc = config['experiment']['search-stop-conditions']
        self.stop_conditions = [
            int(index.data.X.shape[0]*cond) for cond in self.stop_conditions_perc
        ]
        if n_knns:
            self.n_knns = n_knns
        else:
            self.n_knns = len(knns[list(knns.keys())[0]])
        self.model = type(self.index).__name__
        if output_dir is None:
            output_dir = 'outputs'
        if job_id:
            self.job_id = path.join(output_dir, job_id)
        else:
            self.job_id = path.join(output_dir, get_current_datetime())
        create_dir(self.job_id)

        self.LOG = logging.getLogger(__name__)
        self.LOG.setLevel(logging.INFO)

    def bucket_occupancy(self):
        """Finds out the average and std of number of object in the leaf nodes (buckets)."""
        bucket_info = {}
        bucket_info['n_buckets'] = len(self.index.objects_in_buckets)
        bucket_values = list(self.index.objects_in_buckets.values())
        bucket_info['mean_bucket_occupancy'] = np.array(bucket_values).mean()
        bucket_info['std_bucket_occupancy'] = np.array(bucket_values).std()
        with open(f'{self.job_id}/bucket-summary.json', 'w') as f:
            json.dump(bucket_info, f, indent=4)

    def get_buckets(self, gt_knns: Dict[str, Dict[str, float]], query: str) -> Tuple[List[int], List[int]]:
        """Gets bucket identifiers of query's 30 nearest neighbors within `self.index`.
        Assumes that all of the neighbors can be found within the structure.

        Parameters:
            gt_knns (Dict[str, Dict[str, float]]): Dictionary of queries' 30 nearest neighbors
            query (str): Current query's identifier (as a string)

        Returns:
            Tuple[List[int], List[int]]:
            List of the buckets, without the nan (placeholder) values if present
            and number of their occurencies within the 30 NNs
        """
        nan_placeholder = -1
        if all([k.isnumeric() for k in gt_knns[query].keys()]):
            knn_keys = [int(k) for k in gt_knns[query].keys() if int(k) in self.index_data]
        else:
            knn_keys = [k for k in gt_knns[query].keys() if k in self.index_data]

        try:
            knn_buckets, counts = np.unique(
                np.nan_to_num(
                    self.index.data.y.loc[knn_keys][self.index.pred_labels].to_numpy(
                        dtype=np.float16,
                        na_value=np.nan
                    ),
                    nan=nan_placeholder
                ),
                axis=0,
                return_counts=True
            )
        except KeyError:
            self.LOG.error(f'Some of knn_keys: {knn_keys} are not in the dataset used by LMI.')
        return [tuple([int(b) for b in bucket if b != nan_placeholder]) for bucket in knn_buckets], counts

    def run_evaluate(self):
        """ Runs the searching for `self.queries`, evaluates the perf. with `self.gt_knns` and
        documents the results into `search.csv`.
        """
        search_results = []
        times = []
        results_filename = f'{self.job_id}/search.csv'
        if not file_exists(results_filename):
            write_to_file(
                results_filename,
                'query,condition,knn_score,time,visited_objects'  # noqa: E231
            )
        self.LOG.info(f'Starting the search for {len(self.queries)} queries.')
        for query_idx, (query_name, query_row) in enumerate(self.queries.iterrows()):
            if query_idx != 0 and query_idx % 100 == 0:
                self.LOG.info(f'Evaluated {query_idx}/{len(self.queries)} queries.')
            if query_row.shape[0] != 0:
                query = query_row
            else:
                query = query_name
            search_results, times, visited_objects_all = self.index.search(query, self.stop_conditions)
            knn_buckets, counts = self.get_buckets(self.gt_knns, str(query_name))
            for search_result, condition, time, visited_objects in zip(
                search_results, self.stop_conditions, times, visited_objects_all
            ):
                score = 0
                for bucket, n_of_occurences in zip(knn_buckets, counts):
                    if bucket in search_result:
                        score += n_of_occurences

                write_to_file(
                    f'{self.job_id}/search.csv',f'{query_name},'  # noqa: E231
                    f'{condition},{score / self.n_knns},{time},{visited_objects}'  # noqa: E231
                )
        self.LOG.info(f"Search is finished, results are stored in: '{self.job_id}/search.csv'")

    def generate_summary(self, mem_data_load=None, mem_train=None, mem_finish=None):
        """ Creates the `summary.json` file with aggregate information about the results.

        Parameters
        ----------
        mem_data_load : float
            Memory consumed after the data loading - is saved to `summary.json`
        """
        scores, conditions, times, visited_objects = \
            aggregate_search_info(self.job_id)

        self.LOG.info(f'Consumed memory by evaluating (MB): {mem_finish}')

        info_dict = create_info_dict(
            conditions,
            scores,
            times,
            visited_objects,
            self.job_id,
            self.stop_conditions_perc,
            mem_data_load,
            mem_train,
            mem_finish,
            self.model
        )
        save_json(info_dict, f'{self.job_id}/summary.json')


def aggregate_search_info(job_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Checks if `{job_id}/search.csv` exists and aggregates the searching
    information stored in `{job_id}/search.csv`.

    Parameters
    ----------
    job_id : str
        Directory containing `search.csv` file.

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        knn scores, stop conditions, search times and visited objects
    """
    assert isfile(f'{job_id}/search.csv'), \
        f'`{job_id}/search.csv` does not exist, cannot generate summary.'
    search_df = pd.read_csv(f'{job_id}/search.csv')
    condition_agg = search_df.groupby('condition')
    scores = condition_agg['knn_score'].mean().values
    conditions = condition_agg['condition'].mean().values
    times = condition_agg['time'].mean().values
    visited_objects = condition_agg['visited_objects'].mean().values
    return (
        scores,
        conditions,
        times,
        visited_objects
    )


def create_info_dict(
    conditions: List[int],
    scores: List[float],
    times: List[float],
    visited_objects: List[int],
    experiment: str,
    stop_conds: List[int],
    mem_data_load: float,
    mem_train: float,
    mem_finish: float,
    model=None
) -> Dict:
    """ Summarizes the experiment information to a dict which will form the `summary.json` file."""
    info_dict = {}
    if model:
        info_dict['model'] = model
    info_dict['experiment'] = experiment
    info_dict['stop_conditions_perc'] = stop_conds
    info_dict['results'] = {}
    for cond, score, time, vis_obj in zip(conditions, scores, times, visited_objects):
        info_dict['results'][str(cond)] = {}
        info_dict['results'][str(cond)]['time'] = time
        info_dict['results'][str(cond)]['score'] = score
        info_dict['results'][str(cond)]['visited_objects'] = int(vis_obj)

    cpu_info = cpuinfo.get_cpu_info()
    cpu_brand_raw = cpu_info['brand_raw']
    info_dict['hw_info'] = {}
    info_dict['hw_info']['mem_data_load'] = mem_data_load
    info_dict['hw_info']['mem_train'] = mem_train
    info_dict['hw_info']['mem_exp'] = mem_finish
    info_dict['hw_info']['cpu_brand'] = cpu_brand_raw
    info_dict['hw_info']['cpu_hz'] = cpu_info['hz_advertised_friendly']
    info_dict['hw_info']['cpu_arch'] = cpu_info['arch']
    return info_dict


def generate_summary(
    job_id,
    stop_conditions=[o*100 for o in [0.0005, 0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]]
):
    """ Creates the `summary.json` file with aggregate information about the results.

    Parameters
    ----------
    job_id : str
        Identifier of the job, will serve as name of the output directory
    stop_conditions: List[int]
        Stop conditions used in the experiment

    """
    scores, conditions, times, visited_objects = \
        aggregate_search_info(job_id)

    info_dict = create_info_dict(
        conditions,
        scores,
        times,
        visited_objects,
        job_id,
        stop_conditions,
        -1,
        -1,
        -1
    )
    save_json(info_dict, f'{job_id}/summary.json')
