import pandas as pd
from tqdm import tqdm
import logging
from dlmi.utils import get_logger_config
import time

def get_1nn_hit(db, data, query_id):
    actual_leaf_node = db.lmi.find_object_by_id(query_id)
    pred_leaf_node, prob_distr, n_objects = db.lmi.search(data.loc[query_id], stop_condition_n_leaf_nodes=1)
    return [actual_leaf_node, pred_leaf_node[0], [round(p, 2) for p in prob_distr], n_objects]


def get_knn_hit(db, data, query_id, df_nns, stop_cond_n_leaf_nodes):
    actual_leaf_node = db.lmi.find_object_by_id(query_id)
    pred_leaf_node, prob_distr, n_objects = db.lmi.search(
        data.loc[query_id], stop_condition_n_leaf_nodes=stop_cond_n_leaf_nodes
    )
    object_ids = []
    for found_leaf_node in pred_leaf_node:
        object_ids.extend(db.lmi.nodes[found_leaf_node].object_ids)
    found_nns = set(object_ids).intersection(set(df_nns))
    return [actual_leaf_node, pred_leaf_node, [round(p, 2) for p in prob_distr], n_objects, len(found_nns)]


def get_1nn_perf(db, data, queries):
    results = []
    columns = ['query-pos', 'query-predicted-pos', 'probs', 'n-objects', 'is-hit']
    logging.basicConfig(level=logging.DEBUG, format=get_logger_config())
    LOG = logging.getLogger('get_1nn_perf')
    for query_id in list(queries.index):
        s = time.time()
        res = get_1nn_hit(db, data, query_id)
        LOG.info(f'Search for query={query_id}, time: {time.time()-s}')
        res.append(res[0] == res[1])
        results.append(res)
    results_df = pd.DataFrame(results, columns=columns)
    hits_perc = (results_df['is-hit'].sum() / results_df.shape[0]) * 100
    logging.info(f'[1-NN Hit] computed from {len(queries)} queries -- {hits_perc}%')
    return results_df


def get_knn_perf(db, data, queries, nns, stop_cond_n_leaf_nodes):
    results = []
    columns = ['query-pos', 'query-predicted-pos', 'probs', 'n-objects', 'n-found']
    logging.basicConfig(level=logging.DEBUG, format=get_logger_config())
    LOG = logging.getLogger('get_knn_perf')
    for i, query_id in enumerate(list(queries.index)):
        res = get_knn_hit(db, data, query_id, nns[i], stop_cond_n_leaf_nodes)
        results.append(res)
    results_df = pd.DataFrame(results, columns=columns)
    hits_perc = (results_df['n-found'].mean() / 100) * 100
    logging.info(f'[100-NN Hit] computed from {len(queries)} queries -- {hits_perc}%')
    return results_df
