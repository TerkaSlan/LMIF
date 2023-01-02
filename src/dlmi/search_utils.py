import numpy as np
import pandas as pd
import time
from sklearn.metrics import pairwise_distances
import logging
from dlmi.Logger import get_logger_config
from sklearn.metrics.pairwise import cosine_similarity
import gc


def get_1nn_hit(db, data, query_id, query=None):
    actual_leaf_node = db.lmi.find_object_by_id(query_id)
    query = data.loc[query_id] if query is None else query
    pred_leaf_node, prob_distr, n_objects, time_taken = db.lmi.search(query, stop_condition_leaf=1)
    return [
        query_id,
        actual_leaf_node,
        pred_leaf_node[0],
        [round(p, 2) for p in prob_distr],
        n_objects,
        time_taken
    ]


def get_knn_hit(
    lmi,
    data,
    query_id,
    nn_idxs,
    k,
    metric,
    query=None,
    stop_condition_leaf=None,
    stop_condition_time=None,
    collect_knn_distr=False
):
    """ Invokes the knn search, evaluates the hit true knns
    :param lmi: The LMI object
    :param data: The currently indexed dataset
    :param query_id: The query id
    :param nn_idxs: Indexes of the sorted ground truth knns
    :param stop_condition_leaf: The number of leaf nodes to be visited
    :param stop_condition_time: The time limit for the search
    :param k: The number of knns to be returned

    :return: A list containing the collected results
    """
    assert not (stop_condition_leaf is None and stop_condition_time is None), \
        "Provide at least one stop condition"

    query = data.loc[query_id] if query is None else query
    actual_leaf_nodes = []
    if collect_knn_distr:
        for nn in data.index[nn_idxs][:100]:
            actual_leaf_nodes.append(lmi.find_object_by_id(nn))
    if stop_condition_leaf is not None:
        pred_leaf_nodes, prob_distr, n_objects, time_taken = lmi.search(
            query, stop_condition_leaf=stop_condition_leaf
        )
    elif stop_condition_time is not None:
        pred_leaf_nodes, prob_distr, n_objects, time_taken = lmi.search(
            query, stop_condition_time=stop_condition_time
        )
    object_ids = []
    for pred_leaf_node in pred_leaf_nodes:
        leaf_node = lmi.nodes.get(pred_leaf_node)
        if leaf_node is not None:
            object_ids.extend(leaf_node.object_ids)

    s = time.time()
    found_k_nns = data.loc[object_ids].index[
        sequential_search(query, data.loc[object_ids], k, metric)
    ]
    t_sequential_search = time.time() - s
    found_nns = set(list(found_k_nns)).intersection(set(data.index[nn_idxs][:k]))
    return [
        query_id,
        pred_leaf_nodes,
        [round(p, 2) for p in prob_distr],
        n_objects,
        len(found_nns),
        time_taken,
        t_sequential_search
    ], actual_leaf_nodes


def get_1nn_perf(db, data, queries):
    results = []
    columns = [
        'query',
        'query-pos',
        'query-predicted-pos',
        'probs',
        'n-objects',
        'time'
    ]
    logging.basicConfig(level=logging.DEBUG, format=get_logger_config())
    LOG = logging.getLogger('get_1nn_perf')
    for query_id in list(queries.index):
        s = time.time()
        res = get_1nn_hit(db, data, query_id)
        LOG.info(f'Search for query={query_id}, time: {time.time()-s}')
        results.append(res)
    results_df = pd.DataFrame(results, columns=columns)
    results_df['is-hit'] = results_df['query-pos'] == results_df['query-predicted-pos']
    hits_perc = (results_df['is-hit'].sum() / results_df.shape[0]) * 100
    logging.info(f'[1-NN Hit] computed from {len(queries)} queries -- {hits_perc}%')
    return results_df


def get_knn_perf(
    lmi,
    data,
    queries,
    metric,
    k=100,
    stop_condition_leaf=None,
    stop_condition_time=None,
    collect_knn_distr=True
):
    """ Computes the knn performance of the LMI for a given set of queries and a given k.

    :param lmi: The LMI object
    :param data: The currently indexed dataset
    :param queries: The queries to be used for the performance evaluation
    :param stop_condition_leaf: The number of leaf nodes to be visited

    :return: A dataframe containing the performance results
    """
    results = []
    columns = [
        'query', 'query-predicted-pos', 'probs', 'n-objects', 'n-knns-found', 'time', 'time-seq-search'
    ]
    logging.basicConfig(level=logging.DEBUG, format=get_logger_config())
    LOG = logging.getLogger('get_knn_perf')

    nns = get_objective_knns(queries.values, data, metric)
    knn_distributions = []
    for i, query_id in enumerate(list(queries.index)):
        s = time.time()
        res = get_knn_hit(
            lmi,
            data,
            query_id,
            nns[i],
            k,
            metric,
            queries.loc[query_id].values,
            stop_condition_leaf,
            stop_condition_time,
            collect_knn_distr
        )
        LOG.info(f'K-NN perf for query={query_id}, time: {time.time()-s}')
        if res[1] != []:
            knn_distributions.append(pd.Series(res[1]).value_counts())
        res = res[0]
        results.append(res)

    del nns
    gc.collect()
    results_df = pd.DataFrame(results, columns=columns)
    results_df['recall'] = results_df['n-knns-found'] / k
    hits_perc = (results_df['n-knns-found'].mean() / k) * 100
    logging.info(f'[{k}-NN Hit] computed from {len(queries)} queries -- {hits_perc}%')
    return results_df, knn_distributions


def pairwise_cosine(x, y):
    return (-1) * cosine_similarity(x, y)


def pairwise_euclidean(x, y):
    return pairwise_distances(x, y)


def get_objective_knns(queries, dataset, metric):
    if len(queries.shape) == 1:
        queries = queries.reshape(1, -1)
    if metric == 'Angular':
        dists = pairwise_cosine(queries, dataset)
    elif metric == 'L2':
        dists = pairwise_euclidean(queries, dataset)
    else:
        raise Exception('Unknown metric')
    return np.argsort(dists)


def sequential_search(query, found_objects, k, metric):
    if metric == 'Angular':
        dists = pairwise_cosine([query], found_objects)
    elif metric == 'L2':
        dists = pairwise_euclidean([query], found_objects)
    else:
        raise Exception('Unknown metric')
    return np.argsort(dists)[:, :k][0]
