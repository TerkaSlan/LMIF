import logging
from lmi.utils import get_logger_config
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import time
from lmi.distances.euclidean import get_euclidean_distance


class BaseIndex(object):
    """
    BaseIndex class is the parent of all index classes.
    Sets the basic index properties, such as:
    - number of tree levels
    - descriptors and labels
    - root node ID

    Attributes
    ----------
    n_levels : str
        Specified number of levels in LMI.
        For Supervised learning the value is derived from `labels_df`
        For Unsupervised learning, the value is determined from config's `LMI.n_levels`.
    descriptor_values : int
        The length of the descriptor vectors (`df.shape[1]`).
    labels : List[str]
        The labels of the individual levels ('L1'...'Ln' by default).
    """
    def __init__(self, df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to build the index on (i.e., object descriptors without labels)
        labels_df : pd.DataFrame
            The labels associated with the descriptors.
        """
        self.n_levels = labels_df.shape[1]
        self.descriptor_values = df.shape[1]
        self.labels = sorted(labels_df.columns.to_list())
        self.pred_labels = [f'{label}_pred' for label in self.labels]

        assert df.shape[0] == labels_df.shape[0], \
            f'Data and labels need to have the same number of rows. {df.shape[0]} != {labels_df.shape[0]}'
        labels_df = labels_df.reindex(index=df.index)
        assert all(df.index == labels_df.index), 'Data and labels need to have the same index (i.e., object_id).'
        self.root_node_id = -1
        logging.basicConfig(level=logging.INFO, format=get_logger_config())
        self.LOG = logging.getLogger(__name__)

    def get_object_count_in_buckets(self, labels) -> Dict:
        """ Creates dictionary with object counts in individual buckets.

        Returns
        -------
        count: Dict[Tuple[int], int]
            Mapping between identifier of buckets
            and number of objects that each of these buckets contains.
        """
        labels = [label for label in labels if label in self.data.y.columns]
        nan_placeholder = np.iinfo(np.int16).max
        count = self.data.y.fillna(nan_placeholder).groupby(labels).size().reset_index(name='n')
        count['bucket'] = list(count[labels].itertuples(
            index=False, name=None
            )
        )
        count['bucket'] = count['bucket'].apply(
            lambda x: tuple([label for label in x if label != nan_placeholder])
        )
        return dict(zip(count['bucket'].values, count['n'].values))

    def sort_priority_queue(self, priority_queue: np.ndarray, ascending=False) -> np.ndarray:
        """ Sorts priority queue in descending order, unless specified otherwise.

        Parameters
        ----------
        priority_queue : np.ndarray
            Priority queue - structure governing the search
        ascending : bool
            Dictates ascending sorting. Defaults to False.

        Returns
        ----------
        priority_queue (np.ndarray):
            Sorted Priority queue
        """
        sorted_arr = priority_queue[priority_queue[:, 1].argsort()[::-1]].astype(object)
        if ascending:
            return np.flip(sorted_arr, axis=0)
        else:
            return sorted_arr

    def get_object(self, object: Any, reshape=True) -> np.ndarray:
        """ Gets an object (one row) from the dataset.

        Parameters
        ----------
        object : int, str or np.ndarray/pd.Series
            Object to be searched

        Returns
        ----------
        np.ndarray:
            Retrieved object.
        """
        if type(object) == np.ndarray:
            return object.reshape(1, -1)
        elif type(object) == pd.core.series.Series:
            return object.values.reshape(1, -1)
        else:
            try:
                object = self.data.X.loc[object, ].values
                if reshape:
                    return object.reshape(1, -1)
                else:
                    return object
            except KeyError:
                raise Exception(f'Object with id {object} does not exist in the trained dataset.')

    def pop_node(
        self,
        priority_queue: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Pops node from the Priority queue

        Parameters
        ----------
        priority_queue (np.array):
            Priority queue - structure governing the search

        Returns
        ----------
        Tuple[np.ndarray, np.ndarray]:
            popped node in the form (node_label, probability), priority queue
        """
        popped_node = priority_queue[0]
        priority_queue = np.delete(priority_queue, 0, axis=0)
        return popped_node, priority_queue

    def get_node_label(self, popped_node: np.ndarray) -> Tuple[int]:
        """ Unifies a given `popped_node` label format as tuple of ints.

        Parameters
        ----------
        popped_node (np.ndarray):
            A popped node in the form (node_label, probability)

        Returns
        ----------
        Tuple[int]:
            Node label
        """
        node_label = popped_node[0]
        if isinstance(node_label, float) and node_label != -1:
            node_label = tuple((int(node_label),))
        return node_label

    def search(
        self,
        query: Any,
        stop_conditions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """ Performs search on a build index with a given query until all of the
        (potentially) many stop conditions are met.

        Starts with 'popping' the root node and putting all of its children
        to a Priority queue (PQ) - sorted structure with probabilities
        indicating similarity to the query.
        Then continues popping the top-most nodes and putting their children
        into PQ, then re-sorting (`self.search_node`).

        Parameters
        ----------
        query : int or np.ndarray
            Query to be searched
        stop_conditions : List[int]
            List of the number of objects in visited buckets.
            Used as a threshold to stop searching at this point.

        Returns:
        ----------
        Search history, time checkpoints, number of visited objects
        """

        def initialize_priority_queue() -> np.ndarray:
            """
            Create an initial entry for the root model for the priority queue.

            Returns
            -------
            np.array
                initial priority queue
            """
            return np.array((self.root_node_id, 1.0)).reshape(1, -1)

        start = time.time()

        priority_queue = initialize_priority_queue()
        objects_in_buckets_visited = 0
        current_ckpt = 0
        popped_nodes = []
        search_history = []
        time_checkpoints = []
        visited_objects = []

        query = self.get_object(query)
        while len(priority_queue) != 0:

            priority_queue, popped_node = \
                self.search_node(priority_queue, query)

            popped_nodes.append(popped_node[0])

            if popped_node[0] != -1 and popped_node[0] in self.objects_in_buckets:

                objects_in_buckets_visited += \
                    self.objects_in_buckets[popped_node[0]]

                while objects_in_buckets_visited >= stop_conditions[current_ckpt]:

                    bucket_nodes = [p for p in popped_nodes if p in self.objects_in_buckets]
                    search_history.append(bucket_nodes)
                    time_checkpoints.append(time.time() - start)
                    visited_objects.append(objects_in_buckets_visited)

                    self.LOG.debug(
                        f'Finished searching for query.'
                        f'stop condition={stop_conditions[current_ckpt]}, '
                        f'visited {objects_in_buckets_visited} objects in total, '
                        f'took {round(time_checkpoints[-1], 2)}s.'
                    )
                    if current_ckpt == len(stop_conditions) - 1:
                        return search_history, time_checkpoints, visited_objects
                    else:
                        current_ckpt += 1

        return [p for p in popped_nodes if p in self.objects_in_buckets], priority_queue

    # TODO: Extend to support Cophir as well
    def search_linear_profiset(
        self,
        query_id: str,
        stop_condition: int,
        return_top=30
    ) -> Tuple[List[int], float, float]:
        """ Enriches the searching with linear scan on the level of leaf nodes.
        The output of the search are the specific objects found, sorted by their distances to the query.
        If `return_top` is specified, only the top x objects will be returned.

        Parameters
        -------
        query_id : int
            Query to use in the search
        stop_condition : int
            How many objects should be retrieved by the search
        return_top : int
            How many most similar objects should be returned
            Optional

        Returns
        -------
        Tuple[List[int], float, float]
            List of IDs of the most similar objects, searching time, linear searching time (both in seconds)
        """
        search_results, search_times, _ = self.search(int(query_id.lstrip('0')), [stop_condition])
        process_time_s = time.time()

        def get_objects_from_buckets(search_result):
            final_arr = np.empty([0], dtype=np.int64)
            for bucket in search_result:
                final_arr = np.concatenate((final_arr, self.objects.indices[bucket]))
            return final_arr

        object_idxs = get_objects_from_buckets(search_results[0])
        distances = []
        query = self.data.X.loc[query_id]
        for _, row in self.data.X.iloc[object_idxs].iterrows():
            distances.append(get_euclidean_distance(query, row))

        objects_dists = np.vstack((self.data.X.iloc[object_idxs].index, distances))
        objects_dists_sorted = self.sort_priority_queue(objects_dists, ascending=True)
        most_similar_objects = [int(object_id) for object_id in objects_dists_sorted[1][:return_top]]
        process_time = time.time() - process_time_s

        return [str(o).zfill(10) for o in most_similar_objects], search_times[0], process_time
