from lmi.indexes.BaseIndex import BaseIndex
from lmi.distances.cophir import get_cophir_distance
from lmi.distances.euclidean import get_euclidean_distance
import numpy as np
from typing import List, Tuple
import pandas as pd
from lmi.indexes.lmi_utils import Data


class Mtree(BaseIndex):

    """
    The M-tree indexing structure.

    The building procedure is not available in this framework,
    but we can still simulate searching with some dumped metadata
    from build (DataFrame of pivots).

    Attributes
    ----------
    descriptors : pd.DataFrame
        The current dataset of descriptors as loaded by DataLoader
    labels_df : pd.DataFrame
        Dataset of labels
    dataset : str
        Name of the dataset being used
    dataset_size : str
        Size of the dataset
    leaf_node_capacity : str
        Specific type of the dataset
    """
    def __init__(
        self,
        descriptors: pd.DataFrame,
        labels_df: pd.DataFrame,
        pivots_df: pd.DataFrame,
        dataset: str
    ):
        super().__init__(descriptors, labels_df=labels_df)

        if dataset == 'COPHIR':
            assert pd.api.types.is_integer_dtype(descriptors.dtypes.mode().values[0]), \
                'CoPhIR dataset appears to be normalized -- Mtree cannot compute distances on it.' \
                'Have you set `normalize=False` when loading the dataset?'
        self.dataset = dataset
        self.data = Data(descriptors, labels_df, descriptors.shape[1])
        self.objects_in_buckets = self.get_object_count_in_buckets(self.labels)
        self.pred_labels = self.labels
        self.pivots_df = pivots_df

        for i, label in enumerate(self.labels):
            self.pivots_df[label] = self.pivots_df['node'].apply(lambda x: x[i] if len(x) > i else -1)

    def search_node(
        self,
        priority_queue: List[np.ndarray],
        query: np.ndarray
    ) -> Tuple[List[np.ndarray], Tuple[str, float]]:
        """ Pops a node, evaluates its distance to the children and incorporates
        them to the priority queue.

        As opposed to LMI, M-tree requires a pass through all its children
        pivot representatives and evaluating the distance between them.

        Parameters:
            priority_queue (List[np.ndarray]):
                Priority queue - structure deciding the next node to visit
            query (np.ndarray):
                Query to evaluate the search for

        Returns:
            priority_queue, popped_node (Tuple[List[np.ndarray], Tuple[str, float]]):
                Sorted PQ and currently pocessed node
        """

        popped_node, priority_queue = self.pop_node(priority_queue)
        node_label = self.get_node_label(popped_node)

        def get_children_selection(node_label: str) -> str:
            """ Creates a pd.DataFrame query (not to be confused with the query argument in
            the parent function) to get the children pivots from `self.pivots`.

            Parameters:
                node_label (str): Label of the node being processed

            Returns:
                str: pd.DataFrame selection of children pivots
            """
            level = 1 if node_label == -1 else len(node_label) + 1
            selection = f'level == {level}'
            if level > 1:
                selection += ' & '
                for node, label in zip(node_label, self.labels[:len(node_label)]):
                    selection += f'{label} == {node} & '
                selection = selection[:-2]
            return selection

        children_df = self.pivots_df.query(get_children_selection(node_label))
        distances = []

        for pivot_id, child in children_df.iterrows():
            pivot = self.get_object(pivot_id)

            if self.dataset == 'COPHIR':
                distances.append(get_cophir_distance(query, pivot) - np.float(child['radius']))
            else:
                distances.append(get_euclidean_distance(query, pivot) - np.float(child['radius']))

        node_distances = np.array(([children_df['node'].values, distances]), dtype=object).T
        priority_queue = np.append(node_distances, priority_queue, axis=0)
        return self.sort_priority_queue(priority_queue, ascending=True), popped_node
