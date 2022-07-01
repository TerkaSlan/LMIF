from lmi.indexes.BaseIndex import BaseIndex
import numpy as np
from lmi.distances.cophir import get_cophir_distance
from lmi.distances.euclidean import get_euclidean_distance
import pandas as pd
import time
from typing import List, Tuple, Any
from lmi.indexes.lmi_utils import Data


class Mindex(BaseIndex):
    """
    The M-index indexing structure.

    The building procedure is not available in this framework,
    but we can still simulate searching with some dumped metadata
    from build (DataFrame of pivots).

    Attributes
    ----------
    objects_in_buckets : Dict[Tuple[int], int]
        Mapping between identifier of buckets
        (= nodes in the lower-most level of the tree)
        and number of objects that each of these buckets contains.
        - same as in LMI or M-tree
    existing_regions : Dict[Tuple[int], None]
        List of existing regions (= existing nodes in the tree),
        that are not buckets.
        Stored as a dictionary for searching speed.
    pred_labels : List[str]
        Prediction labels (= same as `labels`)
    """
    def __init__(
        self,
        descriptors: pd.DataFrame,
        labels_df: pd.DataFrame,
        pivots: pd.DataFrame,
        dataset: str
    ):
        """
        Parameters:
            descriptors : pd.DataFrame
                The current dataset of descriptors as loaded by DataLoader
            labels_df : pd.DataFrame
                Dataset of labels
            pivots : pd.DataFrame
                Dataset of the used pivots during building
                (will guide the searching procedure)
                - available in `/storage/brno12-cerit/home/tslaninakova/data/pivots`
            dataset : str
                Name of the dataset being used
        """
        super().__init__(descriptors, labels_df=labels_df)

        if dataset == 'COPHIR':
            assert pd.api.types.is_integer_dtype(descriptors.dtypes.mode().values[0]), \
                'CoPhIR dataset appears to be normalized -- Mindex cannot compute distances on it.' \
                'Have you set `normalize=False` when loading the dataset?'

        self.dataset = dataset
        self.data = Data(descriptors, labels_df, descriptors.shape[1])

        self.objects_in_buckets = self.get_object_count_in_buckets(self.labels)
        self.existing_regions = dict.fromkeys(
            [exist_region[:-1] for exist_region in self.objects_in_buckets.keys()]
        )
        self.pred_labels = self.labels
        self.pivots = pivots
        self.root_node_id = (())

    def calculate_wspd(self, current_node_path: Tuple[int], pivots: np.ndarray) -> float:
        """ Performs the WSPD (Weighted Sum of Pivot Distances) calculation to get
        the estimated best distance of `current_node_path` if it were to be
        prolonged to the maximum level (self.n_levels).

        For a better illustration:
            - Let's suppose that the maximum level is 8
            - we choose a query, compute distances between it and all the pivots (`self.get_distances`),
                and choose the `maximum level`, i.e., 8 of the closest pivots, e.g.: [4,1,24,4,22,5,6,8]
            - if `current_node_path` is (1), WSPD combines the distances of closest pivots
                with the distance of (1) in the following way:
                    dist(1) * 0.75^0 + dist(4) * 0.75^1 + dist(24) * 0.75^2 + ... + dist(8) * 0.75^7
            - if `current_node_path` is (7,15), WSPD combines the distances of closest pivots
                with the distance of (1) in the following way:
                    dist(7) * 0.75^0 + dist(15) * 0.75^1 + dist(4) * 0.75^2 + ... + dist(6) * 0.75^7

            That is, the path is constructed from the `current_node_path` (= a combination of popped node
            and current pivot) and the closest pivots that are not represented in `current_node_path`.

        This algoritm is only invoked if `current_node_path` exists as a region (= node) in the index.

        For WSPD details, see: https://link.springer.com/chapter/10.1007/978-3-030-28730-6_21

        Parameters:
            current_node_path (Tuple[int]):
                Currently processed node. A combination of popped node and current pivot.

            pivots (np.ndarray):
                Pivots representing all the regions.

        Returns:
            Node's distance (float)
        """
        closest_regions = self.drop_used_regions(current_node_path, pivots)
        used_regions = []
        for node in current_node_path:
            region = pivots[pivots[:, 0] == node]
            if region.shape[0] != 0:
                used_regions.extend(region)

        power_list = [pow(0.75, i) for i in range(self.n_levels)]
        estimated_path_distance = 0
        for closest_pivot, power in zip(np.vstack((used_regions, closest_regions)), power_list):
            estimated_path_distance += closest_pivot[1]*power
        return estimated_path_distance

    def drop_used_regions(self, used_regions: Tuple[int, int], closest_regions: np.ndarray) -> np.ndarray:
        """ Finds the regions within `closest_regions` that are not in `used_regions`.

        Parameters:
            used_regions (Tuple[int, int]):
                Identifiers of the current node's path.
            closest_regions (np.ndarray):
                Top `self.n_levels` regions closest to the query.

        Returns:
            np.ndarray: The closest regions to be used to 'extend' the `used_region` path.
        """
        for region in used_regions:
            closest_regions = closest_regions[np.where(closest_regions[:, 0] != region)]
        return closest_regions[:self.n_levels]

    def estimate_deepest_paths(
        self,
        priority_queue: np.ndarray,
        pivots: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int], float], bool]:
        """ Pops the priority queue, checks its path combination with all the
        pivots and if it exists, computes WSPD on them and puts them to priority queue.

        Args:
            priority_queue (np.ndarray) :
                Sorted priority queue containing all the non-visited nodes.
            pivots (np.ndarray) :
                Array -- [[pivot_id, pivot_distance_from_query], ...]

        Returns:
            Tuple[np.ndarray, Tuple[Tuple[int, int], float], bool]:
                - sorted Priority queue,
                - popped node,
                - indication of whether `popped_node` is a bucket
        """
        popped_node, priority_queue = self.pop_node(priority_queue)
        if popped_node[0] in self.objects_in_buckets:
            return priority_queue, popped_node, True

        path_distances = []
        region_names = []
        for pivot in pivots:
            if type(popped_node[0]) is tuple and pivot[0] in popped_node[0]:
                continue
            current_region = popped_node[0] + tuple((pivot[0],))
            if current_region in self.existing_regions or current_region in self.objects_in_buckets:
                estimated_path_distance_pivot = self.calculate_wspd(current_region, pivots)
                path_distances.append(estimated_path_distance_pivot)
                region_names.append(current_region)

        pivot_distances = np.array(([region_names, path_distances]), dtype=object).T
        priority_queue = np.append(pivot_distances, priority_queue, axis=0)

        return self.sort_priority_queue(priority_queue, ascending=True), popped_node, False

    def get_distances(
        self,
        priority_queue: np.ndarray,
        query: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[str, float]]:
        """ Gets the initial distanced of all the pivots to a query.
        This serves as a basis for constructing the priority queue.

        Parameters:
            priority_queue - PQ (np.ndarray): Initial priority queue to guide
            the searching - 2D array of form: [[pivot_id, distance], ...]
            query (np.ndarray): Query's descriptor

        Returns:
            priority_queue, popped_node (Tuple[np.ndarray, Tuple[str, float]]):
                Sorted PQ and popped node
        """
        popped_node, priority_queue = self.pop_node(priority_queue)
        distances = []

        for pivot_descriptor in self.pivots.values:
            if self.dataset == 'COPHIR':
                distances.append(get_cophir_distance(query, pivot_descriptor.T, reshape=False))
            else:
                distances.append(get_euclidean_distance(query, pivot_descriptor.T))

        region_ids = [i for i in range(self.pivots.shape[0])]
        node_distances = np.array(([region_ids, distances]), dtype=object).T
        priority_queue = np.append(node_distances, priority_queue, axis=0)

        return self.sort_priority_queue(priority_queue, ascending=True), popped_node

    def search(
        self,
        query_id: Any,
        stop_conditions: List[int]
    ) -> Tuple[np.ndarray, List[float], List[Tuple[int, int]]]:
        """Performs searching within M-index.
        1. Checks the query's distance to all the pivots (and thereby regions that they represent)
            - `self.get_distances`
        2. Forms an estimation of the deepest existing paths that start with each pivot
           using the WSPD algorithm
            - `self.estimate_deepest_paths`
        3. Creates an initial Priority queue using these estimations after popping the root node
        4. Continues popping the PQ and forming the estimation (2.), with the current node,
           while consulting `self.existing_regions` for child nodes/paths to put to the PQ.

        Args:
            query_id (int): Query's identifier
            stop_conditions (List[int]):
                Number of objects that need to be visited in order to stop
                the (sub-portion) of the search.

        Returns:
            Tuple[np.ndarray, List[float]]:
                Priority queue
        """
        def initialize_priority_queue() -> np.ndarray:
            """
            Create an initial entry for the root model for the priority queue.

            Returns
            -------
            initial priority queue: np.array
            """
            return np.array((self.root_node_id, 0.0), dtype=object).reshape(1, -1)

        start = time.time()

        priority_queue = initialize_priority_queue()
        objects_in_buckets_visited = 0
        current_ckpt = 0
        popped_nodes = []
        search_history = []
        time_checkpoints = []
        visited_objects = []

        if isinstance(query_id, (int, np.integer)) or isinstance(query_id, str):
            query = self.get_object(query_id, reshape=False)
        else:
            try:
                query = query_id.values
            except AttributeError:
                query = query_id

        individual_pivot_distances, _ = self.get_distances(priority_queue, query)
        priority_queue, _, _ = self.estimate_deepest_paths(
            priority_queue,
            individual_pivot_distances
        )

        while len(priority_queue) != 0:
            priority_queue, popped_node, is_bucket = self.estimate_deepest_paths(
                priority_queue,
                individual_pivot_distances
            )
            popped_nodes.append(popped_node[0])
            if is_bucket:
                objects_in_buckets_visited += \
                    self.objects_in_buckets[popped_node[0]]

                while objects_in_buckets_visited >= stop_conditions[current_ckpt]:
                    bucket_nodes = [p for p in popped_nodes if p in self.objects_in_buckets]
                    search_history.append(bucket_nodes)
                    time_checkpoints.append(time.time() - start)
                    visited_objects.append(objects_in_buckets_visited)

                    if isinstance(query_id, (int, np.integer)) or isinstance(query_id, str):
                        log_q = query_id
                    else:
                        log_q = query_id.name
                    self.LOG.debug(
                        f'Finished searching for query {log_q}, '
                        f'stop condition={stop_conditions[current_ckpt]}, '
                        f'visited {objects_in_buckets_visited} objects in total, '
                        f'took {round(time_checkpoints[-1], 2)}s.'
                    )
                    if current_ckpt == len(stop_conditions) - 1:
                        return search_history, time_checkpoints, visited_objects
                    else:
                        current_ckpt += 1
        return priority_queue
