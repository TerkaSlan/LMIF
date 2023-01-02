import numpy as np
import pandas as pd
import time
import itertools
from typing import List, Union
from enum import Enum

from dlmi.partitioning import cluster_kmeans_faiss, cluster_kmedoids
from dlmi.utils import data_to_torch, data_X_to_torch, Tuple
from dlmi.model import NeuralNetwork
from dlmi.Tree import InnerNode, LeafNode
from dlmi.Logger import Logger


class NodeType(Enum):
    INNER_NODE = InnerNode.__name__
    LEAF_NODE = LeafNode.__name__


class BulkLMI(Logger):
    """ The Learned Metric Index implementation."""
    def __init__(
        self,
    ):
        """ Initialize the LMI. The attributes are set with the values provided in the config file.
        Args:
            :param leaf_node_capacity_min: The minimum number of objects that a leaf node can hold.
            :param leaf_node_capacity_max: The maximum number of objects that a leaf node can hold.
            :param children_min: The minimum number of children that an inner node can have.
            :param children_max: The maximum number of children that an inner node can have.
            :param violating_nodes: The number of violating nodes to be considered during the partitioning process.
        """
        self.nodes = {}
        self.pq = []

    def find_object_by_id(self, object_id):
        for node_pos, node in self.nodes.items():
            if node.is_leaf_node() and object_id in node.object_ids:
                return node.position
        return None

    def dump_structure(self) -> pd.DataFrame:
        """ Create a dataframe of LMI's structure."""
        struct_df = pd.DataFrame(
            np.array([
                list(self.nodes.keys()),
                [node.__class__.__name__ for node in list(self.nodes.values())],
                [len(node) for node in list(self.nodes.values())]
            ], dtype=object),
            ['key', 'type', 'children']
        ).T
        struct_df = struct_df.set_index(['key'])
        return struct_df

    def get_n_of_objects(self):
        str_df = self.dump_structure()
        return str_df[str_df['type'] == NodeType.LEAF_NODE.value]['children'].sum()

    def get_n_leaf_nodes(self):
        str_df = self.dump_structure()
        return str_df[str_df['type'] == NodeType.LEAF_NODE.value]['type'].count()

    def get_internal_nodes_pos(self):
        str_df = self.dump_structure()
        return list(str_df[str_df['type'] == NodeType.INNER_NODE.value].index)

    def get_leaf_nodes_pos(self):
        str_df = self.dump_structure()
        return list(str_df[str_df['type'] == NodeType.LEAF_NODE.value].index)

    def get_parent_node(self, position: Tuple[int]) -> Union[Tuple[int], None]:
        """ Returns the parent node based on `position` or None
        if the node has no parent (is root).
        """
        if len(position[:-1]) != 0:
            return self.nodes[position[:-1]]
        else:
            return None

    def insert_node(self, node: InnerNode):
        """ Puts `node` to LMI's `nodes` list and increases its parent's `children` count."""
        self.nodes[node.position] = node

        self.logger.debug(f'Inserted node `{node.position}` into LMI')

        parent_node = self.get_parent_node(node.position)
        if parent_node is not None:
            parent_node.children.append(node)

    def remove_node(self, node: Union[InnerNode, LeafNode]):
        """ Deleted `node` from LMI's `nodes` and decreases its parent's `children` count."""
        self.logger.debug(f'Removed node at `{node.position}`')
        parent_node = self.get_parent_node(node.position)
        if parent_node is not None:
            parent_node.children.remove(node)
        #for inconsistency_type in [Inconsistencies.OVERFLOW, Inconsistencies.UNDERFLOW]:
        #    self.remove_inconsistency(inconsistency_type, node)
        del self.nodes[node.position]


    def search(
        self,
        query,
        stop_condition=None,
        stop_condition_leaf=None,
        stop_condition_time=None
    ) -> Tuple:
        """ Searches for `query` in the LMI. Recursively searches through the tree (starting from root),
        fills out priority queue (sorted by probability == similarity to the query) and checks for a given
        stop condition.

        Args:
            :param query: The query to search for.
            :param stop_condition: Condition to stop the search.
            :param stop_condition_leaf: Condition to stop the search based on number of visited leaf nodes.
            :param stop_condition_time: Condition to stop the search based on the amount of time elapsed.
        Returns:
            :return: A Tuple of (candidate_answer, prob_distributions, n_objects, elapsed_time)
        """
        assert not (
            stop_condition is None and stop_condition_leaf is None and stop_condition_time is None
        ), "At least one stop condition needs to be provided."

        self.pq = []
        candidate_answer = []
        prob_distrs = []
        n_objects = 0
        start_time = time.time()

        def predict_positions(parent_position: Tuple[int], query, n_objects):
            """ Recursive function following the branch based on highest node prediction
            probability until a leaf node is met.
            BASE CASE: Stop condition is met or priority queue is empty
            Args:
                :param parent_position: The node being the parent of the branch to inspect.
                :param query: The query to search for.
                :param n_objects: The number of candidate objects.
            Returns:
                :return: A Tuple of (candidate_answer, prob_distributions, n_objects, time)
            """
            def predict_positions_of_child_nodes(parent_position: Tuple[int], query):
                """ Predicts the position of the child nodes of `parent_position` based on `query`.
                Recursively calls predict_positions with child nodes of `parent_position` if
                priority queue is not empty.

                Args:
                    :param parent_position: The node being the parent of the branch to inspect.
                    :param query: The query to search for.
                """
                # (2.1) Collect predictions from the parent node's model
                probs, positions = self.nodes[parent_position].nn.predict_single(data_X_to_torch(query))
                positions = [parent_position + tuple((position, )) for position in positions]
                prob_distr = np.array([positions, probs], dtype=object).T
                # (2.2) Extend the priority queue with the predictions
                if type(self.pq) is list:
                    self.pq = prob_distr
                else:
                    self.pq = np.concatenate([self.pq, prob_distr], axis=0)
                    self.pq = self.pq[self.pq[:, 1].argsort()[::-1]]
                self.logger.debug(f'Top 3 entries in PQ: {self.pq[:3]}')
                # (2.3) Recursively call predict_positions with the child nodes of `parent_position`
                top_entry = self.pq[0]
                self.pq = self.pq[1:] if len(self.pq) > 1 else []
                if len(self.pq) == 0:
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                prob_distrs.append(top_entry[1])
                return predict_positions(top_entry[0], query, n_objects)

            # (1) If node is a leaf node, return the candidate answer.
            if not hasattr(self.nodes.get(parent_position), 'nn'):
                candidate_answer.append(parent_position)
                n_objects = sum(
                    [len(self.nodes[leaf_node_pos].objects) for leaf_node_pos in candidate_answer if self.nodes.get(leaf_node_pos) != None]
                )
                # (1.1) Check if any stop condition is met.
                if stop_condition is not None and n_objects >= stop_condition:
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                if stop_condition_leaf is not None and (
                  len(candidate_answer) == stop_condition_leaf or
                  self.n_leaf_nodes < stop_condition_leaf
                ):
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                if stop_condition_time is not None and \
                   (time.time() - start_time) >= stop_condition_time:
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                else:
                    # (1.2) Check if priority queue is not empty
                    if len(self.pq) == 0:
                        return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                    # (1.3) Continue with the most probable node in the priority queue.
                    top_entry = self.pq[0]
                    self.pq = self.pq[1:] if len(self.pq) > 1 else []
                    prob_distrs.append(top_entry[1])
                    return predict_positions(top_entry[0], query, n_objects)
            else:
                # (2) If node is not a leaf node, move with the search to its child nodes.
                return predict_positions_of_child_nodes(parent_position, query)

        self.pq = []
        if isinstance(query, pd.core.series.Series):
            query = query.values

        # Starts the recursive search from the root node ((0,)).
        return predict_positions((0,), query, n_objects)

    def insert(self, objects: pd.DataFrame):
        """ Inserts `objects` to the leaf node with the highest probabilistic response."""

        def predict_positions(position: Tuple[int], data, ids: List[int]):
            """ Follows the branch based on highest node prediction probability until
            a leaf node is met. Inserts data to the leaf node.
            """
            def predict_positions_of_child_nodes(position: Tuple[int], data):
                pred_positions = self.nodes[position].nn.predict(data_X_to_torch(data))
                for pred_position_cat in np.unique(pred_positions):
                    predict_positions(
                        position + tuple((pred_position_cat,)),
                        data[np.where(pred_positions == pred_position_cat)],
                        ids[np.where(pred_positions == pred_position_cat)]
                    )

            # found the leaf node to insert data to
            if not hasattr(self.nodes[position], 'nn'):
                self.nodes[position].insert_objects(data, ids)
            else:
                predict_positions_of_child_nodes(position, data)

        if len(self.nodes) == 0:
            node = LeafNode((0,), np.nan, np.nan)
            node.insert_objects(objects.values, objects.index)
            self.insert_node(node)
        else:
            predict_positions((0,), objects.values, objects.index)

        self.n_leaf_nodes = self.get_n_leaf_nodes()


    def create_child_nodes(
        self,
        new_child_positions,
        parent_node_position: Tuple[int],
        objects,
        object_ids: List[int],
        pred_positions,
        unique_pred_positions=None
    ):
        """ Creates new leaf nodes based on number of new positions,
        inserts their respective data objects."""
        if unique_pred_positions is None:
            unique_pred_positions = new_child_positions
        for position_cat in new_child_positions:
            new_leaf_node = LeafNode(
                parent_node_position + tuple((position_cat, )),
                np.nan,
                np.nan
            )
            if position_cat in unique_pred_positions:
                current_cat_index = np.where(pred_positions == position_cat)
                accessed_mapping = map(objects.__getitem__, list(current_cat_index[0]))
                objs_to_insert = list(accessed_mapping)
                accessed_mapping = map(object_ids.__getitem__, list(current_cat_index[0]))
                obj_ids_to_insert = list(accessed_mapping)
                new_leaf_node.insert_objects(
                    objs_to_insert,
                    obj_ids_to_insert
                )
            self.insert_node(new_leaf_node)

    def delete_nodes(self, leaf_nodes: List[Union[LeafNode, InnerNode]]):
        for i in range(len(leaf_nodes)-1, -1, -1):
            leaf_node = leaf_nodes[i]
            self.remove_node(leaf_node)

    @staticmethod
    def train_model(
        objects,
        labels: List[int],
        n_clusters: int,
        nn=None,
        epochs=500
    ) -> Tuple:
        """ Trains a new NeuralNetwork, collects the predictions.
        Args:
            objects: Data objects to train the model on.
            labels: Labels of the data objects.
            n_clusters: Number of classes to train the model on.
            nn: NeuralNetwork to train. If None, a new NeuralNetwork is created.
            epochs: Number of epochs to train the model.
        Returns:
            Tuple: (Trained NeuralNetwork, predictions, losses)
        """
        data_X, data_y = data_to_torch(objects, labels)
        nn = NeuralNetwork(input_dim=len(objects[0]), output_dim=n_clusters, model_type='MLP')
        losses = nn.train(data_X, data_y, epochs=epochs)
        nn.predict(data_X)
        return nn, nn.predict(data_X), losses[-1]

    @staticmethod
    def get_objects(leaf_nodes: List[LeafNode]) -> Tuple[int, List[int]]:
        """ Retrieves all the objects from `leaf_nodes`."""
        objects = np.concatenate(
            [leaf_node.objects for leaf_node in leaf_nodes if len(leaf_node.objects) > 0],
            axis=0
        )
        object_ids = np.concatenate(
            [leaf_node.object_ids for leaf_node in leaf_nodes if len(leaf_node.object_ids) > 0],
            axis=0
        )
        return objects, object_ids

    def deepen(self, leaf_node: LeafNode, n_children: int, info_df: pd.DataFrame, distance_function='L2'):
        """
        Input: Leaf node to be replaced by a new inner node
        Output: Additional clustering and learned model, `n` new leaf nodes
        """
        n_objects = self.get_n_of_objects()
        self.logger.debug(f'==== Deepen with {leaf_node.position}')
        # (1) Clustering
        s_partition = time.time()
        if distance_function == 'L2':
            n_clusters = n_children if len(leaf_node.objects) > n_children else len(leaf_node.objects)
            _, labels = cluster_kmeans_faiss(leaf_node.objects, n_clusters=n_clusters)
        else:
            _, labels = cluster_kmedoids(leaf_node.objects, n_clusters=n_children)
        t_partition = time.time() - s_partition
        self.logger.debug(f'==== Partitioned: {t_partition}')
        info_df.loc[len(info_df.index)] = [
            f'DEEPEN-PART-{leaf_node.position}-{n_children}',
            t_partition,
            np.NaN,
            n_objects
        ]
        # (2) Train model
        s_train = time.time()
        nn, pred_positions, loss = BulkLMI.train_model(
            leaf_node.objects, labels, n_children
        )
        self.logger.debug(f'Training loss: {loss}')
        t_train = time.time() - s_train
        info_df.loc[len(info_df.index)] = [
            f'DEEPEN-TRAIN-{leaf_node.position}-{n_children}',
            t_train,
            np.NaN,
            n_objects
        ]
        s_nodes_cleanup = time.time()
        position = leaf_node.position
        # (3) Delete the obsolete leaf node
        self.delete_nodes([leaf_node])

        # (4) Create new inner node
        node = InnerNode(position, nn, np.nan, np.nan)
        # we'll not check the inconsistencies just yet
        self.insert_node(node)

        # (5) Create new child nodes
        self.create_child_nodes(
            np.arange(n_children),
            node.position,
            leaf_node.objects,
            leaf_node.object_ids,
            pred_positions
        )
        t_nodes_cleanup = time.time() - s_nodes_cleanup
        info_df.loc[len(info_df.index)] = [
            f'DEEPEN-REST-{leaf_node.position}-{n_children}',
            t_nodes_cleanup,
            np.NaN,
            n_objects
        ]
        return info_df
