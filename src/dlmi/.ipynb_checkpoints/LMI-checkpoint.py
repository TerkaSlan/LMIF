import numpy as np
import pandas as pd
import time
import itertools
from typing import List, Union
from enum import Enum
import logging
from dlmi.partitioning import cluster_kmeans_faiss, cluster_kmedoids
from dlmi.utils import data_to_torch, data_X_to_torch, Tuple
from dlmi.model import NeuralNetwork
from dlmi.Tree import InnerNode, LeafNode
from dlmi.Logger import Logger

class Inconsistencies(Enum):
    OVERFLOW = 'overflow'
    UNDERFLOW = 'underflow'


class NodeType(Enum):
    INNER_NODE = InnerNode.__name__
    LEAF_NODE = LeafNode.__name__


class LMI(Logger):
    # TODO: Documentation of the attributes
    def __init__(
        self,
        leaf_node_capacity_min,
        leaf_node_capacity_max,
        children_min,
        children_max,
        violating_nodes
    ):
        self.nodes = {}
        assert leaf_node_capacity_min < leaf_node_capacity_max
        assert leaf_node_capacity_min > 0
        assert children_min < children_max and children_min > 0

        self.ln_cap_min = leaf_node_capacity_min
        self.ln_cap_max = leaf_node_capacity_max
        self.child_n_min = children_min
        self.child_n_max = children_max
        self.violating_nodes = violating_nodes

        self.inconsistencies = {
            NodeType.INNER_NODE.value: {
                Inconsistencies.OVERFLOW.value: {},
                Inconsistencies.UNDERFLOW.value: {}
            },
            NodeType.LEAF_NODE.value: {
                Inconsistencies.OVERFLOW.value: {},
                Inconsistencies.UNDERFLOW.value: {}
            }
        }

        self.pq = []

        #logging.basicConfig(level=logging.INFO, format=get_logger_config())
        #self.LOG = logging.getLogger('a')

    def __len__(self):
        if len(self.nodes) <= 1:
            return 0
        else:
            return max([len(node_pos) for node_pos in list(self.nodes.keys())])-1

    def find_object_by_id(self, object_id):
        for node_pos, node in self.nodes.items():
            if node.is_leaf_node() and object_id in node.object_ids:
                return node.position
        return None

    def dump_structure(self, save_path=None):
        struct_df = pd.DataFrame(
            np.array([
                list(self.nodes.keys()),
                [node.__class__.__name__ for node in list(self.nodes.values())],
                [len(node) for node in list(self.nodes.values())]
            ], dtype=object),
            ['key', 'type', 'children']
        ).T
        struct_df = struct_df.set_index(['key'])
        if save_path is not None:
            # todo: save
            pass
        else:
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

    def get_allowed_n_violating_leaf_nodes(self):
        n_viol_nodes = int(self.get_n_leaf_nodes() * self.violating_nodes)
        return n_viol_nodes if n_viol_nodes > 0 else 1

    def inconsistencies_stats(self, input_df=None, save_path=None):
        node_types = [n.value for n in NodeType]
        inconsistency_types = [i.value for i in Inconsistencies]
        header = [
            f'{comb[0]}-{comb[1]}' for comb in itertools.product(
                node_types, inconsistency_types
            )
        ]
        counts = []
        for node_type in node_types:
            for inconsistency_type in inconsistency_types:
                try:
                    count = pd.DataFrame().from_dict(
                        self.inconsistencies[node_type][inconsistency_type],
                        orient='index'
                    ).count().values[0]
                except IndexError:
                    count = 0
                counts.append(count)
        index = 0 if input_df is None else input_df.index.max() + 1
        res_df = pd.DataFrame([counts], columns=header, index=[index])
        if input_df is not None:
            res_df = pd.concat([input_df, res_df])
        if save_path is not None:
            # todo: save
            pass
        else:
            return res_df

    def relocate_object(self, leaf_node_from, leaf_node_to, object, object_id, single_insert=True):
        idx = leaf_node_from.object_ids.index(object_id)
        del leaf_node_from.objects[idx]
        leaf_node_from.object_ids.remove(object_id)

        if single_insert:
            leaf_node_to.objects.extend([object])
            leaf_node_to.object_ids.extend([object_id])

            self.check_inconsistencies(leaf_node_from)
            self.check_inconsistencies(leaf_node_to)

    def relocate_objects(self, leaf_node_from, leaf_node_to, objects, object_ids):
        for object_id, object in zip(object_ids, objects):
            self.relocate_object(leaf_node_from, leaf_node_to, object, object_id, single_insert=False)

        leaf_node_to.objects.extend(objects)
        leaf_node_to.object_ids.extend(object_ids)

        self.check_inconsistencies(leaf_node_from)
        self.check_inconsistencies(leaf_node_to)
        self.logger.debug(
            f'Relocated {len(objects)} objects from `{leaf_node_from.position}`\
            to `{leaf_node_to.position}` (now containing {len(leaf_node_to.objects)}).'
        )

    def get_parent_node(self, position: Tuple[int]) -> Union[Tuple[int], None]:
        """ Returns the parent node based on `position` or None
        if the node has no parent (is root).
        """
        if len(position[:-1]) != 0:
            return self.nodes[position[:-1]]
        else:
            return None

    def insert_node(self, node: InnerNode, should_check=True):
        """ Puts `node` to LMI's `nodes` list and increases its parent's `children` count."""
        self.nodes[node.position] = node
        if should_check:
            self.check_inconsistencies(node)

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
        for inconsistency_type in [Inconsistencies.OVERFLOW, Inconsistencies.UNDERFLOW]:
            self.remove_inconsistency(inconsistency_type, node)
        del self.nodes[node.position]

    def remove_inconsistency(
        self,
        inconsistency_type: Inconsistencies,
        node: Union[InnerNode, LeafNode]
    ):
        relevant_inconsistency = self.inconsistencies[node.__class__.__name__][inconsistency_type.value]
        if node.position in relevant_inconsistency:
            self.logger.debug(f'Removed inconsistency: {inconsistency_type} from {node.position}')
            del relevant_inconsistency[node.position]

    def add_inconsistency(
        self,
        inconsistency_type: Inconsistencies,
        node: Union[InnerNode, LeafNode]
    ):
        relevant_inconsistency = self.inconsistencies[node.__class__.__name__][inconsistency_type.value]
        if node not in relevant_inconsistency:
            self.logger.debug(f'Added inconsistency: {inconsistency_type} at {node.position}')
            relevant_inconsistency[node.position] = len(node)

    def check_inconsistencies(self, node: Union[InnerNode, LeafNode]):
        """ Determines if a newly detected inconsistency occurs in `node`
        or if a logged inconsistency of `node` got resolved.
        Populates or removes from `self.inconsitencies`.
        """
        inconsistency = self.inconsistencies[node.__class__.__name__]
        for inconsistency_type in [Inconsistencies.OVERFLOW, Inconsistencies.UNDERFLOW]:
            if node.position not in inconsistency[inconsistency_type.value]:
                if (inconsistency_type == Inconsistencies.OVERFLOW and node.is_overflow()) or \
                   (inconsistency_type == Inconsistencies.UNDERFLOW and node.is_underflow()):
                    self.add_inconsistency(inconsistency_type, node)
            else:
                # Existing inconsistency -- is it still existing?
                if (inconsistency_type == Inconsistencies.OVERFLOW and not node.is_overflow()) or \
                   (inconsistency_type == Inconsistencies.UNDERFLOW and not node.is_underflow()):
                    self.remove_inconsistency(inconsistency_type, node)
                else:
                    inconsistency[inconsistency_type.value][node.position] = len(node)

    def search(
        self,
        query,
        stop_condition=None,
        stop_condition_leaf=None,
        stop_condition_time=None
    ) -> List[Tuple[int]]:
        assert not (
            stop_condition is None and stop_condition_leaf is None and stop_condition_time is None
        ), "At least one stop condition needs to be provided."

        candidate_answer = []
        prob_distrs = []
        n_objects = 0
        start_time = time.time()

        def predict_positions(parent_position: Tuple[int], query, n_objects):
            """ Follows the branch based on highest node prediction probability until
            a leaf node is met. Inserts data to the leaf node.
            """
            def predict_positions_of_child_nodes(parent_position: Tuple[int], query):
                self.logger.info(f'Visited node: {parent_position}')
                probs, positions = self.nodes[parent_position].nn.predict_single(data_X_to_torch(query))
                positions = [parent_position + tuple((position, )) for position in positions]
                prob_distr = np.array([positions, probs], dtype=object).T
                if type(self.pq) is list:
                    self.pq = prob_distr
                else:
                    self.pq = np.concatenate([self.pq, prob_distr], axis=0)
                    self.pq = self.pq[self.pq[:, 1].argsort()[::-1]]
                self.logger.debug(f'Top 3 entries in PQ: {self.pq[:3]}')
                top_entry = self.pq[0]
                #self.LOG.debug(top_entry)
                #self.LOG.debug(self.pq)
                self.pq = self.pq[1:] if len(self.pq) > 1 else []
                if self.pq == []:
                    return candidate_answer, prob_distrs, n_objects
                prob_distrs.append(top_entry[1])
                return predict_positions(top_entry[0], query, n_objects)

            if not hasattr(self.nodes[parent_position], 'nn'):
                candidate_answer.append(parent_position)
                # prob_distrs.append(prob_distr)
                n_objects = sum([len(self.nodes[leaf_node_pos].objects) for leaf_node_pos in candidate_answer])
                if stop_condition is not None and n_objects >= stop_condition:
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                if stop_condition_leaf is not None and \
                   (len(candidate_answer) == stop_condition_leaf or \
                    self.get_n_leaf_nodes() < stop_condition_leaf):
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                if stop_condition_time is not None and \
                     (time.time() - start_time) >= stop_condition_time:
                    return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                else:
                    if len(self.pq) == 0:
                        return candidate_answer, prob_distrs, n_objects, time.time() - start_time
                    top_entry = self.pq[0]
                    #self.LOG.debug(top_entry)
                    self.pq = self.pq[1:] if len(self.pq) > 1 else []
                    #self.LOG.debug(self.pq)
                    prob_distrs.append(top_entry[1])
                    return predict_positions(top_entry[0], query, n_objects)
            else:
                return predict_positions_of_child_nodes(parent_position, query)

        self.pq = []
        if isinstance(query, pd.core.series.Series):
            query = query.values
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
                self.check_inconsistencies(self.nodes[position])
            else:
                predict_positions_of_child_nodes(position, data)

        if len(self.nodes) == 0:
            node = LeafNode((0,), self.ln_cap_min, self.ln_cap_max)
            node.insert_objects(objects.values, objects.index)
            self.insert_node(node)

        else:
            predict_positions((0,), objects.values, objects.index)

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
                self.ln_cap_min,
                self.ln_cap_max
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
            self.check_inconsistencies(leaf_node)
            self.remove_node(leaf_node)

    @staticmethod
    def train_model(
        objects,
        labels: List[int],
        n_clusters: int,
        nn=None,
        epochs=500
    ) -> Tuple[NeuralNetwork, int]:
        """ Trains a new NeuralNetwork, collects the predictions."""
        data_X, data_y = data_to_torch(objects, labels)
        #if nn is None:
        nn = NeuralNetwork(input_dim=len(objects[0]), output_dim=n_clusters)
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
            _, labels = cluster_kmeans_faiss(leaf_node.objects, n_clusters=n_children)
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
        nn, pred_positions, loss = LMI.train_model(
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
        node = InnerNode(position, nn, self.child_n_min, self.child_n_max)
        # we'll not check the inconsistencies just yet
        self.insert_node(node, should_check=False)

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
        self.check_inconsistencies(node)
        return info_df

    def retrain(self, node: InnerNode, n_children: int, info_df, distance_function='L2'):
        """ Retrains `node` to have `n_children` descendants.
        Uses all the objects from its (grand)-children LeafNode-s.
        """
        n_objects = self.get_n_of_objects()
        def find_nodes_in_subtree(parent_pos: Tuple[int]) -> List[LeafNode]:
            """ Finds all the (grand-)child nodes of `parent_pos`."""
            leaf_nodes = []
            inner_nodes = []

            def is_parent(parent: Tuple[int], child: Tuple[int]):
                """ Checks if `parent` is a (grand-)parent node of `child` based on their positions."""
                if len(parent) >= len(child):
                    return False
                for i, parent_path in enumerate(parent):
                    if parent_path != child[i]:
                        return False
                return True

            for node_pos in list(self.nodes.keys()):
                if is_parent(parent_pos, node_pos) and not hasattr(self.nodes[node_pos], 'nn'):
                    leaf_nodes.append(self.nodes[node_pos])
                elif is_parent(parent_pos, node_pos) and node_pos != node.position:
                    inner_nodes.append(self.nodes[node_pos])
            return leaf_nodes, inner_nodes

        self.logger.debug(f'==== Retrain with {node.position}')

        assert hasattr(node, 'nn')
        s = time.time()
        # (1) Find leaf nodes in the subtree of `node`
        leaf_nodes, inner_nodes = find_nodes_in_subtree(node.position)
        # (2) Get their objects
        objects, object_ids = LMI.get_objects(leaf_nodes)
        t_collect_objects = time.time() - s
        info_df.loc[len(info_df.index)] = [
            f'RETRAIN-COLL-{node.position}-{n_children}',
            t_collect_objects,
            np.NaN,
            n_objects
        ]
        # (3) Cluster
        s = time.time()
        n_clusters = n_children if len(objects) > n_children else len(objects)
        if distance_function == 'L2':
            _, labels = cluster_kmeans_faiss(objects, n_clusters)
        else:
            _, labels = cluster_kmedoids(objects, n_clusters=n_children)
        t_partition = time.time() - s
        info_df.loc[len(info_df.index)] = [
            f'RETRAIN-PART-{node.position}-{n_children}',
            t_partition,
            np.NaN,
            n_objects
        ]
        s = time.time()
        # (4) Delete the old model, create a new one
        del node.nn.model
        del node.nn
        nn, pred_positions, loss = LMI.train_model(objects, labels, n_clusters)
        node.nn = nn
        t_train = time.time() - s
        self.logger.debug(f'Training loss: {loss}')
        info_df.loc[len(info_df.index)] = [
            f'RETRAIN-TRAIN-{node.position}-{n_children}',
            t_train,
            np.NaN,
            n_objects
        ]
        s = time.time()
        # (5) Delete the old leaf and inner nodes
        self.delete_nodes(leaf_nodes)
        self.delete_nodes(inner_nodes)

        # (6) Create new leaf nodes
        self.create_child_nodes(
            np.unique(labels),
            node.position,
            objects,
            object_ids,
            pred_positions,
            np.unique(pred_positions)
        )
        t_rest = time.time() - s
        info_df.loc[len(info_df.index)] = [
            f'RETRAIN-REST-{node.position}-{n_children}',
            t_rest,
            np.NaN,
            n_objects
        ]
        self.check_inconsistencies(node)
        return info_df

    def shorten(self, leaf_nodes: List[LeafNode]):
        to_be_reinserted = pd.DataFrame([])
        for leaf_node in leaf_nodes:
            self.logger.debug(f'==== Shorten with `{leaf_node.position}`')
            node = self.nodes[leaf_node.get_parent()]
            assert node.is_inner_node()
            assert leaf_node in node.children
            node.nn.model.remove_unit(leaf_node.position[-1])
            # we want to reinsert any objects left
            to_be_reinserted = pd.concat(
                [to_be_reinserted, pd.DataFrame(leaf_node.objects).set_index(pd.Index(leaf_node.object_ids))]
            )
        if len(to_be_reinserted) > 0:
            self.insert(to_be_reinserted)
        self.delete_nodes(leaf_nodes)
