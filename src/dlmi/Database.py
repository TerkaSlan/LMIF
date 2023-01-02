import time
import numpy as np
from dlmi.LMI import LMI, Inconsistencies, NodeType, InnerNode
from dlmi.utils import load_yaml, data_X_to_torch, save_as_pickle, get_size_of_file
from dlmi.Logger import Logger


class Database(Logger):
    """ The orchestrator implmenetation, guiding the building process of LMI. """
    def __init__(self, config, dir_path='../test-dir'):
        """ Initialize the Database object.
        Args:
            config (dict): The configuration dictionary.
        """
        if type(config) == str:
            self.config = load_yaml(config)
        else:
            self.config = config

        self.leaf_node_ideal_occupancy = (
            self.config['LMI']['leaf_node_capacity_max'] +
            self.config['LMI']['leaf_node_capacity_min']
        ) / 2
        self.lmi = LMI(**self.config['LMI'])
        self.dir_path = dir_path
        self.version = 0
        self.distance_function = self.config['Data']['distance_function']

    def insert(self, data, info_df):
        """ Insert a new data into the LMI. Saves infomation about the insert
        into info_df, dumps the LMI into a pickle file.
        Invokes the reorganization process, saves new info_df and LMI again.
        Args:
            data (np.array): The data to be inserted.
            info_df (pd.DataFrame): The dataframe to save the insert info.
        Returns:
            info_df (pd.DataFrame): The updated info dataframe.
        """
        self.logger.info(f'[INSERT] [{data.shape[0]}]')
        insert_start = time.time()
        self.lmi.insert(data)
        insert_time = time.time() - insert_start

        save_as_pickle(f'{self.dir_path}/index-{self.version}.pkl', self.lmi)
        index_size = get_size_of_file(f'{self.dir_path}/index-{self.version}.pkl')

        n_objects = self.lmi.get_n_of_objects()
        info_df.loc[len(info_df.index)] = [
            'INSERT',
            insert_time,
            index_size,
            n_objects
        ]
        # Reorganize is invoked
        reorganization_start = time.time()
        info_df = self.reorganize(
            (0, ),
            self.lmi.inconsistencies,
            info_df
        )
        reorganization_time = time.time() - reorganization_start

        save_as_pickle(f'{self.dir_path}/index-{self.version}.pkl', self.lmi)
        index_size = get_size_of_file(f'{self.dir_path}/index-{self.version}.pkl')
        info_df.loc[len(info_df.index)] = [
            'REORGANIZATION',
            reorganization_time,
            index_size,
            n_objects
        ]
        self.version += 1
        return info_df

    def get_overflows(self, node_pos=None):
        overflows = self.lmi.inconsistencies[
            NodeType.LEAF_NODE.value
        ][Inconsistencies.OVERFLOW.value]
        if node_pos:
            overflows = {
                o: v for o, v in overflows.items() if
                self.lmi.get_parent_node(o).position == node_pos
            }
        return overflows

    def get_severe_underflows(self):
        """ Get the severe underflows from the LMI.
        Underflows are severe if the number of objects in the node is less than 5
        """
        severe = {}
        for k, v in self.lmi.inconsistencies[
            NodeType.LEAF_NODE.value
        ][Inconsistencies.UNDERFLOW.value].items():
            if v < 5:
                severe[k] = v
        return severe

    def __relocate_objects(self, node):
        """ Relocate the objects from the node to target node based on 1-NN search.
        Invoked after detection of severe underflows.
        """
        for source in node.children:
            if source.is_leaf_node() and len(source.objects) < 1:
                self.lmi.shorten([source])

            elif source.is_inner_node() and len(source) > 0:
                self.__relocate_objects(source)

            elif len(source) > 0:
                source_objects_torch = data_X_to_torch(source.objects)
                source_objects_orig = np.array(source.objects)
                source_objects_ids = np.array(source.object_ids)

                pred_cats = node.nn.predict(source_objects_torch)
                uniques = np.unique(pred_cats)
                for cat in uniques:
                    if cat != source.position[-1]:
                        objects = source_objects_orig[np.where(pred_cats == cat)]
                        ids = source_objects_ids[np.where(pred_cats == cat)]
                        for query_object, query_id in zip(objects, ids):
                            #pred_leaf_node, prob_distr, n_objects, time_taken
                            cand_answer, _, _, _ = self.lmi.search(
                                query_object, stop_condition_leaf=1
                            )
                            self.lmi.relocate_object(
                                source, self.lmi.nodes[cand_answer[0]], query_object, query_id
                            )

    def reorganize(self, node_position, inconsistencies, info_df):  # noqa: C901
        """ The main method for invoking restructuring the internal shape of LMI.
        Args:
            node_position (tuple): The position of the node to be restructured.
            inconsistencies (dict): The inconsistencies of the LMI.
            info_df (pd.DataFrame): The dataframe to save the reorganization info.
        Returns:
            info_df (pd.DataFrame): The updated info dataframe.

        HEURISTICS:
        - H1: Prefer broad over deep structure (respecting the constraints)
        - H2: Ignore non-severe underflows
        - H3: Deal immediately with severe underflows (<5 objects / node)
        - H4: Overflows up to max self.lmi.violating_nodes

        STEPS:
        - (1): If `node` is an overflowing LEAF node, split it
        - (2): If `node` is a non-full inner node, retrain or split
        """
        def address_underflows():
            """ Calls the shorten operation for every node in the severe underflows
            """
            while self.get_severe_underflows() != {}:
                self.lmi.shorten(
                    [self.lmi.nodes[u] for u in self.get_severe_underflows()]
                )

        def get_n_children_estimate(node: InnerNode):
            """ Estimates how many child nodes should be created for the node."""
            n_ch_nodes = int(len(node) // self.leaf_node_ideal_occupancy) + 1
            n_ch_nodes = n_ch_nodes if n_ch_nodes >= self.lmi.child_n_min else self.lmi.child_n_min
            return n_ch_nodes

        def collect_to_retrain_data():
            """ Collects the data from the overflowing nodes."""
            to_retrain_nodes = []
            objects_to_retrain = []
            for overflown_node_pos, overflown_node_objects in self.get_overflows().items():
                to_retrain_nodes.append(overflown_node_pos)
                objects_to_retrain.append(len(self.lmi.nodes[overflown_node_pos]))
            return objects_to_retrain, to_retrain_nodes

        node = self.lmi.nodes[node_position]
        # (1) Node is an overflowing leaf node -> split/deepen
        if type(node).__name__ == NodeType.LEAF_NODE.value and node.is_overflow():
            n_child_nodes = get_n_children_estimate(node)
            self.logger.info(f'OVERFLOW at {node.position} | {len(node)}')
            self.logger.info(f'[DEEPEN] [{node.position}] [{n_child_nodes}]')
            info_df = self.lmi.deepen(node, n_child_nodes, info_df, self.distance_function)

        processed_internal_nodes = []
        # Iterate through all the internal nodes in the tree
        while set(self.lmi.get_internal_nodes_pos()) != set(processed_internal_nodes):
            node_pos = self.lmi.get_internal_nodes_pos()[len(processed_internal_nodes)]
            node = self.lmi.nodes[node_pos]
            if type(node).__name__ != NodeType.LEAF_NODE.value:
                n_attempts = 0
                max_attempts = 5
                if not node.is_full():
                    # (2)
                    # Do retrained if we need to solve for overflows
                    # (= we're over leaf violation limit) and there is a
                    # overflow in the current node
                    while len(self.get_overflows()) > self.lmi.get_allowed_leaf_violations() \
                      and len(self.get_overflows(node_pos)) != 0:
                        self.logger.info(
                            f'Total n. overflows: {len(self.get_overflows())}, \
                            n. allowed: {self.lmi.get_allowed_leaf_violations()} \
                            overflows in current node ({node_pos}): \
                            {len(self.get_overflows(node_pos))}'
                        )
                        objects_to_retrain, nodes_to_retrain = collect_to_retrain_data()
                        n_child_nodes = int(
                            sum(objects_to_retrain) // self.leaf_node_ideal_occupancy
                        )
                        n_nodes = len(node) - len(nodes_to_retrain) + n_child_nodes
                        n_nodes = n_nodes \
                            if n_nodes < self.lmi.child_n_max else self.lmi.child_n_max

                        self.logger.info(f'[RETRAIN] [{node.position}] [{n_nodes}]')
                        info_df = self.lmi.retrain(
                            node, n_nodes, info_df, self.distance_function
                        )
                        address_underflows()
                        if node.is_full() or n_nodes == self.lmi.child_n_max:
                            node.set_is_full()
                            break
                        if n_attempts >= max_attempts:
                            self.logger.info(f'Reached max attempts ({max_attempts}) with RETRAIN')
                            break
                        n_attempts += 1

                # If we cannot address overflows with retrain or the node is full, we split
                if node.is_full() or n_attempts >= max_attempts:
                    overflows = self.get_overflows(node_pos).copy()
                    for overflown_node_pos, overflown_node_objects in overflows.items():
                        self.logger.debug(
                            f'Addressing OVERFLOW at {overflown_node_pos} |'
                            f' {len(self.lmi.nodes[overflown_node_pos])}]'
                        )

                        n_child_nodes = get_n_children_estimate(self.lmi.nodes[overflown_node_pos])
                        self.logger.info(f'[DEEPEN] [{overflown_node_pos}] [{n_child_nodes}]')
                        info_df = self.lmi.deepen(
                            self.lmi.nodes[overflown_node_pos], n_child_nodes, info_df
                        )

                processed_internal_nodes.append(node_pos)

            # If there are any severe underflows, call shorten and relocate the objects.
            while self.get_severe_underflows() != {}:
                address_underflows()
                self.__relocate_objects(node)

        return info_df
