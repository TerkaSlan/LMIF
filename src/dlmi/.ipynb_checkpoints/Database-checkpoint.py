import logging
import time
import numpy as np
from dlmi.LMI import LMI, Inconsistencies, NodeType, InnerNode
from dlmi.utils import load_yaml, data_X_to_torch, save_as_pickle, get_size_of_file
from dlmi.Logger import Logger


class Database(Logger):
    def __init__(self, config_path, dir_path='../test-dir'):
        self.config = load_yaml(config_path)
        #logging.basicConfig(level=logging.INFO, format=get_logger_config())
        #self.LOG = logging.getLogger(self.__class__.__name__)
        # to be replaced by loading config
        self.leaf_node_ideal_occupancy = (
            self.config['LMI']['leaf_node_capacity_max'] + \
            self.config['LMI']['leaf_node_capacity_min']
        ) / 2
        self.lmi = LMI(**self.config['LMI'])
        self.dir_path = dir_path
        self.version = 0
        self.distance_function = self.config['Data']['distance_function']

    def insert(self, data, info_df):
        # invoke LMI's insert
        # based on violations,
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
        reorganization_start = time.time()
        #info_df = self.__restructure_with_retrain(
        # TODO: Call recursively?
        info_df = self.reorganize(
            (0, ),
            self.lmi.inconsistencies,
            info_df
        )
        reorganization_time = time.time() - reorganization_start
        # TODO abstract into a function
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
        overflows = self.lmi.inconsistencies[NodeType.LEAF_NODE.value][Inconsistencies.OVERFLOW.value]
        if node_pos:
            overflows = {
                o:v for o,v in overflows.items() if self.lmi.get_parent_node(o).position == node_pos
            }
        return overflows

    def get_severe_underflows(self):
        severe = {}
        for k, v in self.lmi.inconsistencies[NodeType.LEAF_NODE.value][Inconsistencies.UNDERFLOW.value].items():
            if v < 5:
                severe[k] = v
        return severe

    # IN CONSTRUCTION
    # TODO -- Figure out a recursive way to relocate objects without having to do 1-NN search
    def __relocate_objects(self, node):
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
                            cand_answer, _ = self.lmi.search(query_object, stop_condition_leaf=1)
                            self.lmi.relocate_object(source, self.lmi.nodes[cand_answer[0]], query_object, query_id)


    def reorganize(self, node_position, inconsistencies, info_df):
        """
        HEURISTICS:
        - H1: Prefer broad over deep structure (respecting the constraints)
        - H2: Ignore non-severe underflows (if LMI has 1 level)
        - H3: Deal immediately with severe underflows (<5 objects / node)
        - H4: Overflows up to max self.lmi.violating_nodes

        STEPS:
        - (1): If `node` is an overflowing LEAF node, split it
        - (2): If `node` is a non-full inner node, retrain
        """
        def address_underflows():
            while self.get_severe_underflows() != {}:
                #self.logger.info(f'Addressing severe UNDERFLOWS at {self.get_severe_underflows()}')
                self.lmi.shorten(
                    [self.lmi.nodes[underflown_pos] for underflown_pos in self.get_severe_underflows()]
                )

        def get_n_children_estimate(node: InnerNode):
            n_ch_nodes = int(len(node) // self.leaf_node_ideal_occupancy) + 1
            n_ch_nodes = n_ch_nodes if n_ch_nodes >= self.lmi.child_n_min else self.lmi.child_n_min
            return n_ch_nodes
        
        def collect_to_retrain_data():
            to_retrain_nodes = []
            objects_to_retrain = []
            for overflown_node_pos, overflown_node_objects in self.get_overflows().items():
                to_retrain_nodes.append(overflown_node_pos)
                #self.logger.debug(
                #    f'OVERFLOW at {overflown_node_pos}'\
                #    f' | {len(self.lmi.nodes[overflown_node_pos])}]'
                #)
                objects_to_retrain.append(len(self.lmi.nodes[overflown_node_pos]))
            return objects_to_retrain, to_retrain_nodes

        # (1) -- Always a situation of root node being a leaf node
        node = self.lmi.nodes[node_position]
        if type(node).__name__ == NodeType.LEAF_NODE.value and node.is_overflow():
            n_child_nodes = get_n_children_estimate(node)
            self.logger.info(f'OVERFLOW at {node.position} | {len(node)}')
            self.logger.info(f'[DEEPEN] [{node.position}] [{n_child_nodes}]')
            info_df = self.lmi.deepen(node, n_child_nodes, info_df, self.distance_function)

        processed_internal_nodes = []
        while set(self.lmi.get_internal_nodes_pos()) != set(processed_internal_nodes):
            node_pos = self.lmi.get_internal_nodes_pos()[len(processed_internal_nodes)]
            node = self.lmi.nodes[node_pos]
            self.logger.info(f'Processing node: {node_pos}')
            if type(node).__name__ != NodeType.LEAF_NODE.value:
                # (2)
                n_attempts = 0; max_attempts = 5
                if not node.is_full():
                    # TODO: Get only overflows related to `node`
                    while len(self.get_overflows()) > self.lmi.get_allowed_n_violating_leaf_nodes():
                        self.logger.info(f'n. overflows: {self.get_overflows()}, n. allowed: {self.lmi.get_allowed_n_violating_leaf_nodes()}')
                        objects_to_retrain, nodes_to_retrain = collect_to_retrain_data()
                        n_child_nodes = int(
                            sum(objects_to_retrain) // self.leaf_node_ideal_occupancy
                        )
                        n_nodes = len(node) - len(nodes_to_retrain) + n_child_nodes
                        n_nodes = n_nodes if n_nodes < self.lmi.child_n_max else self.lmi.child_n_max

                        self.logger.info(f'[RETRAIN] [{node.position}] [{n_nodes}]')
                        info_df = self.lmi.retrain(
                            node, n_nodes, info_df, self.distance_function
                        )
                        address_underflows()
                        if node.is_full() or n_nodes == self.lmi.child_n_max:
                            node.set_is_full()
                            #self.__relocate_objects(node)
                            break
                        if n_attempts >= max_attempts:
                            self.logger.info(f'Reached max attempts ({max_attempts}) with RETRAIN')
                            #self.__relocate_objects(node)
                            break
                        n_attempts += 1
                # (3)
                if node.is_full() or n_attempts >= max_attempts:
                    # while loop for overflows
                    overflows = self.get_overflows(node_pos).copy()
                    for overflown_node_pos, overflown_node_objects in overflows.items():
                        self.logger.debug(
                            f'Addressing OVERFLOW at {overflown_node_pos} |'\
                            f' {len(self.lmi.nodes[overflown_node_pos])}]'
                        )

                        n_child_nodes = get_n_children_estimate(self.lmi.nodes[overflown_node_pos]) 
                        self.logger.info(f'[DEEPEN] [{overflown_node_pos}] [{n_child_nodes}]')
                        info_df = self.lmi.deepen(self.lmi.nodes[overflown_node_pos], n_child_nodes, info_df)
                
                processed_internal_nodes.append(node_pos)
            
            while self.get_severe_underflows() != {}:
                address_underflows()
                self.__relocate_objects(node)

        return info_df


    def __restructure_with_retrain(self, nodes, inconsistencies, info_df):
        """
        HEURISTICS:
        - H1: Prefer broad over deep structure (respecting the constraints)
        - H2: Ignore non-severe underflows (if LMI has 1 level)
        - H3: Deal immediately with severe underflows (<5 objects / node)
        - H4: Overflows up to max self.lmi.violating_nodes

        STEPS:
        - (1): If there's only one (root) leaf node in the structure, do DEEPEN
        - (2): If there is one level root -> children leaf nodes, do RETRAIN up to root node overflow violation
        """
        max_retrain_iterations = 5
        def address_underflows():
            while self.get_severe_underflows() != {}:
                self.logger.info(f'Addressing severe UNDERFLOWS at {self.get_severe_underflows()}')
                self.lmi.shorten(
                    [self.lmi.nodes[underflown_pos] for underflown_pos in self.get_severe_underflows()]
                )

        root_node = nodes[(0, )]
        # (1) First reorganization of the structure -- LMI consists of 1 (root) leaf node at this point
        if type(root_node).__name__ == NodeType.LEAF_NODE.value and root_node.is_overflow():
            n_child_nodes = int(len(root_node) // self.leaf_node_ideal_occupancy) + 1
            n_child_nodes = n_child_nodes if n_child_nodes >= self.lmi.child_n_min else self.lmi.child_n_min
            self.logger.info(f'OVERFLOW at {root_node.position} | {len(root_node)}')
            self.logger.info(f'[DEEPEN] [{root_node.position}] [{n_child_nodes}]')
            info_df = self.lmi.deepen(root_node, n_child_nodes, info_df, self.distance_function)
        # (2)
        elif type(root_node).__name__ != NodeType.LEAF_NODE.value:
            n_iterations = 0
            if not root_node.is_full():
                # H2
                while len(self.get_overflows()) != 0 and not root_node.is_full() and n_iterations < max_retrain_iterations:
                    to_retrain_nodes = []
                    to_retrain_objects = []
                    for overflown_node_pos, overflown_node_objects in self.get_overflows().items():
                        if overflown_node_objects > self.lmi.ln_cap_max:
                            to_retrain_nodes.append(overflown_node_pos)
                            self.logger.debug(
                                f'OVERFLOW at {overflown_node_pos}'\
                                f' | {len(self.lmi.nodes[overflown_node_pos])}]'
                            )
                            to_retrain_objects.append(len(self.lmi.nodes[overflown_node_pos]))

                    n_child_nodes = int(
                        sum(to_retrain_objects) // self.leaf_node_ideal_occupancy
                    )
                    target_n_nodes = len(root_node) - len(to_retrain_nodes) + n_child_nodes
                    target_n_nodes = target_n_nodes if target_n_nodes < self.lmi.child_n_max else self.lmi.child_n_max
                    #if target_n_nodes <= self.config['children_max']:
                    self.logger.info(
                        f'[RETRAIN] [{len(to_retrain_objects)}] [{root_node.position}]'\
                        f' [{target_n_nodes}]'
                    )
                    info_df = self.lmi.retrain(
                        root_node,
                        target_n_nodes,
                        info_df,
                        self.distance_function
                    )
                    address_underflows()
                    """
                    else:
                        # (1) Try to free up some categories by removing underflows:
                        self.logger.debug(
                            f'CANNOT call [RETRAIN]'\
                            f" root node max capacity reached ({self.config['children_max']})."
                        )
                        self.logger.debug(f"FREEING some space by addressing underflows")
                        address_underflows()
                        target_n_nodes = len(root_node) - len(to_retrain_nodes) + n_child_nodes
                        self.__relocate_objects(root_node)

                        # (2) If not, decrease the number of child nodes:
                        if target_n_nodes > self.config['children_max']:
                            old_n_child_nodes = n_child_nodes
                            n_child_nodes = self.config['children_max'] - len(root_node) + len(to_retrain_objects)
                            target_n_nodes = len(root_node) - len(to_retrain_nodes) + n_child_nodes
                            self.logger.debug(
                                f'DESCREASING target n. of child nodes'\
                                f'from {old_n_child_nodes} to {n_child_nodes}'
                            )

                        self.logger.info(
                            f'[RETRAIN] [{root_node.position}]'\
                            f' [{target_n_nodes}]'
                        )
                        info_df = self.lmi.retrain(
                            root_node,
                            len(target_n_nodes),
                            info_df
                        )
                    """
                    n_iterations += 1
            # Do DEEPEN as we reach overflow on the first level
            #if root_node.is_full() or self.get_overflows() != {}:
            if root_node.is_full() or n_iterations == max_retrain_iterations:
                overflows = self.get_overflows().copy()
                for overflown_node_pos, overflown_node_objects in overflows.items():
                    self.logger.debug(
                        f'Addressing OVERFLOW at {overflown_node_pos} |'\
                        f' {len(self.lmi.nodes[overflown_node_pos])}]'
                    )
                    n_child_nodes = int(
                        len(self.lmi.nodes[overflown_node_pos]) // self.leaf_node_ideal_occupancy
                    ) + 1
                    n_child_nodes = n_child_nodes + 1 if n_child_nodes <= 1 else n_child_nodes
                    self.logger.info(f'[DEEPEN] [{overflown_node_pos}] [{n_child_nodes}]')
                    info_df = self.lmi.deepen(self.lmi.nodes[overflown_node_pos], n_child_nodes, info_df, self.distance_function)

            while self.get_severe_underflows() != {}:
                address_underflows()
                self.__relocate_objects(nodes[(0, )])
        
        return info_df