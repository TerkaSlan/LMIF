from typing import List, Tuple
import logging
from dlmi.model import NeuralNetwork
from dlmi.utils import get_logger_config


class LeafNode():
    # TODO: Documentation of the attributes
    def __init__(
        self,
        position: Tuple[int],
        min_leaf_node_capacity: int,
        max_leaf_node_capacity: int
    ):
        self.objects = []
        self.object_ids = []
        self.position = position
        self.ln_cap_max = max_leaf_node_capacity
        self.ln_cap_min = min_leaf_node_capacity

        logging.basicConfig(level=logging.INFO, format=get_logger_config())
        self.LOG = logging.getLogger(self.__class__.__name__)
        self.LOG.debug(f'Created leaf node at `{position}`')

    def insert_objects(self, data, ids: List[int]):
        self.objects.extend(data)
        self.object_ids.extend(ids)

    def __len__(self):
        return len(self.objects)

    def get_level(self):
        return len(self.position)-1

    def is_overflow(self) -> bool:
        return len(self.objects) > self.ln_cap_max

    def is_underflow(self) -> bool:
        return len(self.objects) < self.ln_cap_min

    def is_inner_node(self) -> bool:
        return False

    def is_leaf_node(self) -> bool:
        return True
    
    def get_parent(self) -> Tuple[int]:
        return self.position[:-1]

class InnerNode():
    # TODO: Documentation of the attributes
    def __init__(
        self,
        position: Tuple[int],
        nn: NeuralNetwork,
        child_n_min: int,
        child_n_max: int
    ):
        self.nn = nn
        self.position = position
        self.children = []
        self.child_n_max = child_n_max
        self.child_n_min = child_n_min

        self.LOG = logging.getLogger(self.__class__.__name__)
        self.LOG.debug(f'Created inner node at `{position}`')

    def is_overflow(self) -> bool:
        return len(self.children) > self.child_n_max
    
    def is_full(self) -> bool:
        return len(self.children) >= self.child_n_max

    def is_underflow(self) -> bool:
        return len(self.children) < self.child_n_min

    def is_inner_node(self) -> bool:
        return True

    def is_leaf_node(self) -> bool:
        return False

    def __len__(self):
        return len(self.children)

    def get_level(self):
        return len(self.position)-1
