from typing import List, Tuple
from dlmi.model import NeuralNetwork


class LeafNode():
    """ Leaf node of the tree.
    Args:
        position (int): Position of the node in the tree.
        min_leaf_node_capacity (int): Minimum number of samples in a leaf node.
        max_leaf_node_capacity (int): Maximum number of samples in a leaf node.
    """
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
    """ Inner node of the tree.
    Args:
        position (Tuple[int]): Position of the node in the tree.
        nn (NeuralNetwork): Neural network to be used for the node.
        child_n_min (int): Minimum number of children a node can have.
        child_n_max (int): Maximum number of children a node can have.
    """
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

        self.fullness = False

    def is_overflow(self) -> bool:
        return len(self.children) > self.child_n_max

    def is_full(self) -> bool:
        return len(self.children) >= self.child_n_max or self.fullness

    def set_is_full(self) -> bool:
        self.fullness = True

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
