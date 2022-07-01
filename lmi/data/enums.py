"""
Useful enum definitions realted to data.
"""
from enum import Enum


class Dataset(Enum):
    """
    Defines the type of the dataset
    """
    COPHIR = 'CoPhIR'
    MOCAP = 'MoCap'
    PROFISET = 'Profiset'


class OriginalIndex(Enum):
    """
    Defines the type of the original index used.
    """
    MTREE = 'MTree'
    MINDEX = 'MIndex'


class LeafNodeCapacity(Enum):
    """
    Defines the possible values for leaf node capacity --
        max. number of objects in the leaf node.
    A configuration type, necessary only in Supervised LMI.
    """
    LEAF_2000 = 'leaf2000'
    LEAF_200 = 'leaf200'
    LEAF_100 = 'leaf100'
    LEAF_20 = 'leaf20'


class DatasetWidths(Enum):
    """
    Defines the width of the dataset (number of descriptors for every object)
    """
    COPHIR = 284
    PROFISET = 4096
    MOCAP = 4096


class DatasetSizes(Enum):
    """
    Defines the size of the dataset
    """
    SIZE_1M = 1_000_000
    SIZE_100k = 100_000


class kNNFilename(Enum):
    """
    Defines the ground truth kNN file for every dataset size
    """
    SIZE_1M = '1k-for-1M-dataset'
    SIZE_100k = '1k-for-100k-dataset'
