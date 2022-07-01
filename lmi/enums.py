"""
Useful enum definitions realted to experiment visualization.
"""
from enum import Enum

DATASETS = ['CoPhIR', 'Profiset']
MTREE = ['Mtree']
MINDEX = ['Mindex']
SIZE = ['1M']
LEAF_CAPACITIES = ['2000', '200']
MODELS = ['LR', 'RF', 'NN', 'multilabel-NN']
MODELS_TABLES = ['LR', 'NN', 'multilabel-NN', 'RF']
TRAINING_PERCENTAGE = ['', '10perc']
QUERY_PLACEMENT = ['', 'ood']
BUILD_INFO = ['build t. (h)', 'memory (gb)']


class BasicMtree(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = DATASETS
    SIZE = SIZE
    INDEXES = MTREE
    LEAF_CAPACITIES = LEAF_CAPACITIES


class BasicMindex(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = DATASETS
    SIZE = SIZE
    INDEXES = MINDEX
    LEAF_CAPACITIES = LEAF_CAPACITIES


class BasicModels(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = DATASETS
    SIZE = SIZE
    INDEXES = MTREE + MINDEX
    LEAF_CAPACITIES = LEAF_CAPACITIES
    MODELS = MODELS


class BasicPercentage(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = DATASETS
    SIZE = SIZE
    INDEXES = MTREE + MINDEX
    LEAF_CAPACITIES = LEAF_CAPACITIES
    TRAINING_PERCENTAGE = TRAINING_PERCENTAGE


class GMMMtree(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = ['CoPhIR']
    SIZE = SIZE
    INDEXES = MTREE
    LEAF_CAPACITIES = ['2000']
    MODELS = ['LR'] + MTREE


class GMMCoPhIR(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = ['CoPhIR']
    SIZE = SIZE
    MODELS = ['GMM']


class GMMProfiset(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = ['Profiset']
    SIZE = SIZE
    MODELS = ['GMM']


class GMMMindex(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = ['Profiset']
    SIZE = SIZE
    INDEXES = MINDEX
    LEAF_CAPACITIES = ['200']
    MODELS = ['LR'] + MINDEX


class Mindex10perc(Enum):
    """
    Defines types needed for Figure 8 from [1]
    """
    DATASETS = ['CoPhIR']
    SIZE = SIZE
    INDEXES = MINDEX
    LEAF_CAPACITIES = ['2000']
    MODELS = MODELS
    TRAINING_PERCENTAGE = TRAINING_PERCENTAGE


class Mtree10perc(Enum):
    """
    Defines types needed for Figure 8 from [1]
    """
    DATASETS = ['Profiset']
    SIZE = SIZE
    INDEXES = MTREE
    LEAF_CAPACITIES = ['200']
    MODELS = MODELS
    TRAINING_PERCENTAGE = TRAINING_PERCENTAGE


class MindexOOD(Enum):
    """
    Defines types needed for Figure 9 from [1]
    """
    DATASETS = ['CoPhIR']
    SIZE = SIZE
    INDEXES = MINDEX
    LEAF_CAPACITIES = ['2000']
    MODELS = MODELS + MINDEX
    OOD = QUERY_PLACEMENT


class MtreeOOD(Enum):
    """
    Defines types needed for Figure 9 from [1]
    """
    DATASETS = ['Profiset']
    SIZE = SIZE
    INDEXES = MTREE
    LEAF_CAPACITIES = ['200']
    MODELS = MODELS + MTREE
    OOD = QUERY_PLACEMENT


class BasicBenchmarkMtree(Enum):
    """
    Defines types needed for Figure 5
    """
    DATASETS = DATASETS
    SIZE = SIZE
    INDEXES = MTREE
    LEAF_CAPACITIES = LEAF_CAPACITIES
    MODELS = MODELS + MTREE


class BasicBenchmarkMindex(Enum):
    """
    Defines types needed for Figure 6
    """
    DATASETS = DATASETS
    SIZE = SIZE
    INDEXES = MINDEX
    LEAF_CAPACITIES = LEAF_CAPACITIES
    MODELS = MODELS + MINDEX


class Table1(Enum):
    INDEXES = MTREE + MINDEX
    LEAF_CAPACITIES = LEAF_CAPACITIES
    MODELS = MODELS_TABLES


class IndexTable1(Enum):
    BUILD_INFO = BUILD_INFO
    DATASETS = DATASETS


class Table3(Enum):
    INDEXES = ['CoPhIR-Mindex-2000', 'Profiset-Mtree-200']
    MODELS = MODELS_TABLES


class IndexTable3(Enum):
    BUILD_INFO = BUILD_INFO
    TRAINING_PERCENTAGE = ['100%', '10%']


class Mocap(Enum):
    """
    Defines types needed for Table 1 from [1]
    """
    DATASETS = ['MoCaP']
    INDEXES = MINDEX
    LEAF_CAPACITIES = ['2000']
    MODELS = MODELS + MINDEX
