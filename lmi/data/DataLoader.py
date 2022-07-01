"""
Data loaders are an interface between the data on the disk and the codebase.
We used 3 datasets to test LMI in our experiments: CoPhIR, Profiset and MOCAP.

=========
EXAMPLE USAGE:
=========
from utils import load_yaml
config = load_yaml('./config/config.yml')

loader = CoPhIRDataLoader(config)
df = loader.load_descriptors()
labels = loader.load_labels()
"""
from abc import abstractmethod
from typing import Dict, Tuple
from os import listdir, path
import pandas as pd
import numpy as np
import re

from sklearn import preprocessing
from lmi.data.enums import \
    Dataset, DatasetWidths, DatasetSizes, OriginalIndex, LeafNodeCapacity
from lmi.utils import isfile, load_json, get_logger_config

import logging
import time


class Dataloader:
    """ DataLoader.

    Contains common functionality for all 3 datasets currently used, including:
        - dataset directories
        - labels loading functionality (used in Supervised LMI)
        - abstract function for dataset loading

    Attributes
    ----------
    base_filename : str
        Full path to the base directory of the dataset
    descriptors : str
        Full path to the descriptor portion of the dataset (the data itself)
    object_ids : str
        Full path to the identifiers of the objects
    labels_dirname : str
        Full path to the labels directory. Used only in Supervised LMI.
    queries_filename
        Full path to the randomly chosen queries (objects from the dataset)
    knn_gt_filename
        Full path to the 30-NN ground truth (used for experiment evaluation)
    """
    def __init__(self, config: Dict):
        """
        Parameters
        ----------
        config : Dict
            the json config file
        """
        data = config['data']
        self.base_data_dir = data['data-dir']
        self.dataset = data['dataset']
        self.dataset_size = data['dataset-size']

        dataset_name = f'{Dataset[self.dataset].value}{self.dataset_size}'
        self.base_filename = path.join(self.base_data_dir, 'datasets', dataset_name)
        self.descriptors = f'{self.base_filename}-descriptors.csv'
        self.object_ids = f'{self.base_filename}-objects.txt'

        if config['experiment']['queries-out-of-dataset']:
            ood = 'queries-out-of-dataset'
            self.queries_filename = \
                f'{self.base_data_dir}/queries/{ood}/{Dataset[self.dataset].value}-{ood}-objects.txt'
            self.queries_descriptors = self.queries_filename.replace('objects.txt', 'descriptors.csv')
            self.knn_filename = \
                f'{self.base_data_dir}/ground-truths/{ood}/{Dataset[self.dataset].value}'\
                f'-1k-for-{self.dataset_size}.json'
        else:
            if self.dataset_size == '1M':
                self.queries_filename = \
                    f'{self.base_data_dir}/queries/{Dataset[self.dataset].value}-queries.txt'
            else:
                self.queries_filename = \
                    f'{self.base_data_dir}/queries/{Dataset[self.dataset].value}-{self.dataset_size}-queries.txt'
            self.knn_filename = \
                f'{self.base_data_dir}/ground-truths/{Dataset[self.dataset].value}'\
                f'-1k-for-{self.dataset_size}-dataset.json'

        if 'original' in data:
            self.original_index = data['original']['index']
            self.original_index_cap = data['original']['leaf-node-capacity']
            self.set_pivots_filename()
        else:
            self.original_index = None
            self.original_index_cap = None
            self.pivots_filename = ''

        if not (self.original_index is None and self.original_index_cap is None):
            self.leaf_node_capacity = LeafNodeCapacity[f'LEAF_{self.original_index_cap}'].value
            original_index = OriginalIndex[self.original_index].value
            if 'multilabel' in config['LMI']['model-config']:
                self.labels_dirname = \
                    f'{self.base_data_dir}/labels/knn-labels/'\
                    f'{dataset_name}-{original_index}'\
                    f'-{self.leaf_node_capacity}.pkl'
            else:
                self.labels_dirname = \
                    f'{self.base_data_dir}/labels/{dataset_name}-{original_index}-{self.leaf_node_capacity}'
        else:
            self.labels_dirname = None

        self._shuffle = data['shuffle']
        self._shuffle_seed = data.get('shuffle-seed', np.random.randint(0, 10_000))
        self._normalize = data['normalize']

        logging.basicConfig(level=logging.INFO, format=get_logger_config())
        self.LOG = logging.getLogger(__name__)

    def get_knn_ground_truth(self):
        return load_json(self.knn_filename)

    def get_pivots_filename(self) -> str:
        """Constructs the path to the pivots DataFrame.
        Used in searching with M-tree/M-index.

        Returns:
            pivots' filename (str)
        """
        return self.pivots_filename

    def set_pivots_filename(self):
        """Constructs the path to the pivots DataFrame.
        Used in searching with M-tree/M-index.

        Returns:
            pivots' filename (str)
        """
        self.pivots_filename = \
            f'{self.base_data_dir}/pivots/' \
            f'{OriginalIndex[self.original_index].value}-{Dataset[self.dataset].value}-' \
            f'{self.dataset_size}'
        if self.original_index == 'MTREE':
            self.pivots_filename += f"-{LeafNodeCapacity[f'LEAF_{self.original_index_cap}'].value}.struct"
        else:
            self.pivots_filename += '.struct'

    @abstractmethod
    def load_dataset(self, shuffle: bool) -> pd.DataFrame:
        """ Loads the dataset from the disk into the memory.

        Parameters
        ----------
        shuffle: Boolean
            Decides whether the dataset should be randomly shuffled.

        Returns
        ----------
        df: pd.DataFrame
            Dataset of the loaded descriptors with index set to object_id.
        """
        pass

    def load_labels(self, labels_dtype=np.uint32) -> pd.DataFrame:
        """ Loads a DataFrame with labels and object_ids.
        Retrieves the sorted label filenames, loads them from the
        last one and iterativelly concatenates to the label DataFrame.

        Returns
        ----------
        df: pd.DataFrame
            Dataset of the loaded labels with index set to object_id.
        """

        def get_level_digit(label_filename: str) -> int:
            """ Exctracts the digit portion of label filename:
            `level-2.txt` -> 2

            Parameters
            ----------
            label_filename : str
                The string filename

            Returns
            ----------
            int
                Extracted digit
            """
            return int(re.findall(r'(\d+)', label_filename)[-1])

        def unify_dtypes(df) -> pd.DataFrame:
            """Converts the columns in the labels df into
            integers. Has to use the pandas' UInt16 type to allow
            conversion even for columns that have NaNs in them
            (common in case of M-index).

            Parameters:
                df (pd.DataFrame): labels DataFrame

            Returns:
                pd.DataFrame: Modified DataFrame
            """
            for col, dtype in df.dtypes.items():
                if dtype != np.uint32:
                    df[col] = df[col].astype(pd.UInt16Dtype())
            return df

        if self.labels_dirname is not None and self.labels_dirname.endswith('.pkl'):
            return pd.read_pickle(self.labels_dirname)

        elif self.labels_dirname is not None:

            filenames = listdir(self.labels_dirname)
            filenames = sorted(filenames, key=lambda filename: get_level_digit(filename))
            max_index_level = get_level_digit(filenames[-1])
            complete_label_names = [f'L{i}' for i in range(1, max_index_level+1)]
            label_names = complete_label_names

            time_start = time.time()
            df = pd.DataFrame([])
            filenames.reverse()
            for level, filename in enumerate(filenames):
                if level != 0:
                    label_names = complete_label_names[:-level]

                df_current = pd.read_csv(
                    f'{self.labels_dirname}/{filename}',
                    names=label_names + ['object_id'],
                    sep=r'[.+\s]',
                    dtype=labels_dtype,
                    header=None,
                    engine='python'
                )
                df = pd.concat([df, df_current])
                df = df.drop_duplicates(['object_id'])

            df = df.set_index('object_id')

            if 'MoCap' not in self.labels_dirname:
                assert df.shape == \
                    (DatasetSizes[f'SIZE_{self.dataset_size}'].value, max_index_level),\
                    f'Unexpected label dataset shape: {df.shape}'
                df = unify_dtypes(df)

            self.LOG.debug(f'Loading labels took {round(time.time() - time_start, 2)}s.')
            df = df[~df.index.duplicated(keep='first')]
            return df.sort_index()
        else:
            self.LOG.error('Could not load labels since `self.labels_dirname` is not defined.')
            self.LOG.error('Make sure to properly fill `data -> original` portion when using Supervised LMI.')

    def load_mtree_pivots(self):
        assert isfile(self.pivots_filename), f"'{self.pivots_filename}' was not found."

        struct_df = pd.read_csv(
            self.pivots_filename,
            sep='\t',
            header=None,
            index_col=0,
            names=['node', 'radius'],
            dtype={'node': str, 'radius': np.float64}
        )
        struct_df['node'] = \
            struct_df['node'].apply(lambda n: tuple([int(v) for v in n.split('.')]))
        struct_df['level'] = struct_df['node'].apply(lambda n: len(n))

        return struct_df

    def load_mindex_pivots(self, shuffle=False, normalize=False, crop_n=128) -> pd.DataFrame:
        if 'MoCap' in self.pivots_filename:
            self.pivots_filename = self.pivots_filename.replace('-1M', '')
        descriptors = f"{self.pivots_filename.split('.struct')[0]}-descriptors.csv"
        objects = f"{self.pivots_filename.split('.struct')[0]}-objects.txt"
        pivots = self.load_dataset(descriptors, objects, shuffle, None, normalize)
        if isinstance(pivots, tuple):
            pivots = pivots[0]
        return pivots.head(crop_n)

    def get_queries(self):
        if 'queries-out-of-dataset' in self.queries_filename:
            queries = self.load_dataset(
                self.queries_descriptors,
                self.queries_filename,
                shuffle=False,
                shuffle_seed=None,
                normalize=self._normalize
            )
            if isinstance(queries, tuple):
                return queries[0]
            else:
                return queries
        else:
            queries_df = pd.DataFrame(index=pd.read_csv(self.queries_filename, sep=' ', header=None)[2].values)
            # Drop two problematic query names in MoCaP dataset -- will use 998 queries inst. of 1k
            queries_df.drop(
                ['3278_3278-2952-208_2952_208',
                 '3399_3399-10992-62_10992_62'],
                inplace=True, errors='ignore'
            )
            return queries_df


class CoPhIRDataLoader(Dataloader):
    """
    The CoPhIR data loader.
    Contains the data loading and normalization functionality
    specific for the CoPhIR dataset.
    """
    def __init__(self, config: Dict):
        super().__init__(config)

    def load_descriptors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_dataset(
            self.descriptors,
            self.object_ids,
            self._shuffle,
            self._shuffle_seed,
            self._normalize
        )

    def load_tiny_descriptors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_dataset(
            self.descriptors,
            self.object_ids,
            self._shuffle,
            self._shuffle_seed,
            self._normalize,
            tiny=True
        )

    def load_dataset(
        self,
        descriptors,
        object_ids,
        shuffle,
        shuffle_seed,
        normalize,
        tiny=False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Loads the CoPhIR dataset from the disk into the memory.
        The resulting DataFrame is expected to have `self.dataset_size` rows
        and 282 columns.

        Note that the original LMI implementation used in LMI2021
        worked with 282 columns, ignoring the GPS coordinate descriptors,
        which are not present in every object. Here we're using 0 if not present.

        Assumes that the original CoPhIR dataset was pre-processed using
        `scripts/cophir-to-csv.sh`.

        Returns
        ----------
        df: pd.DataFrame
            Dataset of the loaded descriptors with index set to object_id.
        """
        assert isfile(descriptors) and isfile(object_ids), \
            f'Expected {descriptors} and {object_ids} to exist.'

        converters_dict = {i: np.int16 for i in range(DatasetWidths.COPHIR.value)}
        for geo_coord in [218, 219]:
            converters_dict[geo_coord] = np.float16
        self.LOG.info(f'Loading CoPhIR dataset from {descriptors}.')
        time_start = time.time()

        if tiny:
            df_orig = pd.read_csv(
                descriptors,
                header=None,
                sep=r'[,|;]',
                engine='python',
                dtype=np.int32,
                skiprows=999_990,
                usecols=[i for i in range(284) if i != 218 and i != 219]
            )
        else:
            df_orig = pd.read_csv(
                descriptors,
                header=None,
                sep=r'[,|;]',
                engine='python',
                dtype=np.int32,
                usecols=[i for i in range(284) if i != 218 and i != 219]
            )
        df_orig = df_orig.fillna(0)
        if tiny:
            df_objects = pd.read_csv(object_ids, skiprows=999_990, header=None, dtype=np.uint32)
        else:
            df_objects = pd.read_csv(object_ids, header=None, dtype=np.uint32)

        self.LOG.debug(f'Loading took {round(time.time() - time_start, 2)}s.')

        df_orig = df_orig.set_index([df_objects[0].values])
        if normalize:
            df = pd.DataFrame(self.normalize(df_orig.values))
            df = df.set_index([df_objects[0].values])
            if shuffle:
                df = df.sample(frac=1, random_state=shuffle_seed)
        elif shuffle:
            df = df_orig.sample(frac=1, random_state=shuffle_seed)
        else:
            df = df_orig
        return df, df_orig

    def normalize(
        self,
        objects: np.array,
        attribute_lengths=[12, 64, 80, 62, 64]
    ) -> np.array:
        """ Normalizes the descriptors per descriptor parts.
        Since there are 6 unique descriptor parts within the dataset,
        the normalization is performed by their respective parts --
        array of [12 x `self.dataset_size`], etc.

        Normalization performed is the basic standardization (z-score normalization).

        Parameters
        ----------
        objects: np.array
            Descriptors as numpy arrays.
        attribute_lengths : List[int]
            The lengths of each individual descriptor part.

        Returns
        ----------
        Normalized descriptors of the same shape.
        """
        col_pos = 0
        normalized = []
        for attribute_length in attribute_lengths:
            current = objects[:, col_pos:col_pos+attribute_length]
            normalized.append(preprocessing.scale(current))
            col_pos += attribute_length
        return np.hstack((normalized))


class ProfisetDataLoader(Dataloader):
    """
    The Profiset data loader.
    """
    def __init__(self, data: Dict):
        super().__init__(data)

    def load_descriptors(self) -> pd.DataFrame:
        return self.load_dataset(self.descriptors, self.object_ids, self._shuffle, self._shuffle_seed)

    def load_tiny_descriptors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_dataset(
            self.descriptors,
            self.object_ids,
            self._shuffle,
            self._shuffle_seed,
            self._normalize,
            tiny=True)

    def load_dataset(
        self,
        descriptors,
        object_ids,
        shuffle,
        shuffle_seed,
        normalize=None,
        tiny=False,
        num_tiny=None
    ) -> pd.DataFrame:
        """ Loads the Profiset dataset from the disk into the memory.
        The resulting DataFrame is expected to have `self.dataset_size` rows
        and 4096 columns.

        Returns
        ----------
        df: pd.DataFrame
            Dataset of the loaded descriptors with index set to object_id.
        """
        assert isfile(descriptors) and isfile(object_ids), \
            f'Expected {descriptors} and {object_ids} to exist.'

        self.LOG.info(f'Loading Profiset/MoCap dataset from {descriptors}.')
        time_start = time.time()
        if tiny:
            if num_tiny is None:
                num_tiny = 999_990
            df = pd.read_csv(
                descriptors,
                header=None,
                sep=r'[,| ]',
                engine='python',
                skiprows=num_tiny
            )
            if df.columns.shape[0] > 4096:
                df.drop([4096], axis=1, inplace=True)
        else:
            with open(descriptors) as f:
                data = np.loadtxt((line.replace(',', ' ') for line in f), dtype=np.float16)
            df = pd.DataFrame(data)
        try:
            if tiny:
                if num_tiny is None:
                    num_tiny = 999_990
                df_objects = pd.read_csv(object_ids, header=None, skiprows=num_tiny, dtype=np.uint32)
            else:
                df_objects = pd.read_csv(object_ids, header=None, dtype=np.uint32)
        except ValueError:
            # in case of MoCap the object_ids are strings
            if tiny:
                if num_tiny is None:
                    num_tiny = 999_990
                df_objects = pd.read_csv(object_ids, header=None, skiprows=num_tiny)
            else:
                df_objects = pd.read_csv(object_ids, header=None)

        df = df.set_index([df_objects[0].values])

        if not tiny:
            assert df.shape[1] == DatasetWidths.PROFISET.value, f'Unexpected dataset shape: {df.shape}'

        self.LOG.debug(f'Loading took {round(time.time() - time_start, 2)}s.')

        if shuffle:
            df = df.sample(frac=1, random_state=shuffle_seed)

        df = df[~df.index.duplicated(keep='first')]
        return df

    def get_knn_ground_truth(self):
        return load_json(self.knn_filename)


class MocapDataLoader(ProfisetDataLoader):
    """
    The MOCAP data loader.
    Inherits all of the functionality from the Profiset loader
    since the loading procedure is the same.
    """
    def __init__(self, data: Dict):
        super().__init__(data)
        self.descriptors = self.descriptors.replace('1M', '')
        self.object_ids = self.object_ids.replace('1M', '')
        if self.labels_dirname and 'knn-labels' in self.labels_dirname:
            self.labels_dirname = \
                self.labels_dirname.replace('1M', '').replace('MoCaP', 'MoCap')

    def load_labels(self) -> pd.DataFrame:
        default_dtype = np.uint32
        labels_dtype = {col: default_dtype for col in [f'L{i}' for i in range(8)]}
        labels_dtype['object_id'] = str
        return super().load_labels(labels_dtype)

    def load_tiny_descriptors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_dataset(
            self.descriptors,
            self.object_ids,
            self._shuffle,
            self._shuffle_seed,
            self._normalize,
            tiny=True,
            num_tiny=356_882
        )
