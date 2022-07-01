"""
SimpleDataLoader is the most basic interface between the data on the disk and the codebase.
Created to support LMI training with any type of dataset irregadless of having queries / knn-gt / pivots file.

=========
EXAMPLE USAGE:
=========
from utils import load_yaml
config = load_yaml('./config/config-simple.yml')

loader = SimpleDataLoader(config['data'])
df = loader.load_descriptors()
"""
from typing import Dict
from os import path, listdir
import pandas as pd
import numpy as np
import re

from sklearn import preprocessing
from lmi.utils import isfile, load_json

import logging
import time


class SimpleDataLoader:
    """ The most simple form of data loader. Supports only unsupervised LMI training.

    In case of any specific needs in data loading, rewrite the `pd.read_csv` line or
    create a new class inheriting from this one with a custom `load_dataset` function.

    Attributes
    ----------
    base_data_dir : str
        Full path to the base directory of the dataset
    dataset_path : str
        Full path to the dataset
    queries_filename
        Full path to the randomly chosen queries (objects from the dataset)
    knn_gt_filename
        Full path to the 30-NN ground truth (used for experiment evaluation)
    """
    def __init__(self, data: Dict):
        """
        Parameters
        ----------
        data : Dict
            `data` portion of the json config file
        """

        self.base_data_dir = data['data-dir']
        assert 'dataset-file' in data, \
            'Expected to find `dataset-file` in `config`.'\
            'Are you using the correct config file (e.g. `config-simple.yml`)?'

        self.dataset_path = path.join(self.base_data_dir, data['dataset-file'])
        self.labels_dirname = path.join(self.base_data_dir, data['labels-dir'])
        self.pivots_filename = path.join(self.base_data_dir, data['pivots-filename'])
        if 'queries' in data:
            self.queries_filename = path.join(self.base_data_dir, data['queries'])

        if 'knn-gt' in data:
            self.knn_filename = path.join(self.base_data_dir, data['knn-gt'])

        self._shuffle = data['shuffle']
        self._normalize = data['normalize']

        logging.basicConfig(level=logging.INFO)
        self.LOG = logging.getLogger(__name__)

    def load_descriptors(self, dataset_path=None) -> pd.DataFrame:
        """ Loads the descriptors from the disk into the memory.
        Assumes that headers are present, the values are floating point numbers,
        fills any missing values with '0'.
        If `normalize=True` is specified in config, preforms z-score normalization.

        Returns
        ----------
        df: pd.DataFrame
            Dataset of the loaded descriptors with index set to object_id.
        """
        if dataset_path is None:
            assert isfile(self.dataset_path), f'Expected {self.dataset_path} to exist.'
            dataset_path = self.dataset_path
        else:
            assert isfile(dataset_path), f'Expected {dataset_path} to exist.'

        self.LOG.info(f'Loading dataset from {dataset_path}.')
        time_start = time.time()

        df = pd.read_csv(dataset_path, dtype=np.float32)

        self.LOG.debug(
            f'Loaded dataset of shape: {df.shape}.'
            f' Loading took {round(time.time() - time_start, 2)}s.'
        )

        df = df.fillna(0)
        if self._normalize:
            scaler = preprocessing.StandardScaler()
            df[df.columns] = scaler.fit_transform(df[df.columns])
        if self._shuffle:
            df = df.sample(frac=1)
        return df

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

        if self.labels_dirname.endswith('.pkl'):
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
            self.LOG.debug(f'Loading labels took {round(time.time() - time_start, 2)}s.')
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            if not df.isnull().any().any():
                return df[df.columns].astype(int)
            else:
                return df
        else:
            self.LOG.error('Could not load labels since `self.labels_dirname` is not defined.')
            self.LOG.error('Make sure to properly fill `data -> original` portion when using Supervised LMI.')

    def load_knn_ground_truth(self):
        return load_json(self.knn_filename)

    def load_queries(self):
        return pd.read_csv(self.queries_filename, header=None).values.flatten()

    def load_mtree_pivots(self):
        assert isfile(self.pivots_filename), f"'{self.pivots_filename}' was not found."

        struct_df = pd.read_csv(
            self.pivots_filename,
            sep='\s',  # noqa: W605
            header=None,
            index_col=0,
            names=['node', 'radius'],
            dtype={'node': str, 'radius': np.float64}
        )
        struct_df['node'] = \
            struct_df['node'].apply(lambda n: tuple([int(v) for v in n.split('.')]))
        struct_df['level'] = struct_df['node'].apply(lambda n: len(n))

        return struct_df

    def load_mindex_pivots(self) -> pd.DataFrame:
        return self.load_descriptors(self.pivots_filename)
