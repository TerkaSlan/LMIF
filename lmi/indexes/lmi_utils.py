import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any
Data = namedtuple('Data', 'X y shapes')


def get_model_name(model_name: any, level: int) -> str:
    """ Creates the model's name from int/tuple as a dot-separated string.

    Parameters
    -------
    model_name : int or Tuple
        A single integer or a sequence of integers (Tuple) identifying a node.
    level : int
        Current training level.
    Returns
    -------
    model_name: str
        Dot-separated string model name.
    """
    model_name_str = f'M.{level}.'
    model_name_str += '.'.join([str(x) for x in model_name])
    return model_name_str


def get_trainable_data(
    current_group: pd.DataFrame,
    is_supervised: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Creates a separation of the current dataset to a part that can be
    further trained (has a label at the corresponding level) and those that cannot.
    Is relevant when using the M-index labels (which can be NaN, since the M-index
    tree structure is unbalanced).

    Parameters
    -------
    current_group (pd.DataFrameGroupby):
        The whole group of data to be trained on.
    is_supervised : bool
        Indicator of whether the training is supervised or not

    Returns
    -------
    trainable, non_trainable group (Tuple[pd.GroupBy, pd.GroupBy])
    """
    if is_supervised:
        trainable = current_group.dropna()
        non_trainable = \
            current_group.loc[~current_group.index.isin(current_group.dropna().index)]
    else:
        trainable = current_group
        non_trainable = pd.DataFrame([])

    # If there are more non_trainable data, pre-prune the index
    # to avoid having tons of small buckets make all data non-traininable.
    if non_trainable.shape > trainable.shape:
        non_trainable = current_group
        trainable = pd.Series(dtype=object)

    return trainable, non_trainable


def get_trainable_labels(data_labels: pd.DataFrame, trainable: pd.DataFrame, is_supervised: bool) -> np.ndarray:
    """ Transorms the labels into an array of reasonable data type (int).
    In case of unsupervised training creates an array with NaNs.

    Parameters
    -------
    data_labels : pd.DataFrameGroupby
        Relevant data
    is_supervised : bool
        Indicator of whether the training is supervised or not

    Returns
    -------
    y: np.ndarray (1D)
        Array of labels
    """
    if is_supervised and isinstance(data_labels.iloc[0], list):
        y = np.array(data_labels.loc[trainable.index].values)
    elif is_supervised:
        y = np.array(data_labels.loc[trainable.index].values, dtype=np.uint16)
    else:
        y = np.empty((1, 1))
        y[:] = np.nan
    return y


def split_data(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_percentage: float,
    reindex=True
) -> Tuple[Data, Data]:
    """ Splits the data into training and prediction.
    In the conventional LMI setting, the whole dataset is used for training,
    however, to support shorter training times and smaller memory requirements,
    the structure can be built with smaller subset as well. The rest of the dataset
    is used just to predict the target node placement.

    Parameters
    -------
    df : pd.DataFrame
        Dataset descriptors
    labels_df : pd.DataFrame
        Dataset labels
    train_percentage : float
        Percentage of the dataset to be used for training

    Returns
    -------
    Tuple[Data, Data]
        Split dataset into training and prediction
    """
    if reindex:
        labels_df = labels_df.reindex(index=df.index)
    if train_percentage != 1:
        pred_data, train_data, pred_labels, train_labels = train_test_split(
            df, labels_df, test_size=train_percentage, random_state=10
        )
        train_data_labels = Data(train_data, train_labels, train_data.shape)
        pred_data_labels = Data(pred_data, pred_labels, pred_data.shape)
        if reindex:
            assert all(pred_data.index == pred_labels.index)
            assert all(train_data.index == train_labels.index)
    else:
        train_data_labels = Data(df, labels_df, df.shape)
        pred_data_labels = None
        if reindex:
            assert all(df.index == labels_df.index)
    return train_data_labels, pred_data_labels


def create_fake_labels(df: pd.DataFrame, n_levels: int) -> pd.DataFrame:
    """ Used in unsupervised training. Creates DataFrame of `n_levels`
    columns of NaNs. Is used to store the class assignments.

    Parameters
    -------
    df: pd.DataFrame
        Dataset descriptors
    n_levels: int
        Number of levels in LMI

    Returns
    -------
    pd.DataFrame
        Fake labels DataFrame
    """
    return pd.DataFrame(
        columns=[f'L{i}' for i in range(1, n_levels+1)],
        index=df.index
    )


def create_pred_df(
    predictions: List[int],
    ids: List[Any],
    new_index: pd.Series,
    level_label: str
) -> pd.DataFrame:
    """ Creates prediction DataFrame from collected predictions.

    Parameters
    -------
    predictions : List[int]
        List of predicted categories
    ids : List[Any]
        List of object IDs
    new_index : pd.Series
        Series of new index for result to be reindexed with
    level_label : str
        Current label

    Returns
    -------
    pd.DataFrame
        Fake labels DataFrame
    """
    pred_df = pd.DataFrame(
        predictions,
        index=ids,
        columns=[f'{level_label}_pred']
    )
    return pred_df.reindex(new_index)[f'{level_label}_pred'].astype(pd.UInt16Dtype())


def is_one_class(y: np.ndarray, is_supervised: bool) -> bool:
    """ Decides whether the training labels represent one class.

    Parameters
    -------
    y : np.ndarray (1D)
        Training labels
    is_supervised : bool
        Indicator of whether the training is supervised or not

    Returns
    -------
    bool
        Decision whether the training labels represent one class
    """
    if is_supervised:
        if not isinstance(y[0], np.ndarray) and not isinstance(y[0], list):
            if np.unique(y).shape[0] <= 1 and np.unique(y)[0] != np.nan:
                return True
    return False


def create_data_groups(level: int, dataset: Data, labels: List[str]) -> pd.DataFrame.groupby:
    """ Applies a groupby operation of the dataset based on
        collected probabilities from the previous level.

    Parameters
    -------
    level : int
        Target training level.

    Returns
    -------
    data_groups: pd.DataFrame.groupby
        GroupBy DataFrame based on the predictions.
    """
    group_condition = [f"{labels[j]}_pred" for j in range(level)]
    return dataset.y.groupby(group_condition)


def adjust_group_name(level: int, group_name: Any) -> Tuple:
    """ Transforms the first level group name to tuple.

    Parameters
    -------
    level : int
        Training level.
    group_name : Any
        Current group name - str or Tuple

    Returns
    -------
    group_name: Tuple
        group_name as Tuple
    """
    if level == 1:
        group_name = tuple((group_name,))
    return group_name
