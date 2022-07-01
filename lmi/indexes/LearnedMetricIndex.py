import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from lmi.classifiers.enums import Classifiers
from lmi.classifiers.GMM import train_gmm_classifier
from lmi.classifiers.KMeans import train_kmeans_classifier
from lmi.utils import remove_key
from lmi.indexes.BaseIndex import BaseIndex
from lmi.indexes.lmi_utils import get_trainable_data, get_trainable_labels, get_model_name,\
                      create_fake_labels, create_pred_df, is_one_class,\
                      create_data_groups, adjust_group_name, split_data, Data
# To avoid 'SettingWithCopy' warning
pd.options.mode.chained_assignment = None


class LMI(BaseIndex):
    """
    Learned Metric Index -- architecture containing series of interconnected models

    This class is responsible for two basic operations:
    1. Building -- Training an series of interconnected models (classifiers) -- tree nodes,
                   keeping track of each training object's placing within the tree.
        > Types of building:
            - **Supervised** / **Unsupervised** (see `self.supervised`)
                - is `labels_df` present? If not, create a fake one (filled with NaNs)
                  and use unsupervised classifiers to predict the labels (put to `labels_df`
                  in '_pred' columns)
            - **Full dataset training** / **Subset training** (`self.full_dataset_train`)
                - if 'training-dataset-percentage' is set to value < 1 (and > 0), training is
                  done with a random subset of the passed dataset of the specified size
                  (`self.train_data`). `self.pred_data` is used just to collect the predictions.
    2. Searching -- Navitaging through the structure by classifier's probability distribution,
                    collecting the answer.
        > Types of searching:
            - **Query from dataset** / **Query out of dataset**
                - TBD

    Attributes
    -------
    model_stack : List[List[Classifier]]
        Structure storing the trained classifiers -- the built LMI

    model_mapping : Dict[List[int], int]
        Mapping between the nodes and their classifier representation
        stored in `model_stack`.

    objects_in_buckets : Dict[Tuple[int], int]
        Mapping between identifier of buckets
        (= nodes in the lower-most level of the tree)
        and number of objects that each of these buckets contains.
    """
    def __init__(self, lmi_config: Dict, df: pd.DataFrame, labels_df=None, reindex=True):
        if labels_df is None:
            labels_df = create_fake_labels(df, lmi_config['n_levels'])
            self.supervised = False
        else:
            self.supervised = True
        super().__init__(df, labels_df)
        self.train_data, self.pred_data = split_data(
            df,
            labels_df,
            lmi_config['training-dataset-percentage'],
            reindex
        )
        if self.pred_data is None:
            self.full_dataset_train = True
        else:
            self.full_dataset_train = False

        self.model_stack = []
        self.model_mapping = {}
        self.objects_in_buckets = {}

        gpu = 'CUDA_VISIBLE_DEVICES'
        self.is_gpu = True if gpu in os.environ and len(os.environ[gpu]) != 0 else False

    # --------------------------------------------------------------
    #                       INDEX BUILDING
    # --------------------------------------------------------------

    def train(self, model_config: Dict[str, Dict[str, str]], rebuild=False):
        if not rebuild and len(self.model_stack) != 0:
            self.LOG.warning('Cannot overwrite contents of model_stack.'
                             'Set rebuild=True to re-train the LMI again.')
            return

        self.model_stack = []
        self.model_mapping = {}

        self.LOG.info(
            'Training model M.0 (root) on dataset'
            f"{self.train_data.X.shape} with {model_config['level-0']}."
        )

        if self.full_dataset_train:
            self.train_full_dataset(model_config)
        else:
            self.train_subset(model_config)

        self.objects_in_buckets = self.get_object_count_in_buckets(self.pred_labels)
        self.LOG.info('Finished training the LMI.')

    def train_full_dataset(self, model_config: Dict[str, Dict[str, str]]):
        """ Trains (builds) the whole LMI.

        The bulding starts from the root node (== 0-th level)
        and continues sequentially towards the leaf models.
        During the training, the following attributes are modified:
            - new models are appended to `self.model_stack`, each
              item is composed of another list and corresponds to one training level.
            - after each level is trained, the predictions are collected and
              saved in `self.data.y`, which are then used to split data for training
              the following level
            - `self.model_mapping` saves the data group identification so that we can map
              it to the corresponding model in `self.model_stack`

        After the training the LMI structure is saves in `self.model_stack`, information
        about occupied buckets is in `self.objects_in_buckets`.

        Parameters
        -------
        model_config: Dict[Dict[str, str]]
            Dictionary of models specification in the form:
                {{'level-0' : {'model': 'RF', 'n_estimators': 200 ...} ...}
            Loaded from `config/model-*.yml`

        rebuild: boolean [Optional]
            Decides whether to rebuild the saved (already trained) LMI structure if not empty
        """

        # --------------------------------------------------------------
        #                  Training the root model
        # --------------------------------------------------------------
        self.train_root_node(model_config['level-0'])
        # --------------------------------------------------------------
        #                  Training the index levels
        # --------------------------------------------------------------
        self.labels = self.labels[:len(list(model_config.keys()))]

        for level, level_label in enumerate(self.labels[1:], start=1):
            self.LOG.info(f"Training level {level} with {model_config[f'level-{level}']}.")
            data_groups = create_data_groups(level, self.train_data, self.labels)

            level_object_ids = []
            level_predictions = []
            level_model_stack = []
            model_mapping_level = []

            for group_name, group in data_groups:
                group_name = adjust_group_name(level, group_name)

                trainable, not_trainable = get_trainable_data(group[level_label], self.supervised)
                X = self.train_data.X.loc[trainable.index].values
                y = get_trainable_labels(self.train_data.y[level_label], trainable, self.supervised)
                if X.shape[0] > 0:
                    self.LOG.debug(
                        f'    Training model {get_model_name(group_name, level)}'
                        f' on {X.shape} samples.'
                    )

                    predictions, model = self.train_node(
                        model_config[f'level-{level}'],
                        X,
                        y,
                        group_name
                    )
                    model_mapping_level.append(group_name)
                    level_model_stack.append(model)
                else:
                    predictions = []

                level_object_ids.extend(
                    trainable.index.to_list() + not_trainable.index.to_list()
                )
                level_predictions.extend(
                    list(predictions) + [np.nan for _ in range(not_trainable.shape[0])]
                )

                assert len(level_object_ids) == len(level_predictions)

            self.train_data.y[f'{level_label}_pred'] = create_pred_df(
                level_predictions,
                level_object_ids,
                self.train_data.y.index,
                level_label
            )

            self.model_mapping[level] = dict.fromkeys(model_mapping_level)
            self.model_mapping[level].update((k, i) for i, k in enumerate(self.model_mapping[level]))
            self.model_stack.append(level_model_stack)

        self.data = self.train_data

    def train_subset(self, model_config: Dict[str, Dict[str, str]]):
        # --------------------------------------------------------------
        #                  Training the root model
        # --------------------------------------------------------------
        self.train_root_node(model_config['level-0'])
        # --------------------------------------------------------------
        #                  Training the index levels
        # --------------------------------------------------------------

        for level, level_label in enumerate(self.labels[1:], start=1):
            self.LOG.info(f"Training level {level} with {model_config[f'level-{level}']}.")
            train_data_groups = create_data_groups(level, self.train_data, self.labels)
            pred_data_groups = create_data_groups(level, self.pred_data, self.labels)

            train_level_object_ids = []
            train_level_predictions = []
            pred_level_object_ids = []
            pred_level_predictions = []

            level_model_stack = []
            model_mapping_level = []

            for group_name, group in train_data_groups:

                try:
                    pred_group = pred_data_groups.get_group(group_name)
                    pred_group_missing = False
                except (KeyError, IndexError):
                    pred_group_missing = True

                group_name = adjust_group_name(level, group_name)

                trainable, not_trainable = get_trainable_data(group[level_label], self.supervised)
                X = self.train_data.X.loc[trainable.index].values
                y = get_trainable_labels(self.train_data.y[level_label], trainable, self.supervised)

                if X.shape[0] > 0:
                    self.LOG.debug(
                        f'    Training model {get_model_name(group_name, level)}'
                        f' on {X.shape} samples.'
                    )

                    train_predictions, model = self.train_node(
                        model_config[f'level-{level}'],
                        X,
                        y,
                        group_name
                    )
                    model_mapping_level.append(group_name)
                    level_model_stack.append(model)
                    if not pred_group_missing:
                        pred_predictions = model.predict_classes(
                            self.pred_data.X.loc[pred_group[level_label].index].values
                        )

                        pred_level_object_ids.extend(pred_group[level_label].index.to_list())
                        pred_level_predictions.extend(list(pred_predictions))
                        assert len(pred_level_object_ids) == len(pred_level_predictions)
                else:
                    pred_predictions = []
                    train_predictions = []

                train_level_object_ids.extend(
                    trainable.index.to_list() + not_trainable.index.to_list()
                )
                train_level_predictions.extend(
                    list(train_predictions) + [np.nan for _ in range(not_trainable.shape[0])]
                )
                assert len(train_level_object_ids) == len(train_level_predictions)

            self.train_data.y[f'{level_label}_pred'] = create_pred_df(
                train_level_predictions,
                train_level_object_ids,
                self.train_data.y.index,
                level_label
            )
            self.pred_data.y[f'{level_label}_pred'] = create_pred_df(
                pred_level_predictions,
                pred_level_object_ids,
                self.pred_data.y.index,
                level_label
            )
            self.model_mapping[level] = dict.fromkeys(model_mapping_level)
            self.model_mapping[level].update((k, i) for i, k in enumerate(self.model_mapping[level]))
            self.model_stack.append(level_model_stack)

        self.data = Data(
            pd.concat([self.train_data.X, self.pred_data.X]),
            pd.concat([self.train_data.y, self.pred_data.y]),
            (self.train_data.X.shape[0] + self.pred_data.X.shape[0], self.pred_data.X.shape[1])
        )

    def train_root_node(self, level):
        predictions, model = self.train_node(
            level,
            self.train_data.X.values,
            self.train_data.y[self.labels[0]].values,
            (())
        )
        self.train_data.y[f'{self.labels[0]}_pred'] = predictions
        if not self.full_dataset_train:
            self.pred_data.y[f'{self.labels[0]}_pred'] = model.predict_classes(self.pred_data.X.values)
        self.model_stack.append(model)

    def train_node(  # noqa: C901
        self,
        level_config: Dict[str, Dict[str, str]],
        X: np.ndarray,
        y: np.ndarray,
        node_id: Tuple[int]
    ) -> Tuple[List, Classifiers]:
        """ Trains a single node (= one classifier).

        1. Prepares the input data
        2. Instantiates the model to use for training
        3. Runs the training function
        4. Returns the trained model with predictions

        If the desired model is multilabel NN, target labels are composed of 30-NN as opposed to 1-NN.

        Parameters
        -------
        level_config: Dict[str, Dict[str,str]]
            Dictionary of models specification
        X: np.ndarray
            Training data
        y: np.ndarray
            Training labels, empty array in case of unuspervised learning
        node_id: Tuple[int]
            Current node identifier

        Returns
        -------
        predictions: np.ndarray
            Array of shape (n_objects x n_descriptor_values)
        classifier: Classifier
            Trained classifier representing the node

        """
        assert X.shape[1] == self.descriptor_values

        if X.shape[0] == 1 or is_one_class(y, self.supervised):
            self.LOG.debug(
                '        Only one data-point or one class to train with.'
                ' Using the DummyClassifier.'
            )
            if 'single-point-node' in level_config:
                classifier = Classifiers['DecisionTree'].value(node_id)
                params = remove_key(level_config, 'model')
            else:
                classifier = Classifiers['Dummy'].value(node_id)
        else:
            if 'model' in level_config:
                params = remove_key(level_config, 'model')
                is_hybrid = False
            else:
                assert 'cluster' in level_config and 'search' in level_config, \
                    f'Unrecognized format of model_config: {level_config}'
                is_hybrid = True

            if is_hybrid and level_config['cluster']['model'] == 'FaissKMeans':
                X = np.ascontiguousarray(X)
                params = remove_key(level_config['cluster'], 'model')
                params['d'] = X.shape[1]
                params['gpu'] = self.is_gpu

                classifier = Classifiers[level_config['cluster']['model']].value(node_id, **params)
            elif level_config['model'] == 'LogReg':
                if 'single-point-node' in params:
                    params = remove_key(params, 'single-point-node')
                if 'class_weights' not in params:
                    params['class_weights'] = False
                classifier = Classifiers[level_config['model']].value(node_id, y, **params)
            else:
                classifier = Classifiers[level_config['model']].value(node_id, **params)

        predictions = []
        if type(classifier).__name__ == 'LMINeuralNetwork':
            classifier.set_is_gpu(self.is_gpu)
            classifier.build_network(X)
            y = classifier.add_output_layer(y, activation='softmax')
            classifier.compile(params)
            predictions = classifier.train(X, y, params['epochs'])

        elif type(classifier).__name__ == 'LMIMultilabelNeuralNetwork':
            classifier.set_is_gpu(self.is_gpu)
            classifier.build_network(X)
            y = classifier.add_output_layer(y, activation='sigmoid')
            classifier.compile(params)
            predictions = classifier.train(X, y, params['epochs'], use_class_weights=False)

        elif type(classifier).__name__ in ('LMIGaussianMixtureModel', 'LMIBayesianGaussianMixtureModel'):
            classifier, predictions = train_gmm_classifier(classifier, params, X, node_id)

        elif type(classifier).__name__ == 'LMIKMeans':
            classifier, predictions = train_kmeans_classifier(classifier, params, X, node_id)

        elif type(classifier).__name__ == 'LMIFaissKMeans':
            classifier, predictions = train_kmeans_classifier(classifier, params, X, node_id)
            if type(classifier).__name__ != 'LMIDummyClassifier':
                params = remove_key(level_config['search'], 'model')
                if np.unique(predictions).shape[0] <= 1 and np.unique(predictions)[0] != np.nan:
                    self.LOG.debug('Only one class to train with, using the DummyClassifier.')
                    classifier = Classifiers['Dummy'].value(node_id)
                    predictions = [predictions]
                else:
                    if 'class_weights' not in params:
                        params['class_weights'] = False
                    if 'single-point-node' in params:
                        params = remove_key(params, 'single-point-node')
                    classifier = Classifiers[level_config['search']['model']].value(node_id, y, **params)
                predictions = classifier.fit(X, predictions)

        elif type(classifier).__name__ in (
            'LMIRandomForest',
            'LMILogisticRegression',
            'LMIDecisionTreeClassifier',
            'LMIDummyClassifier'
        ):
            predictions = classifier.fit(X, y)
        else:
            raise NotImplementedError(f'Classifier {classifier} is not found within the implemented classifiers.')
        return predictions, classifier

    # --------------------------------------------------------------
    #                           SEARCHING
    # --------------------------------------------------------------

    def get_probability_distribution(
        self,
        node_id: Tuple[int],
        query: np.ndarray,
        level: int
    ) -> np.ndarray:
        """ Collects probabilities returned by a classifier (= node) for a given query.

        Parameters
        -------
        node_id : Tuple[int]
            Identifier of the node searched
        query : np.ndarray (1D)
            Searched object

        Returns
        -------
        predictions : np.ndarray
            Discrete prob. distribution over the node children (= predicted classes).
        """

        if level == 0:
            model = self.model_stack[level]
            return model.predict_query(query)
        elif level <= len(self.model_mapping):
            model_stack_index = self.model_mapping[level].get(node_id)
            if model_stack_index is not None:
                model = self.model_stack[level][model_stack_index]
                assert model.node_id == node_id, \
                    f"Searched node ID: {node_id} and model's node ID: {model.node_id} don't match."\
                    f"model_stack_index: {model_stack_index} | level: {level}."
                return model.predict_query(query)
        else:
            return []

    def search_node(
        self,
        priority_queue: np.ndarray,
        query: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int]]:
        """ Pop the top node from priority queue,
            collects prob. distribution for its children,
            adds to the priority queue and sorts it.

        Parameters
        -------
        priority_queue : List[Tuple[int]]
            Priority queue - structure deciding the next node to visit.
        query : np.ndarray
            Searched object

        Returns
        -------
        priority_queue : List[Tuple[int]]
            Modified Priority queue
        node_id : Tuple[int]
            Identifier of the popped node
        """

        popped_node, priority_queue = self.pop_node(priority_queue)
        node_label = self.get_node_label(popped_node)
        popped_node[0] = node_label
        level = 0 if node_label == -1 else len(node_label)
        self.LOG.debug(f'-> {node_label}, p={popped_node[1]} | PQ: {priority_queue[:3]}')

        if level != self.n_levels:
            prob_distr = self.get_probability_distribution(node_label, query, level)
            if prob_distr is [] or prob_distr is None:
                return priority_queue, popped_node
            priority_queue = np.append(prob_distr, priority_queue, axis=0)
        else:
            return priority_queue, popped_node

        return self.sort_priority_queue(priority_queue), popped_node
