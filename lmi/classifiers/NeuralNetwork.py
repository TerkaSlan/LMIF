import numpy as np
from tensorflow.keras import layers, models, optimizers
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
import logging
from lmi.utils import get_logger_config, one_hot_frequency_encode, encode_input_labels
import os
import tensorflow as tf
from typing import Dict
from tensorflow.keras.regularizers import l2


def limit_tf_cpu_usage():
    """ Limits CPU usage to permitted value set by
        an environment variable `NCPUS`.

        Returns
        ----------
        cpu_count : str
            Number of CPUs to be used.
    """
    cpu_count = os.environ['NCPUS'] if 'NCPUS' in os.environ else 1
    try:
        tf.config.threading.set_inter_op_parallelism_threads(int(cpu_count))
        tf.config.threading.set_intra_op_parallelism_threads(int(cpu_count))
    except AttributeError:
        # In case of TF version being <2, ignore the cpu_count setting
        pass
    os.environ['OMP_NUM_THREADS'] = cpu_count
    os.environ['TF_NUM_INTRAOP_THREADS'] = cpu_count
    os.environ['TF_NUM_INTEROP_THREADS'] = cpu_count
    return cpu_count


class LMINeuralNetwork:

    def __init__(
        self,
        node_id,
        **kwargs
    ):
        logging.basicConfig(level=logging.INFO, format=get_logger_config())
        self.LOG = logging.getLogger(__name__)
        self.node_id = node_id
        self.model = models.Sequential()
        self.encoder = preprocessing.LabelEncoder()
        self.build_layers_info = kwargs['hidden_layers']['dense']

    def set_is_gpu(self, is_gpu):
        if not is_gpu and 'NCPUS' in os.environ:
            limit_tf_cpu_usage()

    def build_network(self, X):
        input_dim = X.shape[1]
        for dense_info in self.build_layers_info:
            self.add_dense_layer(
                units=dense_info['units'],
                activation=dense_info['activation'],
                dropout=dense_info['dropout'],
                input_dim=input_dim,
                regularizer=True if 'regularizer' in dense_info and dense_info['regularizer'] else False
            )
            input_dim = None

    def add_dense_layer(self, units: int, activation='relu', dropout=None, input_dim=None, regularizer=False):
        """
        Adds a Dense layer to the model, potentially followed by a Dropout
        layer.

        Parameters
        ----------
        units : int
            Same as Dense's `units`.
        activation : str , optional
            Same as Dense's `activation`. Default 'relu'.
        dropout : float, optional
            Same as Dropout's `rate`, or None if no dropout is to be used.
            Default None.
        """
        if regularizer:
            reg = l2(0.0001)
            self.model.add(
                layers.Dense(
                    units,
                    activation=activation,
                    input_dim=input_dim,
                    kernel_regularizer=reg,
                    bias_regularizer=reg,
                    activity_regularizer=reg
                )
            )
        else:
            self.model.add(layers.Dense(units, activation=activation, input_dim=input_dim))
        if dropout is not None:
            self.model.add(layers.Dropout(dropout))

    def add_output_layer(self, y: np.ndarray, activation='softmax'):
        """
        Adds the last layer.

        Parameters
        ----------
        y : np.ndarray
            Labels
        activation : str , optional
            Same as Dense's `activation`. Default 'relu'.
        """
        self.add_dense_layer(np.unique(y).shape[0], activation)
        return y

    def compile(self, config: Dict):
        if config['optimizer'] == 'adam':
            opt = optimizers.Adam(learning_rate=config['learning_rate'])
        else:
            self.LOG.warn(f"Unknown optimizer: {config['optimizer']}")

        self.model.compile(loss=config['loss'], metrics=['accuracy'], optimizer=opt)

    def encode_input_labels(self, y: np.ndarray) -> np.ndarray:
        y, encoder = encode_input_labels(y, self.encoder)
        self.encoder = encoder
        return y

    def train(
        self,
        X: np.array,
        y: np.array,
        epochs: int,
        use_class_weights=True
    ):
        """ Trains a Neural Network. Expects model_dict to contain
        hyperparameters *opt* and *ep* (depth and number of estimators)

        Parameters
        -------
        rf_model: Dict
            Dictionary of model specification
        X: Numpy array
            Training values
        y: Numpy array
            Training labels

        Returns
        -------
        predictions: Numpy array
            Array of model predictions

        encoder: LabelEncoder
            Mapping of actual labels to labels used in training
        """
        y = self.encode_input_labels(y)

        self.node_classes_ = [self.node_id + (node, ) for node in self.encoder.classes_]

        if use_class_weights:
            class_weight = compute_class_weight(class_weight='balanced', classes=y, y=np.unique(y))
        else:
            class_weight = None

        self.model.fit(X, y, epochs=epochs, class_weight=class_weight, verbose=0)
        preds = [np.argmax(p) for p in self.model.predict(X)]
        preds = self.encoder.inverse_transform(preds)
        return preds

    def predict_query(self, query: np.ndarray) -> np.ndarray:
        """ Collects predictions for a query (= one object/data point).

        Parameters
        -------
        query: np.ndarray
            Query - expected shape: (1, n_descriptors)

        Returns
        -------
        np.array[classes, probabilities]
            2D Array of classes and their assigned proabilities.
        """
        assert query.shape[0] == 1
        prob_distr = self.predict(query)[0]
        while len(self.node_classes_) > prob_distr.shape[0]:
            prob_distr = np.append(prob_distr, [0])
        else:
            prob_distr = prob_distr[:len(self.node_classes_)]
        assert len(self.node_classes_) == prob_distr.shape[0]
        return np.array((self.node_classes_, prob_distr), dtype=object).T

    def predict(self, data):
        return self.model.predict(data)

    def predict_classes(self, data: np.ndarray):
        """ Given a trained classifier, predict's data classes.

        Parameters
        -------
        data : np.ndarray
            Data to be predicted
        """
        if data.shape[0] > 0:
            predictions = self.model.predict_classes(data)

            if np.max(predictions) > len(self.encoder.classes_)-1:
                predictions = list(predictions)
                idxs = [i for i, x in enumerate(predictions) if x > len(self.encoder.classes_)-1]
                for idx in idxs:
                    predictions[idx] = 0

            return self.encoder.inverse_transform(predictions)
        else:
            return None


class LMIMultilabelNeuralNetwork(LMINeuralNetwork):

    def __init__(
        self,
        node_id,
        **kwargs
    ):
        super().__init__(node_id, **kwargs)

    def add_output_layer(self, y: np.ndarray, activation='sigmoid'):
        """
        Adds the last layer. For multilabel training we're
        using sigmoid instead of softmax with one-hot encoded labels.

        Parameters
        ----------
        y : np.ndarray
            Labels
        activation : str , optional
            Same as Dense's `activation`. Default 'relu'.
        """
        self.labels_1d = np.concatenate(np.array([np.unique(v) for v in y], dtype=object), axis=0).astype(np.int64)
        self.y_classes = np.unique(self.labels_1d)
        self.encoder.fit(self.labels_1d)
        self.encoder_dict = dict(zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_)))
        y = self.encode_input_labels(y)
        self.add_dense_layer(y.shape[1], activation)
        return y

    def encode_input_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Encodes the input labels with one-hot frequency encoding.

        Parameters
        ----------
        y : np.ndarray
            Labels

        Returns
        -------
        np.ndarray
            One-hot frequency encoded labels
        """
        mapper_dict = dict(zip(self.labels_1d, self.encoder.transform(self.labels_1d)))

        def mp(entry):
            return mapper_dict[entry] if entry in mapper_dict else entry

        mp = np.vectorize(mp)
        encoded = [mp(v) for v in y]
        return one_hot_frequency_encode(encoded, len(self.encoder.classes_))

    def train(
        self,
        X: np.array,
        y: np.array,
        epochs: int,
        use_class_weights=False
    ):
        """ Trains a Neural Network. Expects model_dict to contain
        hyperparameters *opt* and *ep* (depth and number of estimators)

        Parameters
        -------
        rf_model: Dict
            Dictionary of model specification
        X: Numpy array
            Training values
        y: Numpy array
            Training labels

        Returns
        -------
        predictions: Numpy array
            Array of model predictions

        encoder: LabelEncoder
            Mapping of actual labels to labels used in training
        """
        self.node_classes_ = [self.node_id + (node, ) for node in self.encoder.classes_]

        if use_class_weights:
            class_weight = compute_class_weight(class_weight='balanced', classes=y, y=np.unique(y))
        else:
            class_weight = None
        # with tf.device('/CPU:0'):
        self.model.fit(X, y, epochs=epochs, class_weight=class_weight, verbose=0)
        predictions = [np.argmax(p) for p in self.model.predict(X)]

        if np.max(predictions) > len(self.encoder.classes_)-1:
            predictions = list(predictions)
            idxs = [i for i, x in enumerate(predictions) if x > len(self.encoder.classes_)-1]
            for idx in idxs:
                predictions[idx] = 0
        return self.encoder.inverse_transform(predictions)
