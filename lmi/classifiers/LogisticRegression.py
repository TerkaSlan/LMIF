import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from warnings import simplefilter
from typing import List
from sklearn.exceptions import ConvergenceWarning
import os
from lmi.utils import encode_input_labels
simplefilter("ignore", category=ConvergenceWarning)
# Filter out convergence warning:
#   Since running LogReg on lots of data and lot of iterations
#   may be expensive, we allow the failure of convergence
#   and needn't notify about it in the logs.
simplefilter(action='ignore', category=FutureWarning)


def custom_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


class LMILogisticRegression(LogisticRegression):

    def __init__(
        self,
        node_id,
        y,
        max_iter,
        C,
        class_weights,
        **kwargs
    ):
        multi_class = 'ovr'
        n_jobs = int(os.environ['NCPUS']) if 'NCPUS' in os.environ else 1
        if class_weights:
            d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
        else:
            d_class_weights = 'balanced'

        super().__init__(
            max_iter=max_iter,
            C=C,
            class_weight=d_class_weights,
            multi_class=multi_class,
            n_jobs=n_jobs,
            **kwargs
        )
        self.encoder = preprocessing.LabelEncoder()
        self.node_id = node_id

    def encode_input_labels(self, y: np.ndarray) -> np.ndarray:
        y, encoder = encode_input_labels(y, self.encoder)
        self.encoder = encoder
        return y

    def fit(
        self,
        X: np.array,
        y: np.array
    ) -> List[int]:
        """ Trains a Logistic regression model.
        Expects model_dict to contain hyperparameter *ep* (number of epochs)

        Parameters
        -------
        X: Numpy array
            Training values
        y: Numpy array
            Training labels

        Returns
        -------
        predictions: Numpy array
            Array of model predictions
        """
        y = self.encode_input_labels(y)
        super().fit(X, y)
        self.node_classes_ = [self.node_id + (node, ) for node in self.encoder.classes_]
        preds = self.predict(X)
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
        np.ndarray[classes, probabilities]
            2D Array of classes and their assigned proabilities.
        """
        assert query.shape[0] == 1
        predictions = custom_softmax(np.dot(self.coef_, query[0]) + self.intercept_)
        while len(self.node_classes_) > predictions.shape[0]:
            predictions = np.append(predictions, [0])
        assert len(self.node_classes_) == predictions.shape[0]
        return np.array((self.node_classes_, predictions), dtype=object).T

    def predict_classes(self, data: np.ndarray):
        """ Given a trained classifier, predict's data classes.

        Parameters
        -------
        data : np.ndarray
            Data to be predicted
        """
        if data.shape[0] > 0:
            preds = self.predict(data)
            preds = self.encoder.inverse_transform(preds)
            return preds
        else:
            return None
