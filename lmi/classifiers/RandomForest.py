
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import preprocessing
from lmi.utils import encode_input_labels
import os


class LMIRandomForest(RandomForestClassifier):

    def __init__(self, node_id, **kwargs):
        n_jobs = int(os.environ['NCPUS']) if 'NCPUS' in os.environ else 1
        super().__init__(n_jobs=n_jobs, **kwargs)
        self.node_id = node_id
        self.encoder = preprocessing.LabelEncoder()

    def encode_input_labels(self, y: np.ndarray) -> np.ndarray:
        y, encoder = encode_input_labels(y, self.encoder)
        self.encoder = encoder
        return y

    def fit(
        self,
        X: np.array,
        y: np.array
    ):
        """
        Creates and trains a RandomForest classifier.
        Expects model_config to contain hyperparameters *depth* and *n_estimators*
            == depth and number of estimators (trees)

        Parameters
        -------
        X: np.array (2D)
            Training values
        y: np.array (1D)
            Training labels

        Returns
        -------
        predictions : np.array
            Array of model predictions
        """
        y = self.encode_input_labels(y)
        super().fit(X, y)
        self.node_classes_ = [self.node_id + (node, ) for node in self.encoder.inverse_transform(self.classes_)]
        preds = self.predict(X)
        preds = self.encoder.inverse_transform(preds)
        return preds

    def predict_query(self, query: np.ndarray):
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
        predictions = super().predict_proba(query)[0]
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
