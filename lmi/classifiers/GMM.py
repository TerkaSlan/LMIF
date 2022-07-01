from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
from lmi.classifiers.DummyClassifier import LMIDummyClassifier
from typing import Any, Dict, Tuple, List


def train_gmm_classifier(
    classifier: Any,
    params: Dict,
    X: np.ndarray,
    node_id: Tuple[int]
) -> Tuple[Any, List[int]]:
    """
    Decides which classifier to train in relation to `n_components`.
    If n_components drops to 1 (because '1' was specified in the
    config or it came about as a result of `decrease_n_components`),
    there would be just 1 class on the output -- we use DummyClassifier
    in such case.

    Parameters
    -------
    classifier : Any
        GMM or BayesianGMM classifier instance
    params : Dict
        (Bay)GMM's **kwargs
    X : np.ndarray (2D)
        Dataset's descriptors
    node_id : int
        Node's identifier.

    Returns
    -------
    classifier, predictions : [Any, List[int]]
        Result classifier (GMMs or Dummy), its .fit() result -- class predictions
    """
    if X.shape[0] <= params['n_components']:
        params['n_components'] = classifier.decrease_n_components(
            X.shape[0]
        )

    if params['n_components'] <= 1:
        classifier = LMIDummyClassifier(node_id)
        y = [0 for _ in range(X.shape[0])]
        return classifier, classifier.fit(X, [y])
    else:
        return classifier, classifier.fit(X)


class LMIBayesianGaussianMixtureModel(BayesianGaussianMixture):

    def __init__(self, node_id, **kwargs):
        super().__init__(**kwargs)
        self.node_id = node_id
        self.classes_ = [component for component in range(kwargs['n_components'])]

    def decrease_n_components(self, n_objects: int) -> int:
        """ Clustering algorithms into problems when number of objects
        to train with is greater than number of components (= output classes).
        If this happens, this method decreases `n_components` to half of `n_objects`.

        Parameters
        -------
        n_objects : int
            Number of objects to train on

        Returns
        -------
        int
            Decreased number of components
        """
        decreased_value = n_objects // 2
        self.n_components = decreased_value
        self.classes_ = [component for component in range(self.n_components)]
        return decreased_value

    def fit(
        self,
        X: np.array
    ) -> List[int]:
        """
        Trains a Bayesian GMM classifier.

        Parameters
        -------
        X : np.ndarray (2D)
            Training data

        Returns
        -------
        predictions : np.array
            Array of model predictions
        """
        super().fit(X)
        self.node_classes_ = [self.node_id + (node, ) for node in self.classes_]
        return self.predict_classes(X)

    def predict_query(self, query: np.ndarray) -> np.ndarray:
        """ Collects predictions for a query (= one object/data point).

        Parameters
        -------
        query : np.ndarray
            Query - expected shape: (1, n_descriptors)

        Returns
        -------
        np.array[classes, probabilities]
            2D Array of classes and their assigned proabilities.
        """
        assert query.shape[0] == 1
        predictions = super().predict_proba(query)[0]
        assert len(self.node_classes_) == predictions.shape[0]
        return np.array((self.node_classes_, predictions), dtype=object).T

    def predict_classes(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts the classes from a trained Bayesian GMM classifier.

        Parameters
        -------
        data: np.ndarray (2D)
            Training data

        Returns
        -------
        predictions : np.array
            Array of model predictions
        """
        if data.shape[0] > 0:
            return super().predict(data)
        else:
            return None


class LMIGaussianMixtureModel(GaussianMixture):

    def __init__(self, node_id, **kwargs):
        super().__init__(**kwargs)
        self.node_id = node_id
        self.classes_ = [component for component in range(kwargs['n_components'])]

    def decrease_n_components(self, n_objects: int) -> int:
        """
        Decreases GMM's `n_components` attribute if it's > than `n_objects`

        Parameters
        -------
        n_objects : int
            Number of objects to be trained on.

        Returns
        -------
        decreased_value : int
            Decreased `n_components`.
        """
        decreased_value = n_objects // 2
        self.n_components = decreased_value
        self.classes_ = [component for component in range(self.n_components)]
        return decreased_value

    def fit(
        self,
        X: np.array
    ):
        """
        Creates and trains a GMM classifier.

        Parameters
        -------
        X: np.array (2D)
            Training data
        Returns
        -------
        predictions : np.array
            Array of model predictions
        """
        super().fit(X)
        self.node_classes_ = [self.node_id + (node, ) for node in self.classes_]
        return self.predict_classes(X)

    def predict_query(self, query: np.ndarray) -> np.ndarray:
        """ Collects predictions for a query (= one object/data point).

        Parameters
        -------
        query: np.ndarray (1D)
            Query - expected shape: (1, n_descriptors)

        Returns
        -------
        np.array[classes, probabilities]
            2D Array of classes and their assigned proabilities.
        """
        assert query.shape[0] == 1
        predictions = super().predict_proba(query)[0]
        assert len(self.node_classes_) == predictions.shape[0]
        return np.array((self.node_classes_, predictions), dtype=object).T

    def predict_classes(self, data: np.ndarray) -> np.ndarray:
        """
        Predicts the classes from a trained GMM classifier.

        Parameters
        -------
        data: np.ndarray (2D)
            Training data

        Returns
        -------
        predictions : np.array
            Array of model predictions
        """
        if data.shape[0] > 0:
            return super().predict(data)
        else:
            return None
