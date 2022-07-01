
from sklearn.cluster import KMeans
import numpy as np
from lmi.classifiers.DummyClassifier import LMIDummyClassifier
import faiss
from typing import Any, Dict, Tuple, List


def train_kmeans_classifier(
    classifier: Any,
    params: Dict,
    X: np.ndarray,
    node_id: Tuple[int]
) -> Tuple[Any, List[int]]:
    """
    Decides which classifier to train in relation to `n_clusters`.
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
    if 'n_clusters' in params:
        clusters = params['n_clusters']
    elif 'k' in params:
        clusters = params['k']
    else:
        raise KeyError(
            f'The \'number of clusters clusters\' parameter in {params} was not found.'
            'Permitted keys: \'n_clusters\' and \'k\'.'
        )
    if X.shape[0] <= clusters:
        clusters = classifier.decrease_n_components(
            X.shape[0]
        )

    if clusters <= 1:
        classifier = LMIDummyClassifier(node_id)
        y = [0 for _ in range(X.shape[0])]
        return classifier, classifier.fit(X, y)
    else:
        return classifier, classifier.fit(X)


class LMIFaissKMeans(faiss.Kmeans):

    def __init__(self, node_id, **kwargs):
        super().__init__(**kwargs)
        self.node_id = node_id
        self.classes_ = [component for component in range(kwargs['k'])]

    def decrease_n_components(self, n_objects: int) -> int:
        """ Clustering algorithms run into problems when number of objects
        to train with is greater than number of components (= output classes).
        If this happens, this method decreases `k` to half of `n_objects`.

        Parameters
        -------
        n_objects: int
            Number of objects to train on

        Returns
        -------
        int
            Decreased number of components
        """
        decreased_value = n_objects // 2
        self.k = decreased_value
        self.classes_ = [component for component in range(self.k)]
        return decreased_value

    def fit(
        self,
        X: np.ndarray
    ) -> List[int]:
        """
        Trains a Faiss K-Means classifier.

        Parameters
        -------
        X: np contiguous array array (2D)
            Training values

        Returns
        -------
        predictions : np.array
            Array of model predictions
        """
        X = X.astype(np.float32)
        super().train(X)
        self.node_classes_ = [self.node_id + (node, ) for node in self.classes_]
        return self.index.search(X, 1)[1].T[0]

    def predict_query(self, query: np.ndarray) -> np.ndarray:
        """ Collects predictions for a query (= one object/data point).

        Parameters
        -------
        query: np.ndarray (1D)
            Input object

        Returns
        -------
        np.array[classes, probabilities]
            2D Array of classes and their assigned proabilities.
        """
        assert query.shape[0] == 1
        dist_distr = self.predict(query)
        assert len(self.node_classes_) == dist_distr.shape[0]
        return np.array((self.node_classes_, dist_distr), dtype=object).T

    def predict(self, data: np.ndarray) -> np.ndarray:
        """ Normalizes the predicted distances to [0,1] range
        to simulate a probability distribution.

        Parameters
        -------
        data : np.ndarray
            Data to be predicted

        Returns
        -------
        np.ndarray
            Normalized predictions
        """
        data = data.astype(np.float32)
        dist_distr = self.index.search(data, data.shape[0])
        dist_distr = max(dist_distr) - dist_distr
        dist_distr += 1e-8
        dist_distr /= sum(dist_distr)
        return dist_distr


class LMIKMeans(KMeans):

    def __init__(self, node_id, **kwargs):
        super().__init__(**kwargs)
        self.node_id = node_id
        self.classes_ = [component for component in range(kwargs['n_clusters'])]

    def decrease_n_components(self, n_objects):
        """ Clustering algorithms run into problems when number of objects
        to train with is greater than number of components (= output classes).
        If this happens, this method decreases `k` to half of `n_objects`.

        Parameters
        -------
        n_objects: int
            Number of objects to train on

        Returns
        -------
        int
            Decreased number of components
        """
        decreased_value = n_objects // 2
        self.n_clusters = decreased_value
        self.classes_ = [component for component in range(self.n_clusters)]
        return decreased_value

    def fit(
        self,
        X: np.array
    ):
        """
        Trains a K-Means classifier.

        Parameters
        -------
        X: np.array (2D)
            Training values

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
        np.ndarray[classes, probabilities]
            2D Array of classes and their assigned proabilities.
        """
        assert query.shape[0] == 1
        dist_distr = self.predict(query)
        assert len(self.node_classes_) == dist_distr.shape[0]
        return np.array((self.node_classes_, dist_distr), dtype=object).T

    def predict(self, data):
        """ Normalizes the predicted distances to [0,1] range
        to simulate a probability distribution.

        Parameters
        -------
        data : np.ndarray
            Data to be predicted

        Returns
        -------
        np.ndarray
            Normalized predictions
        """
        dist_distr = self.transform(data)[0]
        dist_distr = max(dist_distr) - dist_distr
        dist_distr += 1e-8
        dist_distr /= sum(dist_distr)
        return dist_distr

    def predict_classes(self, data: np.ndarray):
        """ Given a trained classifier, predict's data classes.

        Parameters
        -------
        data : np.ndarray
            Data to be predicted
        """
        if data.shape[0] > 0:
            return super().predict(data)
        else:
            return None
