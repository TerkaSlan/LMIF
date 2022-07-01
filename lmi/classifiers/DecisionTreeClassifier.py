
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from typing import List


class LMIDecisionTreeClassifier(DecisionTreeClassifier):
    """

    DecisionTreeClassifier
    Attributes
    -------
    node_id : Tuple[int]
        Identifier of the node that the classifier represents
    node_classes_ : List[Tuple[int]]
        Identifiers of the created child nodes
    """
    def __init__(self, node_id, **kwargs):
        super().__init__()
        self.node_id = node_id
        self.node_classes_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> List:
        """
        Trains the classifier, populates node_classes_.

        Parameters
        -------
        X: np.ndarray (2D)
            Training values (descriptors)
        y: np.ndarray (1D)
            Training labels

        Returns
        -------
        predictions : List
            List of model predictions
        """
        if isinstance(y[0], list) or isinstance(y[0], np.ndarray):
            y = [0 for _ in range(X.shape[0])]
        elif y is None or np.isnan(y[0]):
            y = np.array([0])

        super().fit(X, y)
        self.node_classes_ = [self.node_id + (node, ) for node in self.classes_]
        return self.predict_classes(X)

    def predict_query(self, query: np.ndarray) -> np.ndarray:
        """
        Collects predictions (= discrete probability distribution over classes / child nodes)
        for a query (a single object / training example).

        Parameters
        -------
        query: np.ndarray (1D)
            Descriptor for one object from the dataset.

        Returns
        -------
        [node identifiers, model predictions] : np.ndarray (2D)
            Array of node identifiers and model predictions
        """
        assert query.shape[0] == 1
        predictions = self.predict_proba(query)[0]
        assert len(self.node_classes_) == predictions.shape[0]
        return np.array((self.node_classes_, predictions), dtype=object).T

    def predict_classes(self, data: np.ndarray) -> np.ndarray:
        """
        Given a trained classifier, predicts classes for `data`.

        Parameters
        -------
        data: np.ndarray (2D)
            Descriptors for objects from the dataset.

        Returns
        -------
        np.ndarray (2D)
            Array of predicted classes
        """
        if data.shape[0] > 0:
            return super().predict(data)
        else:
            return None
