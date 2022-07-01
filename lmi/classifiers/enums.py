"""
Useful enum definitions realted to classifiers.
"""
from enum import Enum
from lmi.classifiers.RandomForest import LMIRandomForest
from lmi.classifiers.LogisticRegression import LMILogisticRegression
from lmi.classifiers.NeuralNetwork import LMINeuralNetwork, LMIMultilabelNeuralNetwork
from lmi.classifiers.GMM import LMIGaussianMixtureModel, LMIBayesianGaussianMixtureModel
from lmi.classifiers.KMeans import LMIKMeans, LMIFaissKMeans
from lmi.classifiers.DummyClassifier import LMIDummyClassifier
from lmi.classifiers.DecisionTreeClassifier import LMIDecisionTreeClassifier


class Classifiers(Enum):
    RF = LMIRandomForest
    LogReg = LMILogisticRegression
    NN = LMINeuralNetwork
    MultilabelNN = LMIMultilabelNeuralNetwork
    GMM = LMIGaussianMixtureModel
    BayesianGMM = LMIBayesianGaussianMixtureModel
    KMeans = LMIKMeans
    FaissKMeans = LMIFaissKMeans
    Dummy = LMIDummyClassifier
    DecisionTree = LMIDecisionTreeClassifier
