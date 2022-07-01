import numpy as np


def get_euclidean_distance(object_1, object_2):
    assert object_1.shape == object_2.shape
    return np.linalg.norm(object_1-object_2)
