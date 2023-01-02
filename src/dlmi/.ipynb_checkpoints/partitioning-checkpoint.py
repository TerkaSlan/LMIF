from sklearn.cluster import KMeans
import faiss
from typing import Tuple
import numpy as np
import kmedoids
from scipy.spatial import distance as dist


def cluster_kmeans(data, n_clusters=10) -> Tuple[KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans, kmeans.predict(data)


def cluster_kmeans_faiss(data, n_clusters=10) -> Tuple[KMeans]:
    kmeans = faiss.Kmeans(d=np.array(data).shape[1], k=n_clusters)
    X = np.array(data).astype(np.float32)
    kmeans.train(X)
    return kmeans, kmeans.index.search(X, 1)[1].T[0]


def cluster_kmedoids(data, n_clusters=10) -> Tuple:
    # https://stackoverflow.com/a/34738077
    def pairwise_angular(x, y):
        costheta = 1 - dist.cdist(x, y, 'cosine')
        return np.arccos(costheta)
    
    dists = pairwise_angular(data, data)
    fp = kmedoids.fasterpam(dists, n_clusters)
    return fp, np.array(fp.labels, dtype=np.int64)