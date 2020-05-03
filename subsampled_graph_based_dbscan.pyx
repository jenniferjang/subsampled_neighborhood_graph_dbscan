import numpy as np
cimport numpy as np
from libcpp cimport bool
import random
import math
from sklearn.neighbors import KDTree
from datetime import datetime
from random import sample


cdef extern from "find_core_neighbors.h":
    void find_core_neighbors_cy(int c,
                                int * neighbors,
                                int * num_neighbors,
                                bool * is_core_point,
                                int * core_neighbors,
                                int * num_core_neighbors)

cdef find_core_neighbors_np(c,
                            np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
                            np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
                            np.ndarray[bool, ndim=1, mode="c"] is_core_pt,
                            np.ndarray[np.int32_t, ndim=1, mode="c"] core_neighbors,
                            np.ndarray[np.int32_t, ndim=1, mode="c"] num_core_neighbors):
    find_core_neighbors_cy(c,
                           <int *> np.PyArray_DATA(neighbors),
                           <int *> np.PyArray_DATA(num_neighbors),
                           <bool *> np.PyArray_DATA(is_core_pt),
                           <int *> np.PyArray_DATA(core_neighbors),
                           <int *> np.PyArray_DATA(num_core_neighbors))


cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int c, 
                   int n,
                   int * X_core,
                   int * neighbors,
                   int * num_neighbors,
                   int * result)

cdef DBSCAN_np(c, n,
               np.ndarray[np.int32_t, ndim=1, mode="c"] X_core,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(c, n,
              <int *> np.PyArray_DATA(X_core),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(num_neighbors),
              <int *> np.PyArray_DATA(result))


cdef extern from "cluster_remaining.h":
    void cluster_remaining_cy(int n,
                              int * neighbors,
                              float * distances,
                              int * num_neighbors,
                              bool * is_core_point,
                              int * result)

cdef cluster_remaining_np(n,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
                          np.ndarray[float, ndim=1, mode="c"] distances,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
                          np.ndarray[bool, ndim=1, mode="c"] is_core_pt,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(n,
                         <int *> np.PyArray_DATA(neighbors),
                         <float *> np.PyArray_DATA(distances),
                         <int *> np.PyArray_DATA(num_neighbors),
                         <bool *> np.PyArray_DATA(is_core_pt),
                         <int *> np.PyArray_DATA(result))


class SubsampledGraphBasedDBSCAN:
    """
    Parameters
    ----------
    
    p: The sample fraction, which determines m, the number of points to sample

    eps: Radius for determining neighbors and edges in the density graph

    minPts: Number of neighbors required for a point to be labeled a core point. Works
            in conjunction with eps_density

    """

    def __init__(self, p, eps, minPts):
        self.p = p
        self.eps = eps
        self.minPts = minPts

    def fit_predict(self, X, neighbors, distances, num_neighbors):
        """
        Determines the clusters in three steps.
        First step is to sample points from X using either the
        k-centers greedy sampling technique or a uniform
        sample technique. The next step is to run DBSCAN on the
        sampled points using the k-NN densities. Finally, all the 
        remaining points are clustered to the closest cluster.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
           Euclidean space

        Returns
        ----------
        (n, ) cluster labels
        """

        # X = np.ascontiguousarray(X)
        # n, d = X.shape

        n = num_neighbors.shape[0]
        e = neighbors.shape[0]

        # Find core points
        is_core_pt = np.where(num_neighbors >= self.minPts, 1, 0)
        core_pts = np.arange(n)[is_core_pt]
        c = core_pts.shape[0]

        # Get the core neighbors for each core point
        core_neighbors = np.full(e, -1, dtype=np.int32)
        num_core_neighbors = np.full(c, 0, dtype=np.int32)
        find_core_neighbors_np(c,
                               neighbors,
                               num_neighbors,
                               is_core_pt,
                               core_neighbors,
                               num_core_neighbors)
        
        # Cluster core points
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c,
                  n,
                  core_pts,
                  core_neighbors,
                  num_core_neighbors,
                  result)

        # Cluster the border points
        cluster_remaining_np(n,
                             neighbors,
                             distances, 
                             num_neighbors,
                             is_core_pt,
                             result)

        return result
