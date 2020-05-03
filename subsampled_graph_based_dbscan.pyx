import numpy as np
cimport numpy as np
from libcpp cimport bool
import random
import math
from sklearn.neighbors import KDTree
from datetime import datetime
from random import sample

cdef extern from "find_core_pts.h":
    void find_core_pts_cy(int n,
                          int minPts,
                          int * num_neighbors,
                          bool * is_core_pt)

cdef find_core_pts_np(n,
                      minPts, 
                      np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
                      np.ndarray[bool, ndim=1, mode="c"] is_core_pt):
    find_core_pts_cy(n,
                     minPts,
                     <int *> np.PyArray_DATA(num_neighbors),
                     <bool *> np.PyArray_DATA(is_core_pt))


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


cdef extern from "find_closest_core_pt.h":
    void find_closest_core_pt_cy(int n,
                                 int d,
                                 float * X,
                                 int * neighbors,
                                 int * num_neighbors,
                                 bool * is_core_point,
                                 int * closest_core_pt,
                                 float * dist_sq_to_core_pt)

cdef find_closest_core_pt_np(n,
                             d,
                             np.ndarray[np.float, ndim=1, mode="c"] X,
                             np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
                             np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
                             np.ndarray[bool, ndim=1, mode="c"] is_core_pt,
                             np.ndarray[np.int32_t, ndim=1, mode="c"] closest_core_pt,
                             np.ndarray[np.float, ndim=1, mode="c"] dist_sq_to_core_pt):
    find_closest_core_pt_cy(n,
                            d,
                            <float *> np.PyArray_DATA(X),
                            <int *> np.PyArray_DATA(neighbors),
                            <int *> np.PyArray_DATA(num_neighbors),
                            <bool *> np.PyArray_DATA(is_core_pt),
                            <int *> np.PyArray_DATA(closest_core_pt),
                            <float *> np.PyArray_DATA(dist_sq_to_core_pt))


cdef extern from "cluster_remaining.h":
    void cluster_remaining_cy(int n,
                              int * closest_point,
                              int * result)

cdef cluster_remaining_np(n, 
                          np.ndarray[int, ndim=1, mode="c"] closest_point,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(n,
                         <int *> np.PyArray_DATA(closest_point),
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

    def fit_predict(self, X, neighbors, num_neighbors, cluster_border=True):
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
        init: String. Either "k-center" for the K-center greedy
              sampling technique or "uniform" for a uniform random
              sampling technique
        cluster_outliers: Boolean. Whether we should cluster the 
              remaining points

        Returns
        ----------
        (n, ) cluster labels
        """

        X = np.ascontiguousarray(X)
        n, d = X.shape

        # Find the core points
        # X_tree = KDTree(X)
        # neighbors, radii = X_tree.query_radius(X, r=self.eps, return_distance=True, sort_results=True)
        # X_core_ind = []
        # for i in range(len(neighbors)):
        #   neighbors[i] = sample(neighbors[i], int(len(neighbors[i]) * self.p))
        #   if len(neighbors[i]) >= self.minPts * self.p:
        #     X_core_ind.append(i)

        is_core_pt = np.full(n, 0, dtype=np.bool)
        find_core_pts_np(n,
                         self.minPts,
                         num_neighbors,
                         is_core_pt)

        core_pts = np.arange(n)[is_core_pt]
        c = core_pts.shape[0]

        core_neighbors = np.full(c * c, -1, dtype=np.int32)
        num_core_neighbors = np.full(c, 0, dtype=np.bool)
        find_core_neighbors_np(c,
                               neighbors,
                               num_neighbors,
                               is_core_pt,
                               core_neighbors,
                               num_core_neighbors)
        
        # Cluster the core points
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c,
                  n,
                  core_pts,
                  core_neighbors,
                  num_core_neighbors,
                  result)

        # Find the closest core point to every data point
        closest_core_pt = np.full(n, 0, dtype=np.int32)
        dist_sq_to_core_pt = np.full(n, 0, dtype=np.float32)

        find_closest_core_pt_np(n,
                                d,
                                X,
                                neighbors,
                                num_neighbors,
                                is_core_pt,
                                closest_core_pt,
                                dist_sq_to_core_pt)

        # Cluster the remaining points
        cluster_remaining_np(n, 
                             closest_core_pt, 
                             result)
        
        # Cluster border points
        if not cluster_border:
          result[dist_sq_to_core_pt[:,0] > self.eps] = -1

        return result
