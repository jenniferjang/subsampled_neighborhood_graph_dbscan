import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.utility cimport pair


cdef extern from "construct_neighborhood_graph.h":
    cdef pair[vector[int], vector[float]] construct_neighborhood_graph_cy(int n,
                                                                          float p, 
                                                                          float * X,
                                                                          int * num_neighbors)

cdef construct_neighborhood_graph_np(n, 
                                     p,
                                     np.ndarray[float, ndim=2, mode="c"] X,
                                     np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors):
    return construct_neighborhood_graph_cy(n,
                                           p,
                                           <float *> np.PyArray_DATA(X),
                                           <int *> np.PyArray_DATA(num_neighbors))


cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int n, 
                   int * is_core_pt,
                   int * neighbors,
                   int * num_neighbors_cum,
                   int * result)

cdef DBSCAN_np(n, 
               np.ndarray[np.int32_t, ndim=1, mode="c"] is_core_pt,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors_cum,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(n,
              <int *> np.PyArray_DATA(is_core_pt),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(num_neighbors_cum),
              <int *> np.PyArray_DATA(result))


cdef extern from "cluster_remaining.h":
    void cluster_remaining_cy(int n,
                              int * neighbors,
                              int * num_neighbors_cum,
                              float * distances,
                              int * is_core_pt,
                              int * result)

cdef cluster_remaining_np(n,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors_cum,
                          np.ndarray[float, ndim=1, mode="c"] distances,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] is_core_pt,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(n,
                         <int *> np.PyArray_DATA(neighbors),
                         <int *> np.PyArray_DATA(num_neighbors_cum),
                         <float *> np.PyArray_DATA(distances),
                         <int *> np.PyArray_DATA(is_core_pt),
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

    def fit_predict(self, X):
        """

        Parameters
        ----------
        

        Returns
        ----------
        (n, ) cluster labels
        """

        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        
        # Construct the neighborhood graph
        num_neighbors = np.full(n, -1, dtype=np.int32)
        neighbors, distances = construct_neighborhood_graph_np(n,
                                                               self.p,
                                                               X,
                                                               num_neighbors)
        neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        distances = np.ascontiguousarray(distances, dtype=np.float32)
        
        # Find core points
        is_core_pt = (num_neighbors >= self.minPts * self.p).astype(np.int32)

        num_neighbors_cum = np.cumsum(num_neighbors, dtype=np.int32)
        
        # Cluster core points
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(n,
                  is_core_pt,
                  neighbors,
                  num_neighbors_cum,
                  result)

        
        # Cluster the border points
        cluster_remaining_np(n,
                             neighbors,
                             num_neighbors_cum, 
                             distances,
                             is_core_pt,
                             result)

        return result
