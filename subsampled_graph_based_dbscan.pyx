import numpy as np
cimport numpy as np
from sklearn.neighbors import KDTree
from datetime import datetime


cdef extern from "construct_neighborhood_graph.h":
    void construct_neighborhood_graph_cy(float p, int pn, int n,
                                                                          int d, 
                                                                          float eps, 
                                                                          float * X,
                                                                          int * neighbors,
                                                                          int * num_neighbors)

cdef construct_neighborhood_graph_np(p, pn,
                                     n, 
                                     d,
                                     eps,
                                     np.ndarray[float, ndim=2, mode="c"] X,
                                     np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
                                     np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors):
    construct_neighborhood_graph_cy(p, pn, n,
                                           d,
                                           eps,
                                           <float *> np.PyArray_DATA(X),
                                           <int *> np.PyArray_DATA(neighbors),
                                           <int *> np.PyArray_DATA(num_neighbors))


cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int pn, int n, 
                   int * is_core_pt,
                   int * neighbors,
                   int * num_neighbors_cum,
                   int * result)

cdef DBSCAN_np(pn, n, 
               np.ndarray[np.int32_t, ndim=1, mode="c"] is_core_pt,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors_cum,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(pn, n,
              <int *> np.PyArray_DATA(is_core_pt),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(num_neighbors_cum),
              <int *> np.PyArray_DATA(result))


cdef extern from "cluster_remaining.h":
    void cluster_remaining_cy(int pn, int n,
                              int * neighbors,
                              int * num_neighbors_cum,
                              int * is_core_pt,
                              int * result)

cdef cluster_remaining_np(pn, n,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors_cum,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] is_core_pt,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(pn, n,
                         <int *> np.PyArray_DATA(neighbors),
                         <int *> np.PyArray_DATA(num_neighbors_cum),
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
        pn = int(max(1, self.p * n))
        
        # Construct the neighborhood graph
        neighbors = np.full(pn * n, -1, dtype=np.int32)
        num_neighbors = np.full(n, 0, dtype=np.int32)
        construct_neighborhood_graph_np(self.p, pn,
                                        n,
                                        d, 
                                        self.eps,
                                        X,
                                        neighbors,
                                        num_neighbors)

        #print list(neighbors), list(distances), list(num_neighbors)

        num_neighbors_cum = np.cumsum(num_neighbors, dtype=np.int32)

        # Find core points
        is_core_pt = (num_neighbors >= max(2, self.minPts * self.p)).astype(np.int32)

        #print list(is_core_pt)
        # Cluster core points
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(pn, n,
                  is_core_pt,
                  neighbors,
                  num_neighbors_cum,
                  result)

        # Cluster the border points
        cluster_remaining_np(pn, n,
                             neighbors,
                             num_neighbors_cum,
                             is_core_pt,
                             result)

        return result
