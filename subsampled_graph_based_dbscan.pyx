import numpy as np
cimport numpy as np


cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int c, 
                   int * is_core_pt,
                   int * neighbors,
                   int * num_neighbors_cum,
                   int * result)

cdef DBSCAN_np(c, 
               np.ndarray[np.int32_t, ndim=1, mode="c"] is_core_pt,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors_cum,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(c, 
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

    def fit_predict(self, neighbors, distances):
        """

        Parameters
        ----------
        

        Returns
        ----------
        (n, ) cluster labels
        """

        num_neighbors = np.ascontiguousarray([len(x) for x in neighbors])
        num_neighbors_cum = np.cumsum(num_neighbors, dtype=np.int32)

        neighbors = np.ascontiguousarray(np.concatenate(neighbors), dtype=np.int32)
        distances = np.ascontiguousarray(np.concatenate(distances), dtype=np.float32)
        
        n = num_neighbors.shape[0]

        # Find core points
        is_core_pt = (num_neighbors >= self.minPts).astype(np.int32)
        
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
