import numpy as np
cimport numpy as np
from sklearn.neighbors import KDTree
from datetime import datetime


cdef extern from "subsampled_graph_based_dbscan_preallocated.h":
    void SubsampledGraphBasedDBSCAN_preallocated_cy(float p, 
                                                    int n,
                                                    int d, 
                                                    float eps, 
                                                    int minPts,
                                                    float * X,
                                                    int * result)

cdef SubsampledGraphBasedDBSCAN_preallocated_np(p, 
                                                n, 
                                                d,
                                                eps,
                                                minPts,
                                                np.ndarray[float, ndim=2, mode="c"] X,
                                                np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    SubsampledGraphBasedDBSCAN_preallocated_cy(p, 
                                               n,
                                               d,
                                               eps,
                                               minPts,
                                               <float *> np.PyArray_DATA(X),
                                               <int *> np.PyArray_DATA(result))


cdef extern from "subsampled_graph_based_dbscan.h":
    void SubsampledGraphBasedDBSCAN_cy(float p, 
                                       int n,
                                       int d, 
                                       float eps, 
                                       int minPts,
                                       float * X,
                                       int * result)

cdef SubsampledGraphBasedDBSCAN_np(p, 
                                   n, 
                                   d,
                                   eps,
                                   minPts,
                                   np.ndarray[float, ndim=2, mode="c"] X,
                                   np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    SubsampledGraphBasedDBSCAN_cy(p, 
                                  n,
                                  d,
                                  eps,
                                   minPts,
                                  <float *> np.PyArray_DATA(X),
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


    def fit_predict(self, X, preallocated=True):
        """

        Parameters 
        ----------
        

        Returns
        ----------
        (n, ) cluster labels
        """

        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        result = np.full(n, -1, dtype=np.int32)

        if preallocated:
          SubsampledGraphBasedDBSCAN_preallocated_np(self.p, n, d, self.eps, self.minPts, X, result)
        else:
          SubsampledGraphBasedDBSCAN_np(self.p, n, d, self.eps, self.minPts, X, result)

        return result
