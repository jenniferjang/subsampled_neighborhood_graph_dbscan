from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import paired_distances
import numpy as np

def subsampled_neighbors(X, epsilon, s,
                         return_distances=False,
                         metric='euclidean'):
  n = X.shape[0]

  # Sample the neighbors. Does not ensure that vertices are neighbors of itself
  x = np.random.choice(np.arange(n), size=int(n * n * s), replace=True)
  y = np.random.choice(np.arange(n), size=int(n * n * s), replace=True)
  distances = paired_distances(X[x], X[y], metric=metric)

  neighbors_x = x[distances <= epsilon]
  neighbors_y = y[distances <= epsilon]

  neighborhood = csr_matrix((np.ones(len(neighbors_x)), (neighbors_x, neighbors_y)), shape=(n, n), dtype=bool)

  # Make the matrix symmetric
  neighborhood_t = neighborhood.transpose()
  neighborhood += neighborhood_t

  if return_distances:
    rows, cols = neighborhood.nonzero()
    neighb_dist = csr_matrix((paired_distances(X[rows], X[cols], metric=metric), (rows, cols)), shape=(n, n), dtype=np.float)
    return (neighborhood, neighb_dist)

  else:
    return neighborhood

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]])
neighbors, distances = subsampled_neighbors(X, 4.0, 1.0, return_distances=True)
print(neighbors.toarray())
print(distances.toarray())