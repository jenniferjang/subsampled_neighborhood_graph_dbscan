from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import paired_distances, euclidean_distances
import numpy as np

def subsampled_neighbors(X, s, metric='euclidean'):
  n = X.shape[0]

  # Sample the neighbors
  # Because we're sampling with replacement, we don't really get enough points
  x = np.random.choice(np.arange(n), size=int(n * n * s), replace=True)
  y = np.random.choice(np.arange(n), size=int(n * n * s), replace=True)

  # Remove duplicates
  neighbors = np.unique(np.column_stack((x, y)), axis=0)

  # Upper triangularize the matrix
  neighbors = neighbors[neighbors[:, 0] < neighbors[:, 1]]

  rows = neighbors[:, 0]
  columns = neighbors[:, 1]

  distances = paired_distances(X[rows], X[columns], metric=metric)
  neighborhood = csr_matrix((distances, (rows, columns)), shape=(n, n), dtype=np.float)
  
  # Make the matrix symmetric
  neighborhood_t = neighborhood.transpose()
  neighborhood += neighborhood_t

  return neighborhood

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]])

neighbors = subsampled_neighbors(X, 1.0)
print(neighbors.toarray())

neighbors = subsampled_neighbors(X, 0.5)
print(neighbors.toarray())