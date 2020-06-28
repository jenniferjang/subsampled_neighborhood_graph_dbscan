from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import paired_distances
import numpy as np

def subsampled_neighbors(X, epsilon, s,
                         return_distances=False,
                         metric='euclidean'):
  n = X.shape[0]

  neighbors = np.array([], dtype=bool)
  neighborptr = np.array([0], dtype=np.int32)
  distances = np.array([], dtype=np.float)

  for i in range(X.shape[0]):
    # Sample the neighbors
    subsampled = np.append([i], np.random.choice(np.arange(i + 1, n), size=int((n - i -1) * s), replace=False))
    
    # This seems inefficient; I can just rewrite paired_distances to 
    # use only one reference point, but it will be more code
    dist = paired_distances([X[i]] * len(subsampled), X[subsampled], metric=metric)
    
    neighbors = np.append(neighbors, subsampled[dist <= epsilon])
    neighborptr = np.append(neighborptr, len(neighbors))

    if return_distances:
      distances = np.append(distances, dist[dist <= epsilon])

  neighborhood = csr_matrix((np.ones(len(neighbors)), neighbors, neighborptr), dtype=bool)

  # Make the matrix symmetric
  neighborhood_t = neighborhood.transpose()
  neighborhood += neighborhood_t

  if return_distances:
    neighborhood_distances = csr_matrix((distances, neighbors, neighborptr), dtype=np.float)

    # Make the matrix symmetric
    neighborhood_distances_t = neighborhood_distances.transpose()
    neighborhood_distances += neighborhood_distances_t
    
    return (neighborhood, neighborhood_distances)

  else:
    return neighborhood

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]])
neighbors, distances = subsampled_neighbors(X, 4.0, 1.0, return_distances=True)
print(neighbors.toarray())
print(distances.toarray())