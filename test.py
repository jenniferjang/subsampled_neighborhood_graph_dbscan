from sklearn.metrics.pairwise import paired_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors._base import UnsupervisedMixin
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class SubsampledNeighborsTransformer(TransformerMixin, UnsupervisedMixin, 
                                     BaseEstimator):

  def __init__(self, s, metric='euclidean', random_state=None):
    self.s = s
    self.metric = metric
    self.random_state = random_state


  def _fit(self, X):
    if self.s <= 0 or self.s > 1:
      raise ValueError("Sampling rate needs to be in (0, 1]: %s" % self.s)

    self.s_ = self.s

    return self


  def transform(self, X):
    """Transform data into a subsampled graph of neighbors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample data.

    Returns
    -------
    neighborhood : sparse matrix of shape (n_samples, n_samples)
        Non-zero entries in neighborhood[i, j] indicate an edge 
        between X[i] and X[j] with value equal to weight of edge.
        The matrix is of CSR format.
    """

    check_is_fitted(self)

    return self.subsampled_neighbors(X, self.s_, self.metric, self.random_state)


  def fit_transform(self, X, y=None):
    """Fit to data, then transform it.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample data.
    y : ignored

    Returns
    -------
    neighborhood : sparse matrix of shape (n_samples, n_samples)
        Non-zero entries in neighborhood[i, j] indicate an edge 
        between X[i] and X[j] with value equal to weight of edge.
        The matrix is of CSR format.
    """
    
    return self.fit(X).transform(X)


  def subsampled_neighbors(self, X, s, metric='euclidean', random_state=None):
    """Compute the subsampled graph of neighbors for points in X.

    Parameters
    ----------
    X : array-like of shape (n, n_features)
        Sample data.

    s : float
        Sampling probability.

    metric : string or callable, default='euclidean'
        Input to paired_distances function. Can be string specified 
        in PAIRED_DISTANCES, including "euclidean", "manhattan", or 
        "cosine." Alternatively, can be a callable function, which should 
        take two arrays from X as input and return a value indicating 
        the distance between them.

    random_state : int, RandomState instance, default=None
        Seeds the random sampling of lists of vertices. Use an int to 
        make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    neighborhood : sparse matrix of shape (n_samples, n_samples)
        Non-zero entries in neighborhood[i, j] indicate an edge 
        between X[i] and X[j] with value equal to weight of edge.
        The matrix is of CSR format.

    References
    ----------
    - Faster DBSCAN via subsampled similarity queries, 2020
      Heinrich Jiang, Jennifer Jang, Jakub Łącki
      https://arxiv.org/abs/2006.06743

    Notes
    -----
    Each edge in the fully connected graph of X is sampled with probability s
    with replacement. We sample two arrays of n_samples * n_samples * s vertices 
    from X with replacement. Since (i, j) is equivalent to (j, i), we discard any 
    pairs where j >= i. We ensure symmetry by adding the neighborhood matrix to its 
    transpose. 
    """

    from scipy.sparse import csr_matrix

    random_state = check_random_state(random_state)

    n_samples = X.shape[0]

    # Sample the neighbors with replacement
    x = random_state.choice(np.arange(n_samples), size=int(n_samples * n_samples * s), 
      replace=True)
    y = random_state.choice(np.arange(n_samples), size=int(n_samples * n_samples * s), 
      replace=True)

    # Remove duplicates
    neighbors = np.unique(np.column_stack((x, y)), axis=0)

    # Upper triangularize the matrix
    neighbors = neighbors[neighbors[:, 0] < neighbors[:, 1]]

    i = neighbors[:, 0]
    j = neighbors[:, 1]

    # Compute the edge weights for the remaining edges
    if len(neighbors) > 0:
      distances = paired_distances(X[i], X[j], metric=metric)
    else:
      distances = []

    # Create the distance matrix in CSR format 
    neighborhood = csr_matrix((distances, (i, j)), shape=(n_samples, n_samples), 
      dtype=np.float)
    
    # Make the matrix symmetric
    neighborhood += neighborhood.transpose()

    return neighborhood

X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8]])

N = SubsampledNeighborsTransformer(1.0)
neighbors = N.fit(X).transform(X)
print(neighbors.toarray())

N = SubsampledNeighborsTransformer(0.1)
neighbors = N.fit(X).transform(X)
print(neighbors.toarray())