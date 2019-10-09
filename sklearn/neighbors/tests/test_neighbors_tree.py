# License: BSD 3 clause

import pickle
import itertools

import numpy as np
import pytest

from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.neighbors.ball_tree import BallTree, kernel_norm
from sklearn.neighbors.kd_tree import KDTree

from sklearn.utils import check_random_state
from numpy.testing import (assert_array_almost_equal,
                           assert_allclose)

rng = np.random.RandomState(42)
V_mahalanobis = rng.rand(3, 3)
V_mahalanobis = np.dot(V_mahalanobis, V_mahalanobis.T)

DIMENSION = 3

METRICS = {'euclidean': {},
           'manhattan': {},
           'minkowski': dict(p=3),
           'chebyshev': {},
           'seuclidean': dict(V=rng.random_sample(DIMENSION)),
           'wminkowski': dict(p=3, w=rng.random_sample(DIMENSION)),
           'mahalanobis': dict(V=V_mahalanobis)}

KD_TREE_METRICS = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
BALL_TREE_METRICS = list(METRICS)


def dist_func(x1, x2, p):
    return np.sum((x1 - x2) ** p) ** (1. / p)


def brute_force_neighbors(X, Y, k, metric, **kwargs):
    D = DistanceMetric.get_metric(metric, **kwargs).pairwise(Y, X)
    ind = np.argsort(D, axis=1)[:, :k]
    dist = D[np.arange(Y.shape[0])[:, None], ind]
    return dist, ind


@pytest.mark.parametrize(
        'Cls, metric',
        itertools.chain(
            [(KDTree, metric) for metric in KD_TREE_METRICS],
            [(BallTree, metric) for metric in BALL_TREE_METRICS]))
@pytest.mark.parametrize('k', (1, 3, 5))
@pytest.mark.parametrize('dualtree', (True, False))
@pytest.mark.parametrize('breadth_first', (True, False))
def test_nn_tree_query(Cls, metric, k, dualtree, breadth_first):
    rng = check_random_state(0)
    X = rng.random_sample((40, DIMENSION))
    Y = rng.random_sample((10, DIMENSION))

    kwargs = METRICS[metric]

    kdt = Cls(X, leaf_size=1, metric=metric, **kwargs)
    dist1, ind1 = kdt.query(Y, k, dualtree=dualtree,
                            breadth_first=breadth_first)
    dist2, ind2 = brute_force_neighbors(X, Y, k, metric, **kwargs)

    # don't check indices here: if there are any duplicate distances,
    # the indices may not match.  Distances should not have this problem.
    assert_array_almost_equal(dist1, dist2)


@pytest.mark.parametrize(
        "Cls, metric",
        [(KDTree, 'euclidean'), (BallTree, 'euclidean'),
         (BallTree, dist_func)])
@pytest.mark.parametrize('protocol', (0, 1, 2))
def test_pickle(Cls, metric, protocol):
    rng = check_random_state(0)
    X = rng.random_sample((10, 3))

    if hasattr(metric, '__call__'):
        kwargs = {'p': 2}
    else:
        kwargs = {}

    tree1 = Cls(X, leaf_size=1, metric=metric, **kwargs)

    ind1, dist1 = tree1.query(X)

    s = pickle.dumps(tree1, protocol=protocol)
    tree2 = pickle.loads(s)

    ind2, dist2 = tree2.query(X)

    assert_array_almost_equal(ind1, ind2)
    assert_array_almost_equal(dist1, dist2)

    assert isinstance(tree2, Cls)


@pytest.mark.parametrize('Cls', [KDTree, BallTree])
@pytest.mark.parametrize('return_distance', [False, True])
def test_nn_tree_query_radius_distance(Cls, return_distance):
    n_samples, n_features = 100, 10
    rng = check_random_state(0)
    X = 2 * rng.random_sample(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1E-15  # roundoff error can cause test to fail
    bt = Cls(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        res = bt.query_radius([query_pt], r + eps,
                              return_distance=return_distance)
        if return_distance:
            assert isinstance(res, tuple)
            assert len(res) == 2
            ind, dist = res
            ind = ind[0]
            dist = dist[0]

            d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

            assert_array_almost_equal(d, dist)
        else:
            assert isinstance(res, np.ndarray)
            ind = res[0]
            i = np.where(rad <= r + eps)[0]

            ind.sort()
            i.sort()

            assert_array_almost_equal(i, ind)


def compute_kernel_slow(Y, X, kernel, h):
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    norm = kernel_norm(h, X.shape[1], kernel)

    if kernel == 'gaussian':
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == 'tophat':
        return norm * (d < h).sum(-1)
    elif kernel == 'epanechnikov':
        return norm * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == 'exponential':
        return norm * (np.exp(-d / h)).sum(-1)
    elif kernel == 'linear':
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == 'cosine':
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError('kernel not recognized')


@pytest.mark.parametrize('Cls', [KDTree, BallTree])
@pytest.mark.parametrize("kernel", ['gaussian', 'tophat', 'epanechnikov',
                                    'exponential', 'linear', 'cosine'])
@pytest.mark.parametrize("h", [0.01, 0.1, 1])
@pytest.mark.parametrize("rtol", [0, 1E-5])
@pytest.mark.parametrize("atol", [1E-6, 1E-2])
@pytest.mark.parametrize("breadth_first", [True, False])
def test_nn_tree_kde(Cls, kernel, h, rtol, atol, breadth_first):
    n_samples, n_features = 100, 3
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    Y = rng.random_sample((n_samples, n_features))
    bt = BallTree(X, leaf_size=10)

    dens_true = compute_kernel_slow(Y, X, kernel, h)

    dens = bt.kernel_density(Y, h, atol=atol, rtol=rtol,
                             kernel=kernel,
                             breadth_first=breadth_first)
    assert_allclose(dens, dens_true,
                    atol=atol, rtol=max(rtol, 1e-7))
