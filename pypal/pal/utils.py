# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for dealing with Pareto fronts in general"""

import numpy as np
from numba import jit
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans

from ._hypervolume import HypervolumeIndicator


@jit(nopython=True)
def dominance_check(point1, point2) -> bool:
    """One point dominates another if it is not worse in all objectives
    and strictly better in at least one. This here assumes we want to maximize"""
    if np.all(point1 >= point2) and np.any(point1 > point2):
        return True

    return False


@jit(nopython=True)
def dominance_check_jitted(point: np.array, array: np.array) -> bool:
    """Check if point dominates any point in array"""
    arr_sorted = array[array[:, 0].argsort()]
    for i in range(len(arr_sorted)):  # pylint:disable=consider-using-enumerate
        if dominance_check(point, arr_sorted[i]):
            return True
    return False


@jit(nopython=True)
def dominance_check_jitted_2(array: np.array, point: np.array) -> bool:
    """Check if any point in array dominates point"""
    arr_sorted = array[array[:, 0].argsort()[::-1]]
    for i in range(len(arr_sorted)):  # pylint:disable=consider-using-enumerate
        if dominance_check(arr_sorted[i], point):
            return True
    return False


@jit(nopython=True)
def dominance_check_jitted_3(array: np.array, point: np.array, ignore_me: int) -> bool:
    """Check if any point in array dominates point. ignore_me
    since numba does not understand masked arrays"""
    sorted_idx = array[:, 0].argsort()[::-1]
    ignore_me = np.where(sorted_idx == ignore_me)[0][0]
    arr_sorted = array[sorted_idx]
    for i in range(len(arr_sorted)):  # pylint:disable=consider-using-enumerate
        if i != ignore_me:
            if dominance_check(arr_sorted[i], point):
                return True
    return False


def is_pareto_efficient(costs: np.array, return_mask: bool = True) -> np.array:
    """Find the Pareto efficient points
    Based on https://stackoverflow.com/questions/
    32791911/fast-calculation-of-pareto-front-in-python

    Args:
        costs (np.array): An (n_points, n_costs) array
        return_mask (bool, optional): True to return a mask,
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
            Defaults to True.

    Returns:
        np.array: [description]
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient


def exhaust_loop(
    palinstance, y: np.array, batch_size: int = 1
):  # pylint:disable=invalid-name
    """Helper function that takes an initialized PAL instance
    and loops the sampling until there is no unclassified point left.
    This is useful if all measurements are already taken and one
    wants to test the algorithm with different hyperparameters.

    Args:
        palinstance (PALBase): A initialized instance of a class that
            inherited from PALBase and implemented the ._train() and
            ._predict() functions
        y (np.array): Measurements.
            The number of measurements must equal the number of
            points in the design space.
        batch_size (int, optional): Number of indices that will be returned.
                Defaults to 10.

    Returns:
        None. The PAL instance is updated in place
    """
    assert palinstance.design_space_size == len(
        y
    ), "The number of points in the design space must equal the number of measurements"
    while sum(palinstance.unclassified):
        idx = palinstance.run_one_step(batch_size=batch_size)
        if idx is not None:
            palinstance.update_train_set(idx, y[idx])


def get_kmeans_samples(  # pylint:disable=invalid-name
    X: np.array, n_samples: int, **kwargs
) -> np.array:
    """Get the samples that are closest to the k=n_samples centroids

    Args:
        X (np.array): Feature array, on which the KMeans clustering is run
        n_samples (int): number of samples are should be selected
        **kwargs passed to the KMeans

    Returns:
        np.array:  selected_indices
    """

    assert (
        len(X) > n_samples
    ), "The numbers of points that shall be selected (n_samples),\
         needs to be smaller than the length of the feature matrix (X)"

    assert n_samples > 0 and isinstance(
        n_samples, int
    ), "The number of points that shall be selected (n_samples)\
             needs to be an integer greater than 0"

    kmeans = KMeans(n_samples, **kwargs).fit(X)
    cluster_centers = kmeans.cluster_centers_
    closest, _ = metrics.pairwise_distances_argmin_min(cluster_centers, X)

    return closest


def get_maxmin_samples(  # pylint:disable=invalid-name
    X: np.array,
    n_samples: int,
    metric: str = "euclidean",
    init: str = "mean",
    seed: int = None,
    **kwargs
) -> np.array:
    """Greedy maxmin sampling, also known as Kennard-Stone sampling (1).
    Note that a greedy sampling is not guaranteed to give the ideal solution
    and the output will depend on the random initialization (if this is chosen).

    If you need a good solution, you can restart this algorithm multiple times
    with random initialization and different random seeds
    and use a coverage metric to quantify how well
    the space is covered. Some metrics are described in (2). In contrast to the
    code provided with (2) and (3) we do not consider the feature importance for the
    selection as this is typically not known beforehand.

    You might want to standardize your data before applying this sampling function.

    Some more sampling options are provided in our structure_comp (4) Python package.
    Also, this implementation here is quite memory hungry.

    References:
    (1) Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments.
    Technometrics 1969, 11 (1), 137–148. https://doi.org/10.1080/00401706.1969.10490666.
    (2) Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet,
    J. P.; Boyd, P. G.; Lee, Y.; Smit, B.; Kulik, H. J.
    Understanding the Diversity of the Metal-Organic Framework Ecosystem.
    Nature Communications 2020, 11 (1), 4068.
    https://doi.org/10.1038/s41467-020-17755-8.
    (3) Moosavi, S. M.; Chidambaram, A.; Talirz, L.; Haranczyk, M.; Stylianou, K. C.;
    Smit, B. Capturing Chemical Intuition in Synthesis of Metal-Organic Frameworks.
    Nat Commun 2019, 10 (1), 539. https://doi.org/10.1038/s41467-019-08483-9.
    (4) https://github.com/kjappelbaum/structure_comp

    Args:
        X (np.array): Feature array, this is the array
            that is used to perform the sampling

        n_samples (int): number of points that will be selected,
            needs to be lower than the length of X

        metric (str, optional): Distance metric to use for the maxmin calculation.
            Must be a valid option of scipy.spatial.distance.cdist
            (‘braycurtis’, ‘canberra’, ‘chebyshev’,
            ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
            ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’,
            ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
            ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’,
            ‘wminkowski’, ‘yule’). Defaults to 'euclidean'

        init (str, optional): either 'mean', 'median', or 'random'.
            Determines how the initial point is chosen. Defaults to 'center'

        seed (int, optional): seed for the random number generator. Defaults to None.
        **kwargs passed to the cdist

    Returns:
        np.array: selected_indices
    """
    np.random.seed(seed)

    assert (
        len(X) > n_samples
    ), "The numbers of points that shall be selected (n_samples),\
         needs to be smaller than the length of the feature matrix (X)"

    assert n_samples > 0 and isinstance(
        n_samples, int
    ), "The number of points that shall be selected (n_samples)\
         needs to be an integer greater than 0"

    greedy_data = []

    if init == "random":
        index = np.random.randint(0, len(X) - 1)
    elif init == "mean":
        index = np.argmin(np.linalg.norm(X - np.mean(X, axis=0), axis=1))
    else:
        index = np.argmin(np.linalg.norm(X - np.median(X, axis=0), axis=1))
    greedy_data.append(X[index])
    remaining = np.delete(X, index, 0)
    while len(greedy_data) < n_samples:
        dist = distance.cdist(remaining, greedy_data, metric, **kwargs)
        greedy_index = np.argmax(np.min(dist, axis=0))
        greedy_data.append(remaining[greedy_index])
        remaining = np.delete(remaining, greedy_index, 0)

    greedy_indices = []

    for datum in greedy_data:
        greedy_indices.append(np.array(np.where(np.all(X == datum, axis=1)))[0])

    greedy_indices = np.concatenate(greedy_indices).ravel()

    return greedy_indices


def get_hypervolume(
    pareto_front: np.array, reference_vector: np.array, prefactor: float = -1
) -> float:
    """Compute the hypervolume indicator of a Pareto front
    I multiply it with minus one as we assume that we want
    to maximize all objective and then we calculate the area

    f1
    |
    |----|
    |     -|
    |       -|
    ------------ f2

    But the code we use for the hv indicator assumes that the reference vector
    is larger than all the points in the Pareto front.
    For this reason, we then flip all the signs using prefactor

    This indicator is not needed for the epsilon-PAL algorithm itself
    but only to allow tracking a metric that might help the user to see
    if the algorithm converges.
    """
    hv_instance = HypervolumeIndicator(reference_vector)
    volume = hv_instance.compute(pareto_front * prefactor)
    return volume
