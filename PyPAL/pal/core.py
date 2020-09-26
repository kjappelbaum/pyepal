# -*- coding: utf-8 -*-
"""Core functions for PAL"""
from typing import Sequence, Tuple, Union

import numpy as np
from numba import jit

from .utils import dominance_check_jitted_2, is_pareto_efficient


def _get_uncertainity_region(mu: np.array, std: np.array, beta_sqrt: float) -> Tuple[np.array, np.array]:  # pylint:disable=invalid-name
    """

    Args:
        mu (float): mean
        std (float): standard deviation
        beta_sqrt (float): scaling factor

    Returns:
        Tuple[float, float]: lower bound, upper bound
    """
    low_lim, high_lim = mu - beta_sqrt * std, mu + beta_sqrt * std
    return low_lim, high_lim


def _get_uncertainity_regions(mus: np.array, stds: np.array, beta_sqrt: float) -> Union[np.array, np.array]:
    """
    Compute the lower and upper bound of the uncertainty region for each dimension (=target)

    Args:
        mus (np.array): means
        stds (np.array): standard deviations
        beta_sqrt (float): scaling factors

    Returns:
        Union[np.array, np.array]: lower bounds, upper bounds
    """
    low_lims, high_lims = [], []

    for i in range(0, mus.shape[1]):  # pylint:disable=consider-using-enumerate (I find this clearer)
        low_lim, high_lim = _get_uncertainity_region(mus[:, i], stds[:, i], beta_sqrt)
        low_lims.append(low_lim.reshape(-1, 1))
        high_lims.append(high_lim.reshape(-1, 1))

    return np.hstack(low_lims), np.hstack(high_lims)


def _union(lows: np.array, ups: np.array, new_lows: np.array, new_ups: np.array) -> Union[np.array, np.array]:
    """Performing iterative intersection (eq. 6 in PAL paper) in all dimensions.

    Args:
        lows (np.array): lower bounds from previous iteration
        ups (np.array): upper bounds from previous iteration
        new_lows (np.array): lower bounds from current iteration
        new_ups (np.array): upper bounds from current iteration

    Returns:
        Union[np.array, np.array]: lower bounds, upper bounds
    """
    out_lows = []
    out_ups = []

    for i in range(0, lows.shape[1]):  # pylint:disable=consider-using-enumerate (I find this clearer)
        low, up = _union_one_dim(lows[:, i], ups[:, i], new_lows[:, i], new_ups[:, i])  # pylint:disable=invalid-name
        out_lows.append(low.reshape(-1, 1))
        out_ups.append(up.reshape(-1, 1))

    out_lows_array, out_ups_array = np.hstack(out_lows), np.hstack(out_ups)

    return out_lows_array, out_ups_array


@jit(nopython=True)
def _union_one_dim(lows: Sequence, ups: Sequence, new_lows: Sequence, new_ups: Sequence) -> Tuple[np.array, np.array]:
    """Used to intersect the confidence regions, for eq. 6 of the PAL paper.
    "The iterative intersection ensures that all uncertainty regions are non-increasing with t."

    We do not check for the ordering in this function.
    We really assume that the lower limits are the lower limits and the upper limits are the upper limits.

    All arrays must have the same length.

    Args:
        lows (Sequence): lower bounds from previous iteration
        ups (Sequence): upper bounds from previous iteration
        new_lows (Sequence): lower bounds from current iteration
        new_ups (Sequence): upper bounds from current iteration

    Returns:
        Tuple[np.array, np.array]: array of lower limits, array of upper limits
    """
    out_lows = []
    out_ups = []

    for i, low in enumerate(lows):
        # In one dimension we can imagine the following cases where there
        # is zero intersection
        # 1) |--old range--|   |--new range--|, i.e., lower new limit above old upper limit
        # 2) |--new range--|   |--old range--|, i.e., upper new limit below lower old limit
        if (new_lows[i] >= ups[i]) or (new_ups[i] <= low):
            out_lows.append(new_lows[i])
            out_ups.append(new_ups[i])

        # In other cases, we want to intersect the ranges, i.e.
        # |---old range-|-|--new-range--| --> |-|
        # i.e. we take the max of the lower limits and the min of the upper limits
        else:
            out_lows.append(max(low, new_lows[i]))
            out_ups.append(min(ups[i], new_ups[i]))

    return np.array(out_lows), np.array(out_ups)


def _pareto_classify(  # pylint:disable=too-many-arguments, too-many-locals
    pareto_optimal_0: np.array,
    not_pareto_optimal_0: np.array,
    unclassified_0: np.array,
    rectangle_lows: np.array,
    rectangle_ups: np.array,
    x_input: np.array,
    epsilon: list,
) -> Tuple[np.array, np.array, np.array]:
    """Performs the classification part of the algorithm
    (p. 4 of the PAL paper, see algorithm 1/2 of the epsilon-PAL paper)

    One core concept is that once a point is classified it does no longer change the class.

    Args:
        pareto_optimal_0 (list): binary encoded list of points classified as Pareto optimal
        not_pareto_optimal_0 (list): binary encoded list of points classified as non-Pareto optimal
        unclassified_0 (list): binary encoded list of unclassified points
        rectangle_lows (np.array): lower uncertainity boundaries
        rectangle_ups (np.array): upper uncertainity boundaries
        x_input (np.array): feature matrix
        epsilon (list): granularity parameter (one per dimension)

    Returns:
        Tuple[list, list, list]: binary encoded list of Pareto optimal, non-Pareto optimal and unclassified points
    """
    pareto_optimal_t = pareto_optimal_0.copy()
    not_pareto_optimal_t = not_pareto_optimal_0.copy()
    unclassified_t = unclassified_0.copy()

    # This part is only relevant when we have points in set P
    # Then we can use those points to discard points from p_pess (P \cup U)
    if sum(pareto_optimal_0) > 0:
        pareto_indices = np.where(pareto_optimal_0 == 1)[0]
        pareto_pessimistic_lows = rectangle_lows[pareto_indices]  # p_pess(P)
        for i in range(0, len(x_input)):
            if unclassified_t[i] == 1:
                if dominance_check_jitted_2(pareto_pessimistic_lows * (1 + epsilon), rectangle_ups[i]):
                    not_pareto_optimal_t[i] = 1
                    unclassified_t[i] = 0

    # ToDo: can be probably cleaned up a bit
    pareto_unclassified_indices = np.where((pareto_optimal_0 == 1) | (unclassified_t == 1))[0]

    pareto_unclassified_lows = rectangle_lows[pareto_unclassified_indices]

    # assuming maximization
    pareto_unclassified_pessimistic_mask = is_pareto_efficient(-pareto_unclassified_lows)  # disable:pylint:disable=invalid-name
    original_indices = pareto_unclassified_indices[pareto_unclassified_pessimistic_mask]
    pareto_unclassified_pessimistic_points = pareto_unclassified_lows[pareto_unclassified_pessimistic_mask]  # pylint:disable=invalid-name

    for i in range(0, len(x_input)):
        # We can only discard points that are unclassified so far
        # We cannot discard points that are part of p_pess(P \cup U)
        if (unclassified_t[i] == 1) and (i not in original_indices):
            # If the upper bound of the hyperrectangle is not dominating anywhere
            # the pareto pessimitic set, we can discard
            if dominance_check_jitted_2(pareto_unclassified_pessimistic_points * (1 + epsilon), rectangle_ups[i]):
                not_pareto_optimal_t[i] = 1
                unclassified_t[i] = 0

    # now, update the pareto set
    # if there is no other point x' such that max(Rt(x')) >= min(Rt(x))
    # move x to Pareto
    unclassified_indices = np.where((unclassified_t == 1) | (pareto_optimal_t == 1))[0]
    unclassified_ups = np.ma.array(rectangle_ups[unclassified_indices])

    # The index map helps us to mask the current point from the unclassified_ups list
    index_map = dict(zip(unclassified_indices, range(len(unclassified_ups))))

    for i in range(0, len(x_input)):
        # again, we only care about unclassified points
        if unclassified_t[i] == 1:
            # We need to make sure that unclassified_ups does not contain the current point
            unclassified_ups[index_map[i]] = np.ma.masked
            # If there is no other point which up is epsilon dominating the low of the current point,
            # the current point is epsilon-accurate Pareto optimal
            if not dominance_check_jitted_2(unclassified_ups, rectangle_lows[i] * (1 + epsilon)):
                pareto_optimal_t[i] = 1
                unclassified_t[i] = 0

            # now we can demask the entry
            unclassified_ups[index_map[i]] = np.ma.nomask

    return pareto_optimal_t, not_pareto_optimal_t, unclassified_t


@jit(nopython=True)
def _get_max_wt(
    rectangle_lows: np.array,
    rectangle_ups: np.array,
    pareto_optimal_t: np.array,
    unclassified_t: np.array,
    sampled: np.array,
    x_input: np.array,
) -> int:
    """Returns the index in design space with the maximum size of the hyperrectangle.
    Samples only from unclassified or Pareto-optimal points.

    Args:
        rectangle_lows (np.array): Lower, pessimistic, bounds of the hyperrectangles
        rectangle_ups (np.array): Upper, optimistic, bounds of the hyperrectangles
        pareto_optimal_t (np.array): Mask array that is True for the Pareto optimal points
        unclassified_t (np.array): Mask array that is True for the unclassified points
        sampled (np.array): Mask array that is True for the sampled points
        x_input (np.array): Design space

    Returns:
        int: index with maximum size of hyperrectangle
    """
    max_uncertainity = 0
    maxid = -1

    for i in range(0, len(x_input)):
        # Among the points x ∈ Pt ∪ Ut, the one with the largest wt(x) is chosen as the next sample xt to be evaluated.
        # Intuitively, this rule biases the sampling towards exploring,
        # and thus improving the model for, the points most likely to be Pareto-optimal.
        if ((unclassified_t[i] == 1) or (pareto_optimal_t[i] == 1)) and not sampled[i] == 1:
            # weight is the length of the diagonal of the uncertainity region
            uncertainity = np.linalg.norm(rectangle_ups[i, :] - rectangle_lows[i, :])
            if maxid == -1:
                max_uncertainity = uncertainity
                maxid = i
            # the point with the largest weight is chosen as the next sample
            elif uncertainity > max_uncertainity:
                max_uncertainity = uncertainity
                maxid = i

    return maxid
