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


"""Core functions for PAL"""
from typing import Sequence, Tuple, Union

import numpy as np
from numba import jit

from .utils import (
    dominance_check_jitted_2,
    dominance_check_jitted_3,
    is_pareto_efficient,
)


def _get_uncertainty_region(  # pylint:disable=invalid-name
    mu: np.array, std: np.array, beta_sqrt: float
) -> Tuple[np.array, np.array]:
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


def _get_uncertainty_regions(
    mus: np.array, stds: np.array, beta_sqrt: float
) -> Union[np.array, np.array]:
    """
    Compute the lower and upper bound of the uncertainty region
         for each dimension (=target)

    Args:
        mus (np.array): means
        stds (np.array): standard deviations
        beta_sqrt (float): scaling factors

    Returns:
        Union[np.array, np.array]: lower bounds, upper bounds
    """
    low_lims, high_lims = [], []

    for i in range(0, mus.shape[1]):
        low_lim, high_lim = _get_uncertainty_region(mus[:, i], stds[:, i], beta_sqrt)
        low_lims.append(low_lim.reshape(-1, 1))
        high_lims.append(high_lim.reshape(-1, 1))

    return np.hstack(low_lims), np.hstack(high_lims)


def _union(
    lows: np.array, ups: np.array, new_lows: np.array, new_ups: np.array
) -> Union[np.array, np.array]:
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

    for i in range(0, lows.shape[1]):
        low, up = _union_one_dim(  # pylint:disable=invalid-name
            lows[:, i], ups[:, i], new_lows[:, i], new_ups[:, i]
        )
        out_lows.append(low.reshape(-1, 1))
        out_ups.append(up.reshape(-1, 1))

    out_lows_array, out_ups_array = np.hstack(out_lows), np.hstack(out_ups)

    return out_lows_array, out_ups_array


@jit(nopython=True)
def _union_one_dim(
    lows: Sequence, ups: Sequence, new_lows: Sequence, new_ups: Sequence
) -> Tuple[np.array, np.array]:
    """Used to intersect the confidence regions, for eq. 6 of the PAL paper.
    The iterative intersection ensures that all uncertainty regions
    are non-increasing with t.

    We do not check for the ordering in this function.
    We really assume that the lower limits are the lower limits
    and the upper limits are the upper limits.

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
        # 1) |--old range--|   |--new range--|,
        # i.e., lower new limit above old upper limit
        # 2) |--new range--|   |--old range--|,
        # i.e., upper new limit below lower old limit
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
    epsilon: np.array,
) -> Tuple[np.array, np.array, np.array]:
    """Performs the classification part of the algorithm
    (p. 4 of the PAL paper, see algorithm 1/2 of the epsilon-PAL paper)

    One core concept is that once a point is classified,
    it does no longer change the class.

    When we do the comparison with +/- epsilon we always use the absolute values!
    Otherwise, we get inconcistent results depending on the sign!

    Args:
        pareto_optimal_0 (np.array): boolean mask of points classified as Pareto optimal
        not_pareto_optimal_0 (np.array): boolean mask of points
            classified as non-Pareto optimal
        unclassified_0 (np.array): boolean mask of unclassified points
        rectangle_lows (np.array): lower uncertainty boundaries
        rectangle_ups (np.array): upper uncertainty boundaries
        epsilon (np.array): granularity parameter (one per dimension)

    Returns:
        Tuple[list, list, list]: binary encoded list of Pareto optimal,
            non-Pareto optimal and unclassified points
    """
    pareto_optimal_t = pareto_optimal_0.copy()
    not_pareto_optimal_t = not_pareto_optimal_0.copy()
    unclassified_t = unclassified_0.copy()

    # This part is only relevant when we have points in set P
    # Then we can use those points to discard points from p_pess (P \cup U)
    if sum(pareto_optimal_0) > 0:
        pareto_indices = np.where(pareto_optimal_0)[0]
        pareto_pessimistic_lows = rectangle_lows[pareto_indices]  # p_pess(P)
        for i in range(0, len(unclassified_0)):
            if unclassified_t[i] == 1:
                # discard if any lower-bound epsilon dominates the upper bound
                if dominance_check_jitted_2(
                    pareto_pessimistic_lows + np.abs(epsilon * pareto_pessimistic_lows),
                    rectangle_ups[i],
                ):
                    not_pareto_optimal_t[i] = True
                    unclassified_t[i] = False

    pareto_unclassified_indices = np.where(pareto_optimal_0 | unclassified_t)[0]
    pareto_unclassified_lows = rectangle_lows[pareto_unclassified_indices]

    # assuming maximization
    pareto_unclassified_pessimistic_mask = is_pareto_efficient(
        -pareto_unclassified_lows
    )
    original_indices = pareto_unclassified_indices[pareto_unclassified_pessimistic_mask]
    pareto_unclassified_pessimistic_points = pareto_unclassified_lows[
        pareto_unclassified_pessimistic_mask
    ]
    for i in range(0, len(unclassified_t)):  # pylint:disable=consider-using-enumerate
        # We can only discard points that are unclassified so far
        # We cannot discard points that are part of p_pess(P \cup U)
        if unclassified_t[i] and (i not in original_indices):
            # discard if any lower-bound epsilon dominates the upper bound
            if dominance_check_jitted_2(
                epsilon * np.abs(pareto_unclassified_pessimistic_points)
                + pareto_unclassified_pessimistic_points,
                rectangle_ups[i],
            ):
                not_pareto_optimal_t[i] = True
                unclassified_t[i] = False

    # now, update the pareto set
    # if there is no other point x' such that max(Rt(x')) >= min(Rt(x))
    # move x to Pareto
    unclassified_indices = np.where(unclassified_t | pareto_optimal_t)[0]
    unclassified_ups = rectangle_ups[unclassified_indices]

    index_map = {index: i for i, index in enumerate(unclassified_indices)}

    # The index map helps us to mask the current point from the unclassified_ups list
    for i in range(0, len(unclassified_t)):  # pylint:disable=consider-using-enumerate
        # again, we only care about unclassified points
        if unclassified_t[i]:
            # If there is no other point which up is epsilon dominating
            # the low of the current point,
            # the current point is epsilon-accurate Pareto optimal
            if not dominance_check_jitted_3(
                unclassified_ups,
                rectangle_lows[i] + epsilon * np.abs(rectangle_lows[i]),
                index_map[i],
            ):
                pareto_optimal_t[i] = True
                unclassified_t[i] = False

    return pareto_optimal_t, not_pareto_optimal_t, unclassified_t


# ToDo: maybe add jitter to avoid issues when the prediction is exactly 0
@jit(nopython=True)
def _get_max_wt(  # pylint:disable=too-many-arguments
    rectangle_lows: np.array,
    rectangle_ups: np.array,
    means: np.array,
    pareto_optimal_t: np.array,
    unclassified_t: np.array,
    sampled: np.array,
) -> int:
    """Returns the index in design space with the maximum size of the hyperrectangle
    (scaled by the mean predictions, i.e., effectively,
    we use the coefficient of variation).
    Samples only from unclassified or Pareto-optimal points.

    Args:
        rectangle_lows (np.array): Lower, pessimistic, bounds of the hyperrectangles
        rectangle_ups (np.array): Upper, optimistic, bounds of the hyperrectangles
        means (np.array): Mean predictions
        pareto_optimal_t (np.array): Mask array that is True
            for the Pareto optimal points
        unclassified_t (np.array): Mask array that is True for the unclassified points
        sampled (np.array): Mask array that is True for the sampled points

    Returns:
        int: index with maximum size of hyperrectangle
    """
    max_uncertainty = -np.inf
    maxid = 0

    for i in range(0, len(unclassified_t)):  # pylint:disable=consider-using-enumerate
        # Among the points x ∈ Pt ∪ Ut, the one with the largest wt(x)
        # is chosen as the next sample xt to be evaluated.
        # Intuitively, this rule biases the sampling towards exploring,
        # and thus improving the model for, the points most likely to be Pareto-optimal.
        if ((unclassified_t[i] == 1) or (pareto_optimal_t[i] == 1)) and not sampled[
            i
        ] == 1:
            # weight is the length of the diagonal of the uncertainty region
            coeff_var = np.divide(
                rectangle_ups[i, :] - rectangle_lows[i, :], means[i, :]
            )
            uncertainty = np.linalg.norm(coeff_var)
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                maxid = i

    return maxid


def _uncertainty(rectangle_ups, rectangle_lows, means):
    """Hyperrectangle sizes"""
    if (rectangle_lows is None) or (rectangle_ups is None):
        raise ValueError(
            "You must have trained a model and\
             run the prediction to calculate hyperrectangle"
        )
    coeff_var = np.nan_to_num(np.divide(rectangle_ups - rectangle_lows, means))
    uncertainty = np.linalg.norm(coeff_var, axis=1)
    return uncertainty
