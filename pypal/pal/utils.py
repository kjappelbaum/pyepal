# -*- coding: utf-8 -*-
"""Utilities for dealing with Pareto fronts in general"""
import numpy as np
from numba import jit


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


def exhaust_loop(palinstance, y: np.array):  # pylint:disable=invalid-name
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

    Returns:
        None. The PAL instance is updated in place
    """
    assert palinstance.design_space_size == len(
        y
    ), "The number of points in the design space must equal the number of measurements"
    while sum(palinstance.unclassified):
        idx = palinstance.run_one_step()
        if idx is not None:
            palinstance.update_train_set(np.array([idx]), y[idx : idx + 1, :])
