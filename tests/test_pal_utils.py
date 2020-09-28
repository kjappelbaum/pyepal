# -*- coding: utf-8 -*-
"""Testing the Pareto front utilities"""
import numpy as np

from PyPAL.pal.utils import (
    dominance_check,
    dominance_check_jitted_2,
    dominance_check_jitted_3,
)


def test_dominance_check():
    """Testing if the dominance check works"""
    assert not dominance_check(np.array([0, 0]), np.array([0, 0]))
    assert dominance_check(np.array([0, 0, 1]), np.array([0, 0, 0]))
    assert dominance_check(np.array([1, 1, 1]), np.array([0, 0, 0]))
    assert dominance_check(np.array([0, 1, 1]), np.array([0, 0, 0]))

    assert not dominance_check(np.array([4.5, 1]), np.array([3.7, 2]))


def test_dominance_check_jitted_2():
    """Testing if the dominance check array/point works"""
    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    point = np.array([1.0, 1.0, 1.0])

    assert not dominance_check_jitted_2(array, point)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    point = np.array([0.0, 0.0, 1.0])

    assert dominance_check_jitted_2(array, point)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    point = np.array([0.0, 0.0, 0.8])

    assert dominance_check_jitted_2(array, point)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    point = np.array([0.0, 0.0, 1.8])

    assert not dominance_check_jitted_2(array, point)

    pareto_optimal_points = np.array([[0.5, 2], [3, 1], [4, 0.5]])
    new_point = np.array([3.8, 2.1])

    assert not dominance_check_jitted_2(pareto_optimal_points, new_point)

    ups = np.array(
        [
            [1.0, 2.5],
            [3.5, 1.5],
            [4.5, 1.0],
            [1.0, 1.0],
            [2.9, 1.0],
            [2.4, 0.5],
            [1.0, 1.0],
        ]
    )

    assert not dominance_check_jitted_2(ups, np.array([3.7, 2.0]))


def test_dominance_check_jitted_3():
    """Test the domincance check function with workaround around
    masked array"""
    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    point = np.array([1.0, 1.0, 1.0])

    assert not dominance_check_jitted_3(array, point, 2)

    array = np.array([[1.0, 2.5], [3.5, 1.5], [4.5, 1.0], [3.9, 2.2]])
    point = np.array([3.7, 2])

    assert not dominance_check_jitted_3(array, point, 3)
