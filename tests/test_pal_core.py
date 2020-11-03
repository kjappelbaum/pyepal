# -*- coding: utf-8 -*-
# Copyright 2020 PyePAL authors
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
# pylint:disable=unused-import
"""Testing the PAL module"""
import numpy as np

from pyepal.pal.core import (
    _get_max_wt,
    _get_uncertainty_region,
    _get_uncertainty_regions,
    _pareto_classify,
    _uncertainty,
    _union,
    _union_one_dim,
)


def test__get_uncertainty_region():
    """make sure that the uncertainty windows is computed in a reasonable way"""
    mu = 1  # pylint:disable=invalid-name

    low0, high0 = _get_uncertainty_region(mu, 0, 1)
    assert low0 == mu
    assert high0 == mu

    low1, high1 = _get_uncertainty_region(mu, 1, 0)
    assert low1 == mu
    assert high1 == mu

    low2, high2 = _get_uncertainty_region(mu, 0, 0)
    assert low2 == mu
    assert high2 == mu

    low3, high3 = _get_uncertainty_region(mu, 1, 1)
    assert low3 == 0
    assert high3 == 2

    low4, high4 = _get_uncertainty_region(mu, 2, 1)
    assert low4 == -1
    assert high4 == 3


def test__get_uncertainty_regions():
    """The test uncertainty regions for three dimensions"""
    mu = 1  # pylint:disable=invalid-name
    lows, highs = _get_uncertainty_regions(
        np.array([mu, mu, mu]).reshape(-1, 3), np.array([0, 1, 2]).reshape(-1, 3), 1
    )
    lows = lows.flatten()
    highs = highs.flatten()
    assert lows[0] == mu
    assert highs[0] == mu
    assert lows[1] == 0
    assert highs[1] == 2
    assert lows[2] == -1
    assert highs[2] == 3

    lows, highs = _get_uncertainty_regions(
        np.array([mu - 1, mu, mu]).reshape(-1, 3), np.array([0, 1, 2]).reshape(-1, 3), 1
    )
    lows = lows.flatten()
    highs = highs.flatten()

    assert lows[0] == mu - 1
    assert highs[0] == mu - 1
    assert lows[1] == 0
    assert highs[1] == 2
    assert lows[2] == -1
    assert highs[2] == 3


def test__union_one_dim():
    """Make sure that the intersection of the uncertainty regions works"""
    zeros = np.array([0, 0, 0])
    zero_one_one = np.array([0, 1, 1])
    # Case 1: Everything is zero, we should also return zero
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )

    assert (low == zeros).all()
    assert (up == zeros).all()

    # Case 2: This should also work if this is the case for only one material
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]
    )
    assert (low == zero_one_one).all()
    assert (up == zero_one_one).all()

    # Case 3: uncertainty regions do not intersect
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]
    )
    assert (low == [2, 2, 2]).all()
    assert (up == [3, 3, 3]).all()

    # Case 4: We have an intersection
    low, up = _union_one_dim(  # pylint:disable=invalid-name
        [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [3, 3, 3]
    )
    assert (low == np.array([0.5, 0.5, 0.5])).all()
    assert (up == np.array([1, 1, 1])).all()


def test__get_max_wt():
    """Testing the sampling function"""
    lows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    highs = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    means = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    pareto_optimal = np.array([False, False, True, True])
    sampled = np.array([False, False, False, False])
    unclassified = np.array([True, True, False, False])

    max_wt = _get_max_wt(lows, highs, means, pareto_optimal, unclassified, sampled)
    assert max_wt == 2

    pareto_optimal = np.array([False, False, True, True])
    sampled = np.array([False, False, False, False])
    unclassified = np.array([True, True, False, False])

    max_wt = _get_max_wt(lows, highs, means, pareto_optimal, unclassified, sampled)
    assert max_wt == 2

    pareto_optimal = np.array([False, False, True, True])
    sampled = np.array([False, False, True, False])
    unclassified = np.array([True, True, False, False])

    max_wt = _get_max_wt(lows, highs, means, pareto_optimal, unclassified, sampled)
    assert max_wt == 1

    pareto_optimal = np.array([False, False, False, True])
    sampled = np.array([False, False, True, False])
    unclassified = np.array([True, True, False, False])

    max_wt = _get_max_wt(lows, highs, means, pareto_optimal, unclassified, sampled)
    assert max_wt == 1


def test_pareto_classify(binh_korn_points):  # pylint:disable=too-many-locals
    """Test the Pareto classification case on a 2D case,
    which I can easily draw and understand"""

    pareto_optimal_points = np.array([[0.5, 2], [3, 1], [4, 0.5]])
    discarded_points = np.array([[0.5, 0.5]])
    unclassified_points = np.array([[3.8, 2.1], [2.4, 0.5], [2.4, 0.5], [0.5, 0.5]])

    design_space = np.vstack(
        [pareto_optimal_points, discarded_points, unclassified_points]
    )

    is_pareto_optimal = np.array(
        [True] * len(pareto_optimal_points)
        + [False] * len(discarded_points)
        + [False] * len(unclassified_points)
    )

    is_discarded = np.array(
        [False] * len(pareto_optimal_points)
        + [True] * len(discarded_points)
        + [False] * len(unclassified_points)
    )

    is_unclassified = np.array(
        [False] * len(pareto_optimal_points)
        + [False] * len(discarded_points)
        + [True] * len(unclassified_points)
    )

    epsilon = np.array([0, 0])

    stdev = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.1, 0.1],
            [0.5, 0.5],
            [0, 0],
            [2.5, 2.5],
        ]
    )

    rectangle_lows = design_space - stdev
    rectangle_ups = design_space + stdev

    pareto_optimal_t, discarded_t, unclassified_t = _pareto_classify(
        is_pareto_optimal,
        is_discarded,
        is_unclassified,
        rectangle_lows,
        rectangle_ups,
        epsilon,
    )

    assert (
        pareto_optimal_t
        == np.array([True, True, True, False, True, False, False, False])
    ).all()
    assert (
        discarded_t == np.array([False, False, False, True, False, True, True, False])
    ).all()
    assert (
        unclassified_t
        == np.array([False, False, False, False, False, False, False, True])
    ).all()

    # 3D arrays, but 3rd dimenension alsways 0

    pareto_optimal_points = np.array([[0.5, 2, 0], [3, 1, 0], [4, 0.5, 0]])
    discarded_points = np.array([[0.5, 0.5, 0]])
    unclassified_points = np.array(
        [[3.8, 2.1, 0], [2.4, 0.5, 0], [2.4, 0.5, 0], [0.5, 0.5, 0]]
    )

    design_space = np.vstack(
        [pareto_optimal_points, discarded_points, unclassified_points]
    )

    epsilon = np.array([0, 0, 0])

    stdev = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.1, 0.1, 0],
            [0.5, 0.5, 0],
            [0, 0, 0],
            [2.5, 2.5, 0],
        ]
    )

    rectangle_lows = design_space - stdev
    rectangle_ups = design_space + stdev

    pareto_optimal_t, discarded_t, unclassified_t = _pareto_classify(
        is_pareto_optimal,
        is_discarded,
        is_unclassified,
        rectangle_lows,
        rectangle_ups,
        epsilon,
    )

    assert (
        pareto_optimal_t
        == np.array([True, True, True, False, True, False, False, False])
    ).all()
    assert (
        discarded_t == np.array([False, False, False, True, False, True, True, False])
    ).all()
    assert (
        unclassified_t
        == np.array([False, False, False, False, False, False, False, True])
    ).all()

    # Now, also test it on a real Pareto frontier
    _, y_binh_korn = binh_korn_points

    # For no uncertainty, we need to classify all points as Pareto optimal,
    # independent of epsilon
    pareto_optimal_t, discarded_t, unclassified_t = _pareto_classify(
        np.array([False] * len(y_binh_korn)),
        np.array([False] * len(y_binh_korn)),
        np.array([True] * len(y_binh_korn)),
        y_binh_korn,
        y_binh_korn,
        np.array([0.01, 0.01]),
    )

    assert sum(pareto_optimal_t) == len(y_binh_korn)

    pareto_optimal_t, discarded_t, unclassified_t = _pareto_classify(
        np.array([False] * len(y_binh_korn)),
        np.array([False] * len(y_binh_korn)),
        np.array([True] * len(y_binh_korn)),
        y_binh_korn,
        y_binh_korn,
        np.array([0.1, 0.1]),
    )
    assert sum(pareto_optimal_t) == len(y_binh_korn)

    pareto_optimal_t, discarded_t, unclassified_t = _pareto_classify(
        np.array([False] * len(y_binh_korn)),
        np.array([False] * len(y_binh_korn)),
        np.array([True] * len(y_binh_korn)),
        y_binh_korn,
        y_binh_korn,
        np.array([0.8, 0.8]),
    )

    assert sum(pareto_optimal_t) == len(y_binh_korn)

    # Now, testing the scale invariance
    lows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    highs = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )

    scale = np.array([1, 1000, 0.0001, 2])
    means = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    )
    pareto_optimal = np.array([False, False, True, True, True])
    sampled = np.array([False, False, False, False, False])
    unclassified = np.array([True, True, False, False, False])

    max_wt = _get_max_wt(
        lows * scale,
        highs * scale,
        means * scale,
        pareto_optimal,
        unclassified,
        sampled,
    )
    assert max_wt == 2


def test__uncertainty():
    """Test the calculation of the hyperrectangle sizes"""
    lows = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    highs = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    means = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    )

    uncertainites = _uncertainty(highs, lows, means)
    assert len(uncertainites) == len(means)
