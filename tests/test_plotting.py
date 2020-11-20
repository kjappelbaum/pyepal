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
"""Test the plotting module for basic functionality.
We do not test if the plots are readable etc."""


import matplotlib
import numpy as np
import pytest

from pyepal import PALBase
from pyepal.plotting import (
    plot_bar_iterations,
    plot_histogram,
    plot_jointplot,
    plot_pareto_front_2d,
    plot_residuals,
)


def test_plot_bar_iterations():
    """Test assertions and the output class"""
    with pytest.raises(AssertionError):
        plot_bar_iterations([1, 1, 2], [2, 2, 2], [4, 4, 4])

    with pytest.raises(AssertionError):
        plot_bar_iterations(np.array([1, 1, 2]), np.array([2, 2, 2]), np.array([3, 3]))

    axis = plot_bar_iterations(
        np.array([1, 1, 2]), np.array([1, 1, 2]), np.array([1, 1, 2])
    )
    assert isinstance(axis, matplotlib.axes.Axes)


def test_plot_pareto_front_2d(make_random_dataset):
    """Test assertions and output class"""
    palinstance = PALBase(make_random_dataset[0][:3], ["model"], 3)
    palinstance.sampled = np.array([True, False, False])
    palinstance.pareto_optimal = np.array([False] * 3)
    palinstance.discarded = np.array([True, False, False])
    with pytest.raises(AssertionError):
        plot_pareto_front_2d([1, 1, 2], [2, 2, 2], [3, 3, 3], [4, 4, 4], palinstance)

    with pytest.raises(AssertionError):
        plot_pareto_front_2d(
            np.array([1, 1, 2]),
            np.array([2, 2, 2]),
            np.array([3, 3]),
            np.array([3, 3]),
            palinstance,
        )

    plot = plot_pareto_front_2d(
        np.array([1, 1, 2]),
        np.array([2, 2, 2]),
        np.array([3, 3, 1]),
        np.array([3, 3, 1]),
        palinstance,
    )

    assert isinstance(plot, matplotlib.axes.Axes)


def test_plot_histogram(make_random_dataset):
    """Test assertions and output class"""
    palinstance = PALBase(make_random_dataset[0], ["model"], 3)
    palinstance.sampled = np.array([True] * 100)
    palinstance.pareto_optimal = np.array([False] * 80 + [True] * 20)
    palinstance.discarded = np.array([True] * 80 + [False] * 20)

    with pytest.raises(AssertionError):
        plot_histogram(np.array([1, 1]), palinstance)

    with pytest.raises(AssertionError):
        plot_histogram([1] * 100, palinstance)

    plot = plot_histogram(np.array([1] * 100), palinstance)
    assert isinstance(plot, matplotlib.axes.Axes)


def test_plot_residuals(make_random_dataset):
    """Test assertions and output class"""
    palinstance = PALBase(make_random_dataset[0], ["model"], 3)
    palinstance.sampled = np.array([[True, True, True]] * 100)
    palinstance.pareto_optimal = np.array([False] * 80 + [True] * 20)
    palinstance.discarded = np.array([True] * 80 + [False] * 20)
    means = np.array([[1, 1, 1]] * 100)

    with pytest.raises(ValueError):
        plot_residuals(np.array([[1, 2, 1]] * 100), palinstance)

    palinstance.means = means
    with pytest.raises(AssertionError):
        plot_residuals(np.array([1] * 100), palinstance)

    with pytest.raises(AssertionError):
        plot_residuals(np.array([[1, 1, 1]] * 99), palinstance)

    fig = plot_residuals(np.array([[1, 1, 1]] * 100), palinstance)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_jointplot(make_random_dataset):
    """Test assertions and output class"""
    palinstance = PALBase(make_random_dataset[0], ["model"], 3)
    palinstance.sampled = np.array([[True, True, True]] * 100)
    palinstance.pareto_optimal = np.array([False] * 80 + [True] * 20)
    palinstance.discarded = np.array([True] * 80 + [False] * 20)
    means = np.array([[1, 1, 1]] * 100)

    with pytest.raises(ValueError):
        plot_jointplot(np.array([[1, 2, 1]] * 100), palinstance)

    palinstance.means = means
    palinstance.std = means
    palinstance.beta = 1
    with pytest.raises(AssertionError):
        plot_jointplot(np.array([1] * 100), palinstance)

    with pytest.raises(AssertionError):
        plot_jointplot(np.array([[1, 1, 1]] * 99), palinstance)

    fig = plot_jointplot(np.array([[1, 1, 1]] * 100), palinstance)

    assert isinstance(fig, matplotlib.figure.Figure)
