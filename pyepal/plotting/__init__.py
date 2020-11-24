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


"""Plotting utilities"""

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from .. import PALBase

plt.rcParams["font.family"] = "sans-serif"


__all__ = [
    "plot_bar_iterations",
    "plot_pareto_front_2d",
    "plot_residuals",
    "plot_histogram",
    "plot_jointplot",
]


def plot_bar_iterations(  # pylint:disable=invalid-name
    pareto_optimal: np.ndarray,
    non_pareto_points: np.ndarray,
    unclassified_points: np.ndarray,
    ax=None,
):
    """Plot stacked barplots for every step of the iteration.

    Args:
        pareto_optimal (np.ndarray): Number of pareto optimal points
            for every iteration.
        non_pareto_points (np.ndarray): Number of discarded points
            for every iteration
        unclassified_points (np.ndarray): Number of unclassified points
            for every iteration

    Returns:
        ax
    """
    assert (
        len(pareto_optimal) == len(non_pareto_points) == len(unclassified_points)
    ), "Make sure that the arrays have the same length"

    # We need numpy arrays as we assume that we can add the arrays
    # ToDo: We could potentially cast any iterable
    assert isinstance(pareto_optimal, np.ndarray), "The arguments must be numpy arrays"
    assert isinstance(
        non_pareto_points, np.ndarray
    ), "The arguments must be numpy arrays"
    assert isinstance(
        unclassified_points, np.ndarray
    ), "The arguments must be numpy arrays"

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.bar(range(len(pareto_optimal)), unclassified_points, label="unclassified")
    ax.bar(
        range(len(pareto_optimal)),
        non_pareto_points,
        bottom=unclassified_points,
        label="discarded",
    )
    ax.bar(
        range(len(pareto_optimal)),
        pareto_optimal,
        bottom=np.array(non_pareto_points) + np.array(unclassified_points),
        label="Pareto optimal",
    )
    ax.set_xlabel("iteration (after initialization)")
    ax.set_ylabel("number of design points")

    return ax


def plot_pareto_front_2d(  # pylint:disable=too-many-arguments, invalid-name
    y_0: np.ndarray,
    y_1: np.ndarray,
    std_0: np.ndarray,
    std_1: np.ndarray,
    palinstance: PALBase,
    ax=None,
):
    """Plot a 2D pareto front, with the different categories
    indicated in color.

    Args:
        y_0 (np.ndarray): objective 0
        y_1 (np.ndarray): objective 1
        std_0 (np.ndarray): standard deviation objective 0
        std_1 (np.ndarray): standard deviation objective 0
        palinstance (PALBase): PAL instance
        ax (ax, optional): Matplotlib figure axis. Defaults to None.

    Returns:
        ax
    """
    # Here, we need numpy arrays for the indexing
    for array in [y_0, y_1, std_0, std_1]:
        assert isinstance(array, np.ndarray), "array must be a numpy array"
    assert (
        len(y_0)
        == len(y_1)
        == len(std_0)
        == len(std_1)
        == palinstance.number_design_points
    ), "Make sure that the arrays have the same length"

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.errorbar(
        y_0,
        y_1,
        xerr=std_0,
        yerr=std_1,
        c="gray",
        alpha=0.3,
        label="all design points",
        fmt=".",
        capsize=5,
        zorder=0,
    )
    ax.scatter(
        y_0[palinstance.sampled_indices],
        y_1[palinstance.sampled_indices],
        c="blue",
        label="sampled",
        s=20,
        zorder=5,
    )
    ax.scatter(
        y_0[palinstance.discarded],
        y_1[palinstance.discarded],
        c="red",
        label="discarded",
        s=15,
        alpha=1,
        zorder=10,
    )
    ax.scatter(
        y_0[palinstance.pareto_optimal],
        y_1[palinstance.pareto_optimal],
        c="green",
        label="Pareto optimal",
        s=10,
        alpha=0.8,
        zorder=15,
    )

    return ax


def plot_histogram(
    y: np.ndarray, palinstance: PALBase, ax=None
):  # pylint:disable=invalid-name
    """Plot histograms, with maxima scaled to one
    and different categories indicated in color
    for one objective

    Args:
        y (np.ndarray): objective (measurement)
        palinstance (PALBase): instance of a PAL class
        ax (ax): Matplotlib figure axis

    Returns:
        ax
    """
    assert isinstance(y, np.ndarray), "Input array y must be a numpy array"
    assert y.ndim == 1, "Input array y must be one dimensional"
    assert (
        len(y) == palinstance.number_design_points
    ), "Length of y must equal the size of the design space"

    if ax is None:
        _, ax = plt.subplots(1, 1)
    heights, bins = np.histogram(y)
    bin_width = bins[1] - bins[0]
    ax.bar(
        bins[:-1],
        heights / heights.max(),
        bin_width,
        label="all design points",
        color="gray",
        alpha=0.6,
    )
    heights, bins = np.histogram(y[palinstance.sampled_indices])
    bin_width = bins[1] - bins[0]
    ax.bar(
        bins[:-1],
        heights / heights.max(),
        bin_width,
        label="sampled",
        color="blue",
        alpha=0.6,
    )

    heights, bins = np.histogram(y[palinstance.pareto_optimal])
    bin_width = bins[1] - bins[0]
    ax.bar(
        bins[:-1],
        heights / heights.max(),
        bin_width,
        label="Pareto optimal",
        color="green",
        alpha=0.6,
    )

    return ax


def plot_residuals(  # pylint:disable=invalid-name
    y: np.array,
    palinstance: PALBase,
    labels: Union[List[str], None] = None,
    figsize: tuple = (6.0, 4.0),
):
    """Plot signed residual (on y axis) vs fitted (on x axis)
    plot of sampled points.
    Will create suplots for y.ndim > 1.


    Args:
        y (np.array): Two-dimensional array with the objectives (measurements)
        palinstance (PALBase): "trained" PAL instance
        labels (Union[List[str], None], optional): Labels for each objective.
            Defaults to "objective [index]".
        figsize (tuple, optional): Figure size for each individual residual
            vs fitted objective plot. Defaults to (6.0, 4.0).

    Returns:
        fig
    """

    assert isinstance(y, np.ndarray), "Input array y must be a numpy array"
    assert (
        len(y) == palinstance.number_design_points
    ), "Length of y must equal the size of the design space"
    assert y.ndim == 2, "y must be a two-dimensional numpy array"
    assert (
        y.shape[1] == palinstance.ndim
    ), "y needs to be a two-dimensional array which \
        column number equals the number of targets"
    if palinstance.means is None:
        raise ValueError(
            "Predicted means is None. Execute run_one_step() \
                to obtain predicted means for each model."
        )

    num_targets = palinstance.ndim

    fig, ax = plt.subplots(  # pylint:disable=invalid-name
        num_targets,
        1,
        figsize=(figsize[0], num_targets * figsize[1]),
        tight_layout=True,
    )

    for index in range(num_targets):
        sampled = palinstance.sampled[:, index]
        fitted = palinstance.means[sampled, index]
        residuals = y[sampled, index] - fitted
        ax[index].scatter(fitted, residuals)
        ax[index].spines["top"].set_color("none")
        ax[index].spines["right"].set_color("none")
        ax[index].spines["left"].set_smart_bounds(True)
        ax[index].spines["bottom"].set_smart_bounds(True)

    if labels is None:
        labels = [f"objective {i}" for i in range(num_targets)]
    else:
        assert (
            len(labels) == num_targets
        ), "If labels are provided the length of the labels list \
            must equal the number of targets"

    for index in range(num_targets):
        ax[index].set_title(labels[index])

    return fig


def plot_jointplot(  # pylint:disable=invalid-name
    y: np.array,
    palinstance: PALBase,
    labels: Union[List[str], None] = None,
    figsize: tuple = (8.0, 6.0),
):
    """Plot a jointplot of the objective space with histograms on the diagonal
    and 2D-Pareto plots on the off-diagonal.

    Args:
        y (np.array): Two-dimensional array with the objectives (measurements)
        palinstance (PALBase): "trained" PAL instance
        labels (Union[List[str], None], optional): Labels for each objective.
            Defaults to "objective [index]".
        figsize (tuple, optional): Figure size for joint plot. Defaults to (8.0, 6.0).

    Returns:
        fig
    """
    assert isinstance(y, np.ndarray), "Input array y must be a numpy array"
    assert (
        len(y) == palinstance.number_design_points
    ), "Length of y must equal the size of the design space"
    assert y.ndim == 2, "y must be a two-dimensional numpy array"
    assert (
        y.shape[1] == palinstance.ndim
    ), "y needs to be a two-dimensional array which column number \
        equals the number of targets"
    if (
        (palinstance.means is None)
        or (palinstance.std is None)
        or (palinstance.beta is None)
    ):
        raise ValueError(
            "Predicted means is None. Execute run_one_step() \
                to obtain predicted means for each model."
        )

    num_targets = y.shape[1]
    fig, ax = plt.subplots(  # pylint:disable=invalid-name
        num_targets, num_targets, figsize=figsize, tight_layout=True
    )

    for row in range(num_targets):
        for column in range(num_targets):
            if row == column:
                plot_histogram(y[:, row], palinstance, ax[row, column])
            else:
                plot_pareto_front_2d(
                    y[:, row],
                    y[:, column],
                    palinstance.std[:, row] * np.sqrt(palinstance.beta),
                    palinstance.std[:, column] * np.sqrt(palinstance.beta),
                    palinstance,
                    ax=ax[row, column],
                )

            ax[row, column].spines["top"].set_color("none")
            ax[row, column].spines["right"].set_color("none")
            ax[row, column].spines["left"].set_smart_bounds(True)
            ax[row, column].spines["bottom"].set_smart_bounds(True)

    if labels is None:
        labels = [f"objective {i}" for i in range(num_targets)]
    else:
        assert len(labels) == num_targets

    for index in range(num_targets):
        ax[index, 0].set_ylabel(labels[index])
        ax[num_targets - 1, index].set_xlabel(labels[index])

    ax[0, num_targets - 1].legend()

    return fig
