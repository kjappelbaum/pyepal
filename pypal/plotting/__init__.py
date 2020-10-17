# -*- coding: utf-8 -*-
"""Plotting utilities"""
import numpy as np
from typing import Iterable

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"


def plot_bar_iterations(  # pylint:disable=invalid-name
    pareto_optimal: Iterable,
    non_pareto_points: Iterable,
    unclassified_points: Iterable,
    ax=None,
):
    """Plot stacked barplots for every step of the iteration.

    Args:
        pareto_optimal (Iterable): Number of pareto optimal points
            for every iteration.
        non_pareto_points (Iterable): Number of discarded points
            for every iteration
        unclassified_points (Iterable): Number of unclassified points
            for every iteration

    Returns:
        ax
    """
    assert len(pareto_optimal) == len(non_pareto_points) == len(unclassified_points)
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


def plot_pareto_front_2d(y_0: np.array, y_1: np.array, palinstance, ax=None):
    ax.scatter(y_0, y_1, c="gray", alpha=0.6, label="all design points", s=1)
    ax.scatter(
        y_0[palinstance.sampled_indices],
        y_1[palinstance.sampled_indices],
        c="blue",
        label="sampled",
        s=20,
    )
    ax.scatter(
        y_0[palinstance.discarded],
        y_1[palinstance.discarded],
        c="red",
        label="discarded",
        s=10,
        alpha=0.8,
    )
    ax.scatter(
        y_0[palinstance.pareto_optimal],
        y_1[palinstance.pareto_optimal],
        c="green",
        label="Pareto optimal",
        s=10,
        alpha=0.8,
    )


def plot_histogram(y, palinstance, ax):
    ax.hist(y, density=True, label="all design points", color="gray", alpha=0.6)
    ax.hist(y[palinstance.sampled_indices], density=True, label="sampled", color="blue")
    ax.hist(y[palinstance.discarded], density=True, label="discarded", color="red")
    ax.hist(
        y[palinstance.pareto_optimal],
        density=True,
        label="Pareto optimal",
        color="green",
    )


def make_jointplot(
    y: np.array, palinstance, labels: Iterable = None, figsize: tuple = (8.0, 6.0)
):
    """Make a jointplot of the objective space

    Args:
        y (np.array): array with the objectives (measurements)
        palinstance (PALBase): "trained" PAL instance
        labels (Iterable, optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (8.0, 6.0).

    Returns:
        fig
    """

    num_targets = y.shape[1]
    fig, ax = plt.subplots(  # pylint:disable=invalid-name
        num_targets, num_targets, figsize=figsize
    )

    for row in range(num_targets):
        for column in range(num_targets):
            if row == column:
                plot_histogram(y[:, row], palinstance, ax[row, column])
            else:
                plot_pareto_front_2d(
                    y[:, row], y[:, column], palinstance, ax=ax[row, column]
                )

            ax[row, column].spines["top"].set_color("none")
            ax[row, column].spines["right"].set_color("none")
            ax[row, column].spines["left"].set_smart_bounds(True)
            ax[row, column].spines["bottom"].set_smart_bounds(True)

    if labels is None:
        labels = [f"Objective {i}" for i in range(num_targets)]
    else:
        assert len(labels) == num_targets

    for index in range(num_targets):
        ax[index, 0].set_ylabel(labels[index])
        ax[num_targets - 1, index].set_xlabel(labels[index])

    ax[0, 0].legend()
    fig.tight_layout()

    return fig
