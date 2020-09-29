# -*- coding: utf-8 -*-
"""Testing the PAL base class"""
# pylint:disable=protected-access
import numpy as np
import pytest

from pypal.pal.pal_base import PALBase


def test_pal_base(make_random_dataset):
    """Testing basic functionality of the PAL base class"""
    palinstance = PALBase(make_random_dataset[0], ["model"], 3)
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_pareto_optimal_points == 0
    assert palinstance.number_unclassified_points == 100
    assert palinstance.number_sampled_points == 0

    assert len(palinstance.discarded_points) == 0
    assert len(palinstance.pareto_optimal_points) == 0
    assert len(palinstance.unclassified_points) == 100

    assert (
        str(palinstance)
        == "pypal at iteration 0. \
        0 Pareto optimal points, \
        0 discarded points, \
        100 unclassified points."
    )
    assert (
        palinstance._log()
        == "pypal at iteration 0. \
        0 Pareto optimal points, \
        0 discarded points, \
        100 unclassified points."
    )

    assert palinstance._should_optimize_hyperparameters()
    assert not palinstance._has_train_set

    with pytest.raises(ValueError):
        palinstance.sample()

    assert palinstance.y.shape == (100, 3)


def test_update_train_set(make_random_dataset):
    """Check if the update of the training set works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 3)
    assert not palinstance._has_train_set
    assert sum(palinstance.sampled) == 0

    palinstance.update_train_set(np.array([0]), y[0, :].reshape(-1, 3))
    assert palinstance.sampled_idx == np.array([0])
    assert palinstance.number_sampled_points == 1
    assert (palinstance.y[0] == y[0, :]).all()


def test_beta_update(make_random_dataset):
    """testing that the beta update works"""
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 3)

    assert palinstance.beta is None

    palinstance._update_beta()
    assert palinstance.beta is not None

    assert palinstance.beta == 1 / 16 * 2 * np.log(
        3 * 100 * np.square(np.pi) * np.square(1) / (6 * 0.05)
    )


def test_turn_to_maximization(make_random_dataset):
    """Test that flipping the sign for minimization problems works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 3)

    palinstance.update_train_set(np.array([0]), y[0, :].reshape(-1, 3))
    assert (palinstance.y[0] == y[0, :]).all()
    assert (palinstance._y[0] == y[0, :]).all()

    palinstance = PALBase(X, ["model"], 3, goals=[1, 1, -1])

    palinstance.update_train_set(np.array([0]), y[0, :].reshape(-1, 3))
    assert (palinstance.y[0] == y[0, :] * np.array([1, 1, -1])).all()
    assert (palinstance._y[0] == y[0, :]).all()
