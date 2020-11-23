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

"""Testing the PAL base class"""
# pylint:disable=protected-access
import numpy as np
import pytest

from pyepal.pal.pal_base import PALBase


def test_pal_base(make_random_dataset):
    """Testing basic functionality of the PAL base class"""
    palinstance = PALBase(make_random_dataset[0], ["model"], 3)
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_pareto_optimal_points == 0
    assert palinstance.number_unclassified_points == 100
    assert palinstance.number_sampled_points == 0
    assert palinstance.number_design_points == 100
    assert palinstance.should_cross_validate()
    assert len(palinstance.discarded_points) == 0
    assert len(palinstance.pareto_optimal_points) == 0
    assert len(palinstance.unclassified_points) == 100

    with pytest.raises(ValueError):
        palinstance.hyperrectangle_sizes  # pylint:disable=pointless-statement

    assert (
        str(palinstance)
        == "pyepal at iteration 1. \
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
    assert palinstance.sampled.sum() == 0

    palinstance.update_train_set(np.array([0]), y[0, :].reshape(-1, 3))
    assert palinstance.sampled_indices == np.array([0])
    assert palinstance.number_sampled_points == 1
    assert (palinstance.y[0] == y[0, :]).all()


def test_augment_design_space(make_random_dataset):
    """Testing the basic functionality of the augmentation method
    Does NOT test the re-classification step, which needs a model"""
    X, _ = make_random_dataset  # pylint: disable=invalid-name
    X_augmented = np.vstack([X, X])  # pylint: disable=invalid-name
    palinstance = PALBase(X, ["model"], 3)
    # Iteration count to low
    with pytest.raises(ValueError):
        palinstance.augment_design_space(X_augmented)

    palinstance.iteration = 2
    # Incorrect shape
    with pytest.raises(AssertionError):
        palinstance.augment_design_space(X_augmented[:, 2])

    with pytest.raises(ValueError):
        palinstance.augment_design_space(X_augmented[:, :2])

    #  Mock that we already ran that
    lows = np.zeros((100, 3))
    highs = np.zeros((100, 3))

    means = np.full((100, 3), 1)
    palinstance.means = means
    palinstance.std = np.full((100, 3), 0.1)
    pareto_optimal = np.array([False] * 98 + [True, True])
    sampled = np.array([[False] * 3, [False] * 3, [False] * 3, [False] * 3])
    unclassified = np.array([True] * 98 + [False, False])

    palinstance.rectangle_lows = lows
    palinstance.rectangle_ups = highs
    palinstance.sampled = sampled
    palinstance.pareto_optimal = pareto_optimal
    palinstance.unclassified = unclassified

    # As we do not have a model, we cannot test the classification
    palinstance.augment_design_space(X_augmented, clean_classify=False)
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_pareto_optimal_points == 2
    assert palinstance.number_unclassified_points == 298
    assert palinstance.number_sampled_points == 0
    assert palinstance.number_design_points == 300
    assert len(palinstance.means) == 300
    assert len(palinstance.std) == 300


def test_beta_update(make_random_dataset):
    """testing that the beta update works"""
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 3)

    assert palinstance.beta is None

    palinstance._update_beta()
    assert palinstance.beta is not None

    assert palinstance.beta == 1 / 9 * 2 * np.log(
        3 * 100 * np.square(np.pi) * np.square(2) / (6 * 0.05)
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


def test_sample(make_random_dataset):
    """Test the sampling"""
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 4)

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
    palinstance.means = means
    pareto_optimal = np.array([False, False, True, True])
    sampled = np.array([[False] * 4, [False] * 4, [False] * 4, [False] * 4])
    unclassified = np.array([True, True, False, False])

    palinstance.rectangle_lows = lows
    palinstance.rectangle_ups = highs
    palinstance.sampled = sampled
    palinstance.pareto_optimal = pareto_optimal
    palinstance.unclassified = unclassified

    sampled_idx = palinstance.sample()
    assert sampled_idx == 2

    pareto_optimal = np.array([False, False, True, True])
    sampled = np.array([[False] * 4, [False] * 4, [False] * 4, [False] * 4])
    unclassified = np.array([True, True, False, False])

    palinstance.sampled = sampled
    palinstance.pareto_optimal = pareto_optimal
    palinstance.unclassified = unclassified

    sampled_idx = palinstance.sample()
    assert sampled_idx == 2

    pareto_optimal = np.array([False, False, True, True])
    sampled = np.array([[False] * 4, [False] * 4, [True] * 4, [False] * 4])
    unclassified = np.array([True, True, False, False])

    palinstance.sampled = sampled
    palinstance.pareto_optimal = pareto_optimal
    palinstance.unclassified = unclassified

    sampled_idx = palinstance.sample()
    assert sampled_idx == 1

    pareto_optimal = np.array([False, False, False, True])
    sampled = np.array([[False] * 4, [False] * 4, [True] * 4, [False] * 4])
    unclassified = np.array([True, True, False, False])

    palinstance.sampled = sampled
    palinstance.pareto_optimal = pareto_optimal
    palinstance.unclassified = unclassified

    sampled_idx = palinstance.sample()
    assert sampled_idx == 1


def test__update_hyperrectangles(make_random_dataset):
    """Testing if the updating of the hyperrectangles works as expected.
    As above, the core functionality is tested in for the function with the logic.
    Here, we're more interested in seeing how it works with the class object
    """
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 4, beta_scale=1)

    palinstance.means = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    palinstance.std = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
    with pytest.raises(TypeError):
        # Beta is not defined
        palinstance._update_hyperrectangles()

    palinstance._update_beta()
    palinstance._update_hyperrectangles()

    assert palinstance.rectangle_lows is not None
    assert palinstance.rectangle_ups is not None

    assert palinstance.rectangle_lows[0][0] == 0
    assert palinstance.rectangle_ups[0][0] == 0
    assert palinstance.rectangle_lows[0][2] == 1
    assert palinstance.rectangle_ups[0][2] == 1

    assert palinstance.rectangle_lows[1][0] < -1
    assert palinstance.rectangle_ups[1][0] > 1

    assert len(palinstance.hyperrectangle_sizes) == len(palinstance.means)


def test_orchestration_run_one_step(make_random_dataset):
    """Test if the orchestration works.
    In the base class it should raise an error as without
    prediction function we cannot do anything
    """
    X, y = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 3, beta_scale=1)
    sample_idx = np.array([1, 2, 3, 4])
    palinstance.update_train_set(sample_idx, y[sample_idx])
    with pytest.raises(NotImplementedError):
        _ = palinstance.run_one_step()


def test__replace_by_measurements(make_random_dataset):
    """Test that replacing the mean/std by the measured ones works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X, ["model"], 3, beta_scale=1)
    assert palinstance.measurement_uncertainty.sum() == 0
    sample_idx = np.array([1, 2, 3, 4])
    palinstance.update_train_set(sample_idx, y[sample_idx], y[sample_idx])
    palinstance.means = palinstance.measurement_uncertainty
    palinstance.std = palinstance.measurement_uncertainty
    palinstance._replace_by_measurements()
    assert (palinstance.y == palinstance.std).all()


def test__update_coef_var_mask(make_random_dataset):
    """Test that the coefficient of variation mask works as expected"""
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    palinstance = PALBase(X[:2], ["model"], 3, beta_scale=1)

    palinstance.means = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    palinstance.std = np.array([[0, 0, 0, 0], [3, 1, 1, 1]])

    assert (palinstance.coef_var_mask == np.array([True, True])).all()
    palinstance._update_coef_var_mask()

    assert (palinstance.coef_var_mask == np.array([True, False])).all()

    palinstance.means = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    palinstance.std = np.array([[0, 0, 0, 0], [3, 1, 1, 1]])

    assert (palinstance.coef_var_mask == np.array([True, False])).all()
