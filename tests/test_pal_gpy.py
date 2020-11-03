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

"""Testing the PALGPy class"""
import numpy as np
import pytest

from pyepal.models.gpr import build_model
from pyepal.pal.pal_gpy import PALGPy


def test_pal_gpy(make_random_dataset):
    """Test basic functionality of the PALGpy class"""
    with pytest.raises(TypeError):
        palgpy_instance = PALGPy()

    X, y = make_random_dataset  # pylint:disable=invalid-name

    with pytest.raises(ValueError):
        palgpy_instance = PALGPy(X, ["m", "m", "m"], 3)

    m0 = build_model(X, y, 0)  # pylint:disable=invalid-name
    m1 = build_model(X, y, 1)  # pylint:disable=invalid-name
    m2 = build_model(X, y, 2)  # pylint:disable=invalid-name

    palgpy_instance = PALGPy(X, [m0, m1, m2], 3, delta=0.01)
    palgpy_instance.cross_val_points = 0
    assert palgpy_instance.restarts == 20

    palgpy_instance.update_train_set(
        np.array([1, 2, 3, 4, 5]), y[np.array([1, 2, 3, 4, 5]), :]
    )
    assert palgpy_instance.models[0].kern.variance.values[0] == 1
    palgpy_instance._train()  # pylint:disable=protected-access
    assert palgpy_instance.models[0].kern.variance.values[0] == 1
    palgpy_instance._set_hyperparameters()  # pylint:disable=protected-access
    assert palgpy_instance.models[0].kern.variance.values[0] != 1


def test_orchestration_run_one_step(make_random_dataset, binh_korn_points):
    """Test if the orchestration works.
    In the base class it should raise an error as without
    prediction function we cannot do anything
    """
    np.random.seed(10)
    # This random dataset is not really ideal for a Pareto test as there's only one
    # optimal point it appears to me
    X, y = make_random_dataset  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model_0 = build_model(X[sample_idx], y[sample_idx], 0)
    model_1 = build_model(X[sample_idx], y[sample_idx], 1)
    model_2 = build_model(X[sample_idx], y[sample_idx], 2)
    palinstance = PALGPy(
        X,
        [model_0, model_1, model_2],
        3,
        beta_scale=1,
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )
    palinstance.cross_val_points = 0

    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    if idx is not None:
        assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0


def test_orchestration_run_one_step_parallel(binh_korn_points):
    """Test if the parallelization works"""
    np.random.seed(10)
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.01,
        delta=0.01,
        n_jobs=2,
        restarts=3,
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0


def test_minimize_run_one_step(binh_korn_points):
    """Test that the minimization argument does not behave weirdly"""
    np.random.seed(10)
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    y_binh_korn = -y_binh_korn

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        goals=["min", "min"],
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        goals=[-1, -1],
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0

    y_binh_korn = y_binh_korn * np.array([-1, 1])

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        goals=[1, -1],
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0

    # Testing batch sampling

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        goals=[1, -1],
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    assert len(idx) == 10
    assert len(np.unique(idx)) == 10
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0


def test_orchestration_run_one_step_missing_data(binh_korn_points):
    """Test that the model also works with missing observations"""
    np.random.seed(10)
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )
    palinstance.cross_val_points = 0
    # make some of the observations missing
    y_binh_korn[:10, 1] = np.nan

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    idx = palinstance.run_one_step()
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0


def test_crossvalidate(binh_korn_points):
    """Test the crossvalidation routine"""
    np.random.seed(10)

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.01,
        delta=0.01,
        restarts=3,
    )
    palinstance.cross_val_points = 2
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    original_sample_mask = palinstance.sampled

    cross_val_error = palinstance._crossvalidate()  # pylint:disable=protected-access
    assert (palinstance.sampled_indices == sample_idx).all()
    assert (palinstance.sampled == original_sample_mask).all()

    assert isinstance(cross_val_error, float)
    assert np.abs(cross_val_error) > 0


def test_epsilon_sensitivity(binh_korn_points):
    """Simple test if the epsilon changes the result in an expected way"""
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance0 = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.0,
        delta=0.01,
        restarts=3,
    )
    palinstance0.cross_val_points = 0
    palinstance0.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = palinstance0.run_one_step()
    palinstance1 = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.05,
        delta=0.01,
        restarts=3,
    )
    palinstance1.cross_val_points = 0
    palinstance1.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = palinstance1.run_one_step()

    palinstance2 = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.1,
        delta=0.01,
        restarts=3,
    )
    palinstance2.cross_val_points = 0
    palinstance2.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = palinstance2.run_one_step()

    assert palinstance0.number_discarded_points == 0
    assert palinstance1.number_discarded_points == 0
    assert palinstance2.number_discarded_points == 0

    assert (
        palinstance0.number_unclassified_points
        > palinstance1.number_unclassified_points
    )
    assert (
        palinstance1.number_unclassified_points
        > palinstance2.number_unclassified_points
    )
