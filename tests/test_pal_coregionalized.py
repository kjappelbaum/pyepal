# -*- coding: utf-8 -*-
# Copyright 2022 PyePAL authors
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

"""Testing the coregionalized PAL class"""
import numpy as np
import pytest

from pyepal.models.gpr import build_coregionalized_model
from pyepal.pal.pal_coregionalized import PALCoregionalized


def test_pal_coregionalized(make_random_dataset):
    """Test the PAL coregionalized model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    model = build_coregionalized_model(X, y)
    with pytest.raises(ValueError):
        pal_coregionalized = PALCoregionalized(X, ["m"], 3, epsilon=0.01)

    pal_coregionalized = PALCoregionalized(X, [model], 3, epsilon=0.01)

    slice_idx = np.array([0, 1, 2, 3, 4])
    pal_coregionalized.update_train_set(slice_idx, y[slice_idx])

    assert (pal_coregionalized.models[0].kern.B.kappa.values == np.array([0.5, 0.5, 0.5])).all()

    pal_coregionalized._set_hyperparameters()  # pylint:disable=protected-access

    assert pal_coregionalized._should_optimize_hyperparameters()  # pylint:disable=protected-access

    assert (pal_coregionalized.models[0].kern.B.kappa.values != np.array([0.5, 0.5, 0.5])).any()


def test_orchestration_run_one_step(make_random_dataset, binh_korn_points):
    """Test if the orchestration works.
    In the base class it should raise an error as without
    prediction function we cannot do anything
    """
    # This random dataset is not really ideal for a Pareto test as there's only one
    # optimal point it appears to me
    np.random.seed(10)
    X, y = make_random_dataset  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model = build_coregionalized_model(X[sample_idx], y[sample_idx])
    palinstance = PALCoregionalized(
        X, [model], 3, beta_scale=1, epsilon=0.01, delta=0.01, restarts=3
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    # Sometimes, the model might classify everything with certainity with the
    # initial set
    if idx is not None:
        assert len(idx) == 1
        assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model = build_coregionalized_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx])

    palinstance = PALCoregionalized(
        X_binh_korn, [model], 2, beta_scale=1, epsilon=0.01, delta=0.01, restarts=3
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0

    # testing batch sampling
    model = build_coregionalized_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx])
    palinstance = PALCoregionalized(
        X_binh_korn, [model], 2, beta_scale=1, epsilon=0.01, delta=0.01, restarts=3
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)

    if idx is not None:
        assert len(idx) == 10
        assert len(np.unique(idx)) == 10
        assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0


def test_orchestration_run_one_step_missing_data(binh_korn_points):
    """Test that the model also works with missing observations"""
    np.random.seed(10)
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    model = build_coregionalized_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)

    palinstance = PALCoregionalized(
        X_binh_korn, [model], 2, beta_scale=1, epsilon=0.01, delta=0.01, restarts=3
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
