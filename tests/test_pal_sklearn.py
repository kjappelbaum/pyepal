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

"""Testing the PAL sklearn class"""
import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from pyepal.pal.pal_sklearn import PALSklearn

np.random.seed(10)


def test_pal_sklearn(make_random_dataset):
    """Test that we can create a instanec of the PAL sklearn class"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    pal_sklearn_instance = PALSklearn(X, [gpr, gpr, gpr], 3)
    pal_sklearn_instance.update_train_set(
        np.array([1, 2, 3, 4, 5]), y[np.array([1, 2, 3, 4, 5]), :]
    )
    assert pal_sklearn_instance.models[0].kernel.length_scale == 1
    pal_sklearn_instance._train()  # pylint:disable=protected-access
    assert pal_sklearn_instance.models[0].kernel_.length_scale != 1


def test_gridsearch_object(binh_korn_points):
    """Test the initialization of PALSklearn with a GridsearchCV object"""
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    grid_search_0 = GridSearchCV(
        GaussianProcessRegressor(), {"kernel": [RBF(), Matern()]}
    )
    grid_search_1 = GridSearchCV(
        GaussianProcessRegressor(), {"kernel": [RBF(), Matern()]}
    )

    with pytest.raises(ValueError):
        palinstance = PALSklearn(
            X_binh_korn, [grid_search_0, grid_search_1], 2, beta_scale=1
        )

    grid_search_0.fit(X_binh_korn, y_binh_korn[:, 0])
    grid_search_1.fit(X_binh_korn, y_binh_korn[:, 1])

    palinstance = PALSklearn(
        X_binh_korn, [grid_search_0, grid_search_1], 2, beta_scale=1
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for model in palinstance.models:
        assert check_is_fitted(model) is None


def test_orchestration_run_one_step(make_random_dataset, binh_korn_points):
    """Test if the orchestration works.
    In the base class it should raise an error as without
    prediction function we cannot do anything
    """
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    gpr_2 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    palinstance = PALSklearn(X, [gpr_0, gpr_1, gpr_2], 3, beta_scale=1)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points == len(sample_idx)
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
    for model in palinstance.models:
        assert check_is_fitted(model) is None


def test_augment_design_space(make_random_dataset):
    """Test if the reclassification step in the design step
    agumentation method works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    gpr_2 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=5)
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    palinstance = PALSklearn(X, [gpr_0, gpr_1, gpr_2], 3, beta_scale=1)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y[sample_idx])
    _ = palinstance.run_one_step()

    X_new = X + 1  # pylint:disable=invalid-name
    palinstance.augment_design_space(X_new, classify=True, clean_classify=False)
    assert palinstance.number_design_points == 200
    assert palinstance.number_sampled_points == len(sample_idx)

    # Adding new design points should not mess up with the models
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    # Now, test the `clean_classify` flag
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=3)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=3)
    gpr_2 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=3)
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    palinstance = PALSklearn(X, [gpr_0, gpr_1, gpr_2], 3, beta_scale=1)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y[sample_idx])
    _ = palinstance.run_one_step()

    X_new = X + np.full((1, 10), 1)  # pylint:disable=invalid-name
    palinstance.augment_design_space(X_new)
    assert palinstance.number_design_points == 200
    assert palinstance.number_sampled_points == len(sample_idx)


def test_augment_design_space_bk(binh_korn_points, binh_korn_points_finer):
    """Test the augment function by using a finer sampling of the Binh-Korn function
    for augmentation"""
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    (
        X_binh_korn_finer,  # pylint:disable=invalid-name
        _,
    ) = binh_korn_points_finer
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    new_idx = palinstance.run_one_step()
    palinstance.update_train_set(new_idx, y_binh_korn[new_idx])
    number_pareto_optimal_points_old = palinstance.number_pareto_optimal_points
    palinstance.augment_design_space(X_binh_korn_finer)
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_pareto_optimal_points > number_pareto_optimal_points_old


def test_orchestration_run_one_step_batch(  # pylint:disable=too-many-statements
    binh_korn_points,
):
    """Test the batch sampling"""
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    assert len(idx) == 10
    assert len(np.unique(idx)) == 10
    for index in idx:
        assert index not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    # scaling up beta
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 3)
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    for index in idx:
        assert index not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    # smaller initial set
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 3)
    palinstance.cross_val_points = 0
    sample_idx = np.array([1, 10, 20, 40, 70, 90])
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    for index in idx:
        assert index not in [1, 10, 20, 40, 70, 90]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    # smaller initial set and beta scale
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9)
    palinstance.cross_val_points = 0
    sample_idx = np.array([1, 10, 20, 40, 70, 90])
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    for index in idx:
        assert index not in [1, 10, 20, 40, 70, 90]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    # smaller initial set and beta scale and different kernel
    gpr_0 = GaussianProcessRegressor(
        Matern(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        Matern(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9)
    palinstance.cross_val_points = 0
    sample_idx = np.array([1, 10, 20, 40, 70, 90])
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    for index in idx:
        assert index not in [1, 10, 20, 40, 70, 90]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0

    for model in palinstance.models:
        assert check_is_fitted(model) is None


def test_orchestration_run_one_step_parallel(binh_korn_points):
    """Test the parallel processing"""
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    gpr_0 = GaussianProcessRegressor(
        Matern(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        Matern(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9, n_jobs=2)
    sample_idx = np.array([1, 10, 20, 40, 70, 90])
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    palinstance.cross_val_points = 0
    idx = palinstance.run_one_step(batch_size=10)
    for index in idx:
        assert index not in [1, 10, 20, 40, 70, 90]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0

    for model in palinstance.models:
        assert check_is_fitted(model) is None


def test_orchestration_run_one_step_missing_data(binh_korn_points):
    """Test that the model also works with missing observations"""
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)
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
    gpr_0 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )
    gpr_1 = GaussianProcessRegressor(
        RBF(), normalize_y=True, n_restarts_optimizer=5, random_state=10
    )

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    original_sample_mask = palinstance.sampled

    cross_val_error = palinstance._crossvalidate()  # pylint:disable=protected-access
    assert (palinstance.sampled_indices == sample_idx).all()
    assert (palinstance.sampled == original_sample_mask).all()

    assert isinstance(cross_val_error, float)
    assert np.abs(cross_val_error) > 0
