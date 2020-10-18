# -*- coding: utf-8 -*-
"""Testing the PAL sklearn class"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.utils.validation import check_is_fitted

from pypal.pal.pal_sklearn import PALSklearn


def test_pal_sklearn(make_random_dataset):
    """Test that we can create a instanec of the PAL sklearn class"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=2)
    pal_sklearn_instance = PALSklearn(X, [gpr, gpr, gpr], 3)
    pal_sklearn_instance.update_train_set(
        np.array([1, 2, 3, 4, 5]), y[np.array([1, 2, 3, 4, 5]), :]
    )
    assert pal_sklearn_instance.models[0].kernel.length_scale == 1
    pal_sklearn_instance._train()  # pylint:disable=protected-access
    assert pal_sklearn_instance.models[0].kernel_.length_scale != 1


def test_orchestration_run_one_step(make_random_dataset, binh_korn_points):
    """Test if the orchestration works.
    In the base class it should raise an error as without
    prediction function we cannot do anything
    """
    np.random.seed(10)
    # This random dataset is not really ideal for a Pareto test as there's only one
    # optimal point it appears to me
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_2 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    palinstance = PALSklearn(X, [gpr_0, gpr_1, gpr_2], 3, beta_scale=1)

    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for model in palinstance.models:
        assert check_is_fitted(model) is None

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert len(idx) == 1
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
    for model in palinstance.models:
        assert check_is_fitted(model) is None


def test_orchestration_run_one_step_batch(  # pylint:disable=too-many-statements
    binh_korn_points,
):
    """Test the batch sampling"""
    np.random.seed(10)
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)

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
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 3)

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
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 3)
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
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9)
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
    gpr_0 = GaussianProcessRegressor(Matern(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(Matern(), normalize_y=True, n_restarts_optimizer=4)
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9)
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
    np.random.seed(10)
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    gpr_0 = GaussianProcessRegressor(Matern(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(Matern(), normalize_y=True, n_restarts_optimizer=4)
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9, n_jobs=2)
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


def test_orchestration_run_one_step_missing_data(binh_korn_points):
    """Test that the model also works with missing observations"""
    np.random.seed(10)
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)

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
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=4)

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
