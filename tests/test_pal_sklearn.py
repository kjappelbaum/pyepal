# -*- coding: utf-8 -*-
"""Testing the PAL sklearn class"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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
    # This random dataset is not really ideal for a Pareto test as there's only one
    # optimal point it appears to me
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=2)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=2)
    gpr_2 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=2)

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    palinstance = PALSklearn(X, [gpr_0, gpr_1, gpr_2], 3, beta_scale=1)

    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0

    # Batch sampling binh-korn

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0

    # scaling up beta
    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1 / 9)

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step(batch_size=10)
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0


def test_orchestration_run_one_step_missing_data(binh_korn_points):
    """Test that the model also works with missing observations"""
    gpr_0 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=2)
    gpr_1 = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=2)

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

    palinstance = PALSklearn(X_binh_korn, [gpr_0, gpr_1], 2, beta_scale=1)

    # make some of the observations missing
    y_binh_korn[:10, 1] = np.nan

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) == 0
