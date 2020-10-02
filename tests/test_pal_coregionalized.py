# -*- coding: utf-8 -*-
"""Testing the coregionalized PAL class"""
import numpy as np
import pytest

from pypal.models.gpr import build_coregionalized_model
from pypal.pal.pal_coregionalized import PALCoregionalized


def test_pal_coregionalized(make_random_dataset):
    """Test the PAL coregionalized model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    model = build_coregionalized_model(X, y)
    with pytest.raises(ValueError):
        pal_coregionalized = PALCoregionalized(X, ["m"], 3)

    pal_coregionalized = PALCoregionalized(X, [model], 3)

    slice_idx = np.array([0, 1, 2, 3, 4])
    pal_coregionalized.update_train_set(slice_idx, y[slice_idx])

    assert (
        pal_coregionalized.models[0].kern.B.kappa.values == np.array([0.5, 0.5, 0.5])
    ).all()

    pal_coregionalized._set_hyperparameters()  # pylint:disable=protected-access

    assert (
        pal_coregionalized._should_optimize_hyperparameters()  # pylint:disable=protected-access
    )

    assert (
        pal_coregionalized.models[0].kern.B.kappa.values != np.array([0.5, 0.5, 0.5])
    ).any()


def test_orchestration_run_one_step(make_random_dataset, binh_korn_points):
    """Test if the orchestration works.
    In the base class it should raise an error as without
    prediction function we cannot do anything
    """
    # This random dataset is not really ideal for a Pareto test as there's only one
    # optimal point it appears to me
    X, y = make_random_dataset  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model = build_coregionalized_model(X[sample_idx], y[sample_idx])
    palinstance = PALCoregionalized(X, [model], 3, beta_scale=1)

    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model = build_coregionalized_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx])

    palinstance = PALCoregionalized(X_binh_korn, [model], 2, beta_scale=1)

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert sum(palinstance.sampled_idx) > 0
    assert sum(palinstance.unclassified) > 0
    assert sum(palinstance.discarded) > 0
