# -*- coding: utf-8 -*-
"""Testing the PALGPy class"""
import numpy as np
import pytest

from pypal.models.gpr import build_model
from pypal.pal.pal_gpy import PALGPy


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

    palgpy_instance = PALGPy(X, [m0, m1, m2], 3)
    assert palgpy_instance.restarts == 20
    assert not palgpy_instance.parallel

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
    # This random dataset is not really ideal for a Pareto test as there's only one
    # optimal point it appears to me
    X, y = make_random_dataset  # pylint:disable=invalid-name
    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model_0 = build_model(X[sample_idx], y[sample_idx], 0)
    model_1 = build_model(X[sample_idx], y[sample_idx], 1)
    model_2 = build_model(X[sample_idx], y[sample_idx], 2)
    palinstance = PALGPy(X, [model_0, model_1, model_2], 3, beta_scale=1)

    palinstance.update_train_set(sample_idx, y[sample_idx])
    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(X_binh_korn, [model_0, model_1], 2, beta_scale=1)

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert sum(palinstance.sampled) > 0
    assert sum(palinstance.discarded) == 0