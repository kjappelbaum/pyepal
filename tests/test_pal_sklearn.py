# -*- coding: utf-8 -*-
"""Testing the PAL sklearn class"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from PyPAL.pal.pal_sklearn import PALSklearn


def test_pal_sklearn(make_random_dataset):
    """Test that we can create a instanec of the PAL sklearn class"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    gpr = GaussianProcessRegressor(RBF(), normalize_y=True, n_restarts_optimizer=1)
    pal_sklearn_instance = PALSklearn(X, [gpr, gpr, gpr], 3)
    pal_sklearn_instance.update_train_set(
        np.array([1, 2, 3, 4, 5]), y[np.array([1, 2, 3, 4, 5]), :]
    )
    assert pal_sklearn_instance.models[0].kernel.length_scale == 1
    pal_sklearn_instance._train()  # pylint:disable=protected-access
    assert pal_sklearn_instance.models[0].kernel_.length_scale != 1
