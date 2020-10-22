# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
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

# pylint:disable=invalid-name
"""
Wrappers for Gaussian Process Regression models.

We typically use the GPy package as it offers most flexibility
for Gaussian processes in Python.
Typically, we use automatic relevance determination (ARD),
where one lengthscale parameter per input dimension is used.

If your task requires training on larger training sets,
you might consider replacing the models with their sparse
version but for the epsilon-PAL algorithm this typically shouldn't
be needed.

For kernel selection, you can have a look at
https://www.cs.toronto.edu/~duvenaud/cookbook/
Matérn, RBF and RationalQuadrat are good quick and dirty solutions
but have their caveats
"""

from typing import Tuple

import GPy
import numpy as np

from .coregionalized import GPCoregionalizedRegression


def _get_matern_32_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Matern32:
    """Matern-3/2 kernel without ARD"""
    return GPy.kern.Matern32(NFEAT, ARD=ARD, **kwargs)


def _get_matern_52_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Matern52:
    """Matern-5/2 kernel without ARD"""
    return GPy.kern.Matern52(NFEAT, ARD=ARD, **kwargs)


def _get_ratquad_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.RatQuad:
    """Rational quadratic kernel without ARD"""
    return GPy.kern.RatQuad(NFEAT, ARD=ARD, **kwargs)


def build_coregionalized_model(
    X_train: np.array, y_train: np.array, kernel=None, **kwargs
) -> GPy.models.GPCoregionalizedRegression:
    """Wrapper for building a coregionalized GPR, it will have as many
    outputs as y_train.shape[1].
    Each output will have its own noise term"""
    NFEAT = X_train.shape[1]
    num_targets = y_train.shape[1]
    if isinstance(kernel, GPy.kern.src.kern.Kern):
        K = kernel
    else:
        K = _get_matern_52_kernel(NFEAT)
    icm = GPy.util.multioutput.ICM(input_dim=NFEAT, num_outputs=num_targets, kernel=K)

    target_list = [y_train[:, i].reshape(-1, 1) for i in range(num_targets)]
    m = GPCoregionalizedRegression(
        [X_train] * num_targets, target_list, kernel=icm, normalizer=True, **kwargs
    )
    # We constrain the variance of the RBF/Matern ..
    # as the variance is now encoded in the kappa B of the ICM
    # Not constraining it would lead to a degeneracy
    m[".*ICM.*.variance"].constrain_fixed(1.0)
    # initialize the noise model
    m[".*Gaussian_noise_*"] = 0.1
    return m


def build_model(
    X_train: np.array, y_train: np.array, index: int = 0, kernel=None, **kwargs
) -> GPy.models.GPRegression:
    """Build a single-output GPR model"""
    NFEAT = X_train.shape[1]
    if isinstance(kernel, GPy.kern.src.kern.Kern):
        K = kernel
    else:
        K = _get_matern_52_kernel(NFEAT)
    m = GPy.models.GPRegression(
        X_train, y_train[:, index].reshape(-1, 1), kernel=K, normalizer=True, **kwargs
    )
    m[".*Gaussian_noise_*"] = 0.1
    return m


def predict(model: GPy.models.GPRegression, X: np.array) -> Tuple[np.array, np.array]:
    """Wrapper function for the prediction method of a GPy regression model.
    It return the standard deviation instead of the variance"""
    assert isinstance(
        model, GPy.models.GPRegression
    ), "This wrapper function is written for GPy.models.GPRegression"
    mu, var = model.predict(X)
    return mu, np.sqrt(var)


def predict_coregionalized(
    model: GPy.models.GPCoregionalizedRegression, X: np.array, index: int = 0
) -> Tuple[np.array, np.array]:
    """Wrapper function for the prediction method of a coregionalized
    GPy regression model.
    It return the standard deviation instead of the variance"""
    assert isinstance(
        model, (GPy.models.GPCoregionalizedRegression, GPCoregionalizedRegression)
    ), "This wrapper function is written for GPy.models.GPCoregionalizedRegression"
    newX = np.hstack([X, index * np.ones_like(X)])
    mu_c0, var_c0 = model.predict(
        newX,
        Y_metadata={"output_index": index * np.ones((newX.shape[0], 1)).astype(int)},
    )

    return mu_c0, np.sqrt(var_c0)


def set_xy_coregionalized(model, X, y, mask=None):
    """Wrapper to update a coregionalized model with new data"""
    num_target = y.shape[1]
    if mask is None:
        X_array = [X] * num_target
        y_array = [y[:, i].reshape(-1, 1) for i in range(num_target)]

    else:
        X_array = [X[mask[:, i]] for i in range(num_target)]
        y_array = [y[mask[:, i], i].reshape(-1, 1) for i in range(num_target)]

    model.set_XY(X_array, y_array)

    return model
