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


"""Use PAL with the neural tangent library.
This allows to perform
1. Exact Bayesian inference (NNGP)
2. Inference using gradient descent with MSE loss (NTK)
3. Inference using an ensemble of finite-width neural networks

Note that the neural tangent code usually assumes mean-zero Gaussians
"""

from typing import Sequence, Tuple

import jax.numpy as jnp
import neural_tangents as nt
import numpy as np

# from jax import random
# from jax.api import grad, jit, vmap
# from jax.experimental import optimizers
from neural_tangents import stax

from ..models.nt_stax import NT_TUPLE

# Probably we also need to add jaxlib to the dependencies
from .pal_base import PALBase

__all__ = ["PALNT", "NT_TUPLE"]

# We move those functions out of the class so that we can parallelize them
def _set_one_infinite_width_model(
    i: int,
    models: Sequence[NT_TUPLE],
    design_space: np.ndarray,
    objectives: np.ndarray,
    sampled: np.ndarray,
    # predict_fn_kwargs: dict = {},
) -> stax.Callable:
    model = models[i]
    kernel_fn = model.kernel_fn
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn,
        design_space[sampled[:, i]],
        objectives[sampled[:, i], i].reshape(-1, 1),
        # **predict_fn_kwargs,
    )

    return predict_fn


def _predict_one_infinite_width_model(
    i: int, models: Sequence[NT_TUPLE], design_space: np.ndarray, kernel
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    predict_fn = models[i].predict_fn
    mean, covariance = predict_fn(x_test=design_space, get=kernel, compute_cov=True)

    return mean.flatten(), np.sqrt(np.diag(covariance))


class PALNT(PALBase):
    """ε-PAL with neural tangents models
    (in the current implementation, one model per objective)"""

    def __init__(self, *args, **kwargs):
        """Construct the PALNT instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (Sequence[NT_TUPLE]): You need to provide a sequence of
                namedtuples NT_TUPLE (`pyepal.models.nt_stax.NT_TUPLE`).
                The elements of this tuple are the `apply_fn`, `init_fn`,
                `kernel_fn` and `predict_fn` (for latter you can typically
                provide `None`)
            ndim (int): Number of objectives
            epsilon (Union[list, float], optional): Epsilon hyperparameter.
                Defaults to 0.01.
            delta (float, optional): Delta hyperparameter. Defaults to 0.05.
            beta_scale (float, optional): Scaling parameter for beta.
                If not equal to 1, the theoretical guarantees do not necessarily hold.
                Also note that the parametrization depends on the kernel type.
                Defaults to 1/9.
            goals (List[str], optional): If a list, provide "min" for every objective
                that shall be minimized and "max" for every objective
                that shall be maximized. Defaults to None, which means
                that the code maximizes all objectives.
            coef_var_threshold (float, optional): Use only points with
                a coefficient of variation below this threshold
                in the classification step. Defaults to 3.
            kernel (str, optional): The kernel type you want to use ('nngp' or 'ntk').
                Defaults to 'nngp'.
        """
        self.kernel = kwargs.pop("kernel", "nngp")

        super().__init__(*args, **kwargs)

    def _set_data(self):
        for i in range(len(self.models)):
            predict_fn = _set_one_infinite_width_model(
                i, self.models, self.design_space, self.y, self.sampled
            )
            self.models[i].predict_fn = predict_fn

    def _train(self):
        pass

    def _predict(self):
        means, stds = [], []
        for i in range(len(self.models)):
            mean, std = _predict_one_infinite_width_model(
                i, self.models, self.design_space, self.kernel
            )
            means.append(mean)
            stds.append(std)

        self.means = np.hstack(means)
        self.std = np.hstack(stds)
