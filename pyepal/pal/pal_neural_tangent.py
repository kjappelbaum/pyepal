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

__all__ = ["PALNT"]


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
        super().__init__(*args, **kwargs)

    def _train(self):
        ...

    def _predict(self):
        ...
