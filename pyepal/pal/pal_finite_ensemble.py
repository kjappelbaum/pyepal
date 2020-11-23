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


"""Run PAL with the same models for finite ensemble models
and infinite width models (`PALNT`)
"""
from typing import Sequence

import numpy as np
from jax import random
from jax.api import grad, jit, vmap
from sklearn.preprocessing import StandardScaler

from ..models.nt import JaxOptimizer, NTModel
from .pal_base import PALBase


def _ensemble_train_one_finite_width(  # pylint:disable=too-many-arguments, too-many-locals
    i: int,
    models: Sequence[NTModel],
    design_space: np.ndarray,
    objectives: np.ndarray,
    sampled: np.ndarray,
    optimizers: Sequence[JaxOptimizer],
    key: random.PRNGKey,
    training_steps: int = 500,
    ensemble_size: int = 100,
):
    model = models[i]
    optimizer = optimizers[i]
    loss = jit(lambda params, x, y: 0.5 * np.mean((model.apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(optimizer.get_params(state), x, y))

    x_train = design_space[sampled[:, i]]
    y_train = objectives[sampled[:, i], i].reshape(-1, 1)

    def train_network(key):
        _, params = model.init_fn(key, (-1, x_train.shape[1]))
        opt_state = optimizer.opt_init(params)

        for i in range(training_steps):
            opt_state = optimizer.opt_update(
                i, grad_loss(opt_state, x_train, y_train), opt_state
            )

        return optimizer.get_params(opt_state)

    ensemble_key = random.split(key, ensemble_size)
    params = vmap(train_network)(ensemble_key)

    return params


def _ensemble_predict_one_finite_width(i: int, models: Sequence[NTModel], design_space):
    model = models[i]

    ensemble_func = vmap(model.apply_fn, (0, None))(model.params, design_space)

    mean_func = np.reshape(np.mean(ensemble_func, axis=0), (-1,))
    std_func = np.reshape(np.std(ensemble_func, axis=0), (-1,))

    return mean_func, std_func


class PALNTEnsemble(PALBase):
    """Use PAL with and ensemble of finite-width neural networks"""

    def __init__(self, *args, **kwargs):
        self.optimizer = kwargs.pop("optimizer")
        self.training_steps = kwargs.pop("training_steps", 500)
        self.ensemble_size = kwargs.pop("ensemble_size", 100)
        self.key = kwargs.pop("key", random.PRNGKey(10))
        self.design_space_scaler = StandardScaler()
        super().__init__(*args, **kwargs)

    def _train(self):
        ...

    def _predict(self):
        ...
