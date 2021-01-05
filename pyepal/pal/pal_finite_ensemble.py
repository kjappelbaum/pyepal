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
from sklearn.preprocessing import StandardScaler

from ..models.nt import JaxOptimizer, NTModel
from .pal_base import PALBase
from .validate_inputs import (
    validate_nt_models,
    validate_optimizers,
    validate_positive_integer_list,
)


# Again, the idea of having the core as pure functions outside of the class is that
# we could parallelize it easier in this way
def _ensemble_train_one_finite_width(  # pylint:disable=too-many-arguments, too-many-locals
    i: int,
    models: Sequence[NTModel],
    design_space: np.ndarray,
    objectives: np.ndarray,
    sampled: np.ndarray,
    optimizers: Sequence[JaxOptimizer],
    key: object,
    training_steps: Sequence[int],
    ensemble_size: Sequence[int],
):
    from jax import random  # pylint:disable=import-outside-toplevel
    from jax.api import grad, jit, vmap  # pylint:disable=import-outside-toplevel

    model = models[i]
    optimizer = optimizers[i]
    loss = jit(lambda params, x, y: 0.5 * np.mean((model.apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(optimizer.get_params(state), x, y))

    x_train = design_space[sampled[:, i]]

    scaler = StandardScaler()
    y_train = scaler.fit_transform(objectives[sampled[:, i], i].reshape(-1, 1))

    def train_network(key):
        _, params = model.init_fn(key, (-1, x_train.shape[1]))
        opt_state = optimizer.opt_init(params)

        for j in range(training_steps[i]):
            opt_state = optimizer.opt_update(
                j, grad_loss(opt_state, x_train, y_train), opt_state
            )

        return optimizer.get_params(opt_state)

    ensemble_key = random.split(key, ensemble_size[i])
    params = vmap(train_network)(ensemble_key)

    return params, scaler


def _ensemble_predict_one_finite_width(i: int, models: Sequence[NTModel], design_space):
    from jax.api import vmap  # pylint:disable=import-outside-toplevel

    model = models[i]

    ensemble_func = vmap(model.apply_fn, (0, None))(model.params, design_space)

    mean_func = np.reshape(np.mean(ensemble_func, axis=0), (-1,))
    std_func = np.reshape(np.std(ensemble_func, axis=0), (-1,))

    return mean_func, std_func


__all__ = ["PALJaxEnsemble", "NTModel", "JaxOptimizer"]


class PALJaxEnsemble(PALBase):  # pylint:disable=too-many-instance-attributes
    """Use PAL with and ensemble of finite-width neural networks.
    Note that we current assume that there is one model per output,
    i.e., we did not yet implement multihead support.
    """

    def __init__(self, *args, **kwargs):
        """Construct the PALJaxEnsemble instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (Sequence[NTModel]): You need to provide a sequence of
                 NTModel (`pyepal.models.nt.NTModel`).
                The elements of this dataclass are the `apply_fn`, `init_fn`,
                `kernel_fn` and `predict_fn` (for latter you can typically
                provide `None`).
                Can be constructed with
                :py:func:`pyepal.pal.models.nt.build_dense_network`.
            optimizer (Union[JaxOptimizer, Sequence[JaxOptimizer]]):
                Sequence of dataclasses with functions for a JAX optimizer,
                can be constructed with :py:func:`pyepal.pal.models.nt.get_optimizer`.
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
            key (int): Seed to generate the key for the JAX
                pseudo-random number generator. Defaults to 10.
            training_steps (Union[int, Sequence[int]]): Number of epochs,
                the networks are trained. Defaults to 500.
            ensemble_size (Union[int, Sequence[int]]): Size of the ensemble, i.e.,
                over how many randomly initialized neural networks we average
                to obtain estimates of mean and standard deviation.
                Automatically vectorized using `vmap`.
                Defaults to 100.
        """
        from jax import random  # pylint:disable=import-outside-toplevel

        self.optimizers = validate_optimizers(
            kwargs.pop("optimizers"), kwargs.get("ndim")
        )

        self.training_steps = validate_positive_integer_list(
            kwargs.pop("training_steps", 500), kwargs.get("ndim")
        )
        self.ensemble_size = validate_positive_integer_list(
            kwargs.pop("ensemble_size", 100), kwargs.get("ndim")
        )
        self.key = random.PRNGKey(kwargs.pop("key", 10))
        self.design_space_scaler = StandardScaler()
        super().__init__(*args, **kwargs)
        self.models = validate_nt_models(self.models, self.ndim)

    def _set_data(self):
        self.design_space = self.design_space_scaler.fit_transform(self.design_space)

    def _train(self):
        for i in range(len(self.models)):
            params, scaler = _ensemble_train_one_finite_width(
                i,
                self.models,
                self.design_space,
                self.y,
                self.sampled,
                self.optimizers,
                self.key,
                self.training_steps,
                self.ensemble_size,
            )
            self.models[i].params = params
            self.models[i].scaler = scaler
            self.y[:, i] = scaler.transform(self.y[:, i].reshape(-1, 1)).flatten()

    def _predict(self):
        means, stds = [], []
        for i in range(len(self.models)):
            mean, std = _ensemble_predict_one_finite_width(
                i, self.models, self.design_space
            )
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)
