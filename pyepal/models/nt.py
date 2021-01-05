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

"""Utility functions to build neutral tangents models for PALNT

Depending on the dataset there might be some issues with these models,
some tricks are listed in https://github.com/google/neural-tangents/issues/76

1. Use Erf as activation
2. Initialize the weights with larger standard deviation
3. Standardize the data

The first two points are done by default in the `build_dense_network` function

Note that following the law of total variance the prior, intialized via
W_std and b_std give an upper bound on the std of the posterior
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Union


@dataclass
class NTModel:
    """Defining a dataclass for neural tangents models"""

    # Initialization functions construct parameters for neural networks
    # given a random key and an input shape.
    init_fn: Callable
    # Apply functions do computations with finite-width neural networks.
    apply_fn: Callable
    kernel_fn: Callable
    predict_fn: Union[Callable, None] = None
    scaler: Union[Callable, None] = None  # Used to store Standard Scaler objects
    params: Union[list, None] = None  # Used to store parameters for the ensemble models


@dataclass
class JaxOptimizer:
    """Defining a dataclass for a JAX optimizer"""

    opt_init: Callable
    opt_update: Callable
    get_params: Callable


__all__ = ["NTModel", "build_dense_network", "JaxOptimizer"]


def build_dense_network(
    hidden_layers: Sequence[int],
    activations: Union[Sequence, str] = "erf",
    w_std: float = 2.5,
    b_std=1,
) -> NTModel:
    """Utility function to build a simple feedforward network with the
    neural tangents library.

    Args:
        hidden_layers (Sequence[int]): Iterable with the number of neurons.
            For example, [512, 512]
        activations (Union[Sequence, str], optional):
            Iterable with neural_tangents.stax axtivations or "relu" or "erf".
            Defaults to "erf".
        w_std (float): Standard deviation of the weight distribution.
        b_std (float): Standard deviation of the bias distribution.

    Returns:
        NTModel: jiited init, apply and
            kernel functions, predict_function (None)
    """
    from jax.config import config  # pylint:disable=import-outside-toplevel

    config.update("jax_enable_x64", True)
    from jax import jit  # pylint:disable=import-outside-toplevel
    from neural_tangents import stax  # pylint:disable=import-outside-toplevel

    assert len(hidden_layers) >= 1, "You must provide at least one hidden layer"
    if activations is None:
        activations = [stax.Relu() for _ in hidden_layers]
    elif isinstance(activations, str):
        if activations.lower() == "relu":
            activations = [stax.Relu() for _ in hidden_layers]
        elif activations.lower() == "erf":
            activations = [stax.Erf() for _ in hidden_layers]
    else:
        for activation in activations:
            assert callable(
                activation
            ), "You need to provide `neural_tangents.stax` activations"

    assert len(activations) == len(
        hidden_layers
    ), "The number of hidden layers should match the number of nonlinearities"
    stack = []

    for hidden_layer, activation in zip(hidden_layers, activations):
        stack.append(stax.Dense(hidden_layer, W_std=w_std, b_std=b_std))
        stack.append(activation)

    stack.append(stax.Dense(1, W_std=w_std, b_std=b_std))

    init_fn, apply_fn, kernel_fn = stax.serial(*stack)

    return NTModel(init_fn, jit(apply_fn), jit(kernel_fn, static_argnums=(2,)), None)


def get_optimizer(
    learning_rate: float = 1e-4, optimizer="sdg", optimizer_kwargs: dict = None
) -> JaxOptimizer:
    """Return a `JaxOptimizer` dataclass for a JAX optimizer

    Args:
        learning_rate (float, optional): Step size. Defaults to 1e-4.
        optimizer (str, optional): Optimizer type (Allowed types: "adam",
            "adamax", "adagrad", "rmsprop", "sdg"). Defaults to "sdg".
        optimizer_kwargs (dict, optional): Additional keyword arguments
            that are passed to the optimizer. Defaults to None.

    Returns:
        JaxOptimizer
    """
    from jax.config import config  # pylint:disable=import-outside-toplevel

    config.update("jax_enable_x64", True)
    from jax import jit  # pylint:disable=import-outside-toplevel
    from jax.experimental import optimizers  # pylint:disable=import-outside-toplevel

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer.lower()
    if optimizer == "adam":
        opt_init, opt_update, get_params = optimizers.adam(
            learning_rate, **optimizer_kwargs
        )
    elif optimizer == "adagrad":
        opt_init, opt_update, get_params = optimizers.adagrad(
            learning_rate, **optimizer_kwargs
        )
    elif optimizer == "adamax":
        opt_init, opt_update, get_params = optimizers.adamax(
            learning_rate, **optimizer_kwargs
        )
    elif optimizer == "rmsprop":
        opt_init, opt_update, get_params = optimizers.rmsprop(
            learning_rate, **optimizer_kwargs
        )
    else:
        opt_init, opt_update, get_params = optimizers.sgd(
            learning_rate, **optimizer_kwargs
        )

    opt_update = jit(opt_update)

    return JaxOptimizer(opt_init, opt_update, get_params)
