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

"""Utility functions to build neutral tangents models for PALNT"""
from collections import namedtuple
from typing import Sequence, Union

from jax import jit
from neural_tangents import stax

__all__ = ["NT_TUPLE", "build_dense_network"]

NT_TUPLE = namedtuple(  # pylint:disable=invalid-name
    "NT_TUPLE", ("apply_fn", "init_fn", "kernel_fn", "predict_fn")
)


def build_dense_network(
    hidden_layers: Sequence[int], activations: Union[Sequence, str] = "relu"
) -> NT_TUPLE:
    """Utility function to build a simple feedforward network with the
    neural tangents library.

    Args:
        hidden_layers (Sequence[int]): Iterable with the number of neurons.
            For example, [512, 512]
        activations (Union[Sequence, str], optional):
            Iterable with neural_tangents.stax axtivations or "relu" or "erf".
            Defaults to "relu".

    Returns:
        NT_TUPLE: jiited init, apply and
            kernel functions, predict_function (None)
    """
    assert len(hidden_layers) >= 1, "You must provide at least one hidden layer"
    if activations is None:
        activations = [stax.Relu() for x in hidden_layers]
    elif isinstance(activations, str):
        if activations.lower() == "relu":
            activations = [stax.Relu() for x in hidden_layers]
        elif activations.lower() == "erf":
            activations = [stax.Erf() for x in hidden_layers]
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
        stack.append(stax.Dense(hidden_layer))
        stack.append(activation)

    stack.append(stax.Dense(1))

    init_fn, apply_fn, kernel_fn = stax.serial(*stack)

    return NT_TUPLE(jit(init_fn), jit(apply_fn), jit(kernel_fn), None)
