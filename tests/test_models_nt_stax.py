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

"""Testing if the utilities for building NNs with the neural tangent library work"""
import pytest
from neural_tangents import stax

from pyepal.models.nt_stax import build_dense_network


def test_dense_network():
    """Make sure that we return the tuple of three functions"""

    with pytest.raises(AssertionError):
        build_dense_network([])

    with pytest.raises(AssertionError):
        build_dense_network([512, 512], [stax.Relu()])

    with pytest.raises(AssertionError):
        build_dense_network([512, 512], ["relu", "relu"])

    init_fn, apply_fn, kernel_fn, _ = build_dense_network([512])

    assert callable(apply_fn)
    assert callable(init_fn)
    assert callable(kernel_fn)

    # check that we return a named tuple and can access the elements
    n_tuple = build_dense_network([512])

    assert callable(n_tuple.apply_fn)
    assert callable(n_tuple.init_fn)
    assert callable(n_tuple.kernel_fn)
