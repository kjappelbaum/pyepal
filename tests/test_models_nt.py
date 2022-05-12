# -*- coding: utf-8 -*-
# Copyright 2022 PyePAL authors
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

from pyepal.models.nt import build_dense_network, get_optimizer


def test_dense_network():
    """Make sure that we return the tuple of three functions"""

    with pytest.raises(AssertionError):
        build_dense_network([])

    with pytest.raises(AssertionError):
        build_dense_network([512, 512], [stax.Relu()])

    with pytest.raises(AssertionError):
        build_dense_network([512, 512], ["relu", "relu"])

    # check that we return a dataclass and can access the elements
    net = build_dense_network([512])

    assert callable(net.apply_fn)
    assert callable(net.init_fn)
    assert callable(net.kernel_fn)


def test_get_optimizer():
    """Make sure that we can create an optimizer data class"""
    optimizer = get_optimizer()
    assert callable(optimizer.get_params)
    assert callable(optimizer.opt_init)
    assert callable(optimizer.opt_update)
