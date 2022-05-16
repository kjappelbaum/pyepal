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

"""Testing some of the model utilities"""
from typing import Callable

from botorch.models.model import Model

from pyepal.models.botorch_gp import build_model, build_multioutput_model


def test_build_model(make_random_dataset):
    """Test that we can make a basic GPflow model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    y = y[:, 0].reshape(-1, 1)
    m = build_model(X, y)  # pylint:disable=invalid-name
    assert isinstance(m, Callable)
    model, likelihood = m(X, y)
    assert isinstance(model, Model)


def test_build_multioutput_model(make_random_dataset):
    """Test that we can build a coregionalized model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    m = build_multioutput_model(X, y)  # pylint:disable=invalid-name
    assert isinstance(m, Callable)
    model, likelihood = m(X, y)
    assert isinstance(model, Model)
