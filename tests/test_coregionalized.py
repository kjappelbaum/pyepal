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

"""Rudimentary testing of the GPCoregionalized class"""
import numpy as np
import pytest

from pyepal.models.coregionalized import GPCoregionalizedRegression


def test_gpcoregionalizedregression(make_random_dataset):
    """Test that we can create an instance of this class"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    with pytest.raises(TypeError):
        gpcoreg = GPCoregionalizedRegression()  # pylint:disable=no-value-for-parameter

    target_list = [y[:, i].reshape(-1, 1) for i in range(3)]
    gpcoreg = GPCoregionalizedRegression([X] * 3, target_list)
    assert isinstance(gpcoreg, GPCoregionalizedRegression)


def test_set_xy(make_random_dataset):
    """Testing that the set_xy functions throws no error"""
    X, y = make_random_dataset  # pylint:disable=invalid-name

    target_list = [y[:, i].reshape(-1, 1) for i in range(3)]
    gpcoreg = GPCoregionalizedRegression([X] * 3, target_list)
    assert gpcoreg.kern.input_dim == 11

    X = np.vstack([X, X])  # pylint:disable=invalid-name
    y = np.vstack([y, y])  # pylint:disable=invalid-name
    target_list = [y[:, i].reshape(-1, 1) for i in range(3)]

    gpcoreg.set_XY([X] * 3, target_list)
    assert gpcoreg.kern.input_dim == 11
