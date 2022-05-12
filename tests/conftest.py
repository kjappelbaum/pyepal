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

"""Setting up pytest"""
import numpy as np
import pytest
from sklearn.datasets import make_regression


def binh_korn(x, y):  # pylint:disable=invalid-name
    """https://en.wikipedia.org/wiki/Test_functions_for_optimization"""
    obj1 = 4 * x**2 + 4 * y**2
    obj2 = (x - 5) ** 2 + (y - 5) ** 2
    return -obj1, -obj2


@pytest.fixture()
def binh_korn_points():
    """Create a dataset based on the Binh-Korn test function"""
    x = np.linspace(0, 5, 100)  # pylint:disable=invalid-name
    y = np.linspace(0, 3, 100)  # pylint:disable=invalid-name
    array = np.array([binh_korn(xi, yi) for xi, yi in zip(x, y)])
    return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)]), array


@pytest.fixture()
def binh_korn_points_finer():
    """Create a dataset based on the Binh-Korn test function"""
    x = np.linspace(0, 5, 399)  # pylint:disable=invalid-name
    y = np.linspace(0, 3, 399)  # pylint:disable=invalid-name
    array = np.array([binh_korn(xi, yi) for xi, yi in zip(x, y)])
    return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)]), array


@pytest.fixture()
def make_random_dataset(targets=3):
    """Make a dataset with three targets"""
    return make_regression(n_samples=100, n_features=10, n_informative=8, n_targets=targets)


@pytest.fixture()
def make_one_dim_test():
    """Make a dataset with one target"""
    return make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=0)
