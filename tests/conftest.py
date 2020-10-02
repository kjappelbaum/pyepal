# -*- coding: utf-8 -*-
"""Setting up pytest"""
import numpy as np
import pytest
from sklearn.datasets import make_regression


def binh_korn(x, y):  # pylint:disable=invalid-name
    """https://en.wikipedia.org/wiki/Test_functions_for_optimization"""
    obj1 = 4 * x ** 2 + 4 * y ** 2
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
def make_random_dataset(targets=3):
    """Make a dataset with three targets"""
    return make_regression(
        n_samples=100, n_features=10, n_informative=8, n_targets=targets
    )


@pytest.fixture()
def make_one_dim_test():
    """Make a dataset with one target"""
    return make_regression(
        n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=0
    )
