# -*- coding: utf-8 -*-
"""Setting up pytest"""
from __future__ import absolute_import

import pytest
from sklearn.datasets import make_regression


@pytest.fixture()
def make_random_dataset(targets=3):
    return make_regression(n_samples=100, n_features=10, n_informative=8, n_targets=targets)


@pytest.fixture()
def make_one_dim_test():
    return make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=0)
