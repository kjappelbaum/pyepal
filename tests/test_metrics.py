# -*- coding: utf-8 -*-
import numpy as np
import pytest

from pyepal.metrics.stationarity import levene_stationarity_test
from pyepal.metrics.uncertainty import picp


def test_levene_stationarity_test():
    a = np.random.multivariate_normal([0, 0], [[1, 1], [1, 1]], size=(10))
    assert levene_stationarity_test(a)[1] == pytest.approx(1)


def test_picp():
    a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    c = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    assert picp(a, c, b) == pytest.approx(1)
    assert picp(a, c + 2, b + 1) == pytest.approx(0)
