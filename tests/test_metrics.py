import numpy as np
from pyepal.metrics.stationarity import levene_stationarity_test
from pyepal.metrics.uncertainty import picp
import pytest


def test_levene_stationarity_test():
    a = np.random.multivariate_normal([0, 0], [[1, 1], [1, 1]], size=(10))
    assert pytest.approx(levene_stationarity_test(a)[1], 1)


def test_picp():
    a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    c = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    assert pytest.approx(picp(a, c, b), 1)
    assert pytest.approx(picp(a, c + 2, b + 1), 0)
