# -*- coding: utf-8 -*-
"""Rudimentary testing of the GPCoregionalized class"""
import numpy as np
import pytest

from PyPAL.models.coregionalized import GPCoregionalizedRegression


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
