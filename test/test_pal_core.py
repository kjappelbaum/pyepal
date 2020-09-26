# -*- coding: utf-8 -*-
"""Testing the PAL module"""
import numpy as np
import pytest

from PyPAL.pal.core import _get_uncertainity_region, _union_one_dim, pal


def test__get_uncertainity_region():
    """make sure that the uncertainity windows is computed in a reasonable way"""
    mu = 1  # pylint:disable=invalid-name

    low0, high0 = _get_uncertainity_region(mu, 0, 1)
    assert low0 == mu
    assert high0 == mu

    low1, high1 = _get_uncertainity_region(mu, 1, 0)
    assert low1 == mu
    assert high1 == mu

    low2, high2 = _get_uncertainity_region(mu, 0, 0)
    assert low2 == mu
    assert high2 == mu

    low3, high3 = _get_uncertainity_region(mu, 1, 1)
    assert low3 == 0
    assert high3 == 2


def test__union_one_dim():
    """Make sure that the intersection of the uncertainity regions works"""
    zeros = np.array([0, 0, 0])
    zero_one_one = np.array([0, 1, 1])
    # Case 1: Everything is zero, we should also return zero
    low, up = _union_one_dim(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    )  # pylint:disable=invalid-name

    assert (low == zeros).all()
    assert (up == zeros).all()

    # Case 2: This should also work if this is the case for only one material
    low, up = _union_one_dim(
        [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]
    )  # pylint:disable=invalid-name
    assert (low == zero_one_one).all()
    assert (up == zero_one_one).all()

    # Case 3: Uncertainity regions do not intersect
    low, up = _union_one_dim(
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]
    )  # pylint:disable=invalid-name
    assert (low == [2, 2, 2]).all()
    assert (up == [3, 3, 3]).all()

    # Case 4: We have an intersection
    low, up = _union_one_dim(
        [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [3, 3, 3]
    )  # pylint:disable=invalid-name
    assert (low == np.array([0.5, 0.5, 0.5])).all()
    assert (up == np.array([1, 1, 1])).all()
