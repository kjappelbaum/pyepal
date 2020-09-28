# -*- coding: utf-8 -*-
"""Testing some of the model utilities"""
import GPy

from PyPAL.models.coregionalized import GPCoregionalizedRegression
from PyPAL.models.gpr import build_coregionalized_model, build_model


def test_build_model(make_random_dataset):
    """Test that we can make a basic GPy model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    m = build_model(X, y)  # pylint:disable=invalid-name
    assert isinstance(m, GPy.models.GPRegression)
    assert m.normalizer is not None
    assert m.kern.input_dim == 10


def test_build_coregionalized_model(make_random_dataset):
    """Test that we can build a coregionalized model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    m = build_coregionalized_model(X, y)  # pylint:disable=invalid-name
    assert isinstance(m, GPCoregionalizedRegression)
