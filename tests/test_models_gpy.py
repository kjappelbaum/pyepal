# -*- coding: utf-8 -*-
"""Testing some of the model utilities"""
import GPy

from PyPAL.models.coregionalized import GPCoregionalizedRegression
from PyPAL.models.gpr import (
    build_coregionalized_model,
    build_model,
    predict,
    predict_coregionalized,
)


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


def test_predict(make_random_dataset):
    """Test that the prediction function works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    m = build_model(X, y)  # pylint:disable=invalid-name
    prediction = predict(m, X)
    assert len(prediction) == 2
    assert len(prediction[0]) == 100
    assert len(prediction[1]) == 100
    assert prediction[0].shape[1] == 1
    assert prediction[1].shape[1] == 1


def test_predict_coregionalized(make_random_dataset):
    """"Test that the prediction utility for coregionalized models works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    m = build_coregionalized_model(X, y)  # pylint:disable=invalid-name
    prediction = predict_coregionalized(m, X)
    assert len(prediction) == 2
    assert len(prediction[0]) == 100
    assert len(prediction[1]) == 100
    assert prediction[0].shape[1] == 1
    assert prediction[1].shape[1] == 1

    prediction2 = predict_coregionalized(m, X, 1)
    assert len(prediction2) == 2
    assert len(prediction2[0]) == 100
    assert len(prediction2[1]) == 100
    assert prediction2[0].shape[1] == 1
    assert prediction2[1].shape[1] == 1
