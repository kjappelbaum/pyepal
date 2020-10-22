# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
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

"""Testing some of the model utilities"""
import GPy
import numpy as np

from pypal.models.coregionalized import GPCoregionalizedRegression
from pypal.models.gpr import (
    build_coregionalized_model,
    build_model,
    predict,
    predict_coregionalized,
    set_xy_coregionalized,
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


def test_set_xy_coregionalized(make_random_dataset):
    """Test that the utility for updating the data of a coregionalized model works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    m = build_coregionalized_model(X, y)  # pylint:disable=invalid-name
    assert len(m.X) == 300

    X = np.vstack([X, X])  # pylint:disable=invalid-name
    y = np.vstack([y, y])  # pylint:disable=invalid-name

    m = set_xy_coregionalized(m, X, y)  # pylint:disable=invalid-name
    assert len(m.X) == 600
