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


"""Testing the input validation"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pypal.models.gpr import build_coregionalized_model
from pypal.pal.validate_inputs import (
    _validate_sklearn_gpr_model,
    base_validate_models,
    validate_beta_scale,
    validate_coef_var,
    validate_coregionalized_gpy,
    validate_delta,
    validate_epsilon,
    validate_gbdt_models,
    validate_goals,
    validate_gpy_model,
    validate_interquartile_scaler,
    validate_ndim,
    validate_njobs,
    validate_number_models,
)


def test_validate_ndim():
    """Check that the ndmin validation works"""
    with pytest.raises(ValueError):
        validate_ndim(0)
    with pytest.raises(ValueError):
        validate_ndim(-1)
    with pytest.raises(ValueError):
        validate_ndim(0.5)

    assert validate_ndim(1) == 1
    assert validate_ndim(2) == 2


def test_validate_epsilon():
    """Check that the epsilon validation works"""
    with pytest.raises(ValueError):
        validate_epsilon([0.1], 2)
    with pytest.raises(ValueError):
        validate_epsilon([-0.1, 1], 2)

    assert (validate_epsilon(0.1, 2) == np.array([0.1, 0.1])).all()
    assert (validate_epsilon([0.1, 0.1], 2) == np.array([0.1, 0.1])).all()
    assert (validate_epsilon(np.array([0.1, 0.1]), 2) == np.array([0.1, 0.1])).all()


def test_validate_beta_scale():
    """Test beta scaling validation"""
    assert validate_beta_scale(1 / 9) == 1 / 9

    with pytest.raises(ValueError):
        validate_beta_scale(-1)

    assert validate_beta_scale(1) == 1


def test_validate_goals():
    """Test the goals validation"""
    assert (validate_goals(["max", "min"], 2) == np.array([1, -1])).all()
    with pytest.raises(ValueError):
        validate_goals([2, 3], 2)
    with pytest.raises(ValueError):
        validate_goals([1], 2)
    with pytest.raises(ValueError):
        validate_goals(1, 1)

    assert (validate_goals(None, 3) == np.array([1, 1, 1])).all()


def test_validate_number_models():
    """Test the validation of the number of models"""
    with pytest.raises(ValueError):
        validate_number_models([1, 1], 1)
    with pytest.raises(ValueError):
        validate_number_models([1, 1], 3)

    assert validate_number_models([1, 1], 2) is None


def test_base_validate_models():
    """Test that the basic validation of the models works"""
    with pytest.raises(ValueError):
        base_validate_models([])

    assert ["m"] == base_validate_models(["m"])


def test_validate_delta():
    """Test the delta validation"""
    with pytest.raises(ValueError):
        validate_delta(1.1)

    with pytest.raises(ValueError):
        validate_delta(-0.1)

    assert validate_delta(0.1) == 0.1


def test_validate_gpy_models():
    """Raise error when there models are not GPy models"""
    with pytest.raises(ValueError):
        validate_gpy_model(["m"])


def test_validate_coregionalized_gpy(make_random_dataset):
    """Test that the check for coregionalized models works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    model = build_coregionalized_model(X, y)
    with pytest.raises(ValueError):
        validate_coregionalized_gpy(model)

    with pytest.raises(ValueError):
        validate_coregionalized_gpy(["m"])

    assert validate_coregionalized_gpy([model]) is None


def test_validate_njobs():
    """Test that the validation of n_jobs for multiprocessing works"""
    with pytest.raises(ValueError):
        validate_njobs(0.1)
    with pytest.raises(ValueError):
        validate_njobs(0)

    assert validate_njobs(1) == 1
    assert validate_njobs(2) == 2


def test_validate_coef_var():
    """Test that the validate_coef_var works"""

    with pytest.raises(ValueError):
        validate_coef_var(-1)

    with pytest.raises(ValueError):
        validate_coef_var(0)

    with pytest.raises(ValueError):
        validate_coef_var(None)

    assert validate_coef_var(3) == 3


def test__validate_sklearn_gpr_model(make_random_dataset):
    """Test that the model validation for PALSklearn works"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(GaussianProcessClassifier())
    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(GridSearchCV(GaussianProcessClassifier(), {}))
    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(RandomizedSearchCV(GaussianProcessClassifier(), {}))

    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(GridSearchCV(GaussianProcessRegressor(), {}))
    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(RandomizedSearchCV(GaussianProcessRegressor(), {}))

    gpr = GaussianProcessRegressor()

    assert _validate_sklearn_gpr_model(gpr) == gpr

    grid_search = GridSearchCV(
        GaussianProcessRegressor(), {"kernel": [RBF(), Matern()]}
    )

    random_search = RandomizedSearchCV(
        GaussianProcessRegressor(), {"kernel": [RBF(), Matern()]}
    )

    grid_search.fit(X, y)
    random_search.fit(X, y)

    assert _validate_sklearn_gpr_model(grid_search) == grid_search.best_estimator_
    assert _validate_sklearn_gpr_model(random_search) == random_search.best_estimator_

    (
        X_classification,  # pylint:disable=invalid-name
        y_classification,
    ) = make_classification()

    grid_search_class = GridSearchCV(
        GaussianProcessClassifier(), {"kernel": [RBF(), Matern()]}
    )
    random_search_class = RandomizedSearchCV(
        GaussianProcessClassifier(), {"kernel": [RBF(), Matern()]}
    )

    grid_search_class.fit(X_classification, y_classification)
    random_search_class.fit(X_classification, y_classification)

    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(grid_search_class)

    with pytest.raises(ValueError):
        _validate_sklearn_gpr_model(random_search_class)


def test_validate_gbdt_models():
    """Test the input validation for the input of the PALGBDT class"""
    from lightgbm import LGBMRegressor  # pylint:disable=import-outside-toplevel

    lgbm = LGBMRegressor(objective="quantile", alpha=0.1)
    with pytest.raises(ValueError):
        validate_gbdt_models([(lgbm, lgbm), (lgbm, lgbm, lgbm)], 2)

    with pytest.raises(ValueError):
        validate_gbdt_models([(LGBMRegressor(), lgbm, lgbm), (lgbm, lgbm, lgbm)], 2)

    with pytest.raises(ValueError):
        validate_gbdt_models([(lgbm, lgbm, lgbm), (lgbm, lgbm, LGBMRegressor())], 2)

    valid_input = [(lgbm, LGBMRegressor(), lgbm), (lgbm, LGBMRegressor(), lgbm)]
    assert validate_gbdt_models(valid_input, 2) == valid_input


def test_validate_interquartile_scaler():
    """Make sure that the validation of the interquartile_scaler makes sense"""

    with pytest.raises(ValueError):
        validate_interquartile_scaler(None)

    with pytest.raises(ValueError):
        validate_interquartile_scaler(-1)

    assert validate_interquartile_scaler(1.35) == 1.35
