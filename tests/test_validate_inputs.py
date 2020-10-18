# -*- coding: utf-8 -*-
"""Testing the input validation"""
import numpy as np
import pytest

from pypal.models.gpr import build_coregionalized_model
from pypal.pal.validate_inputs import (
    base_validate_models,
    validate_beta_scale,
    validate_coregionalized_gpy,
    validate_delta,
    validate_epsilon,
    validate_goals,
    validate_gpy_model,
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

    assert validate_njobs(1) is None
    assert validate_njobs(2) is None
