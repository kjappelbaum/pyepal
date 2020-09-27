# -*- coding: utf-8 -*-
"""Testing the input validation"""
import numpy as np
import pytest

from PyPAL.pal.validate_inputs import (
    validate_beta_scale,
    validate_epsilon,
    validate_ndim,
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
        validate_beta_scale(1.1)
    with pytest.raises(ValueError):
        validate_beta_scale(-1)
