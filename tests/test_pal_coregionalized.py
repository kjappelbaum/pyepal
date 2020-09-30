# -*- coding: utf-8 -*-
"""Testing the coregionalized PAL class"""
import numpy as np
import pytest

from pypal.models.gpr import build_coregionalized_model
from pypal.pal.pal_coregionalized import PALCoregionalized


def test_pal_coregionalized(make_random_dataset):
    """Test the PAL coregionalized model"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    model = build_coregionalized_model(X, y)
    with pytest.raises(ValueError):
        pal_coregionalized = PALCoregionalized(X, ["m"], 3)

    pal_coregionalized = PALCoregionalized(X, [model], 3)

    slice_idx = np.array([0, 1, 2, 3, 4])
    pal_coregionalized.update_train_set(slice_idx, y[slice_idx])

    assert (
        pal_coregionalized.models[0].kern.B.kappa.values == np.array([0.5, 0.5, 0.5])
    ).all()

    pal_coregionalized._set_hyperparameters()  # pylint:disable=protected-access

    assert (
        pal_coregionalized._should_optimize_hyperparameters()  # pylint:disable=protected-access
    )

    assert (
        pal_coregionalized.models[0].kern.B.kappa.values != np.array([0.5, 0.5, 0.5])
    ).any()
