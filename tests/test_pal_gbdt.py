# -*- coding: utf-8 -*-
"""Test the PALGBDT class"""
import numpy as np
from lightgbm import LGBMRegressor

from pypal.pal.pal_gbdt import PALGBDT


def test_pal_gbdt(binh_korn_points):
    """Test the basic funtionality of the PALGBDT class"""
    models = [
        (
            LGBMRegressor(objective="quantile", alpha=0.3, n_estimators=50),
            LGBMRegressor(n_estimators=50),
            LGBMRegressor(objective="quantile", alpha=0.7, n_estimators=50),
        ),
        (
            LGBMRegressor(objective="quantile", alpha=0.3, n_estimators=50),
            LGBMRegressor(n_estimators=50),
            LGBMRegressor(objective="quantile", alpha=0.7, n_estimators=50),
        ),
    ]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 95])

    palinstance0 = PALGBDT(
        X_binh_korn, models, 2, beta_scale=1 / 20, epsilon=0.05, coef_var_threshold=100
    )
    palinstance0.cross_val_points = 0

    palinstance0.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = palinstance0.run_one_step()
    assert palinstance0.number_discarded_points == 0
    assert palinstance0.number_unclassified_points < 100
