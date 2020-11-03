# -*- coding: utf-8 -*-
# Copyright 2020 PyePAL authors
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

"""Test the PALGBDT class"""
import numpy as np
from lightgbm import LGBMRegressor

from pyepal.pal.pal_gbdt import PALGBDT


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
