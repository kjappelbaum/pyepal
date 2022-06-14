# -*- coding: utf-8 -*-
# Copyright 2022 PyePAL authors
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

"""Testing the CatBoost PAL class"""

from catboost import CatBoostRegressor
from pyepal.pal.pal_catboost import PALCatBoost
import numpy as np


def test_pal_catboost(binh_korn_points):
    models = [
        CatBoostRegressor(
            iterations=5000,
            learning_rate=0.05,
            loss_function="RMSEWithUncertainty",
            posterior_sampling=True,
            silent=True,
        )
        for _ in range(2)
    ]

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 95, 99])

    pal = PALCatBoost(
        X_binh_korn, models, 2, beta_scale=1 / 15, epsilon=0.05, coef_var_threshold=100
    )

    pal.cross_val_points = 0

    pal.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = pal.run_one_step()
    assert pal.number_discarded_points == 0
    assert pal.number_unclassified_points < 100
