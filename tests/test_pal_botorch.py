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

"""Test the PALGBDT class"""
import numpy as np

from pyepal.pal.pal_botorch import PALBoTorch, PALMultiTaskBoTorch
from pyepal.models.botorch_gp import build_model, build_multioutput_model


def test_pal_botorch(binh_korn_points):
    """Test the basic funtionality of the PALGBDT class"""

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.arange(0, 100, 5)

    model_func_1 = build_model(X_binh_korn, y_binh_korn[:, 0].reshape(-1, 1), warped=True)
    model_func_2 = build_model(X_binh_korn, y_binh_korn[:, 1].reshape(-1, 1), warped=True)

    palinstance0 = PALBoTorch(
        X_binh_korn,
        model_functions=[model_func_1, model_func_2],
        ndim=2,
        beta_scale=1,
        epsilon=0.01,
        coef_var_threshold=100,
        power_transformer=True,
        add_observation_noise=True,
    )
    palinstance0.cross_val_points = 0

    palinstance0.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = palinstance0.run_one_step()

    assert palinstance0.number_discarded_points == 0
    assert palinstance0.number_unclassified_points < 100


def test_pal_multitask_botorch(binh_korn_points):
    """Test the basic funtionality of the PALGBDT class"""

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    sample_idx = np.arange(0, 100, 5)

    model_func_1 = build_multioutput_model(X_binh_korn, y_binh_korn, warped=True)

    palinstance0 = PALMultiTaskBoTorch(
        X_binh_korn,
        model_functions=[model_func_1],
        ndim=2,
        beta_scale=1,
        epsilon=0.01,
        coef_var_threshold=100,
        power_transformer=True,
    )
    palinstance0.cross_val_points = 0
    palinstance0.update_train_set(sample_idx, y_binh_korn[sample_idx])
    _ = palinstance0.run_one_step()

    assert palinstance0.number_discarded_points == 0
    assert palinstance0.number_unclassified_points < 100
