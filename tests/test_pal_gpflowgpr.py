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

"""Testing the PALGPflowGPR class"""
import numpy as np

from pyepal.pal.pal_gpflowgpr import PALGPflowGPR


def test_pal_gpflow(binh_korn_points):
    """Test basic functionality of the PALGpy class"""
    import gpflow  # pylint:disable=import-outside-toplevel

    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name
    X_binh_korn = (  # pylint:disable=invalid-name
        X_binh_korn - X_binh_korn.mean()
    ) / X_binh_korn.std()  # pylint:disable=invalid-name
    y_binh_korn = (y_binh_korn - y_binh_korn.mean()) / y_binh_korn.std() + 0.01 * np.random.rand()

    def build_model(x, y):  # pylint:disable=invalid-name
        k = gpflow.kernels.RationalQuadratic()
        m = gpflow.models.GPR(  # pylint:disable=invalid-name
            data=(x, y), kernel=k, mean_function=None
        )
        return m

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx])
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx])

    palinstance = PALGPflowGPR(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        epsilon=0.01,
        delta=0.01,
        opt_kwargs={"maxiter": 50},
    )
    palinstance.cross_val_points = 0
    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])
    idx = palinstance.run_one_step()
    assert idx[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70]
    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0
