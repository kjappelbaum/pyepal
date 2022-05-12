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

"""Test running PAL with neural tangents models"""

from sklearn.preprocessing import StandardScaler

from pyepal.models.nt import build_dense_network
from pyepal.pal.pal_neural_tangent import PALNT
from pyepal.pal.utils import get_kmeans_samples


def test_run_one_step(binh_korn_points):
    """Test the neural tangent network using the Binh-Korn testfunction"""
    X, y = binh_korn_points  # pylint:disable=invalid-name
    X = StandardScaler().fit_transform(X)  # pylint:disable=invalid-name
    y = StandardScaler().fit_transform(y)  # pylint:disable=invalid-name
    # We create one model per objective
    model_tuple_1 = build_dense_network([128])
    model_tuple_2 = build_dense_network([128])
    palinstance = PALNT(X, [model_tuple_1, model_tuple_2], 2, beta_scale=1, kernel="ntk")

    palinstance.cross_val_points = 0
    sample_idx = get_kmeans_samples(X, 5)
    palinstance.update_train_set(sample_idx, y[sample_idx])

    idx = palinstance.run_one_step()
    if idx is not None:
        assert idx[0] not in sample_idx

    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0

    # We create one model per objective
    model_tuple_1 = build_dense_network([128])
    model_tuple_2 = build_dense_network([128])
    palinstance = PALNT(X, [model_tuple_1, model_tuple_2], 2, beta_scale=1, kernel="nngp")

    palinstance.cross_val_points = 0
    sample_idx = get_kmeans_samples(X, 5)
    palinstance.update_train_set(sample_idx, y[sample_idx])

    idx = palinstance.run_one_step()
    if idx is not None:
        assert idx[0] not in sample_idx

    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0
