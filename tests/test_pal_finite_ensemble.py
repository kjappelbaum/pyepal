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

"""Test running PAL with ensembles of finite width neural networks"""

from sklearn.preprocessing import StandardScaler

from pyepal.models.nt import build_dense_network, get_optimizer
from pyepal.pal.pal_finite_ensemble import PALNTEnsemble
from pyepal.pal.utils import get_kmeans_samples


def test_pal_palntensemble(binh_korn_points):
    """Test the ensemble of finite width
    neural networks with the Binh-Korn testfunction"""
    X, y = binh_korn_points  # pylint:disable=invalid-name
    X = StandardScaler().fit_transform(X)  # pylint:disable=invalid-name
    y = StandardScaler().fit_transform(y)  # pylint:disable=invalid-name
    # We create one model per objective

    network_a = build_dense_network([512])
    network_b = build_dense_network([512])

    optimizer_a = get_optimizer()
    optimizer_b = get_optimizer()

    palinstance = PALNTEnsemble(
        X, models=[network_a, network_b], optimizers=[optimizer_a, optimizer_b], ndim=2
    )

    palinstance.cross_val_points = 0
    sample_idx = get_kmeans_samples(X, 5)
    palinstance.update_train_set(sample_idx, y[sample_idx])

    idx = palinstance.run_one_step()
    if idx is not None:
        assert idx[0] not in sample_idx

    assert palinstance.number_sampled_points > 0
    assert sum(palinstance.discarded) == 0
