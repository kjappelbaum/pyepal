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

from botorch.models import SingleTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Warp, Normalize, ChainedInputTransform
from sklearn.preprocessing import PowerTransformer
from botorch.fit import fit_gpytorch_model
from typing import Tuple
import numpy as np
import torch


def build_model(
    X,
    y,
    warped: bool = True,
    input_scaled: bool = True,
    wrap_indices: Tuple[int] = None,
    scaling_indices: Tuple[int] = None,
    covar_module=None,
):
    input_transformations = {}

    if input_scaled:
        if scaling_indices is None:
            scaling_indices = tuple(range(X.shape[1]))
        input_transformations["norm"] = Normalize(
            d=X.shape[1],
            indices=scaling_indices,
        )

    if warped:
        if wrap_indices is None:
            wrap_indices = np.arange(X.shape[1])
        input_transformations["warp"] = Warp(wrap_indices)

    my_input_transformations = ChainedInputTransform(**input_transformations)

    def model_creator(x, y, old_state_dict=None):
        x = torch.tensor(x)
        y = torch.tensor(y)
        gp = SingleTaskGP(x, y, covar_module=covar_module, input_transform=my_input_transformations)
        if old_state_dict is not None:
            gp.load_state_dict(old_state_dict)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        return gp, mll

    return model_creator


def build_multioutput_model(
    X,
    y,
    warped: bool = True,
    input_scaled: bool = True,
    wrap_indices: Tuple[int] = None,
    scaling_indices: Tuple[int] = None,
    covar_module=None,
):
    num_targets = y.shape[1]
    input_transformations = {}

    if input_scaled:
        if scaling_indices is None:
            scaling_indices = tuple(range(X.shape[1]))
        input_transformations["norm"] = Normalize(
            d=X.shape[1] + 1,
            indices=scaling_indices,
        )

    if warped:
        if wrap_indices is None:
            wrap_indices = np.arange(X.shape[1])
        input_transformations["warp"] = Warp(wrap_indices)

    my_input_transformations = ChainedInputTransform(**input_transformations)

    def model_creator(x, y, old_state_dict=None):
        y_stacked = np.vstack([y[:, i].reshape(-1, 1) for i in range(num_targets)])
        x = np.hstack(
            [
                np.vstack([x] * num_targets),
                np.vstack([np.array([i] * x.shape[0]).reshape(-1, 1) for i in range(num_targets)]),
            ]
        )
        x = torch.tensor(x)
        y = torch.tensor(y_stacked)
        gp = MultiTaskGP(
            x,
            y,
            covar_module=covar_module,
            input_transform=my_input_transformations,
            task_feature=-1,
        )
        if old_state_dict is not None:
            gp.load_state_dict(old_state_dict)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        return gp, mll

    return model_creator
