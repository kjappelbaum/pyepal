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


"""Implements a PAL class for GBDT models with virtual ensembles for the uncertainty estimates.

Note that the scaling of the hyperrectangles has been derived
for GPR models (with RBF kernels).
"""
from functools import partial

import numpy as np

from .pal_base import PALBase

__all__ = ["PALCatBoost"]


def _train_model_picklable(i, models, design_space, objectives, sampled):
    model = models[i]
    model.fit(
        design_space[sampled[:, i], :],
        objectives[sampled[:, i], i].ravel(),
    )
    return model


class PALCatBoost(PALBase):
    """PAL class for a list of LightGBM GBDT models"""

    def __init__(self, *args, **kwargs):
        """Construct the PALCatBoost instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (List[Iterable[LGBMRegressor, LGBMRegressor, LGBMRegressor]]:
                Machine learning models. You need to provide a list of
                CatBoost regressors.
            ndim (int): Number of objectives
            epsilon (Union[list, float], optional): Epsilon hyperparameter.
                Defaults to 0.01.
            delta (float, optional): Delta hyperparameter. Defaults to 0.05.
            beta_scale (float, optional): Scaling parameter for beta.
                If not equal to 1, the theoretical guarantees do not necessarily hold.
                Also note that the parametrization depends on the kernel type.
                Defaults to 1/9.
            goals (List[str], optional): If a list, provide "min" for every objective
                that shall be minimized and "max" for every objective
                that shall be maximized. Defaults to None, which means
                that the code maximizes all objectives.
            coef_var_threshold (float, optional): Use only points with
                a coefficient of variation below this threshold
                in the classification step. Defaults to 3.

        """

        super().__init__(*args, **kwargs)

        # self.models = validate_gbdt_models(self.models, self.ndim)

    def _set_data(self):
        pass

    def _train(self):
        train_single_partial = partial(
            _train_model_picklable,
            models=self.models,
            design_space=self.design_space,
            objectives=self.y,
            sampled=self.sampled,
        )
        models = []

        for model in map(train_single_partial, range(len(self.models))):
            models.append(model)

        self.models = models

    def _predict(self):
        means, stds = [], []
        for model in self.models:
            pred = model.predict(self.design_space)

            std = np.sqrt(pred[:, 1])
            mean = pred[:, 0]
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self._means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        # ToDo: Maybe add some optuna helper here.
        pass
