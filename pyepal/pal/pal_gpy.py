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

"""PAL using GPy GPR models"""
import concurrent.futures
from functools import partial

import numpy as np

from .pal_base import PALBase
from .schedules import linear
from .validate_inputs import validate_njobs, validate_number_models

__all__ = ["PALGPy"]


def _train_model_picklable(i, models, restarts):
    model = models[i]
    model.optimize_restarts(restarts)
    return model


class PALGPy(PALBase):
    """PAL class for a list of GPy GPR models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        """Contruct the PALGPy instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (list): Machine learning models
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
            restarts (int): Number of random restarts that are used for hyperparameter
                optimization. Defaults to 20.
            n_jobs (int): Number of parallel processes that are used to fit
                the GPR models. Defaults to 1.
        """
        from .validate_inputs import (  # pylint:disable=import-outside-toplevel
            validate_gpy_model,
        )

        self.restarts = kwargs.pop("restarts", 20)
        self.n_jobs = validate_njobs(kwargs.pop("n_jobs", 1))

        assert isinstance(
            self.restarts, int
        ), "the restarts keyword must be of type int"
        super().__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)
        validate_gpy_model(self.models)

    def _set_data(self):
        for i, model in enumerate(self.models):
            model.set_XY(
                self.design_space[self.sampled[:, i]],
                self.y[self.sampled[:, i], i].reshape(-1, 1),
            )

    def _train(self):
        pass  # There is no training in instance based models

    def _predict(self):

        from ..models.gpr import predict  # pylint:disable=import-outside-toplevel

        means, stds = [], []
        for model in self.models:
            mean, std = predict(model, self.design_space)
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        models = []

        train_model_pickleable_partial = partial(
            _train_model_picklable, models=self.models, restarts=self.restarts
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_jobs
        ) as executor:
            for model in executor.map(train_model_pickleable_partial, range(self.ndim)):
                models.append(model)
        self.models = models

    def _should_optimize_hyperparameters(self) -> bool:
        return linear(self.iteration, 10)
