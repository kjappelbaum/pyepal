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
import concurrent.futures

import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from sklearn.preprocessing import PowerTransformer

from .pal_base import PALBase
from .schedules import linear
from .validate_inputs import validate_njobs, validate_number_models

__all__ = ["PALBoTorch", "PALMultiTaskBoTorch"]


class PALBoTorch(PALBase):
    """PAL class for a list of BoTorch (GPR) models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        """Contruct the PALBoTorch instance

        Args:
            X_design (np.array): Design space (feature matrix)
            model_functions (list): Functions that when called with `x`, `y`, and optionally `old_state_dict` return a model and a likelihood. We need to this due to problems with re-training warm-started models in BOtorch (https://github.com/pytorch/botorch/issues/533).
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
            power_transformer (bool): If True, use Yeo-Johnson transform on the inputs.
                Defaults to True.
            add_observation_noise (bool): If True, add observation noise to predicted
                uncertainties. Defaults to False
        """

        # todo: not nice that we have to provide the model functions as keyword arguments
        power_transformer = kwargs.pop("power_transformer", True)
        self.model_functions = kwargs.pop("model_functions", None)

        self.n_jobs = validate_njobs(kwargs.pop("n_jobs", 1))
        self.add_observation_noise = kwargs.pop("add_observation_noise", False)
        self.warm_start = kwargs.pop("warm_start", False)
        super().__init__(*args, **kwargs, models=[None])
        self.models = [None] * self.ndim

        self.power_transformer = (
            [PowerTransformer() for _ in range(self.ndim)] if power_transformer else None
        )
        validate_number_models(self.models, self.ndim)

    def _set_data(self):
        for i, model_generator in enumerate(self.model_functions):
            if self.warm_start and self.iteration > 1:
                old_state_dict = self.models[i][0].state_dict()
            else:
                old_state_dict = None
            y = self.y[self.sampled[:, i], i].reshape(-1, 1)
            if self.power_transformer is not None:
                y = self.power_transformer[i].fit_transform(y)
            self.models[i] = model_generator(
                self.design_space[self.sampled[:, i]], y, old_state_dict
            )

    def _train(self):
        pass  # There is no training in instance based models

    def _predict(self):
        means, stds = [], []
        for i, model in enumerate(self.models):
            posterior = model[0].posterior(
                torch.tensor(self.design_space), observation_noise=self.add_observation_noise
            )
            mean = posterior.mean.detach().numpy()
            std = posterior.variance.detach().numpy()
            mean = mean.reshape(-1, 1)
            std = std.reshape(-1, 1)
            if self.power_transformer is not None:
                mean = self.power_transformer[i].inverse_transform(mean)
                std = self.power_transformer[i].inverse_transform(std)

            means.append(mean)
            stds.append(np.sqrt(std))

        self._means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        # ToDo: parallelize
        for m in self.models:
            fit_gpytorch_model(m[1])
        # the fit function doesn't return anything, the state of the object is updated...

    def _should_optimize_hyperparameters(self) -> bool:
        return linear(self.iteration, 10)


class PALMultiTaskBoTorch(PALBase):
    """PAL class for a multioutput BoTorch model"""

    def __init__(self, *args, **kwargs):
        """Contruct the PALBoTorch instance

        Args:
            X_design (np.array): Design space (feature matrix)
            model_functions (list): Function that when called with `x`, `y`, and optionally `old_state_dict` returns a model and a likelihood. We need to this due to problems with re-training warm-started models in BOtorch (https://github.com/pytorch/botorch/issues/533).
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
            power_transformer (bool): If True, use Yeo-Johnson transform on the inputs.
                Defaults to True.
        """

        # todo: not nice that we have to provide the model functions as keyword arguments
        power_transformer = kwargs.pop("power_transformer", True)
        self.model_functions = kwargs.pop("model_functions", None)

        self.n_jobs = validate_njobs(kwargs.pop("n_jobs", 1))
        self.warm_start = kwargs.pop("warm_start", False)
        super().__init__(*args, **kwargs, models=[None])
        self.models = [None]

        self.power_transformer = PowerTransformer() if power_transformer else None

    def _set_data(self):
        model_generator = self.model_functions[0]
        if self.warm_start and self.iteration > 1:
            old_state_dict = self.models[0][0].state_dict()
        else:
            old_state_dict = None

        y = self.y[self.sampled_indices]

        if self.power_transformer is not None:
            y = self.power_transformer.fit_transform(y)
        self.models[0] = model_generator(self.design_space[self.sampled_indices], y, old_state_dict)

    def _train(self):
        pass  # There is no training in instance based models

    def _predict(self):
        means, stds = [], []

        posterior = self.models[0][0].posterior(torch.tensor(self.design_space))
        mean = posterior.mean.detach().numpy()
        std = posterior.variance.detach().numpy()
        if self.power_transformer is not None:
            mean = self.power_transformer.inverse_transform(mean)
            std = self.power_transformer.inverse_transform(std)

        means.append(mean)
        stds.append(np.sqrt(std))

        self._means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        # ToDo: parallelize
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        for m in self.models:
            fit_gpytorch_model(m[1])
        # the fit function doesn't return anything, the state of the object is updated...

    def _should_optimize_hyperparameters(self) -> bool:
        return linear(self.iteration, 10)
