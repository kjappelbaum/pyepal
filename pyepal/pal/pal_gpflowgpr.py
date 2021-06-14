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

__all__ = ["PALGPflowGPR"]


def _train_model_picklable(i, models, opt, opt_kwargs):
    print(f"training {i}")
    model = models[i]
    _ = opt.minimize(model.training_loss, model.trainable_variables, options=opt_kwargs)
    return model


class PALGPflowGPR(PALBase):
    """PAL class for a list of GPFlow GPR models, with one model per objective.
    Please consider that there are specific multioutput models
    (https://gpflow.readthedocs.io/en/master/notebooks/advanced/multioutput.html)
    for which the train and prediction function would need to be adjusted.
    You might also consider using streaming GPRs
    (https://github.com/thangbui/streaming_sparse_gp).
    In future releases we might support this case automatically
    (i.e., handle the case in which only one model is provided).
    """

    def __init__(self, *args, **kwargs):
        """Contruct the PALGPflowGPR instance

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
            opt (function, optional): Optimizer function for the GPR parameters.
                If None (default), then we will use ` gpflow.optimizers.Scipy()`
            opt_kwargs (dict, optional): Keyword arguments passed to the optimizer.
                If None, PyePAL will pass `{"maxiter": 100}`
            n_jobs (int): Number of parallel threads that are used to fit
                the GPR models. Defaults to 1.
        """
        import gpflow  # pylint:disable=import-outside-toplevel

        self.n_jobs = validate_njobs(kwargs.pop("n_jobs", 1))
        self.opt = kwargs.pop("opt", gpflow.optimizers.Scipy())
        self.opt_kwargs = kwargs.pop("opt_kwargs", {"maxiter": 100})
        super().__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)
        # validate_gpy_model(self.models)

    def _set_data(self):
        from gpflow.models.util import (  # pylint:disable=import-outside-toplevel
            data_input_to_tensor,
        )

        for i, model in enumerate(self.models):
            model.data = data_input_to_tensor(
                (
                    self.design_space[self.sampled[:, i]],
                    self.y[self.sampled[:, i], i].reshape(-1, 1),
                )
            )

    def _train(self):
        models = []
        train_model_pickleable_partial = partial(
            _train_model_picklable,
            models=self.models,
            opt=self.opt,
            opt_kwargs=self.opt_kwargs,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.n_jobs,
        ) as executor:
            for model in executor.map(train_model_pickleable_partial, range(self.ndim)):
                models.append(model)
        self.models = models
        print("training done")

    def _predict(self):
        means, stds = [], []
        for model in self.models:
            mean, std = model.predict_f(self.design_space)
            mean = mean.numpy()
            std = std.numpy()
            means.append(mean.reshape(-1, 1))
            stds.append(np.sqrt(std.reshape(-1, 1)))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        pass

    def _should_optimize_hyperparameters(self) -> bool:
        return linear(self.iteration, 10)
