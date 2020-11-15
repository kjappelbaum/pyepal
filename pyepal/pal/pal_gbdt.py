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


"""Implements a PAL class for GBDT models which can predict uncertainity intervals
when used with quantile loss.
For an example of GBDT with quantile loss see
Jablonka, Kevin Maik; Moosavi, Seyed Mohamad; Asgari, Mehrdad; Ireland, Christopher;
Patiny, Luc; Smit, Berend (2020): A Data-Driven Perspective on
the Colours of Metal-Organic Frameworks. ChemRxiv. Preprint.
https://doi.org/10.26434/chemrxiv.13033217.v1

For general information about quantile regression
see https://en.wikipedia.org/wiki/Quantile_regression

Note that the scaling of the hyperrectangles has been derived
for GPR models (with RBF kernels).
"""
import math
from functools import partial

import numpy as np

from .pal_base import PALBase
from .validate_inputs import validate_gbdt_models, validate_interquartile_scaler

__all__ = ["PALGBDT"]


def _train_model_picklable(i, models, design_space, objectives, sampled):
    model = models[i]
    objective = math.floor(i / 3)
    model.fit(
        design_space[sampled[:, objective], :],
        objectives[sampled[:, objective], objective].ravel(),
    )
    return model


class PALGBDT(PALBase):
    """PAL class for a list of LightGBM GBDT models"""

    def __init__(self, *args, **kwargs):
        """Construct the PALGBDT instance

        Args:
            X_design (np.array): Design space (feature matrix)
            models (List[Iterable[LGBMRegressor, LGBMRegressor, LGBMRegressor]]:
                Machine learning models. You need to provide a list of iterables.
                One iterable per objective and every iterable contains three
                LGBMRegressors. The first one for the lower uncertainty limits,
                the middle one for the median and the last one for the upper limit.
                To create appropriate models, you need to use the quantile loss.
                If you want to parallelize training, we recommend that you use
                the LightGBM parallelization and fit the models for the different
                objectives in serial fashion.s
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
            interquartile_scaler (float, optional): Used to convert the difference
                between the upper and lower quantile into a standard deviation.
                This, is std = (up-low)/interquartile_scaler. Defaults to 1.35,
                following Wan, X., Wang, W., Liu, J. et al.
                Estimating the sample mean and standard deviation from the sample size,
                median, range and/or interquartile range.
                BMC Med Res Methodol 14, 135 (2014).
                https://doi.org/10.1186/1471-2288-14-135

        """
        self.interquartile_scaler = validate_interquartile_scaler(
            kwargs.pop("interquartile_scaler", 1.35)
        )

        super().__init__(*args, **kwargs)

        self.models = validate_gbdt_models(self.models, self.ndim)

    def _set_data(self):
        pass

    def _train(self):
        models_flat = []
        # not using list(sum(self.models, ())) such that we do not
        # have to assume the type
        for model_tuples in self.models:
            for model in model_tuples:
                model.n_jobs = 1
                model.nthreads = 1
                models_flat.append(model)

        train_single_partial = partial(
            _train_model_picklable,
            models=models_flat,
            design_space=self.design_space,
            objectives=self.y,
            sampled=self.sampled,
        )
        models = []

        for model in map(train_single_partial, range(len(models_flat))):
            models.append(model)

        model_tuples = []
        for i in range(0, len(models), 3):
            chunk = models[i : i + 3]
            model_tuples.append(chunk)
        self.models = model_tuples

    def _predict(self):
        means, stds = [], []
        for model_tuple in self.models:
            mean = model_tuple[1].predict(self.design_space)
            upper = model_tuple[2].predict(self.design_space)
            lower = model_tuple[0].predict(self.design_space)
            std = np.abs(upper - lower) / self.interquartile_scaler

            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        # ToDo
        pass
