# -*- coding: utf-8 -*-
"""PAL using GPy GPR models"""
import concurrent.futures
from functools import partial

import numpy as np

from ..models.gpr import predict
from .pal_base import PALBase
from .schedules import linear
from .validate_inputs import validate_gpy_model, validate_njobs, validate_number_models


def _train_model_picklable(i, models, restarts):
    model = models[i]
    model.optimize_restarts(restarts)
    return model


class PALGPy(PALBase):
    """PAL class for a list of GPy GPR models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        self.restarts = kwargs.pop("restarts", 20)

        n_jobs = kwargs.pop("n_jobs", 1)
        validate_njobs(n_jobs)
        self.n_jobs = n_jobs

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
