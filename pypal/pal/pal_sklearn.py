# -*- coding: utf-8 -*-
"""PAL using Sklearn GPR models"""
import concurrent.futures
from functools import partial

import numpy as np

from .pal_base import PALBase
from .validate_inputs import validate_njobs, validate_number_models


def _train_model_picklable(i, models, design_space, objectives, sampled):
    model = models[i]
    model.fit(
        design_space[sampled[:, i]],
        objectives[sampled[:, i], i].reshape(-1, 1),
    )
    return model


class PALSklearn(PALBase):
    """PAL class for a list of Sklearn (GPR) models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        n_jobs = kwargs.pop("n_jobs", 1)
        validate_njobs(n_jobs)
        self.n_jobs = n_jobs
        super().__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)

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
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_jobs
        ) as executor:
            for model in executor.map(train_single_partial, range(self.ndim)):
                models.append(model)
        self.models = models

    def _predict(self):
        means, stds = [], []
        for model in self.models:
            mean, std = model.predict(self.design_space, return_std=True)
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        pass
