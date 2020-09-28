# -*- coding: utf-8 -*-
"""PAL using GPy GPR models"""
import numpy as np

from .pal_base import PALBase
from .validate_inputs import validate_number_models


class PALSklearn(PALBase):
    """PAL class for a list of Sklearn (GPR) models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)

    def _set_data(self):
        pass

    def _train(self):
        for i, model in enumerate(self.models):
            model.fit(
                self.design_space[self.sampled], self.y[self.sampled, i].reshape(-1, 1)
            )

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
