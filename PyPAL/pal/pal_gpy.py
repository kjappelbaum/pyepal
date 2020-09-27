# -*- coding: utf-8 -*-
"""PAL using GPy GPR models"""
import numpy as np

from .pal_base import PALBase
from .validate_inputs import validate_number_models


class PALGPy(PALBase):
    """PAL class for a list of GPy GPR models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        self.restarts = kwargs.pop("restarts", 20)
        self.parallel = kwargs.pop("parallel", False)
        assert isinstance(
            self.parallel, bool
        ), "the parallel keyword must be of type bool"
        assert isinstance(
            self.restarts, int
        ), "the restarts keyword must be of type int"
        super().__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)

    def _set_data(self):
        for model in self.models:
            model.set_xy(self.design_space[self.sampled], self.y[self.sampled])

    def _train(self):
        pass  # There is no training in instance based models

    def _predict(self):
        means, stds = [], []
        for model in self.models:
            mean, std = model.predict(self.design_space)
            means.append(mean.reshape(-1, 1))
            stds.append(np.sqrt(std).reshape(-1, 1))

        self.means = np.hstack(mean)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        for model in self.models:
            model.optimize_restarts(self.restarts, parallel=self.parallel)
