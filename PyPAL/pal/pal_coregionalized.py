# -*- coding: utf-8 -*-
"""PAL for coregionalized GPR models"""

import numpy as np

from ..models.gpr import predict_coregionalized, set_xy_coregionalized
from .pal_base import PALBase
from .schedules import exp_decay


class PALCoregionalized(PALBase):
    """PAL class for a coregionalized GPR model"""

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

    def _set_data(self):
        self.models[0] = set_xy_coregionalized(
            self.models[0],
            self.design_space_size[self.sampled_indices],
            self.y[self.sampled_indices],
        )

    def _train(self):
        pass

    def _predict(self):
        means, stds = [], []
        for i in range(self.ndim):
            mean, std = predict_coregionalized(self.models[0], self.design_space, i)
            means.append(mean.reshape(-1, 1))
            stds.append(np.sqrt(std).reshape(-1, 1))

        self.means = np.hstack(means)
        self.std = np.hstack(stds)

    def _set_hyperparameters(self):
        self.models[0].optimize_restarts(self.restarts, parallel=self.parallel)

    def _should_optimize_hyperparameters(self) -> bool:
        return exp_decay(self.iteration, 2)
