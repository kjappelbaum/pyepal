# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np


class PALBase:

    def __init__(
        self,
        X_design: np.array,
        models: list,
        ndim: int,
        epsilon: list,
        delta: float,
        beta_scale: float,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.beta_scale = beta_scale
        self.pareto_optimal = np.array([False] * len(X_design))
        self.sampled = np.array([False] * len(X_design))
        self.unclassified = np.array([True] * len(X_design))
        self.rectangle_ups = []
        self.rectangle_lows = []
        self.models = models
        self.iteration = 0
        self.ndim = ndim
        self.design_space_size = len(X_design)
        self.mu = []
        self.std = []
        self.design_space = X_design

    @property
    def pareto_optimal_points(self):
        return self.design_space[self.pareto_optimal]

    @property
    def sampled_points(self):
        return self.design_space[self.sampled]

    @property
    def pareto_optimal_indices(self):
        return np.where(self.pareto_optimal == True)[0]

    @property
    def sampled_indices(self):
        return np.where(self.sampled == True)[0]

    @property
    def number_pareto_optimal_points(self):
        return sum(self.pareto_optimal)

    def _update_beta(self):
        self.beta = (self.beta_scale * 2 *
                     np.log(self.ndim * self.design_space_size * np.square(np.pi) * np.square(self.iteration + 1) /
                            (6 * self.delta)))

    def _log(self):
        pass

    def _should_optimize_hyperparameters(self, iteration: int) -> bool:
        return True

    def _predict(self):
        pass

    def _set_hyperparameters(self, X: np.array, y: np.array):
        pass

    def _train(self, X_train: np.array, y_train: np.array) -> list:
        pass

    def _update_hyperrectangles(self):
        pass

    def _classify(self):
        ...

    def run_one_step(self):
        """Inner part of the loop"""
        self._train()
        self._predict()
        self._update_hyperrectangles()
        self._update_beta()
        self._classify()
        self.sample()
        self.iteration += 1

    def update_train_set(self):
        pass

    def sample(self):
        pass
