# -*- coding: utf-8 -*-
"""Base class for PAL"""

import numpy as np

from .core import _get_max_wt, _get_uncertainity_regions, _pareto_classify, _union
from .validate_inputs import (
    validate_ndim,
    validate_epsilon,
    validate_delta,
    validate_beta_scale,
    validate_goals,
    base_validate_models,
)
from typing import List


class PALBase:  # pytlint:disable=too-many-instance-attributes
    """PAL base class"""

    def __init__(  # pylint:disable=too-many-arguments
        self,
        X_design: np.array,
        models: list,
        ndim: int,
        epsilon: list = 0.5,
        delta: float = 0.05,
        beta_scale: float = 1 / 16,
        goals: List[string] = None,
    ):
        self.ndim = validate_ndim(ndim)
        self.epsilon = validate_epsilon(epsilon, self.ndim)
        self.delta = validate_delta(delta)
        self.beta_scale = validate_beta_scale(beta_scale)
        self.pareto_optimal = np.array([False] * len(X_design))
        self.discarded = np.array([False] * len(X_design))
        self.sampled = np.array([False] * len(X_design))
        self.unclassified = np.array([True] * len(X_design))
        self.rectangle_ups: np.array = None
        self.rectangle_lows: np.array = None
        self.models = base_validate_models(models)
        self.iteration = 0
        self.design_space_size = len(X_design)
        self.means: np.array = None
        self.std: np.array = None
        self.design_space = X_design
        self.beta = None
        self.goals = validate_goals(
            goals
        )  # we should take here a list of maximize/minimize. In the algortihm itself we will always assume maximization but this gives the user to specify different goals for different objectives without the need of manually dealing with flipping the sign

    def __repr__(self):
        return f"PyPAL at iteration {self.iteration}. {self.number_pareto_optimal_points} Pareto optimal points, {self.number_discarded_points} discarded points, {self.number_unclassified_points} unclassified points."

    @property
    def pareto_optimal_points(self):
        return self.design_space[self.pareto_optimal]

    @property
    def sampled_points(self):
        return self.design_space[self.sampled]

    @property
    def discarded_points(self):
        return self.design_space[self.discarded]

    @property
    def pareto_optimal_indices(self):
        return np.where(self.pareto_optimal == True)[0]

    @property
    def sampled_indices(self):
        return np.where(self.sampled == True)[0]

    @property
    def discarded_indices(self):
        return np.where(self.discarded == True)[0]

    @property
    def number_pareto_optimal_points(self):
        return sum(self.pareto_optimal)

    @property
    def number_discarded_points(self):
        return sum(self.discarded)

    @property
    def number_unclassified_points(self):
        return sum(self.unclassified)

    def _update_beta(self):
        self.beta = (
            self.beta_scale
            * 2
            * np.log(
                self.ndim
                * self.design_space_size
                * np.square(np.pi)
                * np.square(self.iteration + 1)
                / (6 * self.delta)
            )
        )

    def _log(self):
        return self.__repr__

    def _should_optimize_hyperparameters(self) -> bool:
        return True

    def _predict(self):
        pass

    def _set_hyperparameters(self, X: np.array, y: np.array):
        pass

    def _turn_to_maximization(self):
        pass

    def _train(self):
        pass

    def _update_hyperrectangles(self):
        lows, ups = _get_uncertainity_regions(self.means, self.std, np.sqrt(self.beta))
        if self.iteration == 0:
            # initialization
            self.rectangle_lows, self.rectangle_ups = lows, ups
        else:
            self.rectangle_lows, self.rectangle_ups = _union(
                self.rectangle_lows, self.rectangle_ups, lows, ups
            )

    def _classify(self):
        self.pareto_optimal, self.discarded, self.unclassified = _pareto_classify(
            self.pareto_optimal,
            self.discarded,
            self.unclassified,
            self.rectangle_lows,
            self.rectangle_ups,
            self.design_space,
            self.epsilon,
        )

    def run_one_step(self):
        """Inner part of the loop"""
        if self._should_optimize_hyperparameters():
            self._set_hyperparameters()
        self._train()
        self._predict()
        self._update_beta()
        self._update_hyperrectangles()
        self._classify()
        sampled_idx = self.sample()
        self._log()
        self.iteration += 1

    def update_train_set(self):
        pass

    def sample(self):
        sampled_idx = _get_max_wt(
            self.rectangle_lows,
            self.rectangle_ups,
            self.pareto_optimal,
            self.unclassified,
            self.sampled,
            self.design_space,
        )

        return sampled_idx
