# -*- coding: utf-8 -*-
"""Base class for PAL"""

from typing import List, Union

import numpy as np

from .core import _get_max_wt, _get_uncertainity_regions, _pareto_classify, _union
from .validate_inputs import (
    base_validate_models,
    validate_beta_scale,
    validate_delta,
    validate_epsilon,
    validate_goals,
    validate_ndim,
)


class PALBase:  # pylint:disable=too-many-instance-attributes
    """PAL base class"""

    def __init__(  # pylint:disable=too-many-arguments
        self,
        X_design: np.array,
        models: list,
        ndim: int,
        epsilon: Union[list, float] = 0.5,
        delta: float = 0.05,
        beta_scale: float = 1 / 16,
        goals: List[str] = None,
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
        self.goals = validate_goals(goals, ndim)
        self.sampled_idx = 0
        self.y = np.array(  # pylint:disable=invalid-name
            [[np.nan] * self.design_space_size] * self.ndim
        )
        self._has_train_set = False

    def __repr__(self):
        return f"PyPAL at iteration {self.iteration}. \
        {self.number_pareto_optimal_points} Pareto optimal points, \
        {self.number_discarded_points} discarded points, \
        {self.number_unclassified_points} unclassified points."

    @property
    def pareto_optimal_points(self):
        """Return the pareto optimal points"""
        return self.design_space[self.pareto_optimal]

    @property
    def sampled_points(self):
        """Return the sampled points"""
        return self.design_space[self.sampled]

    @property
    def discarded_points(self):
        """Return the discarded points"""
        return self.design_space[self.discarded]

    @property
    def pareto_optimal_indices(self):
        """Return the indices of the Pareto optimal points"""
        return np.where(
            self.pareto_optimal == True  # pylint:disable=singleton-comparison
        )[0]

    @property
    def sampled_indices(self):
        """Return the indices of the sampled points"""
        return np.where(self.sampled == True)[0]  # pylint:disable=singleton-comparison

    @property
    def discarded_indices(self):
        """Return the indices of the discarded points"""
        return np.where(self.discarded == True)[  # pylint:disable=singleton-comparison
            0
        ]

    @property
    def number_pareto_optimal_points(self):
        """Return the number of Pareto optimal points"""
        return sum(self.pareto_optimal)

    @property
    def number_discarded_points(self):
        """Return the nnumber of discarded points"""
        return sum(self.discarded)

    @property
    def number_unclassified_points(self):
        """Return the number of unclassified points"""
        return sum(self.unclassified)

    def _update_beta(self):
        """Update beta according to section 7.2. of the epsilon-PAL paper"""
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
        return self.__repr__()

    def _should_optimize_hyperparameters(self) -> bool:  # pylint:disable=no-self-use
        return True

    def _predict(self):
        pass

    def _set_hyperparameters(self):
        pass

    def _turn_to_maximization(self):
        pass

    def _train(self):
        pass

    def _set_data(self):
        pass

    def _update_hyperrectangles(self):
        """Computes new hyperrectangles based on beta,
        the means and the standard deviations.
        If the iteration is > 0,
        then it uses iterative intersection to ensure that the size of the
        hyperrectangles is decreasing.
        """
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

    def run_one_step(self) -> int:
        """Inner part of the loop"""
        if not self._has_train_set:
            raise ValueError(
                "You need to provide a points to train the model on.\
               Before the first iteration, call the update_train_set() function."
            )
        self._set_data()
        if self._should_optimize_hyperparameters():
            self._set_hyperparameters()
        self._train()
        self._predict()
        self._update_beta()
        self._update_hyperrectangles()
        self._classify()
        sampled_idx = self.sample()
        self.sampled_idx = sampled_idx
        self._log()
        self.iteration += 1
        return sampled_idx

    # ToDo: Implement me / think about most reasonable API
    def update_train_set(self):
        """Update the training set based on the included indices and measurements"""
        self._has_train_set = True

    def sample(self) -> int:
        """Runs the sampling step based on the size of the hyperrectangle.
        I.e., favoring exploration.

        Returns:
            int: Index of next point to evaulate in desing space
        """
        sampled_idx = _get_max_wt(
            self.rectangle_lows,
            self.rectangle_ups,
            self.pareto_optimal,
            self.unclassified,
            self.sampled,
        )

        return sampled_idx
