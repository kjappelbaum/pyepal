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
        epsilon: Union[list, float] = 0.01,
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
        self.y = np.zeros(  # pylint:disable=invalid-name
            (self.design_space_size, self.ndim)
        )
        self.measurement_uncertainity = np.zeros((self.design_space_size, self.ndim))
        self._has_train_set = False
        self._y = self.y

    def __repr__(self):
        return f"pypal at iteration {self.iteration}. \
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
    def unclassified_points(self):
        """Return the discarded points"""
        return self.design_space[self.unclassified]

    @property
    def pareto_optimal_indices(self):
        """Return the indices of the Pareto optimal points"""
        return np.where(self.pareto_optimal)[0]

    @property
    def sampled_indices(self):
        """Return the indices of the sampled points"""
        return np.where(self.sampled)[0]

    @property
    def discarded_indices(self):
        """Return the indices of the discarded points"""
        return np.where(self.discarded)[0]

    @property
    def unclassified_indices(self):
        """Return the indices of the unclassified points"""
        return np.where(self.unclassified)[0]

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

    @property
    def number_sampled_points(self):
        """Return the number of sampled points"""
        return sum(self.sampled)

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
        pass

    def _should_optimize_hyperparameters(self) -> bool:  # pylint:disable=no-self-use
        return True

    def _predict(self):
        raise NotImplementedError("The predict function is not implemented")

    def _set_hyperparameters(self):
        pass

    def _turn_to_maximization(self):
        self.y = self._y * self.goals

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
            self.epsilon,
        )

    def _replace_by_measurements(self):
        """Implements one "trick". Instead of using the GPR
        predictions for the sampled points we use the data that
        was actually measured and the actual uncertainty.
        This is different from the PAL implementation proposed
        by Zuluaga et al. This could make issues when the measurements
        are outliers"""
        self.means[self.sampled] = self.y[self.sampled]
        self.std[self.sampled] = self.measurement_uncertainity[self.sampled]

    def run_one_step(self) -> Union[int, None]:
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
        self._replace_by_measurements()
        self._update_hyperrectangles()
        self._classify()
        if sum(self.unclassified):
            sampled_idx = self.sample()
            self.sampled_idx = sampled_idx
            self._log()
            self.iteration += 1
            return sampled_idx
        print("Done. No unclassified point left")
        return None

    def update_train_set(
        self,
        indices: np.ndarray,
        measurements: np.ndarray,
        measurement_uncertainity: np.ndarray = None,
    ):
        """Update training set following a measurement

        Args:
            indices (np.ndarray): Indices of design space at which
                the measurements were taken
            measurements (np.ndarray): Measured values, 2D array.
                the length must equal the length of the inidices array.
                the second direction must equal the number of objectives
            measurement_uncertainity (np.ndarray): uncertainty in the measuremens,
                if not provided (None) will be zero. If it is not None, it must be
                an array with the same shape as the measurements
        """
        self._has_train_set = True
        assert measurements.shape[1] == self.ndim
        assert len(indices) == len(measurements)
        if measurement_uncertainity is not None:
            assert measurement_uncertainity.shape == measurements.shape
        else:
            measurement_uncertainity = np.zeros(measurements.shape)
        self._y[indices] = measurements
        self.measurement_uncertainity[indices] = measurement_uncertainity
        self.sampled[indices] = True
        self._turn_to_maximization()

    def sample(self) -> int:
        """Runs the sampling step based on the size of the hyperrectangle.
        I.e., favoring exploration.

        Returns:
            int: Index of next point to evaulate in desing space
        """
        if (self.rectangle_lows is None) | (self.rectangle_ups is None):
            raise ValueError(
                "You need to have uncertainty rectangles\
                     before you can peform the sampling"
            )

        sampled_idx = _get_max_wt(
            self.rectangle_lows,
            self.rectangle_ups,
            self.means,
            self.pareto_optimal,
            self.unclassified,
            self.sampled,
        )

        return sampled_idx
