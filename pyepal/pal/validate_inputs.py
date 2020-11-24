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


"""Methods to validate inputs for the PAL classes"""
import collections
import warnings
from copy import deepcopy
from typing import Any, Iterable, List, Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

__all__ = [
    "base_validate_models",
    "validate_beta_scale",
    "validate_coef_var",
    "validate_coregionalized_gpy",
    "validate_delta",
    "validate_epsilon",
    "validate_gbdt_models",
    "validate_goals",
    "validate_gpy_model",
    "validate_interquartile_scaler",
    "validate_ndim",
    "validate_njobs",
    "validate_nt_models",
    "validate_number_models",
    "validate_optimizers",
    "validate_positive_integer_list",
    "validate_sklearn_gpr_models",
]


def validate_ndim(ndim: Any) -> int:
    """Make sure that the number of dimensions makes sense

    Args:
        ndim (Any): number of dimensions

    Raises:
        ValueError: If the number of dimensions is not an integer
        ValueError: If the number of dimensions is not greater than 0

    Returns:
        int: the number of dimensions
    """
    if not isinstance(ndim, int):
        raise ValueError("The number of dimensions, ndim, must be a positive integer")

    if ndim <= 0:
        raise ValueError("ndmin must be greater than 0")

    return ndim


def validate_delta(delta: Any) -> float:
    """Make sure that delta is in a reasonable range

    Args:
        delta (Any): Delta hyperparameter

    Raises:
        ValueError: Delta must be in [0,1].

    Returns:
        float: delta
    """
    if (delta > 1) | (delta < 0):
        raise ValueError("The delta values must be in [0,1]")

    return delta


def validate_beta_scale(beta_scale: Any) -> float:
    """

    Args:
        beta_scale (Any): scaling factor for beta

    Raises:
        ValueError: If beta is smaller than 0

    Returns:
        float: scaling factor for beta
    """
    if beta_scale < 0:
        raise ValueError("The beta_scale values must be positive")

    return beta_scale


def validate_epsilon(epsilon: Any, ndim: int) -> np.ndarray:
    """Validate epsilon and return a np.array

    Args:
        epsilon (Any): Epsilon hyperparameter
        ndim (int): Number of dimensions/objectives

    Raises:
        ValueError: If epsilon is a list there must be one float per dimension
        ValueError: Epsilon must be in [0,1]
        ValueError: If epsilon is an array there must be one float per dimension

    Returns:
        np.ndarray:  Array of one epsilon per objective
    """
    if isinstance(epsilon, list):
        if len(epsilon) != ndim:
            raise ValueError(
                "If epsilon is provided as a list,\
                     there must be one float per dimension"
            )

        for value in epsilon:
            if (value > 1) | (value < 0):
                raise ValueError("The epsilon values must be in [0,1]")
        return np.array(epsilon)

    if isinstance(epsilon, np.ndarray):
        if len(epsilon) != ndim:
            raise ValueError(
                "If epsilon is provided as a array,\
                     there must be one float per dimension"
            )

        for value in epsilon:
            if (value > 1) | (value < 0):
                raise ValueError("The epsilon values must be in [0,1]")
        return epsilon

    if (epsilon > 1) | (epsilon < 0):
        raise ValueError("The epsilon values must be in [0,1]")

    warnings.warn(
        """Only one epsilon value provided,
will automatically expand to use the same value in every dimension""",
        UserWarning,
    )
    return np.array([epsilon] * ndim)


def validate_goals(  # pylint:disable=too-many-branches
    goals: Any, ndim: int
) -> np.ndarray:
    """Create a valid array of goals. 1 for maximization, -1
        for objectives that are to be minimized.

    Args:
        goals (Any): List of goals,
            typically provideded as strings 'max' for maximization
            and 'min' for minimization
        ndim (int): number of dimensions

    Raises:
        ValueError: If goals is a list and the length is not equal to ndim
        ValueError: If goals is a list and the elements
            are not strings 'min', 'max' or -1 and 1

    Returns:
        np.ndarray: Array of -1 and 1
    """
    if goals is None:
        warnings.warn(
            "No goals provided, will assume that every dimension should be maximized",
            UserWarning,
        )

        return np.array([1] * ndim)
    if isinstance(goals, list):
        if len(goals) != ndim:
            raise ValueError("If goals is a list, the length must be equal to the ndim")
        for goal in goals:
            if not isinstance(goal, str) | (goal != 1) | (goal != -1):
                raise ValueError("If goals is a list, it must be a list of strings")

        clean_goals = []
        for goal in goals:
            if isinstance(goal, str):
                if "max" in goal.lower():
                    clean_goals.append(1)
                elif "min" in goal.lower():
                    clean_goals.append(-1)
                else:
                    raise ValueError("The strings in the goals list must be min or max")
            elif isinstance(goal, int):
                if goal == 1:
                    clean_goals.append(1)
                elif goal == -1:
                    clean_goals.append(-1)
                else:
                    raise ValueError("The ints in the goals list must be 1 or -1")

        assert len(clean_goals) == ndim
        return np.array(clean_goals)

    raise ValueError(
        "Goal can be set to None or must be a list of strings\
             or -1 and 1 of length equal to ndim"
    )


def base_validate_models(models: Any) -> list:
    """Currently no validation as the predict and train function
    are implemented independet of the base class"""
    if models:
        return models

    raise ValueError("You must provide some models to initialize pyepal")


def validate_number_models(models: Any, ndim: int):
    """Make sure that there are as many models as objectives

    Args:
        models (Any): List of models
        ndim (int): Number of objectives

    Raises:
        ValueError:  If the number of models does not equal the number of objectives
    """
    if not isinstance(models, list):
        raise ValueError("You must provide a list of models. One model per objective")
    if len(models) != ndim:
        raise ValueError("You must provide a list of models. One model per objective")


def validate_gpy_model(models: Any):
    """Make sure that all elements of the list a GPRegression models"""
    import GPy  # pylint:disable=import-outside-toplevel

    for model in models:
        if not isinstance(model, GPy.models.GPRegression):
            raise ValueError("The models must be an instance of GPy.model")


def validate_coregionalized_gpy(models: Any):
    """Make sure that model is a coregionalized GPR model"""
    if not isinstance(models, list):
        raise ValueError("You must provide a list of models with one element")

    from ..models.coregionalized import (  # pylint:disable=import-outside-toplevel
        GPCoregionalizedRegression,
    )

    if not isinstance(models[0], GPCoregionalizedRegression):
        raise ValueError(
            "Model must be a GPCoregionalized regression object from this package!"
        )


def validate_njobs(njobs: Any) -> int:
    """Make sure that njobs is an int > 1"""
    if not isinstance(njobs, int):
        raise ValueError("njobs musst be of type int")
    if njobs < 1:
        raise ValueError("njobs must be a number greater equal 1")
    return njobs


def validate_coef_var(coef_var: Any):
    """Make sure that the coef_var makes sense"""
    if not isinstance(coef_var, (float, int)):
        raise ValueError("coef_var must be of type float or int")
    if coef_var <= 0:
        raise ValueError("coef_var must be greater 0")

    return coef_var


def _validate_sklearn_gpr_model(model: Any) -> GaussianProcessRegressor:
    """Make sure that we deal with a GaussianProcessRegressor instance,
    if it is a fitted random or grid search instance, extract the model"""
    if isinstance(model, (RandomizedSearchCV, GridSearchCV)):
        try:
            if isinstance(model.best_estimator_, GaussianProcessRegressor):
                return model.best_estimator_

            raise ValueError(
                """If you provide a grid or random search instance,
it needs to contain a GaussianProcessRegressor instance."""
            )
        except AttributeError as not_fitted_exception:
            raise ValueError(
                "If you provide a grid or random search instance it needs to be fitted."
            ) from not_fitted_exception
    elif isinstance(model, GaussianProcessRegressor):
        return model
    raise ValueError("You need to provide a GaussianProcessRegressor instance.")


def validate_sklearn_gpr_models(
    models: Any, ndim: int
) -> List[GaussianProcessRegressor]:
    """Make sure that there is a list of GPR models, one model per objective"""
    validate_number_models(models, ndim)
    models_validated = []

    for model in models:
        models_validated.append(_validate_sklearn_gpr_model(model))

    return models_validated


def _validate_quantile_loss(lightgbmregressor):
    try:
        alpha = lightgbmregressor.alpha
        loss = lightgbmregressor.objective
    except AttributeError as missing_attribute:
        raise ValueError(
            """Make sure that you initialize at
least the first and last model with quantile loss.
"""
        ) from missing_attribute
    if loss != "quantile":
        raise ValueError(
            """Make sure that you initialize at
least the first and last model with quantile loss.
"""
        )
    assert alpha > 0


def validate_gbdt_models(models: Any, ndim: int) -> List[Iterable]:
    """Make sure that the number of iterables is equal to the number of objectives
    and that every iterable contains three LGBMRegressors.
    Also, we check that at least the first and last models use quantile loss"""

    validate_number_models(models, ndim)
    from lightgbm import LGBMRegressor  # pylint:disable=import-outside-toplevel

    for model_tuple in models:
        if len(model_tuple) != 3:
            raise ValueError(
                """The model list must contain
tuples with three LGBMRegressor instances.
        """
            )

        for counter, model in enumerate(model_tuple):
            if not isinstance(model, LGBMRegressor):
                raise ValueError(
                    """The model list must contain
tuples with three LGBMRegressor instances.
        """
                )

            if counter != 1:
                _validate_quantile_loss(model)

    return models


def validate_interquartile_scaler(interquartile_scaler: Any) -> float:
    """Make sure that the interquartile_scaler makes sense"""
    if not isinstance(interquartile_scaler, (float, int)):
        raise ValueError("interquartile_scaler must be a number.")

    if interquartile_scaler < 0:
        raise ValueError("interquartile_scaler must be a number greater 0.")

    return interquartile_scaler


def _is_jaxoptimizer(optimizer: Any) -> bool:
    from ..models.nt import JaxOptimizer  # pylint:disable=import-outside-toplevel

    return isinstance(optimizer, JaxOptimizer)


def validate_optimizers(optimizers: Any, ndim: int) -> Sequence:
    """Make sure that we can work with a Sequence if JaxOptimizer"""
    if not isinstance(optimizers, collections.Sequence):
        if not _is_jaxoptimizer(optimizers):
            raise ValueError(
                "You need to provide a `pyepal.models.nt.JaxOptimizer` instance"
            )
        return [deepcopy(optimizers) for _ in range(ndim)]

    if not len(optimizers) == ndim:
        raise ValueError(
            "If you provide a sequence it must have one optimizer per objective."
        )
    for optimizer in optimizers:
        if not _is_jaxoptimizer(optimizer):
            raise ValueError(
                "You need to provide a `pyepal.models.nt.JaxOptimizer` instance"
            )
    return optimizers


def validate_nt_models(models: Any, ndim: int) -> Sequence:
    """Make sure that we can work with a sequence of
    :py:func:`pyepal.pal.models.nt.NTModel`"""
    from pyepal.models.nt import NTModel  # pylint:disable=import-outside-toplevel

    if not isinstance(models, collections.Sequence):
        raise ValueError(
            "You need to provide a sequence of `pyepal.models.nt.NTModel` instances"
        )

    for model in models:
        if not len(models) == ndim:
            raise ValueError("You need to provide one model per objective.")
        if not isinstance(model, NTModel):
            raise ValueError(
                "You need to provide a sequence of `pyepal.models.nt.NTModel` instances"
            )

    return models


def validate_positive_integer_list(
    seq: Any, ndim: int, parameter_name: str = "Parameter"
) -> Sequence[int]:
    """Can be used, e.g., to validate and standardize the ensemble size
    and epochs input"""

    if not isinstance(seq, collections.Sequence):
        if (not isinstance(seq, int)) or (seq < 1):
            raise ValueError("{} must be a positive integer".format(parameter_name))
        return [seq] * ndim

    if not len(seq) == ndim:
        raise ValueError(
            "If you provide a sequence for {} its length must match \
                the number of objectives".format(
                parameter_name
            )
        )
    for elem in seq:
        if (not isinstance(elem, int)) or (elem < 1):
            raise ValueError("{} must be a positive integer".format(parameter_name))

    return seq
