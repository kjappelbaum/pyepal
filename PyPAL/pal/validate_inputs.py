# -*- coding: utf-8 -*-
"""Methods to validate inputs for the PAL classes"""
import warnings


def validate_ndim(ndim):
    if not isinstance(ndim, int):
        raise ValueError('The number of dimensions, ndim, must be a positive integer')

    if ndim <= 0:
        raise ValueError('ndmin must be greater than 0')

    return ndim


def validate_delta(delta):
    if delta >= 1 | delta < 0:
        raise ValueError('The delta values must be in (0,1]')

    return delta


def validate_beta_scale(beta_scale):
    if beta_scale >= 1 | beta_scale < 0:
        raise ValueError('The beta_scale values must be in (0,1]')

    return beta_scale


def validate_epsilon(epsilon, ndim):
    if isinstance(epsilon, list):
        if len(epsilon) != ndim:
            raise ValueError('If epsilon is provided as a list, there must be one float per dimension')

        for value in epsilon:
            if value >= 1 | value < 0:
                raise ValueError('The epsilon values must be in (0,1]')
        return epsilon

    if epsilon >= 1 | epsilon < 0:
        raise ValueError('The epsilon values must be in (0,1]')

    warnings.warn(
        'Only one epsilon value provided, will automatically expand to use the same value in every dimension',
        UserWarning,
    )
    return [epsilon] * len(ndim)


def validate_goals(goals, ndim):
    if goals is None:
        warnings.warn(
            'No goals provided, will assume that every dimension should be maximized',
            UserWarning,
        )

        return ['max'] * ndim
    if isinstance(goals, list):
        if len(goals) != ndim:
            raise ValueError('If goals is a list, the length must be equal to the ndmin')
        for goal in goals:
            if not isinstance(goal, str):
                raise ValueError('If goals is a list, it must be a list of strings')

        clean_goals = []
        for goal in goals:
            if 'max' in goal.lower():
                clean_goals.append('max')
            elif 'min' in goal.lower():
                clean_goals.append('min')
            else:
                raise ValueError('The strings in the goals list must be min or max')
        assert len(clean_goals) == ndim
        return clean_goals

    raise ValueError('Goal can be set to None or must be a list of strings of length equal to ndim')


def base_validate_models(models):
    """Currently no validation as the predict and train function are implemented independet of the base class"""
    if models:
        return models

    raise ValueError('You must provide some models to initialize PyPAL')


def validate_number_models(models, ndim):
    if not isinstance(models, list) | len(models) != ndim:
        raise ValueError('You must provide a list of models. One model per objective')
