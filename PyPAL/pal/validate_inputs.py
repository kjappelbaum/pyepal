# -*- coding: utf-8 -*-
def validate_ndim(ndim):
    if not isinstance(ndim, int):
        raise ValueError("The number of dimensions, ndim, must be a positive integer")

    if ndim <= 0:
        raise ValueError("ndmin must be greater than 0")

    return ndim


def validate_delta(delta):
    if delta >= 1 | delta < 0:
        raise ValueError("The delta values must be in (0,1]")
    else:
        return delta


def validate_beta_scale(beta_scale):
    if beta_scale >= 1 | beta_scale < 0:
        raise ValueError("The beta_scale values must be in (0,1]")
    else:
        return beta_scale


def validate_epsilon(epsilon, ndim):
    if isinstance(epsilon, list):
        if len(epsilon) != ndim:
            raise ValueError(
                "If epsilon is provided as a list, there must be one float per dimension"
            )
        else:
            for value in epsilon:
                if value >= 1 | value < 0:
                    raise ValueError("The epsilon values must be in (0,1]")
        return epsilon
    else:
        if epsilon >= 1 | epsilon < 0:
            raise ValueError("The epsilon values must be in (0,1]")
        else:
            return [epsilon] * len(ndim)


def validate_goals(goals, ndim):
    ...