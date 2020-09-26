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
            raise Warning(
                "Only one epsilon value provided, will automatically expand to use the same value in every dimension"
            )
            return [epsilon] * len(ndim)


def validate_goals(goals, ndim):
    if goals is None:
        raise Warning(
            "No goals provided, will assume that every dimension should be maximized"
        )

        return ["max"] * ndim
    elif isinstance(goals, list):
        if len(goals) != ndim:
            raise ValueError(
                "If goals is a list, the length must be equal to the ndmin"
            )
        for goal in goals:
            if not isinstance(goal, str):
                raise ValueError("If goals is a list, it must be a list of strings")

        clean_goals = []
        for goal in goals:
            if "max" in goal.lower():
                clean_goals.append("max")
            elif "min" in goal.lower():
                clean_goals.append("min")
            else:
                raise ValueError("The strings in the goals list must be min or max")
        assert len(clean_goals) == ndim
        return clean_goals
    else:
        raise ValueError(
            "Goal can be set to None or must be a list of strings of length equal to ndim"
        )


def base_validate_models(models):
    """Currently no validation as the predict and train function are implemented independet of the base class"""
    if models:
        return models
    else:
        raise ValueError("You must provide some models to initialize PyPAL")

