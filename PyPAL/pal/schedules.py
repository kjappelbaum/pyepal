# -*- coding: utf-8 -*-
"""Provides some scheduling functions
that can be used to implement the _should_optimize_hyperparameters function"""
import math

import numpy as np


def linear(iteration: int, frequency: int = 10) -> bool:
    """Optimize hyperparameters at equally spaced intervals

    Args:
        iteration (int): current iteration
        frequency (int, optional): Spacing between the True outputs. Defaults to 10.

    Returns:
        bool: True if iteration can be divided by frequency without remainder
    """
    if iteration % frequency == 0:
        return True
    return False


def exp_decay(iteration: int, base: int = 10) -> bool:
    """Optimize hyperparameters at logartihmically spaced intervals

    Args:
        iteration (int): current iteration
        base (int, optional): Base of the logarithm. Defaults to 10.

    Returns:
        bool: True if iteration is on the log scaled grid
    """
    if iteration == 0:
        return True

    result = math.log(iteration, base)
    if np.abs(result - round(result)) < 0.00001:
        return True
    return False
