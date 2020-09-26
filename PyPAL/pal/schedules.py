# -*- coding: utf-8 -*-
"""Provides some scheduling functions that can be used to implement the _should_optimize_hyperparameters function"""
import math


def linear(iteration, frequency: int = 10):
    if iteration % frequency == 0:
        return True
    return False


def exp_decay(iteration, base: int = 10):
    if math.log(iteration, base).is_integer():
        return True
    return False
