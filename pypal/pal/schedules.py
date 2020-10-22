# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
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
    if iteration == 1:
        return True
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
    if iteration == 1:
        return True

    result = math.log(iteration, base)
    if np.abs(result - round(result)) < 0.00001:
        return True
    return False
