# -*- coding: utf-8 -*-
# Copyright 2022 PyePAL authors
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

import numpy as np


def picp(y_true: np.ndarray, y_lower: np.array, y_upper: np.array) -> float:
    """
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval.
    Taken from https://github.com/IBM/UQ360/blob/main/uq360/metrics/regression_metrics.py

    Args:
        y_true: Ground truth
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)
