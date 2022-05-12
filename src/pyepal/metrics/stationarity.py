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
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import levene


def levene_stationarity_test(
    X: np.array,
    excluded_cols: Union[List[int], None] = None,
    center: str = "median",
    proportiontocut: float = 0.05,
) -> Tuple[float, float]:
    """Levene's test for stationarity - commmon assumption for Gaussian processes
    with ||x-x'|| in the kernel.

    Args:
        X (np.array): Feature matrix.
        excluded_cols (Union[List[int], None], optional): Optional list of column indices to exclude.
            Defaults to None.
        center (str, optional): One from {'mean’, ‘median’, ‘trimmed’}.
            Which function of the data to use in the test. The default is ‘median’.
             Defaults to "median".
            * ‘median’ : Recommended for skewed (non-normal) distributions>
            * ‘mean’ : Recommended for symmetric, moderate-tailed distributions.
            * ‘trimmed’ : Recommended for heavy-tailed distributions.
        proportiontocut (float, optional): When center is ‘trimmed’,
            this gives the proportion of data points to cut from each end.
            (See scipy.stats.trim_mean.). Defaults to 0.05.

    Returns:
        Tuple[float, float]: The test statistic and the p-value for the test.
    """
    if excluded_cols is None:
        excluded_cols = []
    cols = [i for i in range(X.shape[1]) if i not in excluded_cols]

    return levene(*X[:, cols].T, center=center, proportiontocut=proportiontocut)
