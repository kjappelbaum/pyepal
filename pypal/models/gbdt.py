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

"""Some utilities to work with LightGBM models"""
from typing import List, Tuple, Union

import numpy as np
from lightgbm import LGBMRegressor


def build_gbdt_tuple(
    alphas: Union[List[float], np.ndarray, Tuple[float, float, float]] = [
        0.25,
        0.5,
        0.75,
    ],
    **kwargs  # pylint:disable=dangerous-default-value
) -> Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]:
    """Build a Tuple of LGBMRegressors in the correct
    format for PALGBDT

    Args:
        alphas (Union[List[float], np.ndarray, Tuple[float]], optional):
            Quantile levels.
            Defaults to [0.16, 0.5, 0.84]
            (about 68% of the probability under a
            normal curve lies between 𝜇±𝜎.)

    Returns:
        Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]
    """
    return (
        LGBMRegressor(objective="quantile", alpha=alphas[0], **kwargs),
        LGBMRegressor(objective="quantile", alpha=alphas[1], **kwargs),
        LGBMRegressor(objective="quantile", alpha=alphas[2], **kwargs),
    )
