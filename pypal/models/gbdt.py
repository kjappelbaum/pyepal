# -*- coding: utf-8 -*-
"""Some utilities to work with LightGBM models"""
from typing import List, Tuple, Union

import numpy as np
from lightgbm import LGBMRegressor


def build_gbdt_tuple(
    alphas: Union[List[float], np.ndarray] = [0.16, 0.5, 0.84],
    **kwargs  # pylint:disable=dangerous-default-value
) -> Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]:
    """Build a Tuple of LGBMRegressors in the correct
    format for PALGBDT

    Args:
        alphas (Union[List[float], np.ndarray], optional): Quantile levels.
            Defaults to [0.16, 0.5, 0.84]
            (about 68% of the probability under a
            normal curve lies between 𝜇±𝜎.)

    Returns:
        Tuple[LGBMRegressor, LGBMRegressor, LGBMRegressor]
    """
    return (
        LGBMRegressor(loss="quantile", alpha=alphas[0], **kwargs),
        LGBMRegressor(loss="quantile", alpha=alphas[1], **kwargs),
        LGBMRegressor(loss="quantile", alpha=alphas[2], **kwargs),
    )
