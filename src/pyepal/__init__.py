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


"""PyePAL"""
from .models.nt import JaxOptimizer, NTModel
from .pal.pal_base import PALBase
from .pal.pal_coregionalized import PALCoregionalized
from .pal.pal_finite_ensemble import PALJaxEnsemble
from .pal.pal_gbdt import PALGBDT
from .pal.pal_gpflowgpr import PALGPflowGPR
from .pal.pal_gpy import PALGPy
from .pal.pal_neural_tangent import PALNT
from .pal.pal_sklearn import PALSklearn
from .pal.utils import exhaust_loop, get_hypervolume, get_kmeans_samples, get_maxmin_samples
from .version import VERSION

__version__ = VERSION

__all__ = [
    "PALBase",
    "PALCoregionalized",
    "PALGBDT",
    "PALGPy",
    "PALGPflowGPR",
    "PALSklearn",
    "PALJaxEnsemble",
    "PALNT",
    "NTModel",
    "JaxOptimizer",
    "exhaust_loop",
    "get_hypervolume",
    "get_kmeans_samples",
    "get_maxmin_samples",
]
