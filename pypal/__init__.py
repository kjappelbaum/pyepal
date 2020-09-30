# -*- coding: utf-8 -*-
"""Python PAL"""
from ._version import get_versions
from .pal.pal_base import PALBase
from .pal.pal_coregionalized import PALCoregionalized
from .pal.pal_gpy import PALGPy
from .pal.pal_sklearn import PALSklearn

__version__ = get_versions()["version"]
del get_versions
