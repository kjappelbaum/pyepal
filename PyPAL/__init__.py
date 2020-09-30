# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""pypal"""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .pal.pal_base import PALBase
from .pal.pal_coregionalized import PALCoregionalized
from .pal.pal_gpy import PALGPy
from .pal.pal_sklearn import PALSklearn

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
