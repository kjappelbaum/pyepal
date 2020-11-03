# -*- coding: utf-8 -*-
# Copyright 2020 PyePAL authors
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
"""Model agnostic Python implementation of the epsilon-PAL algorithm"""

from setuptools import find_packages, setup

import versioneer

with open("requirements.txt", "r") as fh:
    REQUIREMENTS = [line.strip() for line in fh]


with open("README.md", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name="pyepal",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/kjappelbaum/PyePAL",
    license="Apache 2.0",
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    extras_require={
        "testing": ["pytest", "pytest-cov<2.11"],
        "docs": [
            "sphinx",
            "pydata-sphinx-theme",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
        ],
        "pre-commit": [
            "pre-commit",
            "black",
            "prospector",
            "pylint",
            "versioneer",
            "isort",
            "gitchangelog",
        ],
        "GPy": ["GPy==1.9.9", "matplotlib"],
        "GBDT": ["lightgbm>=3.0.0"],
    },
    author="PyePAL authors",
    author_email="kevin.jablonka@epfl.ch, brian.yoo@basf.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
