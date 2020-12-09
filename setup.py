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
    REQUIREMENTS = [line.strip().split(";")[0] for line in fh]


with open("README.md", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

gpy_requirements = ["GPy==1.9.9", "matplotlib"]
gbdt_requirements = ["lightgbm~=3.0.0"]
neural_tangents_requirements = ["neural_tangents~=0.3.5", "jaxlib~=0.1.57"]
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
        "testing": ["pytest~=6.1.0", "pytest-cov~=2.11"],
        "docs": [
            "sphinx~=3.2.1",
            "sphinx-book-theme~=0.0.39",
            "sphinx-autodoc-typehints~=1.11.0",
            "sphinx-copybutton~=0.3.0",
        ],
        "pre-commit": [
            "pre-commit~=2.7.1",
            "black~=20.8",
            "pylint~=2.6",
            "versioneer~=0.18",
            "isort~=5.5.3",
        ],
        "GPy": gpy_requirements,
        "GBDT": gbdt_requirements,
        "neural_tangents": neural_tangents_requirements,
        "all": neural_tangents_requirements + gbdt_requirements + gpy_requirements,
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
