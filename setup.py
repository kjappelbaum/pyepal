# -*- coding: utf-8 -*-
"""Model agnostic Python implementation of the epsilon-PAL algorithm"""

from setuptools import find_packages, setup

import versioneer

with open("requirements.txt", "r") as fh:
    REQUIREMENTS = [line.strip() for line in fh]

setup(
    name="pypal",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url="",
    license="MIT",
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    extras_require={
        "testing": ["pytest", "pytest-cov<2.11"],
        "docs": ["sphinx", "pydata-sphinx-theme", "sphinx-autodoc-typehints"],
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
    },
    author="Kevin M. Jablonka, Brian Yoo, Berend Smit",
    author_email="kevin.jablonka@epfl.ch, brian.yoo@basf.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 1 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
