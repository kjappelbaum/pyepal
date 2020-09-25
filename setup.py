# -*- coding: utf-8 -*-
"""Model agnostic Python implementation of the epsilon-PAL algorithm"""

from setuptools import find_packages, setup

import versioneer

with open('requirements.txt', 'r') as fh:
    requirements = [line.strip() for line in fh]  # pylint:disable=invalid-name

setup(
    name='PyPAL',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    url='',
    license='MIT',
    python_requires='<3.8>=3.7',
    install_requires=requirements,
    extras_require={
        'testing': ['pytest', 'pytest-cov<2.11'],
        'docs': ['guzzle_sphinx_theme'],
        'pre-commit': [
            'pre-commit',
            'yapf',
            'prospector',
            'pylint',
            'versioneer',
            'isort',
            'gitchangelog',
        ],
    },
    author='Kevin M. Jablonka, Brian Yoo, Berend Smit',
    author_email='kevin.jablonka@epfl.ch, brian.yoo@basf.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
