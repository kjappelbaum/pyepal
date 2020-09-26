# PyPAL

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PyPAL)
![GitHub](https://img.shields.io/github/license/kjappelbaum/PyPAL)
![Gitter](https://img.shields.io/gitter/room/kjappelbaum/PyPAL)
![GitHub last commit](https://img.shields.io/github/last-commit/kjappelbaum/PyPAL)

Generalized Python implementation of the ε-PAL algorithm [[1](#1), [2](#2)].

## Installation

To install the latest development version from the head use

```(bash)
pip install git+
```

To install a stable release use

```(bash)
pip install PyPAL
```

Developers can install the extras `[testing, docs, pre-commit]`.

## Usage

The main logic is implemented in the `PALBase` class. There are some pre-built classes for common use-cases (`GPy`, `sklearn`) that inherit from this class.

### Pre-Built classes

### Custom classes

## References

1. <a name="1"></a> Zuluaga, M.; Krause, A.; Püschel, M. E-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem. Journal of Machine Learning Research 2016, 17 (104), 1–32.
2. <a name="2"></a> Zuluaga, M.; Krause, A.; Sergent, G.; Puschel, M. Active Learning for Multi-Objective Optimization. 9.

## Acknowledgments
