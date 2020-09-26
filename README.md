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

#### scikit-learn

If you want to use a list of [sklearn]() models, you cam use the `PALSklearn` class.

#### GPy

If you want to use a list of [GPy](https://sheffieldml.github.io/GPy/) models, you can use the `PALGPy` class.

#### Coregionalized GPR

Coregionalized GPR models can harvest correlations between the objectives and also work in the cases in which some of the objectives are not measured for all samples.

### Custom classes

You will need to implement the `_train()` and `_predict()` functions if you inherit from `PALBase`. If you want to tune the hyperparameters of your models while new training points are added, you can implement a schedule by setting the `_should_optimize_hyperparameters()` function and the `_set_hyperparameters()` function which sets the hyperparameters for the model(s).

A basic example of how a custom class can be implemented is the `PALGPy` class:

```(python)
class PALGPy(PALBase):
    def __init__(self, *args, **kwargs):
        self.restarts = kwargs.pop("restarts", 20) # if provided as keyword argument use it, otherwise use 20 as default
        self.parallel = kwargs.pop("parallel", False)
        assert isinstance(
            self.parallel, bool
        ), "the parallel keyword must be of type bool"
        assert isinstance(
            self.restarts, int
        ), "the restarts keyword must be of type int"
        super(PALGPy, self).__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)

    def _set_data(self):
        for model in self.models:
            model.set_xy(self.design_space[self.sampled], self.y[self.sampled])

    def _train(self):
        pass  # There is no training in instance based models

    def _predict(self):
        mus, stds = [], []
        for model in self.models:
            mu, std = model.predict(self.design_space)
            mus.append(mu.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(mus)
        self.stds = np.hstack(stds)

    def _set_hyperparameters(self):
        for model in self.models:
            model.optimize_restarts(self.restarts, parallel=self.parallel)
```

For scheduling for the hyperparameter optimization we have some predefined schedules in the `PyPAL.pal.schedules` module.

### Test the algorithms

If the full design space is known, you can use a while loop to fully explore the space.
For the theoretical guarantees to hold, you'll need to sample until all uncertainties are below epsilon. In practice, it is usually enough to require as termination criterion that there a no unclassified samples left.

## References

1. <a name="1"></a> Zuluaga, M.; Krause, A.; Püschel, M. E-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem. Journal of Machine Learning Research 2016, 17 (104), 1–32.
2. <a name="2"></a> Zuluaga, M.; Sergent, G.; Krause, A.; Püschel, M. Active Learning for Multi-Objective Optimization; Dasgupta, S., McAllester, D., Eds.; Proceedings of machine learning research; PMLR: Atlanta, Georgia, USA, 2013; Vol. 28, pp 462–470.

## Acknowledgments

The research was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement 666983, MaGic), by the NCCR-MARVEL, funded by the Swiss National Science Foundation, and by the Swiss National Science Foundation (SNSF) under Grant 200021_172759. Part of the work was performed during the Explore Together internship program at BASF.
