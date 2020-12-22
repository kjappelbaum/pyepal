<p align="center">
 <img src="pyepal_logo.png" />
</p>

|                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Continuous integration      | ![Python package](https://github.com/kjappelbaum/pyepal/workflows/Python%20package/badge.svg) ![pre-commit](https://github.com/kjappelbaum/pyepal/workflows/pre-commit/badge.svg)                                                                                                                                                                                                                                                                                                                                                                                                       |
| Code health                | [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Maintainability](https://api.codeclimate.com/v1/badges/db9b3f21528574dfb141/maintainability)](https://codeclimate.com/github/kjappelbaum/pyepal/maintainability) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kjappelbaum/pyepal.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kjappelbaum/pyepal/context:python) ![GitHub last commit](https://img.shields.io/github/last-commit/kjappelbaum/pyepal) [![codecov](https://codecov.io/gh/kjappelbaum/pyepal/branch/master/graph/badge.svg?token=BL2CF4HQ06)](https://codecov.io/gh/kjappelbaum/pyepal)|
| Documentation and tutorial | [![Documentation Status](https://readthedocs.org/projects/pyepal/badge/?version=latest)](https://pyepal.readthedocs.io/en/latest/?badge=latest) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kjappelbaum/pyepal/HEAD?filepath=examples)                                                                                                                                                                                                                                                                                                                  |
| Social                     | [![Gitter](https://badges.gitter.im/kjappelbaum/pyepal.svg)](https://gitter.im/kjappelbaum/pyepal?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Python                     | ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyepal) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)                                                                                                                                                                                                                                                                                                                                                                                                |
| License                    | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [Citation](#citation) |[![Paper DOI](https://img.shields.io/badge/DOI-10.26434/chemrxiv.13200197.v1-blue.svg)](https://chemrxiv.org/articles/preprint/Bias_Free_Multiobjective_Active_Learning_for_Materials_Design_and_Discovery/13200197) [![Zenodo archive](https://zenodo.org/badge/253408969.svg)](https://zenodo.org/badge/latestdoi/253408969) |


Generalized Python implementation of a modified version of the ε-PAL algorithm [[1](#1), [2](#2)].

For more detailed docs [go here](https://pyepal.readthedocs.io/en/latest/?badge=latest).

## Installation

To install the latest stable release use

```(bash)
pip install pyepal
```

to install the latest development version from the head use

```(bash)
pip install git+https://github.com/kjappelbaum/pyepal.git
```

Developers can install the extras `[testing, docs, pre-commit]`. Installation should take only a few minutes.

### Additional Notes

- On MacOS you might need to install `libomp` (e.g., `brew install libomp`) for multithreading in some of the models.

- We currently support Python 3.7 and 3.8.


## Usage

The main logic is implemented in the `PALBase` class. There are some prebuilt classes for common use cases (`GPy`, `sklearn`) that inherit from this class.
For more details about how to use the code and notes about the tutorials [see the docs](https://kjappelbaum.github.io/pyepal/).

### Pre-Built classes

#### scikit-learn

If you want to use a list of [sklearn](https://scikit-learn.org/stable/index.html) models, you can use the `PALSklearn` class. To use it for one step,
you can follow the following code snippet. The basic principle is the same for all the different `PAL` classes.

```python
from pyepal import PALSklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# For each objective, initialize a model
gpr_objective_0 = GaussianProcessRegressor(RBF())
gpr_objective_1 = GaussianProcessRegressor(RBF())

# The minimal input to create a PAL instance is a list of models,
# the design space (X, in ML terms "feature matrix") and the number of objectives
palsklearn_instance = PALSklearn(X, [gpr_objective_0, gpr_objective_1], 2)

# the next step is to provide some initial measurements.
# You can do this with the update_train_set function, which you
# can use throughout the active learning process to update the training set.
# For this, provide a numpy array of indices in your design space
# and the corresponding measurements
sampled_indices = np.array([1,2,3])
measurements = np.array([[1,2],
                        [0.8, 1],
                        [7,1]])
palsklearn_instance.update_train_set(sampled_indices, measurements)

# Now, you're ready to run the first iteration.
# This will return the next index to sample and update all the attributes
# If there are no unclassified samples left, it will return None and
# print a statement saying that the classification is completed
index_to_sample = palsklearn_instance.run_one_step()
```

#### GPy

If you want to use a list of [GPy](https://sheffieldml.github.io/GPy/) models, you can use the `PALGPy` class.

#### Coregionalized GPR

Coregionalized GPR models can utilize correlations between the objectives and also work in the cases in which some of the objectives are not measured for all samples.

### Custom classes

You will need to implement the `_train()` and `_predict()` functions if you inherit from `PALBase`. If you want to tune the hyperparameters of your models while new training points are added, you can implement a schedule by setting the `_should_optimize_hyperparameters()` function and the `_set_hyperparameters()` function, which sets the hyperparameters for the model(s).

If you need to train a model, use `self.design_space` as the feature matrix and `self.y` as the target vector. Note that in `self.y` all objectives are turned into maximization problems. That is, if one of your problems is a minimization problem, PyePAL will flip its sign in `self.y`.

A basic example of how a custom class can be implemented is the `PALSklearn` class:

```python
class PALSklearn(PALBase):
    """PAL class for a list of Sklearn (GPR) models, with one model per objective"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        validate_number_models(self.models, self.ndim)

    def _train(self):
        for i, model in enumerate(self.models):
            model.fit(self.design_space[self.sampled], self.y[self.sampled, i].reshape(-1,1))

    def _predict(self):
        means, stds = [], []
        for model in self.models:
            mean, std = model.predict(self.design_space, return_std=True)
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        self.means = np.hstack(mean)
        self.std = np.hstack(stds)
```

For scheduling of the hyperparameter optimization, we have some predefined schedules in the `pyepal.pal.schedules` module.

### Test the algorithms

If the full design space is known, you can use a while loop to fully explore the space with PyePAL.
For the theoretical guarantees of PyePAL to hold, you'll need to sample until all uncertainties are below epsilon. In practice, it is usually enough to require as a termination criterion that there are no unclassified samples left. For this you can use the following snippet

```python
from pyepal.utils import exhaust_loop
from pyepal.models.gpr import build_model

# indices for initialization
sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])

# build one model per objective
model_0 = build_model(X[sample_idx], y[sample_idx], 0)
model_1 = build_model(X[sample_idx], y[sample_idx], 1)

# initialize the PAL instance
palinstance = PALGPy(X, [model_0, model_1], 2, beta_scale=1)
palinstance.update_train_set(sample_idx, y[sample_idx])

# This will run the sampling and training as long as there
# are unclassified samples
exhaust_loop(palinstance, y)
```

To measure the performance, you can use the `get_hypervolume` function from `pyepal.pal.utils`. More indicators are implemented in packages like [deap](https://github.com/DEAP/deap), [pagmo](https://github.com/esa/pagmo), or [pymoo](https://github.com/msu-coinlab/pymoo/tree/master).

## References

1. <a name="1"></a> Zuluaga, M.; Krause, A.; Püschel, M. E-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem. Journal of Machine Learning Research 2016, 17 (104), 1–32.
2. <a name="2"></a> Zuluaga, M.; Sergent, G.; Krause, A.; Püschel, M. Active Learning for Multi-Objective Optimization; Dasgupta, S., McAllester, D., Eds.; Proceedings of machine learning research; PMLR: Atlanta, Georgia, USA, 2013; Vol. 28, pp 462–470.

## Citation
<a name="citation"></a>

If you find this code useful for your work, please cite:

- Our paper that describes the implementation and an application to materials discovery: [Jablonka, K. M.; Giriprasad, M. J.; Wang, S.; Smit, B.; Yoo, B. Bias Free Multiobjective Active Learning for Materials Design and Discovery, ChemRxiv 2020 (10.26434/chemrxiv.13200197.v1).](https://chemrxiv.org/articles/preprint/Bias_Free_Multiobjective_Active_Learning_for_Materials_Design_and_Discovery/13200197)

- The original paper that describes the ε-PAL algorithm: [Zuluaga, M.; Krause, A.; Püschel, M. E-PAL: An Active Learning Approach to the Multi-Objective Optimization Problem. Journal of Machine Learning Research 2016, 17 (104), 1–32.](https://jmlr.csail.mit.edu/papers/volume17/15-047/15-047.pdf)

## Acknowledgments

The research was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([grant agreement 666983, MaGic](https://cordis.europa.eu/project/id/666983)), by the [NCCR-MARVEL](https://www.nccr-marvel.ch/), funded by the Swiss National Science Foundation, and by the Swiss National Science Foundation (SNSF) under Grant 200021_172759. Part of the work was performed as part of the [Explore Together internship program at BASF](https://www.basf.com/global/en/careers/students/explore-together.html).
