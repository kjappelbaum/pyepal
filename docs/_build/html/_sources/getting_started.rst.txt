Getting Started
================

Installation
---------------

You can install `pypal` from `PyPi` using

.. code-block:: python

    pip install pypal

We recommend that you install `pypi` in a dedicated `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ or `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

If you want to use the latest version from GitHub, you can install it using

.. code-block:: python

    pip install git+https://github.com/kjappelbaum/pypal.git


Running an active learning experiment
---------------------------------------

The `examples` directory contains a `Jupyter notebook with an example <https://github.com/kjappelbaum/pypal/blob/master/examples/test_pal.ipynb>`_ that you can also run on MyBinder.

If you use a Gaussian process model built with `sklearn` or `GPy` you can use a pre-built class and follow the following steps:

1. For each objective create a model (if you want to use a coregionalized model you, of course, only need to create one)

2. Sample a few initial points from your design space. In practice, you can use the `get_maxmin_samples` or `get_kmeans_samples` utilities for that. Assuming that `X` is a `np.array` if the descriptors/features

    .. code-block:: python

        from pypal import get_kmeans_samples, get_maxmin_samples

        # This selects the 10 points closest to the centroids of a k=10 means clustering
        indices = get_kmeans_samples(X, 10)

        # This selects the 10 farthest points in feature space
        indices = get_maxmin_samples(X, 10)

3. Now, you can intialize the instance of one `PAL` class. If we use a `sklearn` Gaussian process model, we would use

    .. code-block:: python

        from pypal import PALSklearn

        # Each of these models is an instance of sklearn.gaussian_process.GaussianProcessRegressor
        models = [gpr0, gpr1, gpr2]

        # We always need to provide the feature matrix (X), a list of models, and the number of objectives
        palinstance = PALSklearn(X, models, 3)

        # Now, we can also feed in the first measurements
        # this here assumes that in y you have all measurements and you now
        # provide the ones which index is in the indices array
        palinstance.update_train_set(indices, y[indices])

        # Now you can run one step
        next_idx = palinstance.run_one_step()

    At this level you have a range of different options you can set.

    - `epsilon`: in a `np.array` you can provide one :math:`\epsilon` per dimension. This allows you to set looser tolerance for some objectives. Note that :math:`\epsilon_i \in [0,1]`.
    - `delta`: allows you to specify the :math:`\delta` hyperparameter (:math:`\delta \in [0,1]`). Increasing this value will spped up the convergence.
    - `beta_scale`: allows you to provide an empirical scaling parameter for beta. The theoretical guarantees in the PAL paper are derived for this parameter set to 1. But in practice, you can achieve much faster convergence by setting it to a number :math:`0< \beta_\mathrm{scale} \ll 1`. As shown in the figure below, :math:`\beta` depends on :math:`\delta`

        .. image:: _static/beta.png
            :width: 600
            :alt: Beta as function of hyperparameters

    - `goal`: By default, `pypal` assumes that you want to maximize every objective. If this is not the case, you can set the `goal` argument using a list of "min" and "max", using "min" to specify that you want to minimize the ith objective and "max" to indicate that you want to maximize this objective.

In case you have missing observations, i.e., you measured only two of three outputs at sometimes you need to report the missing observations as `np.nan`, i.e., the call could look like

.. code-block:: python

    palinstance.update_train_set(np.array([1,2]), np.array([1, 2, 3], [np.nan, 1, 2, 0]])

for a case in which we performed measurements for samples 1 and 2 of our design space but didn't measure the first target for sample 2.

Hyperparameter optimization
.............................
Usually, the hyperparameters of a machine learning model should be optimized as new training data is added, in particular the kernel hyperparameters of a Gaussian process regression model. But since this is usually a computationally expensive process, you do not want to do this every iteration. The timing of the hyperparameter optimization is internally set by the `_should_optimize_hyperparameter` function that by default uses a schedule that will optimize the hyperparameter every 10th iteration. If you want to change this behavior, you can override this function.

Logging
........
You will see basic information like the current iteration and the classification status if you print the `PAL` object

.. code:: python

    print(palinstance)

    # returns: pypal at iteration 1. 10 Pareto optimal points, 1304 discarded points, 200 unclassified points.


In case you want to also know the hypervolume, you can use the `get_hypervolume` function

.. code:: python

    hv = get_hypervolume(palinstance.means[palinstance.pareto_optimal])


Exploring a space where all objectives are known
.................................................

In some cases, you already know all measurements you may want to run PAL with different settings and test how the algorithm performs.
In this case you can use the `exhaust_loop` wrapper.

.. code-block:: python

    from pypal import PALSklearn, exhaust_loop
    models = [gpr0, gpr1, gpr2]
    palinstance = PALSklearn(X, models, 3)

    exhaust_loop(palinstance, y)

This will continue calling `run_one_step()` until there is no unclassified sample left.


Batch sampling
................

By default, the `run_one_step` function of the PAL classes will return a `np.array` with only one index for the point in the design space for which the next experiment should be performed. In some cases, you want to run multiple experiments in batches before you run a new iteration of the PyPAL. In this case, you use the `batch_size` keyword argument and change it to an integer greater than one.

.. code-block:: python

    next_idx = palinstance.run_one_step(batch_size=10)
    # next_idx will be a np.array of length 10

Of course, also the `exhaust_loop` supports the `batch_size` keyword argument

.. code-block:: python

    palinstance = PALSklearn(X, models, 3)

    # sample always 10 points and do this until there is no unclassified
    # point left
    exhaust_loop(palinstance, y, batch_size=10)


Implementing a new PAL class
------------------------------

If you want to use `pypal` with a model that we do not support yet, i.e., not `GPy` or `sklearn` Gaussian process regression, it is easy to write your own class. For this, you need to inherit from `PALBase` and implement your of `_train` and `_predict` functions (and maybe also the `_set_hyperparameters` and `_should_optimize_hyperparameters` functions) using the `design_space` and `y` attributes of the class.

For instance, if we develop some multioutput model that has a `train()` and a `predict()` method we could simply do

.. code-block:: python

    from pypal import PALBase

    class PALMyModel(PALBase):
        def _train(self):
            self.models[0].train(self.design_space[self.sampled], self.y[self.sampled])

        def _predict(self):
            self.mu, self.std = self.models[0].predict(self.design_space)


Note that we typically provide the models, even if it is only one, in a list to keep the API consistent.

In some instances, you might want to perform an operation in parallel, e.g., train the models for different objectives in parallel. One convenient way to do this in Python is `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_. The only hitch is that this approach requires that the function is picklable. To ensure is, you may want to implement the function that is to be run in parallel outside the class. For example, you could use the following design pattern

.. code-block:: python

    from pypal import PALBase
    import concurrent.futures
    from functools import partial

    def _train_model_picklable(i, models, design_space, objectives, sampled):
        model = models[i]
        model.fit(
            design_space[sampled[:, i]],
            objectives[sampled[:, i], i].reshape(-1, 1),
        )
        return model

    class MyPal(PALBase):
        def __init__(self, *args, **kwargs):
            n_jobs = kwargs.pop("n_jobs", 1)
            validate_njobs(n_jobs)
            self.n_jobs = n_jobs
            super().__init__(*args, **kwargs)

            validate_number_models(self.models, self.ndim)

        def _train(self):
            train_single_partial = partial(
                _train_model_picklable,
                models=self.models,
                design_space=self.design_space,
                objectives=self.y,
                sampled=self.sampled,
            )
            models = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_jobs
            ) as executor:
                for model in executor.map(train_single_partial, range(self.ndim)):
                    models.append(model)
            self.models = models
