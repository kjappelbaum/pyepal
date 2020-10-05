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

1. For each objective create a model (if you want to use a coregionalized model you of course only need to create one)

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

Implementing a new PAL class
------------------------------
