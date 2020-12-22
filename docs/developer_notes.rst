Developer notes
================

Contribution Guidelines
-------------------------

Commit messages
.................

- To automatically generate the changelog and releases we use `conventional commits <https://www.conventionalcommits.org/en/v1.0.0-beta.2/>`_ use the prefix :code:`feat` for new features, :code:`chore` `for updating grunt tasks etc; no production code change <https://stackoverflow.com/questions/26944762/when-to-use-chore-as-type-of-commit-message>`_, :code:`fix` for bug fixes and :code:`docs` for changes to the documentation. Use :code:`feat!:`, or :code:`fix!:`, :code:`refactor!:`, etc., to represent a breaking change (indicated by the `!`). This will result in bump of the SemVer major version number.


Python code
.................

Please install the pre-commit hooks using

.. code::bash

    pip install pre-commit
    pre-commit install .


to automatically

- format the code with `black <https://github.com/psf/black>`_
- sort the imports with `isort <https://pycqa.github.io/isort/>`_
- lint the code with `pylint <https://pylint.org/>`_

We use type hints, which we feel is a good way of documentation and helps us find bugs using `mypy <http://mypy-lang.org/>`_.

Some of the pre-commit hooks modify the files, e.g., they trim whitespaces or format the code. If they modify your file, you will have
to run :code:`git add` and :code:`git commit` again. To skip the pre-commit checks (not recommended) you can use :code:`git commit --no-verify`.

New features
.................

Please make a new branch for the development of new features. `Rebase on the upstream master <https://medium.com/@ruthmpardee/git-fork-workflow-using-rebase-587a144be470>`_ and include a test for your new feature. (The CI checks for a drop in code coverage.)


Releases
.................

Releases are automated using a GitHub actions based on the commit message. Maintainers manually upload the release to PyPi.


.. _new_pal_class:

Implementing a new PAL class
-----------------------------

If you want to use PyePAL  with a model that we do not support yet, i.e., not :code:`GPy` or :code:`sklearn` Gaussian process regression, it is easy to write your own class. For this, you will need to inherit from :py:class:`~pyepal.pal.pal_base.PALBase` and implement your  :py:obj:`~pyepal.pal.pal_base.PALBase._train` and  :py:func:`~pyepal.pal.pal_base.PALBase._predict` functions (and maybe also the  :py:obj:`pyepal.pal.pal_base.PALBase._set_hyperparameters` and  :py:obj:`pyepal.pal.pal_base.PALBase._should_optimize_hyperparameters` functions) using the :code:`design_space` and :code:`y` attributes of the class.

For instance, if we develop some multioutput model that has a :code:`train()` and a :code:`predict()` method, we could simply use the following design pattern

.. code-block:: python

    from pyepal import PALBase

    class PALMyModel(PALBase):
        def _train(self):
            self.models[0].train(self.design_space[self.sampled], self.y[self.sampled])

        def _predict(self):
            self.mu, self.std = self.models[0].predict(self.design_space)


Note that we typically provide the models, even if it is only one, in a list to keep the API consistent.

In some instances, you may want to perform an operation in parallel, e.g., train the models for different objectives in parallel. One convenient way to do this in Python is by using `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_. The only caveat to this that this approach requires that the function is picklable. To ensure this, you may want to implement the function that you want to parallelize, outside the class. For example, you could use the following design pattern

.. code-block:: python

    from pyepal import PALBase
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
