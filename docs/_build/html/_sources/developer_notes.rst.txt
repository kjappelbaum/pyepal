Developer notes
================

Commit messages
----------------

- To automatically generate the changelog and version numbers we use `conventional commits <https://www.conventionalcommits.org/en/v1.0.0-beta.2/>`_ use the prefix :code:`feat` for new features, :code:`chore` `for updating grunt tasks etc; no production code change <https://stackoverflow.com/questions/26944762/when-to-use-chore-as-type-of-commit-message>`_, :code:`fix` for bug fixes and :code:`docs` for changes to the documentation.

Python code
--------------

Please install the pre-commit hooks to automatically

- format the code with `black <https://github.com/psf/black>`_
- sort the imports with `isort <https://pycqa.github.io/isort/>`_
- lint the code with `prospector <http://prospector.landscape.io/en/master/>`_

We use type hints, which we feel is a good way of documentation and helps us find bugs using `mypy <http://mypy-lang.org/>`_.

New features
--------------

Please make a new branch for the development of new features. `Rebase on the upstream master <https://medium.com/@ruthmpardee/git-fork-workflow-using-rebase-587a144be470>`_ and include a test for your new feature. (The CI checks for a drop in code coverage.)


Documentation
---------------

Currently, documentation is hosted on GitHub pages. Build it locally using :code:`make html` in the :code:`doc` directory and then push it to GitHub pages using

.. code-block:: bash

   git subtree push --prefix docs/_build/html  origin gh-pages
