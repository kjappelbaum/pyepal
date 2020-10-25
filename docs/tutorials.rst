Tutorials
============

To explore different use cases of PyPAL, we recommend checking out the `example notebooks <https://github.com/kjappelbaum/pypal/tree/master/examples>`_.
All notebooks can be run without installation on MyBinder. In the folder, you find the notebooks with the output cells, if you rerun them, the execution
should not take more than a few minutes.


1. One active learning step using GPR models built with :code:`GPy`
----------------------------------------------------------------------

.. image:: _static/basic_pypal_screenshot.png
    :width: 400px
    :align: center

Topic covered
................

- building a pal_coregionalized GPR model using :py:meth:`pypal.pal.models.gpr.build_coregionalized_model`
-  using coregionalized models with :py:obj:`pypal.pal.pal_coregionalized.PALCoregionalized`
- attributes of the :code:`PAL` instance
- :py:meth:`pypal.pal.utils.exhaust_loop`


2. Active learning with "measure" function and :code:`sklearn` models
-----------------------------------------------------------------------

.. image:: _static/active_learning_screenshot.png
    :width: 400px
    :align: center

Topic covered
................

- using :code:`sklearn`  models with :py:obj:`pypal.pal.pal_sklearn.PALSklearn`
- selecting an initial set with :py:meth:`pypal.pal.utils.get_maxmin_samples`
- plotting with :py:meth:`pypal.plotting.make_jointplot`



3. Quantile regression
-----------------------

.. image:: _static/quantile_regression_screenshot.png
    :width: 400px
    :align: center


- Using LightGBM models with quantile loss with :py:obj:`pypal.pal.pal_gbdt.PALGBDT`
- selecting an initial set with :py:meth:`pypal.pal.utils.get_kmeans_samples`
