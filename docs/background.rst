Background
===========

This package implements a modified version of the `ε-PAL algorithm from Zuluaga et al. <https://jmlr.org/papers/v17/15-047.html>`_ in an object oriented interface for finding the Pareto efficient points in any number of dimensions with any model that can output a standard deviation and a mean.

This implementation has the following features:

- We make sure that the sampling is scale invariant and that the algorithm can deal with positive and negative objective values.

- Instead of using the predicted :math:`\hat{\mu}` and :math:`\hat{\sigma}` also for the sampled points we use the measured :math:`\mu` and :math:`\sigma`.

- This implementation is directly applicable to :math:`n`-dimensional problems.

- It can be easily used with any kind of model. For example, one might replace the Gaussian process model with a neural network with `Dropout Monte Carlo <http://proceedings.mlr.press/v48/gal16.pdf>`_ for the uncertainity estimate.

In our own work, we used this algorithm for materials discovery.


Intuition behind the algorithm
--------------------------------
