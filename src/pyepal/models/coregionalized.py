# -*- coding: utf-8 -*-
# Copyright (c) 2012 - 2014 the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Custom version of GPCoregionalizedRegression with set_XY that works.
Needs to be used until the upstream PR gets merged"""
import numpy as np
from GPy import kern, util
from GPy.core import GP
from paramz import ObsAr

__all__ = ["GPCoregionalizedRegression"]


class GPCoregionalizedRegression(GP):  # pylint:disable=too-many-ancestors
    """
    Gaussian Process model for heteroscedastic multioutput regression
    This is a thin wrapper around the models.GP class, with a set of sensible defaults
    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W'
        (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        X_list,
        Y_list,
        kernel=None,
        normalizer=None,
        likelihoods_list=None,
        name="GPCR",
        W_rank=1,
        kernel_name="coreg",
    ):

        # Input and Output
        (
            X,  # pylint:disable=invalid-name
            Y,  # pylint:disable=invalid-name
            self.output_index,
        ) = util.multioutput.build_XY(X_list, Y_list)
        Ny = len(Y_list)  # pylint:disable=invalid-name

        # Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1] - 1)

            kernel = util.multioutput.ICM(
                input_dim=X.shape[1] - 1,
                num_outputs=Ny,
                kernel=kernel,
                W_rank=W_rank,
                name=kernel_name,
            )

        # Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list, self.output_index, likelihoods_list)

        super(GPCoregionalizedRegression, self).__init__(  # pylint:disable=super-with-arguments
            X,
            Y,
            kernel,
            likelihood,
            Y_metadata={"output_index": self.output_index},
            normalizer=normalizer,
        )

    def set_XY(self, X=None, Y=None):
        if isinstance(X, list):
            X, _, self.output_index = util.multioutput.build_XY(X, None)
        if isinstance(Y, list):
            _, Y, self.output_index = util.multioutput.build_XY(Y, Y)

        self.update_model(False)
        if Y is not None:
            if self.normalizer is not None:
                self.normalizer.scale_by(Y)
                self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
                self.Y = Y
            else:
                self.Y = ObsAr(Y)
                self.Y_normalized = self.Y
        if X is not None:
            self.X = ObsAr(X)

        self.Y_metadata = {
            "output_index": self.output_index,
            "trials": np.ones(self.output_index.shape),
        }

        self.update_model(True)
