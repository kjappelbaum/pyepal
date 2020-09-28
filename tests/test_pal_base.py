# -*- coding: utf-8 -*-
"""Testing the PAL base class"""
# pylint:disable=protected-access
import pytest

from PyPAL.pal.pal_base import PALBase


def test_pal_base(make_random_dataset):
    """Testing basic functionality of the PAL base class"""
    palinstance = PALBase(make_random_dataset[0], ["model"], 3)
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_pareto_optimal_points == 0
    assert palinstance.number_unclassified_points == 100

    assert len(palinstance.discarded_points) == 0
    assert len(palinstance.pareto_optimal_points) == 0
    assert len(palinstance.unclassified_points) == 100

    assert (
        str(palinstance)
        == "PyPAL at iteration 0. \
        0 Pareto optimal points, \
        0 discarded points, \
        100 unclassified points."
    )
    assert (
        palinstance._log()
        == "PyPAL at iteration 0. \
        0 Pareto optimal points, \
        0 discarded points, \
        100 unclassified points."
    )

    assert palinstance._should_optimize_hyperparameters()
    assert not palinstance._has_train_set

    with pytest.raises(ValueError):
        palinstance.sample()
