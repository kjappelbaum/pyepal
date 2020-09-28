# -*- coding: utf-8 -*-
"""Testing the scheduling utility functions"""
from PyPAL.pal.schedules import exp_decay, linear


def test_linear():
    """Testing the scheduler with equally spaced intervals"""
    assert linear(10, 10)
    assert not linear(9, 10)
    assert not linear(11, 10)
    assert linear(0, 10)
    assert not linear(8, 10)


def test_exp_decay():
    """Testing the logarithmically spaced schedule"""
    assert exp_decay(0, 10)
    assert exp_decay(100)
    assert exp_decay(1000)
    assert not exp_decay(200)

    assert exp_decay(0, 2)
    assert exp_decay(2, 2)
    assert exp_decay(16, 2)
