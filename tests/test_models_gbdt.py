# -*- coding: utf-8 -*-
"""Testing the GBDT utilities"""
from pypal.models.gbdt import build_gbdt_tuple


def test_build_gbdt_tuple():
    """Test of the construction of a tuple
    of GBDTs for quantile regression works"""
    gbdt_tuple = build_gbdt_tuple()
    assert len(gbdt_tuple) == 3
    for model in gbdt_tuple:
        assert model.loss == "quantile"  # pylint:disable=no-member
        assert model.alpha > 0  # pylint:disable=no-member
