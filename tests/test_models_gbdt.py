# -*- coding: utf-8 -*-
# Copyright 2020 PyePAL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing the GBDT utilities"""
from pyepal.models.gbdt import build_gbdt_tuple


def test_build_gbdt_tuple():
    """Test of the construction of a tuple
    of GBDTs for quantile regression works"""
    gbdt_tuple = build_gbdt_tuple()
    assert len(gbdt_tuple) == 3
    for model in gbdt_tuple:
        assert model.objective == "quantile"
        assert model.alpha > 0  # pylint:disable=no-member
