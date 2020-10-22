# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
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


"""Testing the scheduling utility functions"""
from pypal.pal.schedules import exp_decay, linear


def test_linear():
    """Testing the scheduler with equally spaced intervals"""
    assert linear(10, 10)
    assert not linear(9, 10)
    assert not linear(11, 10)
    assert linear(1, 10)
    assert not linear(8, 10)


def test_exp_decay():
    """Testing the logarithmically spaced schedule"""
    assert exp_decay(1, 10)
    assert exp_decay(100)
    assert exp_decay(1000)
    assert not exp_decay(200)

    assert exp_decay(1, 2)
    assert exp_decay(2, 2)
    assert exp_decay(16, 2)
