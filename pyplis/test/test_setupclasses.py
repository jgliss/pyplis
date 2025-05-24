# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""Test module for setupclasses.py
"""
from pyplis.setupclasses import FilterSetup

def test_filter_setup():
    stp = FilterSetup()
    assert len(stp) == 1, len(stp)
    assert stp.ids_on == ['on']
    assert stp['on'] == stp.on_band

    from pyplis import Filter
    flst = [Filter('bla', type='on', acronym='blub'),
            Filter('off', type='off', acronym='kk')]

    stp['bla'] = flst[0]
    assert len(stp) == 2, len(stp)
    assert stp.default_key_on == 'on'
    stp.default_key_on = 'bla'
    assert stp.default_key_on == 'bla'

    stp = FilterSetup(filter_list=flst, one_more=Filter('add'))
    assert len(stp) == 3
    assert stp.default_key_off == 'off'
    assert stp.default_key_on == 'bla'

    assert stp.ids_on == ['bla', 'one_more'], stp.ids_on
