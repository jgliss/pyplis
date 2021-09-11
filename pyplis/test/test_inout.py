# -*- coding: utf-8 -*-
"""Test environment for setupclasses.py module.

Author: Jonas Gliss
Email: jonasgliss@gmail.com
License: GPLv3+
"""
import pytest
import os
from pyplis import inout as mod

def test_create_temporary_copy(tmpdir):
    fpath = os.path.join(tmpdir, 'file.txt')
    with open(fpath, 'w') as f:
        f.write('bla')
    assert os.path.exists(fpath)
    loc = mod.create_temporary_copy(fpath)
    assert os.path.exists(loc)
    with open(loc) as f:
        assert f.readline() == 'bla'
