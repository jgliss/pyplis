# -*- coding: utf-8 -*-
u"""Pyplis test module for dataset.py base module of Pyplis.

Author: Jonas Gliss
Email: jonasgliss@gmail.com
License: GPLv3+
"""
from pyplis import Dataset, __dir__ as pyplis__dir__, Img
from os.path import join
import pytest

EC2_IMG_PATH = join(pyplis__dir__, "data", "test_201509160708_F01_335.fts")


@pytest.fixture(scope="module")
def ec2_img():
    """Load and return test image."""
    return Img(EC2_IMG_PATH)


def test_empty_dataset():
    ds = Dataset()
    info = ds.check_filename_info_access(EC2_IMG_PATH)
    target_vals = ["on", 4, 0]
    actual_vals = [list(ds.lists_access_info.keys())[0],
                   len(info),
                   sum(info.values())]

    assert actual_vals == target_vals
