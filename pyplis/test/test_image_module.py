# -*- coding: utf-8 -*-
"""Pyplis test module for image.py base module of Pyplis.

Author: Jonas Gliss
Email: jonasgliss@gmail.com
License: GPLv3+
"""
from pyplis import Img, __dir__ as pyplis__dir__
from os.path import join, exists
from numpy import nan, zeros
from numpy.testing import assert_allclose
import pytest
import math

EC2_IMG_PATH = join(pyplis__dir__, "data", "test_201509160708_F01_335.fts")


def test_empty_img():
    """Test creation of empty image."""
    img = Img()
    assert img.img is None


def test_path_example_img():
    """Test if EC2 test image path is available."""
    val = exists(EC2_IMG_PATH)
    assert val is True


@pytest.fixture(scope="function")
def ec2_img():
    """Load and return test image."""
    return Img(EC2_IMG_PATH)


def test_img_numerical_props(ec2_img):
    """Test basic numerical properties (mean, min, max, std) of test image."""
    arr = [ec2_img.mean(), ec2_img.min(), ec2_img.max(), ec2_img.std()]
    assert_allclose(arr, [2413.0872, 1174.0, 3917.0, 450.4161], rtol=1e-4)


def test_pyramid_crop(ec2_img):
    """Test changing pyramid level in test image."""
    shapes = []
    shapes.append(ec2_img.pyr_down(3).shape)
    shapes.append(ec2_img.pyr_up(3).shape)
    shapes.append(ec2_img.crop([10, 10, 20, 20]).shape)
    assert shapes == [(128, 168), (1024, 1344), (10, 10)]


def test_meta_info(ec2_img):
    """Test if all relevant meta information is loaded."""
    assert ec2_img.shape == (1024, 1344)


@pytest.fixture(scope="function")
def binary_mask():
    """Binary mask for ec2 image."""
    mask = zeros((1024, 1344))
    mask[0:500, :] = 1
    return mask


@pytest.mark.parametrize("fill_value", [0.0, 10, -1.234])
def test_masked_img(ec2_img, binary_mask, fill_value):
    """Test if masking works."""
    masked_img = ec2_img.get_masked_img(mask=binary_mask,
                                        fill_value=fill_value)
    assert masked_img[200, 500] == fill_value


def test_masked_img_nan(ec2_img, binary_mask):
    """Test if masking works."""
    masked_img = ec2_img.get_masked_img(mask=binary_mask, fill_value=nan)
    assert math.isnan(masked_img[200, 500])
