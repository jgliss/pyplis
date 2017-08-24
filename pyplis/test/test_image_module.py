# -*- coding: utf-8 -*-
"""
Pyplis test module for image.py base module of Pyplis
"""

import pytest
from pyplis import Img

def test_empty_img():
    img=Img()
    assert img.img == None

