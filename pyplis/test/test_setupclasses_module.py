# -*- coding: utf-8 -*-
"""
Test environment for setupclasses.py module
"""

from pyplis import setupclasses as scl

def test_load_ecII():
    """This method checks if default information for EC2 camera can be loaded"""
    cam = scl.Camera("ecII")    
    assert cam.time_info_str == '%Y%m%d%H%M%S%f'
    
