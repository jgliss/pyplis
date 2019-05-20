# -*- coding: utf-8 -*-
"""Test environment for setupclasses.py module.

Author: Jonas Gliss
Email: jonasgliss@gmail.com
License: GPLv3+
"""
from __future__ import (absolute_import, division)
from pyplis import setupclasses as scl


def test_load_ecII():
    """Check if default information for EC2 camera can be loaded."""
    cam = scl.Camera("ecII")
    info_is = [cam.cam_id, cam.delim, cam.time_info_pos, cam.time_info_str,
               cam.filter_id_pos, cam.texp_pos, cam.file_type,
               cam.main_filter_id, cam.meas_type_pos, cam.darkcorr_opt,
               cam.focal_length, cam.pix_height, cam.pix_width,
               cam.pixnum_x, cam.pixnum_y]
    info = ["ecII", "_", 3, '%Y%m%d%H%M%S%f', 4, None, "fts", "on", 4,
            1, None, 4.65e-6, 4.65e-6, 1344, 1024]
    assert info_is == info
