# -*- coding: utf-8 -*-
u"""Pyplis test module for camera_base_info.py base module of Pyplis.
"""
from __future__ import (absolute_import, division)
from pyplis.camera_base_info import CameraBaseInfo
from datetime import datetime
from pytest import fixture
from distutils import dir_util
import os


@fixture
def datadir(tmpdir, request):
    """Search for test data files.

    Search for a folder with the same name of test module and, if available,
    move all contents to a temporary directory so tests can use them freely.
    Adapted from https://stackoverflow.com/a/29631801
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def test_get_img_meta_from_filename(datadir):
    cam = CameraBaseInfo()
    flags = cam._fname_access_flags
    assert flags["filter_id"] is False
    assert flags["texp"] is False
    assert flags["meas_type"] is False
    assert flags["start_acq"] is False

    cam.filename_regexp = r'^.*_(?P<date>.*)_(?P<meas_type>(?P<filter_id>.*))_.*'  # noqa: E501
    cam.time_info_str = "%Y%m%d%H%M%S%f"
    filename = datadir.join('EC2_1106307_1R02_2015091607003032_F01_Etna.fts')

    (acq_time, filter_id, meas_type,
        texp, warnings) = cam.get_img_meta_from_filename(str(filename))

    assert flags["filter_id"] is True
    assert flags["texp"] is False
    assert flags["meas_type"] is True
    assert flags["start_acq"] is True
    assert acq_time == datetime(2015, 9, 16, 7, 0, 30, 320000)
    assert filter_id == meas_type == 'F01'
    assert texp is None
    assert len(warnings) == 1


def test_parse_filename_parsers():
    """Test and compare regex w/ std parser."""
    filename = 'EC2_1106307_1R02_2015091607003032_F01_Etna.fts'

    config = {
        "delim": "_",
        "time_info_pos": 3,
        "time_info_subnum": 1,
        "time_info_str": "%Y%m%d%H%M%S%f",
        "filter_id_pos": 4,
        "fid_subnum_max": 1,
        "meas_type_pos": 4,
        "mtype_subnum_max": 1,
        "texp_pos": "",
        "texp_unit": "ms"
    }

    cam = CameraBaseInfo()
    cam.parse_filename(filename, config)

    values = cam.parse_filename(filename, config)
    assert values["date"] == '2015091607003032'
    assert values["filter_id"] == 'F01'
    assert values["meas_type"] == 'F01'
    assert values["texp"] is None

    # compare regexp parser

    config = {
        "filename_regexp":
            r'^.*_(?P<date>.*)_(?P<meas_type>(?P<filter_id>.*))_.*'
    }
    values_r = cam.parse_filename_regexp(filename, config)

    assert values == values_r
