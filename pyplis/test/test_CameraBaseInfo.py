# -*- coding: utf-8 -*-
"""Pyplis test module for camera_base_info.py base module of Pyplis.
"""
from pyplis.camera_base_info import CameraBaseInfo
from datetime import datetime

def test_get_img_meta_from_filename():
    cam = CameraBaseInfo()
    flags = cam._fname_access_flags
    assert flags["filter_id"] is False
    assert flags["texp"] is False
    assert flags["meas_type"] is False
    assert flags["start_acq"] is False

    cam.filename_regexp = r'^.*_(?P<date>.*)_(?P<meas_type>(?P<filter_id>.*))_.*'  # noqa: E501
    cam.time_info_str = "%Y%m%d%H%M%S%f"
    filepath = '/tmp/EC2_1106307_1R02_2015091607003032_F01_Etna.fts'

    (acq_time, filter_id, meas_type,
        texp, warnings) = cam.get_img_meta_from_filename(file_path=filepath)

    assert flags["filter_id"] is True
    assert flags["texp"] is False
    assert flags["meas_type"] is True
    assert flags["start_acq"] is True
    assert acq_time == datetime(2015, 9, 16, 7, 0, 30, 320000)
    assert filter_id == meas_type == 'F01'
    assert texp is None
    assert len(warnings) == 0


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

    extracted_metadata = {
            "start_acq": None, #datetime object
            "filter_id": None, #str
            "meas_type": None, #str
            "texp": None, #float
            "date": None #str
        }
    cam = CameraBaseInfo()

    extracted_metadata = cam.parse_meta_from_filename(filename, extracted_metadata, config)
    assert extracted_metadata["date"] == '2015091607003032'
    assert extracted_metadata["filter_id"] == 'F01'
    assert extracted_metadata["meas_type"] == 'F01'
    assert extracted_metadata["texp"] is None

    # compare regexp parser

    config = {
        "filename_regexp":
            r'^.*_(?P<date>.*)_(?P<meas_type>(?P<filter_id>.*))_.*'
    }
    values_r = cam.parse_meta_from_filename_regexp(filename, extracted_metadata, config)

    assert extracted_metadata == values_r
