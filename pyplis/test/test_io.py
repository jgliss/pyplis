# -*- coding: utf-8 -*-
from pyplis import setupclasses as scl
from pyplis.inout import get_source_info_online
from collections import OrderedDict as od

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

def test_get_source_info_online():
    """Check if volcano location info can be retrieved from API""" 
    expected_etna = {'etna': od([
        ('name', 'Etna'),
        ('country', 'Italy'),
        ('region', 'Italy'),
        ('lat', 37.748),
        ('lon', 14.999),
        ('altitude', 3357),
        ('type', 'Stratovolcano'),
        ('status', 'Historical'),
        ('last_eruption', 'D1')])}

    expected_vesuvius = {"vesuvius": od([
        ('name', 'Vesuvius'),
        ('country', 'Italy'),
        ('region', 'Italy'),
        ('lat', 40.821),
        ('lon', 14.426),
        ('altitude', 1281),
        ('type', 'Complex volcano'),
        ('status', 'Historical'),
        ('last_eruption', 'D2')])}

    result_etna = get_source_info_online("etna")
    result_vesuvius = get_source_info_online("vesuvius")

    assert result_etna == expected_etna
    assert result_vesuvius != expected_etna
    assert result_vesuvius == expected_vesuvius
