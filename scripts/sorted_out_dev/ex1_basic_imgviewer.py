# -*- coding: utf-8 -*-
"""
PISCOPE example script 1 - Camera specification
"""
from os.path import join
import piscope

img_dir = "../data/piscope_etna_testdata/images/"
ecII_filename = "EC2_1106307_1R02_2015091607080439_F01_Etnaxxxxxxxxxxxx.fts"

cam = piscope.setup.Camera("ecII")

start_acq, filter_id, meas_type, texp, warnings =\
            cam.get_img_meta_from_filename(join(img_dir, ecII_filename))
print cam

