# -*- coding: utf-8 -*-
"""
Example script 12 - Emission rate retrieval from AA image list
"""

import piscope
from ex4_prepare_aa_imglist import prepare_aa_image_list, save_path
from os.path import join, exists
from matplotlib.pyplot import close

close("all")

calib_file = join(save_path, "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts")
corr_mask_path = join(save_path, "aa_corr_mask.fts")
if not exists(calib_file):
    raise IOError("Calibration file could not be found at specified location:\n"
        "%s\nYou might need to run example 6 first")

### Load AA list
aa_list = prepare_aa_image_list()
aa_list.pyrlevel = 1

### Load DOAS calbration data and FOV information (see example 6)
doas = piscope.doascalib.DoasCalibData()
doas.load_from_fits(file_path=calib_file)
doas.fit_calib_polynomial()

#Load AA corr mask
aa_corr_mask = piscope.Img(corr_mask_path)
aa_corr_mask.to_pyrlevel(1)
aa_corr_mask.show()






