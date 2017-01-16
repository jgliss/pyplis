# -*- coding: utf-8 -*-
"""
piscope example script no. 7 - Combined DOAS and cell calibration
"""

import piscope 
from os.path import join, exists
from ex1_measurement_setup_plume_data import create_dataset
from ex5_auto_cellcalib import perform_auto_cell_calib
from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
                                                                path_bg_off
from ex6_doas_calibration import my_dat #folder where FOV file is saved

calib_file = join(my_dat, "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts")

if not exists(calib_file):
    raise IOError("Calibration file could not be found at specified location:\n"
        "%s\nYou might need to run example 6 first")

doas = piscope.doascalib.DoasCalibData()
doas.load_from_fits(file_path=calib_file)
doas.fit_calib_polynomial()
ax = doas.plot()

cell = perform_auto_cell_calib()
posx, posy = doas.fov.pixel_position_center(abs_coords=True)
radius = doas.fov.pixel_extend(abs_coords=True)

ax = cell.plot_calib_curve("aa", pos_x_abs= posx, pos_y_abs=\
                                            posy, radius_abs=radius, ax = ax)
                                                        

#==============================================================================
# dataset = create_dataset()
# aa_list = prepare_aa_image_list(dataset, path_bg_on, path_bg_off)
#==============================================================================






                                                                
