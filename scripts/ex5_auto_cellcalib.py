# -*- coding: utf-8 -*-
"""
piscope example script no. 5: Automatic cell calibration

Sript showing how to work with cell calibration data and automatic retrieval of
plume background images.

"""

import piscope as piscope
from datetime import datetime
from os.path import join
from os import getcwd
import matplotlib.pyplot as plt

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

def perform_auto_cell_calib():
    ### Calibration time stamps
    start = datetime(2015, 9, 16, 6, 59, 00)
    stop  = datetime(2015, 9, 16, 7, 3, 00)
    
    ### Gas CDs in cells and cell ids
    # See supplementary package data about DOAS fit retrieval
    calib_cells= {'a37'  :   [8.59e17, 2.00e17],
                  'a53'  :   [4.15e17, 1.00e17],
                  'a57'  :   [19.24e17, 3.00e17]}
    
    # the camera used 
    cam_id = "ecII"
    
    # The camera filter setup is different from the ECII default setup and is
    # therefore defined explicitely
    filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
              piscope.utils.Filter(type = "off", acronym = "F02")]
    
    ### create camera setup, this includes the filename convention for image separation
    cam = piscope.setup.Camera(cam_id = cam_id, filter_list = filters)
    
    ### Create CellCalibSetup class for initiation of CellCalib object
    setup = piscope.setup.AutoCellCalibSetup(\
                                    calib_cells, img_dir, start, stop, cam) 
    
    ### Create CellCalib object, read on...
    # This is a DataSet object and performs file separation, dark/offset list 
    # assignment, etc. for all images in the specified time window (corresponding
    # to the cell calibration data set, so after initiation)
    c = piscope.cellcalib.CellCalibEngine(setup)
    c.find_and_assign_cells_all_filter_lists()
    
    c.prepare_tau_stack("on", pyrlevel = 2)
    c.prepare_tau_stack("off", pyrlevel = 2)
    c.prepare_aa_stack()
    return c


if __name__ == "__main__":
    c = perform_auto_cell_calib()
    ### Determine exemplary tau time series from on band stack at pixel 100, 100, radius=10
    fig, axes = plt.subplots(1,3, figsize=(20,6))
    ### Plot search result of on
    c.plot_cell_search_result("on", include_tit = False, ax = axes[0])
    c.plot_cell_search_result("off", include_tit = False, ax = axes[1])
    c.plot_all_calib_curves(1344/2, 512, 20, ax = axes[2])
    axes[0].set_title("A) Cell search result on band", fontsize = 18)
    axes[1].set_title("B) Cell search result off band", fontsize = 18)
    axes[2].set_title("C) Calibration polynomials", fontsize = 18)
    fig.tight_layout()
    plt.show()
    fig.savefig(join(save_path, "ex5_out_1.png"))