# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliß (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Pyplis example script no. 5 - Automatic cell calibration

Script showing how to use the automatic cell calibration engine which only 
requires to specify start / stop time stamps of a calibration window. Based on 
that sub time windows for each cell as well as suitable background images are
detected and separated into individual image lists (for all filters, i.e. here
on / off).

"""
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

import pyplis
from datetime import datetime
from time import time
from os.path import join
from matplotlib.pyplot import show, close

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, IMG_DIR, OPTPARSE

### SCRIPT FUNCTION DEFINITIONS
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
    filters= [pyplis.utils.Filter(type = "on", acronym = "F01"),
              pyplis.utils.Filter(type = "off", acronym = "F02")]
    
    ### create camera setup, this includes the filename convention for image separation
    cam = pyplis.setupclasses.Camera(cam_id = cam_id, filter_list = filters)
    
    ### Create CellCalibSetup class for initiation of CellCalib object
    setup = pyplis.setupclasses.MeasSetup(IMG_DIR, start, stop,
                                           camera=cam,
                                           cell_info_dict=calib_cells) 
    
    ### Create CellCalibEngine object, read on...
    # This is a DataSet object and performs file separation and creation of 
    # on / off, dark / offset lists for all images in the specified time window 
    c = pyplis.cellcalib.CellCalibEngine(setup)
    
    # the following high level method calls several functions in the 
    # CellCalibEngine class, most importantly the method find_cells for on and
    # off band image time series, which detects sub time windows for each cell
    # and background images. After the individual time windows are detected for
    # each cell and filter, the method _assign_calib_specs is called, which
    # assigns the SO2 CD amount (specified above in dictionary calib_cells) 
    # to the detected sub time windows (both for on and off) based on the depth 
    # of the intensity dip (in the onband) for each sub time window (should 
    # become clear from the plot produced in this script). Then it creates 
    # CellImgList objects for each of the cells and for the detected background
    # images (i.e. resulting in (M + 1) x 2 lists, with M being the number of 
    # detected intensity dips, the + 1 is the corresponding background list and
    # times 2 for on / off)
    c.find_and_assign_cells_all_filter_lists()
    # prepares CellCalibData object for tau on band (at pyramid level 2)
    c.prepare_tau_calib("on", pyrlevel=2)
    # prepares CellCalibData object for tau off band (at pyramid level 2)
    c.prepare_tau_calib("off", pyrlevel=2)
    # from the previous 2, prepare CellCalibData object for tau_aa
    c.prepare_aa_calib()
    return c

### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    start = time()
    c = perform_auto_cell_calib()
    stop = time()
    ### Plot search result of on
    ax0 = c.plot_cell_search_result("on", include_tit=False)
    ax1 = c.plot_cell_search_result("off", include_tit=False)
    # Plot all calibration curves for center pixel and in a radial 
    # neighbourhood of 20 pixels
    ax2 = c.plot_all_calib_curves(pos_x_abs=672, pos_y_abs=512, radius_abs=20)
    ax2.set_xlim([0, 0.7])
    ax2.set_ylim([0, 2.25e18])
    ### IMPORTANT STUFF FINISHED    
    if SAVEFIGS:
        ax0.figure.savefig(join(SAVE_DIR, "ex05_2_out_1.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        ax1.figure.savefig(join(SAVE_DIR, "ex05_2_out_2.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        ax2.figure.savefig(join(SAVE_DIR, "ex05_2_out_3.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)     
                    
    ax0.set_title("Cell search result on band", fontsize = 18)
    ax1.set_title("Cell search result off band", fontsize = 18)
    ax2.set_title("Calibration polynomials", fontsize = 18)
    
    # You can explicitely access the individual CellCalibData objects
    aa_calib = c.calib_data["aa"]
    
    aa_calib.fit_calib_polynomial(100, 100, 15)
    # print some useful attributes of the calibration
    print ("Properties of AA cell calibration object:\n"
           "Polynomial: %s\n"
           "Slope / err: %.3e / %.3e"
           %(aa_calib.poly, aa_calib.slope, aa_calib.slope_err))
    
    print "Time elapsed for preparing calibration data: %.4f s" %(stop-start)
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"
