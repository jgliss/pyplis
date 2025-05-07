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
"""Pyplis example script no. 5 - Automatic cell calibration.

Script showing how to use the automatic cell calibration engine which only
requires to specify start / stop time stamps of a calibration window. Based on
that sub time windows for each cell as well as suitable background images are
detected and separated into individual image lists (for all filters, i.e. here
on / off).

"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pyplis
from datetime import datetime
from time import time

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, IMG_DIR, ARGPARSER

# File path for storing cell AA calibration data including sensitivity
# correction mask
AA_CALIB_FILE = SAVE_DIR / "ex05_cellcalib_aa.fts"

# SCRIPT FUNCTION DEFINITIONS
def perform_auto_cell_calib():
    # Calibration time stamps
    start = datetime(2015, 9, 16, 6, 59, 00)
    stop = datetime(2015, 9, 16, 7, 3, 00)

    # Gas CDs in cells and cell ids
    # See supplementary package data about DOAS fit retrieval
    calib_cells = {'a37': [8.59e17, 2.00e17],
                   'a53': [4.15e17, 1.00e17],
                   'a57': [19.24e17, 3.00e17]}

    # the camera used
    cam_id = "ecII"

    # The camera filter setup is different from the ECII default setup and is
    # therefore defined explicitely
    filters = [pyplis.utils.Filter(type="on", acronym="F01"),
               pyplis.utils.Filter(type="off", acronym="F02")]

    # create camera setup, this includes the filename convention for
    # image separation
    cam = pyplis.setupclasses.Camera(cam_id=cam_id, filter_list=filters)

    # Create CellCalibSetup class for initiation of CellCalib object
    setup = pyplis.setupclasses.MeasSetup(IMG_DIR, start, stop,
                                          camera=cam,
                                          cell_info_dict=calib_cells)

    # Create CellCalibEngine object, read on...
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

    return c


# SCRIPT MAIN FUNCTION
def main():
    plt.close("all")
    start = time()
    c = perform_auto_cell_calib()

    # prepare CellCalibData objects for on, off and aa images. These can
    # be accessed via c.calib_data[key] where key is "on", "off", "aa"
    c.prepare_calib_data(pos_x_abs=None,  # change for a specific pix
                         pos_y_abs=None,  # change for a specific pix
                         radius_abs=1,  # radius of retrieval disk
                         on_id="on",  # ImgList ID of onband filter
                         off_id="off")  # ImgList ID of offband filter
    stop = time()
    # Plot search result of on
    ax0 = c.plot_cell_search_result("on", include_tit=False)
    ax1 = c.plot_cell_search_result("off", include_tit=False)
    # Plot all calibration curves for center pixel and in a radial
    # neighbourhood of 20 pixels
    ax2 = c.plot_all_calib_curves()
    ax2.set_xlim([0, 0.7])
    ax2.set_ylim([0, 2.25e18])
    ax0.set_title("Cell search result on band", fontsize=18)
    ax1.set_title("Cell search result off band", fontsize=18)
    ax2.set_title("Calibration polynomials", fontsize=18)
    # IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        ax0.figure.savefig(SAVE_DIR / f"ex05_2_out_1.{FORMAT}", format=FORMAT, dpi=DPI)
        ax1.figure.savefig(SAVE_DIR / f"ex05_2_out_2.{FORMAT}", format=FORMAT, dpi=DPI)
        ax2.figure.savefig(SAVE_DIR / f"ex05_2_out_3.{FORMAT}", format=FORMAT, dpi=DPI)

    # You can explicitely access the individual CellCalibData objects
    aa_calib = c.calib_data["aa"]

    aa_calib.fit_calib_data()
    c.plot_calib_curve("on")
    mask = c.get_sensitivity_corr_mask("aa")

    aa_calib.save_as_fits(AA_CALIB_FILE)
    print(f"Time elapsed for preparing calibration data: {stop - start:.4f} s")

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        calib_reload = pyplis.CellCalibData()
        calib_reload.load_from_fits(AA_CALIB_FILE)
        calib_reload.fit_calib_data(polyorder=2, through_origin=True)

        timestamps = np.array([datetime(2015, 9, 16, 7, 0, 25, 127400), datetime(2015, 9, 16, 7, 0, 58, 637400), datetime(2015, 9, 16, 7, 1, 32, 647400)])
        npt.assert_array_equal(timestamps, calib_reload.time_stamps)

        # test some basic features of calibraiton dataset (e.g. different
        # ImgList classes for on and off and the different cells)
        npt.assert_array_equal([c.cell_search_performed,
                                c.cell_lists["on"]["a37"].nof,
                                c.cell_lists["on"]["a53"].nof,
                                c.cell_lists["on"]["a57"].nof,
                                c.cell_lists["off"]["a37"].nof,
                                c.cell_lists["off"]["a53"].nof,
                                c.cell_lists["off"]["a57"].nof,
                                calib_reload.calib_id,
                                calib_reload.pos_x_abs,
                                calib_reload.pos_y_abs],
                               [True, 2, 3, 3, 2, 3, 3, "aa", 672, 512])
        d = c._cell_info_auto_search

        vals_approx = [d["a37"][0],
                       d["a53"][0],
                       d["a57"][0],
                       aa_calib.calib_fun(0, *aa_calib.calib_coeffs),
                       calib_reload.calib_fun(0, *calib_reload.calib_coeffs)]
        desired=[8.59e+17,4.15e+17,1.924e+18,-1.831327e+16,0.0]
        npt.assert_allclose(actual=vals_approx,desired=desired,rtol=1e-5)

        # explicitely check calibration data for on, off and aa (plotted in
        # this script)
        actual = [c.calib_data["on"].calib_coeffs.mean(),
                  c.calib_data["off"].calib_coeffs.mean(),
                  aa_calib.calib_coeffs.mean(),
                  calib_reload.calib_coeffs.mean()]
        desired=[1.892681e+18, -3.742539e+19, 2.153654e+18, 2.119768e+18]
        npt.assert_allclose(actual=actual,desired=desired,rtol=1e-5)
        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()