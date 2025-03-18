# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)
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
"""Pyplis introduction script no. 5: manual cell calibration.

Perform manual cell calibration based on 3 cell images and one background
image. The calibration data consists of 3 cells which were put in front of the
lense successively and a background images both before and after the
cell images.

This script creates an empty CellCalibEngine object in which 6 cell images
are assigned (3 on and 3 off band) with their corresponding SO2 column
densities. Further, 2 background images are assigned for each filter, one
before and one after the cells were put in front of the camera. These are used
to determine tau images from the cell images. Variations in the background
intensities are corrected for (for details see manuscript).

Note, that this calibration does not include a dark correction of the images
before the calibration, therefore, the results are slightly different compared
to the results from ex05_cell_calib_auto.py.
"""
import pathlib
import pyplis
import matplotlib.pyplot as plt
from time import time

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER, IMG_DIR

# SPECIFY IMAGE PATHS FOR EACH CELL AND BACKGROUND IMAGES
BG_BEFORE_ON = "EC2_1106307_1R02_2015091607000845_F01_Etna.fts"
BG_BEFORE_OFF = "EC2_1106307_1R02_2015091607001021_F02_Etna.fts"

A53_ON = "EC2_1106307_1R02_2015091607003032_F01_Etna.fts"
A53_OFF = "EC2_1106307_1R02_2015091607003216_F02_Etna.fts"

A37_ON = "EC2_1106307_1R02_2015091607005847_F01_Etna.fts"
A37_OFF = "EC2_1106307_1R02_2015091607010023_F02_Etna.fts"

A57_ON = "EC2_1106307_1R02_2015091607013835_F01_Etna.fts"
A57_OFF = "EC2_1106307_1R02_2015091607014019_F02_Etna.fts"

BG_AFTER_ON = "EC2_1106307_1R02_2015091607020256_F01_Etna.fts"
BG_AFTER_OFF = "EC2_1106307_1R02_2015091607020440_F02_Etna.fts"

def main():
    plt.close("all")
    start = time()
    # define camera for ECII custom image import
    cam = pyplis.Camera("ecII")
    cellcalib = pyplis.cellcalib.CellCalibEngine(cam)

    # now add all cell images manually, specifying paths, the SO2 column
    # densities of each cell, and the corresponding cell ID as well as image
    # type (on, off)
    cellcalib.set_cell_images(img_paths=IMG_DIR / A53_ON,
                              cell_gas_cd=4.15e17,
                              cell_id="a53", filter_id="on")

    cellcalib.set_cell_images(img_paths=IMG_DIR / A53_OFF,
                              cell_gas_cd=4.15e17,
                              cell_id="a53", filter_id="off")

    cellcalib.set_cell_images(img_paths=IMG_DIR / A37_ON,
                              cell_gas_cd=8.59e17,
                              cell_id="a37", filter_id="on")

    cellcalib.set_cell_images(img_paths=IMG_DIR / A37_OFF,
                              cell_gas_cd=8.59e17,
                              cell_id="a37", filter_id="off")

    cellcalib.set_cell_images(img_paths=IMG_DIR / A57_ON,
                              cell_gas_cd=1.92e18,
                              cell_id="a57", filter_id="on")

    cellcalib.set_cell_images(img_paths=IMG_DIR / A57_OFF,
                              cell_gas_cd=1.92e18,
                              cell_id="a57", filter_id="off")

    # put the onband background images into a Python list ....
    bg_on_paths = [
        IMG_DIR / BG_BEFORE_ON, 
        IMG_DIR / BG_AFTER_ON
        ]

    # ... and add them to the calibration object ...
    cellcalib.set_bg_images(img_paths=bg_on_paths, filter_id="on")

    # ... same for off band background images
    bg_off_paths = [
        IMG_DIR / BG_BEFORE_OFF, 
        IMG_DIR / BG_AFTER_OFF
        ]
    
    cellcalib.set_bg_images(img_paths=bg_off_paths, filter_id="off")

    # Prepare calibration data (i.e. CellCalibData objets) for on, off and
    # AA images. This function determines tau images for each cell using
    # background images scaled to the present background intensities at the
    # acq. time stamp of each cell using temporal interpolation of the provided
    # background images. This results in 3 tau images for each filter (on, off)
    # and from that, 3 AA images are determined. Each of these collection of
    # 3 tau images (on, off, AA) is then stored within a CellCalibData object
    # which can be accessed using e.g. cellcalib.calib_data["aa"]
    cellcalib.prepare_calib_data(on_id="on", off_id="off", darkcorr=False)
    stop = time()

    ax = cellcalib.plot_all_calib_curves()
    ax.set_title("Manual cell calibration\n(NO dark correction performed)")

    aa_calib = cellcalib.calib_data["aa"]

    print(f"Time elapsed for preparing calibration data: {stop - start:.4f} s")
    
    ### IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        outfile = SAVE_DIR / f"ex0_7_out_1.{FORMAT}"
        ax.figure.savefig(outfile, format=FORMAT, dpi=DPI)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt

        slope, offs = aa_calib.calib_coeffs

        actual = [cellcalib.tau_stacks["aa"].sum(),
                  cellcalib.tau_stacks["aa"].mean(),
                  aa_calib.cd_vec.sum(),
                  slope,
                  offs]

        desired = [1007480.56, 0.24401481, 3.194000052267778e+18, 4.779782684462987e+18, -2.7244424951657216e+16]
        npt.assert_allclose(actual=actual, desired=desired, rtol=1e-5)
        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()