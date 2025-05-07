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
"""Pyplis example script no. 8 - Plume velocity retrieval by cross correlation.
"""
from matplotlib.pyplot import close, show, subplots
from time import time
from pathlib import Path

from pyplis.plumespeed import VeloCrossCorrEngine

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER, LINES

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

# SCRIPT OPTONS

# distance in pixels between two lines used for cross correlation analysis
OFFSET_PIXNUM = 40

# start / stop indices of considered images in image list (only relevant if PCS
# profiles are reloaded, i.e. Opt RELOAD=True)
START_IDX = 10
STOP_IDX = 200

# PCS line for which velocity is supposed to retrieved
PCS = LINES[0]  # orange "young_plume" line

# Color of PCS offset line used to perform cross correlation analysis
# (relevant for illustration)
COLOR_PCS_OFFS = "c"

# RELEVANT DIRECTORIES AND PATHS
# the time series of PCS profiles for both lines are stored as
# ProfileTimeSeriesImg objects using the following names. The images
# contain the PCS profiles (y-axis) for each image in the list (y-axis)
# and have thus dimension MXN where M denotes the pixel number of the lines
# and N denotes the total number of images from which the profiles are
# extracted. The images will be stored in SAVE_DIR after this script is run
# once. After that, re-running the script and applying the cross-correlation
# analysis will be much faster, since the profiles are imported from the
# two precomupted images and do not need to be extracted by looping over
# the image list.
PCS_PROFILES_PIC_NAME = "ex08_ica_tseries_pcs.fts"
OFFSET_PROFILES_PIC_NAME = "ex08_ica_tseries_offset.fts"


# SCRIPT MAIN FUNCTION
def main():
    close("all")
    axes = []
    # prepare the AA image list (see ex4)
    aa_list = prepare_aa_image_list()
    aa_list.pyrlevel = 1

    t0 = time()
    cc = VeloCrossCorrEngine(pcs=PCS, imglist=aa_list)
    cc.create_parallel_pcs_offset(offset_pix=40,
                                  color=COLOR_PCS_OFFS,
                                  linestyle="--")
    
    cc.get_pcs_tseries_from_list(start_idx=START_IDX, stop_idx=STOP_IDX)
    cc.save_pcs_profile_images(save_dir=SAVE_DIR,
                                fname1=PCS_PROFILES_PIC_NAME,
                                fname2=OFFSET_PROFILES_PIC_NAME)
    
    cc.load_pcs_profile_img(SAVE_DIR / PCS_PROFILES_PIC_NAME, line_id="pcs")
    cc.load_pcs_profile_img(SAVE_DIR / OFFSET_PROFILES_PIC_NAME, line_id="pcs_offset")
    t1 = time()
    # the run method of the high level VeloCrossCorrEngine class is
    # basically a wrapper method for the low-level find_signal_correlation
    # function which is part of the plumespeed.py module. Before calling
    # the latter, the ICA time-series are extracted from the two
    # ProfileTimeSeriesImg objects which were computed above from the
    # ImgList class containing AA images, and which are stored as FITS
    # files for fast re-computing of this script. The following run
    # command passes valid input parameters to the find_signal_correlation
    # method.
    velo = cc.run(cut_border_idx=10,
                  reg_grid_tres=100,
                  freq_unit="L",
                  sigma_smooth=2,
                  plot=0)
    t2 = time()
    _, ax = subplots(1, 2, figsize=(20, 6))
    axes.append(cc.plot_pcs_lines(ax=ax[0]))
    cc.plot_ica_tseries_overlay(ax=ax[1])
    axes.append(cc.plot_corrcoeff_tseries())

    print(f"Result performance analysis\n"
          f"Number of images: {STOP_IDX - START_IDX}\n"
          f"Create ICA images: {t1 - t0:.3f} s\n"
          f"Cross-corr analysis: {t2 - t1:.3f} s")

    print(f"Retrieved plume velocity of v = {velo:.2f} m/s")

    # IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        for k in range(len(axes)):
            outfile= SAVE_DIR / f"ex08_out_{k + 1}.{FORMAT}"
            axes[k].figure.savefig(outfile, format=FORMAT, dpi=DPI)

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        results = cc.results
        npt.assert_array_equal([len(results), 
                                len(results["coeffs"]), 
                                len(results["ica_tseries_offset"]),
                                len(results["ica_tseries_shift"])],
                               [6, 1578, 7891, 7712])

        npt.assert_allclose(actual=[results["velo"], results["lag"]],
                            desired=[4.434, 17.9],
                            rtol=1e-2)
        print(f"All tests passed in script: {Path(__file__).name}")
    try:
        if int(options.show) == 1:
            show()
    except Exception:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()