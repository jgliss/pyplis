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
"""Pyplis example script no. 7 - AA sensitivity correction masks.

In this script, cell and DOAS calibration (see previous 2 scripts) of the Etna
test dataset are opposed. Furthermore, it is illustrated, how to create
correction masks for pixel variations in the SO2 sensitivity due to shifts in
the filter transmission windows.

The cell calibration is re-performed (using method ``perform_auto_cell_calib``)
from example script 5. The results from the DOAS calibration
(see prev. example) were stored as a FITS file (including FOV information)
and the results are imported here.
"""
import pyplis
from pathlib import Path
import numpy as np
from matplotlib.pyplot import close, subplots, show
from matplotlib.patches import Circle

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex05_cell_calib_auto import perform_auto_cell_calib
from ex04_prep_aa_imglist import prepare_aa_image_list

CELL_AA_CALIB_FILE = SAVE_DIR / "ex05_cellcalib_aa.fts"
# RELEVANT DIRECTORIES AND PATHS

# fits file containing DOAS calibration information (from ex6)
DOAS_CALIB_FILE = SAVE_DIR / "ex06_doascalib_aa.fts"

# SCRIPT FUNCTION DEFINITIONS
def draw_doas_fov(fov_x, fov_y, fov_extend, ax):
    # add FOV position to plot of examplary AA image
    c = Circle((fov_x, fov_y), fov_extend, ec="k", fc="lime", alpha=.5)
    ax.add_artist(c)
    ax.text(fov_x, (fov_y - fov_extend * 1.3), "DOAS FOV")
    ax.set_xlim([0, 1343]), ax.set_ylim([1023, 0])
    return ax


def plot_pcs_comparison(aa_init, aa_imgs_corr, pcs1, pcs2):
    fig, axes = subplots(1, 2, figsize=(18, 6))
    p10 = pcs1.get_line_profile(aa_init.img)
    p20 = pcs2.get_line_profile(aa_init.img)

    num = len(p10)

    axes[0].set_title(f"Line {pcs1.line_id}")
    axes[1].set_title(f"Line {pcs2.line_id}")
    phi_p1_init = sum(p10) / num
    phi_p2_init = sum(p20) / num
    axes[0].plot(p10, "-", label=f"Init φ={phi_p1_init:.3f}")
    axes[1].plot(p20, "-", label=f"Init φ={phi_p2_init:.3f}")
    # store main results for testing in main()
    results_p1 = [(np.nan, float(phi_p1_init))]
    results_p2 = [(np.nan, float(phi_p2_init))]
    
    for cd, aa_corr in aa_imgs_corr.items():
        p1 = pcs1.get_line_profile(aa_corr.img)
        p2 = pcs2.get_line_profile(aa_corr.img)
        phi_p1 = sum(p1) / num
        phi_p2 = sum(p2) / num
        results_p1.append((float(cd), float(phi_p1)))
        results_p2.append((float(cd), float(phi_p2)))

        axes[0].plot(p1, "-", label=f"Cell CD: {cd:.2e} φ={phi_p1:.3f}")
        axes[1].plot(p2, "-", label=f"Cell CD: {cd:.2e} φ={phi_p2:.3f}")

    axes[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
    axes[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
    axes[0].grid()
    axes[1].grid()
    return fig, axes, results_p1, results_p2


# SCRIPT MAIN FUNCTION
def main():
    close("all")

    if not DOAS_CALIB_FILE.exists():
        raise IOError("Calibration file could not be found at specified "
                      "location:\n %s\nYou might need to run example 6 first")

    # Load AA list
    aa_list = prepare_aa_image_list()
    aa_list.add_gaussian_blurring(2)

    # Load DOAS calbration data and FOV information (see example 6)
    doascalib = pyplis.doascalib.DoasCalibData()
    doascalib.load_from_fits(file_path=DOAS_CALIB_FILE)
    doascalib.fit_calib_data()

    # Get DOAS FOV parameters in absolute coordinates
    fov_x, fov_y = doascalib.fov.pixel_position_center(abs_coords=True)
    fov_extend = doascalib.fov.pixel_extend(abs_coords=True)

    # Load cell calibration (see example 5)
    cellcalib = perform_auto_cell_calib()

    # get cell calibration
    cellcalib.prepare_calib_data(
        pos_x_abs=fov_x,  # change if you want it for a specific pix
        pos_y_abs=fov_y,  # change if you want it for a specific pix
        radius_abs=fov_extend,  # radius of retrieval disk
        on_id="on",  # ImgList ID of onband filter
        off_id="off")  # ImgList ID of offband filter

    cell_aa_calib = cellcalib.calib_data["aa"]

    # Define lines on image for plume profiles
    pcs1 = pyplis.LineOnImage(620, 700, 940, 280, line_id="center")
    pcs2 = pyplis.LineOnImage(40, 40, 40, 600, line_id="edge")

    # Plot DOAS calibration polynomial
    ax0 = doascalib.plot(add_label_str="DOAS calib")
    ax0 = cellcalib.calib_data["aa"].plot(ax=ax0, c="r", add_label_str="Cell calib")
    ax0.set_title("")
    ax0.set_xlim([0, 0.5])

    # Get current AA image from image list
    aa_init_img = aa_list.current_img()

    # now determine sensitivity correction masks from the different cells
    masks = {}
    aa_imgs_corr = {}
    for cd in cell_aa_calib.cd_vec:
        mask = cellcalib.get_sensitivity_corr_mask("aa",
                                                   pos_x_abs=fov_x,
                                                   pos_y_abs=fov_y,
                                                   radius_abs=fov_extend,
                                                   cell_cd_closest=cd)
        masks[cd] = mask
        aa_imgs_corr[cd] = pyplis.Img(aa_init_img.img / mask.img)

    # get mask corresponding to minimum cell CD
    mask = list(masks.values())[np.argmin(list(masks.keys()))]

    # assing mask to aa_list
    aa_list.senscorr_mask = mask

    # activate AA sensitivity correction in list
    aa_list.sensitivity_corr_mode = True

    # set DOAS calibration data in list ...
    aa_list.calib_data = doascalib

    # ... and activate calibration mode
    aa_list.calib_mode = True
    ax = aa_list.current_img().show(zlabel=r"$S_{SO2}$ [cm$^{-2}$]")

    # plot the two lines into the exemplary AA image
    pcs1.plot_line_on_grid(ax=ax, color="r")
    pcs2.plot_line_on_grid(ax=ax, color="g")
    ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
    ax = draw_doas_fov(fov_x, fov_y, fov_extend, ax=ax)

    fig, _, results_p1, results_p2 = plot_pcs_comparison(aa_init_img, aa_imgs_corr, pcs1, pcs2)

    # IMPORTANT STUFF FINISHED

    if SAVEFIGS:
        ax0.figure.savefig(SAVE_DIR / f"ex07_out_1.{FORMAT}", format=FORMAT, dpi=DPI)
        ax.figure.savefig(SAVE_DIR / f"ex07_out_2.{FORMAT}", format=FORMAT, dpi=DPI)
        fig.savefig(SAVE_DIR / f"ex07_out_3.{FORMAT}", format=FORMAT, dpi=DPI)

    # Save the sensitivity correction mask from the cell with the lowest SO2 CD
    so2min = np.min(list(masks.keys()))
    mask = masks[so2min]
    mask.save_as_fits(SAVE_DIR, "ex07_aa_corr_mask")

    # assign mask in doascalib and resave
    doascalib.senscorr_mask = mask
    doascalib.save_as_fits(DOAS_CALIB_FILE)

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        assert len(doascalib.tau_vec) == 88
        assert len(doascalib.cd_vec) == 88

        npt.assert_allclose(
            actual=[
                doascalib.tau_vec.mean(),
                doascalib.cd_vec.mean()
            ],
            desired=[
                0.10984407750550997,
                1.2129406193969774e+18
            ],
            rtol=1e-7
        )

        npt.assert_allclose(
            actual=[
                cellcalib.calib_data["aa"].cd_vec,
                cellcalib.calib_data["aa"].tau_vec,
                cellcalib.calib_data["on"].tau_vec,
                cellcalib.calib_data["off"].tau_vec
            ],
            desired=[
                [4.150e+17, 8.590e+17, 1.924e+18],
                [0.11438841, 0.20960312, 0.45484966],
                [0.25561193, 0.34218314, 0.58974224],
                [0.14122352, 0.13258003, 0.13489263]
            ],
            rtol=1e-7
        )

        npt.assert_allclose(
            actual=results_p1,
            desired=[
                (np.nan, 0.06704542158954685), 
                (4.15e+17, 0.06718042466335893), 
                (8.59e+17, 0.0672710026562296), 
                (1.924e+18, 0.06713881262426598)],
            rtol=1e-7
        )

        npt.assert_allclose(
            actual=results_p2,
            desired=[
                (np.nan, 0.09656624766757335), 
                (4.15e+17, 0.08434758808651711), 
                (8.59e+17, 0.08673337715111279), 
                (1.924e+18, 0.08756188495164571)],
            rtol=1e-7
        )
        print(f"All tests passed in script: {Path(__file__).name}")
    try:
        if int(options.show) == 1:
            show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()