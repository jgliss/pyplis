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
"""Pyplis example script no. 3 - Plume background analysis.

This example script introduces features related to plume background modelling
and tau image calculations.
"""
from typing import Tuple
from matplotlib.figure import Figure
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pyplis

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, IMG_DIR, ARGPARSER

# SCRIPT OPTIONS

# If this is True, then sky reference areas are set in auto mode (note that
# in this case, the tests at the end of the script will fail!)
USE_AUTO_SETTINGS = False

# intensity threshold to init mask for bg surface fit
POLYFIT_2D_MASK_THRESH = 2600

# Choose the background correction modes you want to use

BG_CORR_MODES = [0,  # 2D poly surface fit (without sky radiance image)
                 1,  # Scaling of sky radiance image
                 4,  # Scaling + linear gradient correction in x & y direction
                 6]  # Scaling + quadr. gradient correction in x & y direction


# Image file paths relevant for this script
PLUME_FILE = IMG_DIR / 'EC2_1106307_1R02_2015091607065477_F01_Etna.fts'
BG_FILE = IMG_DIR /'EC2_1106307_1R02_2015091607022602_F01_Etna.fts'
OFFSET_FILE = IMG_DIR / 'EC2_1106307_1R02_2015091607064723_D0L_Etna.fts'
DARK_FILE = IMG_DIR / 'EC2_1106307_1R02_2015091607064865_D1L_Etna.fts'

def init_background_model():
    """Create background model and define relevant sky reference areas."""
    # Create background modelling object
    m = pyplis.plumebackground.PlumeBackgroundModel()

    # Define default gas free areas in plume image
    w, h = 40, 40  # width/height of rectangles

    m.scale_rect = [1280, 20, 1280 + w, 20 + h]
    m.xgrad_rect = [20, 20, 20 + w, 20 + h]
    m.ygrad_rect = [1280, 660, 1280 + w, 660 + h]

    # Define coordinates of horizontal and vertical profile lines

    # row number of profile line for horizontal corretions in the sky
    # gradient...
    m.xgrad_line_rownum = 40
    # ... and start / stop columns for the corrections
    m.xgrad_line_startcol = 20
    m.xgrad_line_stopcol = 1323

    # col number of profile line for vertical corretions in the sky gradient...
    m.ygrad_line_colnum = 1300
    # ... and start / stop rows for the corrections
    m.ygrad_line_startrow = 10
    m.ygrad_line_stoprow = 700
    # Order of polyonmial fit applied for the gradient correction
    m.ygrad_line_polyorder = 2

    return m


def load_and_prepare_images():
    """Load images defined above and prepare them for the background analysis.

    Returns
    -------
        - Img, plume image
        - Img, plume image vignetting corrected
        - Img, sky radiance image

    """
    # get custom load method for ECII
    fun = pyplis.custom_image_import.load_ecII_fits
    # Load the image objects and peform dark correction
    plume, bg = pyplis.Img(PLUME_FILE, fun), pyplis.Img(BG_FILE, fun)
    dark, offset = pyplis.Img(DARK_FILE, fun), pyplis.Img(OFFSET_FILE, fun)

    # Model dark image for tExp of plume image
    dark_plume = pyplis.image.model_dark_image(plume.meta["texp"],
                                               dark, offset)
    # Model dark image for tExp of background image
    dark_bg = pyplis.image.model_dark_image(bg.meta["texp"],
                                            dark, offset)

    plume.subtract_dark_image(dark_plume)
    bg.subtract_dark_image(dark_bg)
    # Blur the images (sigma = 1)
    plume.add_gaussian_blurring(1)
    bg.add_gaussian_blurring(1)

    # Create vignetting correction mask from background image
    vign = bg.img / bg.img.max()  # NOTE: potentially includes y & x gradients
    plume_vigncorr = pyplis.Img(plume.img / vign)
    return plume, plume_vigncorr, bg

def autosettings_vs_manual_settings(
        bg_model: pyplis.PlumeBackgroundModel,
        plume_img: pyplis.Img) -> Tuple[dict, Figure]:
    """Perform automatic retrieval of sky reference areas.

    If you are lazy... (i.e. you dont want to define all these reference areas)
    then you could also use the auto search function, a comparison is plotted
    here.
    """
    auto_params = pyplis.plumebackground.find_sky_reference_areas(plume_img)
    current_params = bg_model.settings_dict()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].set_title("Manually set parameters")
    pyplis.plumebackground.plot_sky_reference_areas(plume_img, current_params,
                                                    ax=axes[0])
    pyplis.plumebackground.plot_sky_reference_areas(plume_img, auto_params,
                                                    ax=axes[1])
    axes[1].set_title("Automatically set parameters")

    return auto_params, fig


def plot_pcs_profiles_4_tau_images(tau0, tau1, tau2, tau3, pcs_line):
    """Plot PCS profiles for all 4 methods."""
    fig, ax = plt.subplots(1, 1)
    tau_imgs = [tau0, tau1, tau2, tau3]

    for k in range(4):
        img = tau_imgs[k]
        profile = pcs_line.get_line_profile(img)
        ax.plot(profile, "-", label=r"Mode %d: $\phi=%.3f$"
                % (BG_CORR_MODES[k], np.mean(profile)))

    ax.grid()
    ax.set_ylabel(r"$\tau_{on}$", fontsize=20)
    ax.set_xlim([0, pcs_line.length()])
    ax.set_xticklabels([])
    ax.set_xlabel("PCS", fontsize=16)
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=12)
    return fig


def main():
    plt.close("all")

    # Create a background model with relevant sky reference areas
    bg_model = init_background_model()

    # Define exemplary plume cross section line
    pcs_line = pyplis.LineOnImage(x0=530,
                                  y0=730,
                                  x1=890,
                                  y1=300,
                                  line_id="example PCS",
                                  color="lime")

    plume, plume_vigncorr, bg = load_and_prepare_images()

    auto_params, fig0 = autosettings_vs_manual_settings(
        bg_model=bg_model, plume_img=plume)

    # Script option
    if USE_AUTO_SETTINGS:
        bg_model.update(**auto_params)

    # Model 4 tau images

    # list to store figures of tau plotted tau images
    _tau_figs = []

    # mask for corr mode 0 (i.e. 2D polyfit)
    mask = np.ones(plume_vigncorr.img.shape, dtype=np.float32)
    mask[plume_vigncorr.img < POLYFIT_2D_MASK_THRESH] = 0

    # First method: retrieve tau image using poly surface fit
    tau0 = bg_model.get_tau_image(plume_vigncorr,
                                  mode=BG_CORR_MODES[0],
                                  surface_fit_mask=mask,
                                  surface_fit_polyorder=1)

    # Plot the result and append the figure to _tau_figs
    _tau_figs.append(bg_model.plot_tau_result(tau0, PCS=pcs_line))

    # Second method: scale background image to plume image in "scale" rect
    tau1 = bg_model.get_tau_image(plume, bg, mode=BG_CORR_MODES[1])
    _tau_figs.append(bg_model.plot_tau_result(tau1, PCS=pcs_line))

    # Third method: Linear correction for radiance differences based on two
    # rectangles (scale, ygrad)
    tau2 = bg_model.get_tau_image(plume, bg, mode=BG_CORR_MODES[2])
    _tau_figs.append(bg_model.plot_tau_result(tau2, PCS=pcs_line))

    # 4th method: 2nd order polynomial fit along vertical profile line
    # For this method, determine tau on tau off and AA image
    tau3 = bg_model.get_tau_image(plume, bg, mode=BG_CORR_MODES[3])
    _tau_figs.append(bg_model.plot_tau_result(tau3, PCS=pcs_line))

    fig6 = plot_pcs_profiles_4_tau_images(tau0, tau1, tau2, tau3, pcs_line)

    if SAVEFIGS:
        outfile = SAVE_DIR / f"ex03_out_1.{FORMAT}"
        fig0.savefig(outfile, format=FORMAT,
                     dpi=DPI, transparent=True)
        
        for k in range(len(_tau_figs)):
            outfile = SAVE_DIR / f"ex03_out_{k+2}.{FORMAT}"
            _tau_figs[k].savefig(outfile, format=FORMAT, dpi=DPI)
        outfile = outfile = SAVE_DIR / f"ex03_out_{k+3}.{FORMAT}"
        fig6.savefig(outfile, format=FORMAT, dpi=DPI)
    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        m = bg_model

        # test settings for clear sky reference areas
        npt.assert_array_equal([2680, 3960, 160, 6, 1300, 10, 700, 40, 20,
                                1323, 567584],
                               [sum(m.scale_rect),
                                sum(m.ygrad_rect),
                                sum(m.xgrad_rect),
                                m.mode,
                                m.ygrad_line_colnum,
                                m.ygrad_line_startrow,
                                m.ygrad_line_stoprow,
                                m.xgrad_line_rownum,
                                m.xgrad_line_startcol,
                                m.xgrad_line_stopcol,
                                int(m.surface_fit_mask.sum())])

        m.update(**auto_params)
        # test settings for clear sky reference areas
        npt.assert_array_equal([2682, 4142, 1380, 6, 1337, 1, 790, 6, 672,
                                1343, 567584],
                               [sum(m.scale_rect),
                                sum(m.ygrad_rect),
                                sum(m.xgrad_rect),
                                m.mode,
                                m.ygrad_line_colnum,
                                m.ygrad_line_startrow,
                                m.ygrad_line_stoprow,
                                m.xgrad_line_rownum,
                                m.xgrad_line_startcol,
                                m.xgrad_line_stopcol,
                                int(m.surface_fit_mask.sum())])

        # test all tau-modelling results
        actual = [tau0.mean(), tau1.mean(), tau2.mean(), tau3.mean()]
        npt.assert_allclose(actual=actual,
                            desired=[0.11395558008662043,
                                     0.25279653,
                                     0.13842879832119934,
                                     0.13943940574220634],
                            rtol=1e-7)
        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()