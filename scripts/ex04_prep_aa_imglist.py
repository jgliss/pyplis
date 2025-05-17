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
"""Pyplis example script no. 4 - Prepare AA image list (from onband list).

Script showing how to work in AA mode using ImgList object
"""
import pyplis
import pathlib
import matplotlib.pyplot as plt
from time import time

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, IMG_DIR, ARGPARSER, PCS1

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex01_analysis_setup import create_dataset
from ex02_meas_geometry import find_viewing_direction

# SCRIPT FUNCTION DEFINITIONS
def prepare_aa_image_list(bg_corr_mode=6):
    """Get and prepare onband list for aa image mode.

    The relevant gas free areas for background image modelling are set
    automatically (see also ex. 3 for details)

    :return: - on list in AA mode
    """
    dataset = create_dataset()
    geom, _ = find_viewing_direction(dataset.meas_geometry, False)

    # Set plume background images for on and off
    # this is the same image which is also used for example script
    # ex03_plume_background.py demonstrating the plume background routines
    path_bg_on = IMG_DIR / 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts'
    path_bg_off = IMG_DIR / 'EC2_1106307_1R02_2015091607022820_F02_Etna.fts'

    # Get on and off lists and activate dark correction
    on_list = dataset.get_list("on")
    off_list = dataset.get_list("off")

    # Deactivate automatic reload in list while changing some list
    # attributes
    on_list.auto_reload = False
    off_list.auto_reload = False

    on_list.darkcorr_mode = True
    off_list.darkcorr_mode = True

    # Prepare on and offband background images
    bg_on = pyplis.Img(path_bg_on)
    bg_on.subtract_dark_image(on_list.get_dark_image())

    bg_off = pyplis.Img(path_bg_off)
    bg_off.subtract_dark_image(off_list.get_dark_image())

    # set the background images within the lists
    on_list.set_bg_img(bg_on)
    off_list.set_bg_img(bg_off)

    # automatically set gas free areas
    on_list.bg_model.set_missing_ref_areas(on_list.current_img())
    # Now update some of the information from the automatically set sky ref
    # areas
    on_list.bg_model.xgrad_line_startcol = 20
    on_list.bg_model.xgrad_line_rownum = 25
    off_list.bg_model.xgrad_line_startcol = 20
    off_list.bg_model.xgrad_line_rownum = 25

    # set background modelling mode
    on_list.bg_model.mode = bg_corr_mode
    off_list.bg_model.mode = bg_corr_mode

    on_list.aa_mode = True  # activate AA mode

    off_list.auto_reload = True
    on_list.auto_reload = True
    on_list.meas_geometry = geom
    return on_list


def main():
    plt.close("all")
    aa_list = prepare_aa_image_list()

    t0 = time()
    # Deactivate auto reload while changing some settings (else, after each
    # of the following operations the images are reloaded and edited, which)
    aa_list.auto_reload = False
    aa_list.goto_img(50)

    aa_list.add_gaussian_blurring(1)
    # aa_list.pyrlevel = 2
    aa_list.roi_abs = [300, 300, 1120, 1000]
    aa_list.crop = True
    # now reactive image reload in list (loads image no. 50 with all changes
    # that where set in the previous lines)
    aa_list.auto_reload = True

    # store some results for tests below
    shape_log, mean_log = sum(aa_list.current_img().shape), aa_list.current_img().mean()

    ax = aa_list.show_current(zlabel=r"$\tau_{AA}$")
    print(f"Elapsed time: {time() - t0} s")

    aa_list.crop = False
    ax1 = aa_list.bg_model.plot_sky_reference_areas(aa_list.current_img())
    fig = aa_list.bg_model.plot_tau_result(aa_list.current_img())

    # import plume cross section and computed integrated optical density
    # for current image
    img = aa_list.current_img()
    ax2 = img.show(vmin=-0.1, vmax=0.3)
    pcs = PCS1.convert(to_pyrlevel=0)
    pcs.plot_line_on_grid(ax=ax2)
    pix_steps = aa_list.meas_geometry.compute_all_integration_step_lengths()[0]
    integrated_aa = pcs.integrate_profile(img, pix_steps)

    # IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        outfile = SAVE_DIR / f"ex04_out_1.{FORMAT}"
        ax.figure.savefig(outfile, format=FORMAT, dpi=DPI)

        outfile = SAVE_DIR / f"ex04_out_2.{FORMAT}"
        ax1.figure.savefig(outfile, format=FORMAT, dpi=DPI)

        outfile = SAVE_DIR / f"ex04_out_3.{FORMAT}"
        fig.savefig(outfile, format=FORMAT, dpi=DPI)
        
    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt

        m = aa_list.bg_model
        npt.assert_array_equal([2682, 4144, 1380, 6, 1337, 1, 791, 25, 20,
                                1343],
                               [sum(m.scale_rect),
                                sum(m.ygrad_rect),
                                sum(m.xgrad_rect),
                                m.mode,
                                m.ygrad_line_colnum,
                                m.ygrad_line_startrow,
                                m.ygrad_line_stoprow,
                                m.xgrad_line_rownum,
                                m.xgrad_line_startcol,
                                m.xgrad_line_stopcol])

        actual = [aa_list.meas_geometry.cam_elev,
                  aa_list.meas_geometry.cam_azim,
                  aa_list.meas_geometry.plume_dist()[0, 0],
                  aa_list.current_img().mean(),
                  shape_log, mean_log]

        npt.assert_allclose(actual=actual,
                            desired=[15.477542212645357,
                                     279.30130009369515,
                                     10731.024327931776,
                                     0.009083584068527644,
                                     1520,
                                     0.014380159209694215],
                            rtol=1e-7)
        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()