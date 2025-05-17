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
"""Pyplis example script no. 9 - Optical flow Plume velocity retrieval."""
import pyplis
from pathlib import Path
# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER, LINES
from matplotlib.pyplot import (close, show, subplots, figure, xticks, yticks,
                               sca, rcParams)

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

rcParams["font.size"] = 16

def analyse_and_plot(lst, lines):
    fig = figure(figsize=(14, 8))

    ax0 = fig.add_axes([0.01, 0.15, 0.59, 0.8])
    ax1 = fig.add_axes([0.61, 0.15, 0.16, 0.8])
    ax2 = fig.add_axes([0.78, 0.15, 0.16, 0.8])

    mask = lst.get_thresh_mask(0.05)
    fl = lst.optflow
    fl.plot(ax=ax0)
    for line in lines:
        m = mask * line.get_rotated_roi_mask(fl.flow.shape[:2])
        line.plot_line_on_grid(ax=ax0, include_normal=1,
                               include_roi_rot=1)
        
        _, mu, sigma = fl.plot_orientation_histo(
            pix_mask=m,
            apply_fit=True, 
            ax=ax1,
            color=line.color)
        ax1.legend_.remove()
        low, high = mu - sigma, mu + sigma
        fl.plot_length_histo(
            pix_mask=m, 
            ax=ax2, 
            dir_low=low,
            dir_high=high, 
            color=line.color)
        
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax0.set_title("")

    ymax = max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
    ax1.set_title("")
    ax1.set_xlabel(r"$\varphi\,[^\circ]$", fontsize=20)
    ax1.set_ylim([0, ymax])
    ax1.get_yaxis().set_ticks([])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel(r"$|\mathbf{f}|$ [pix]", fontsize=20)
    ax2.set_ylabel("Counts / bin", fontsize=20)
    ax2.set_ylim([0, ymax])

    ax2.set_title("")
    ax2.legend_.remove()

    sca(ax1)
    xticks(rotation=40, ha="right")
    sca(ax2)
    yticks(rotation=90, va="center")

    return fig


# SCRIPT MAIN FUNCTION
def main():
    close("all")
    figs = []
    # Prepare aa image list (see example 4)
    aa_list = prepare_aa_image_list()

    # the aa image list includes the measurement geometry, get pixel
    # distance image where pixel values correspond to step widths in the plume,
    # obviously, the distance values depend on the downscaling factor, which
    # is calculated from the analysis pyramid level (PYRLEVEL)
    dist_img, _, _ = aa_list.meas_geometry.compute_all_integration_step_lengths(pyrlevel=1)
    # set the pyramid level in the list
    aa_list.pyrlevel = 1
    
    # Access to the optical flow module in the image list. If optflow_mode is
    # active in the list, then, whenever the list index changes (e.g. using
    # list.goto_next(), or list.goto_img(100)), the optical flow field is
    # calculated between the current list image and the next one
    fl = aa_list.optflow
    # (! note: fl is only a pointer, i.e. the "=" is not making a copy of the
    # object, meaning, that whenever something changes in "fl", it also does
    # in "aa_list.optflow")

    # Now activate optical flow calculation in list (this slows down the
    # speed of the analysis, since the optical flow calculation is
    # comparatively slow
    s = aa_list.optflow.settings
    s.hist_dir_gnum_max = 10
    s.hist_dir_binres = 10
    s.hist_sigma_tol = 2

    s.roi_rad_abs = [0, 0, 1344, 730]

    aa_list.optflow_mode = True

    plume_mask = pyplis.Img(aa_list.get_thresh_mask(0.05))
    plume_mask.show(tit="AA threshold mask")

    figs.append(analyse_and_plot(aa_list, LINES))

    PCS1, PCS2 = LINES
    
    figs.append(fl.plot_flow_histograms(PCS1, plume_mask.img))
    figs.append(fl.plot_flow_histograms(PCS2, plume_mask.img))

    # Show an image containing plume speed magnitudes (ignoring direction)
    velo_img = pyplis.Img(fl.to_plume_speed(dist_img))
    velo_img.show(vmin=0, vmax=10, cmap="Greens",
                  tit="Optical flow plume velocities",
                  zlabel="Plume velo [m/s]")

    # Create two objects used to store time series information about the
    # retrieved plume properties
    plume_props_l1 = pyplis.plumespeed.LocalPlumeProperties(PCS1.line_id)
    plume_props_l2 = pyplis.plumespeed.LocalPlumeProperties(PCS2.line_id)

    aa_list.goto_img(0)
    stop_idx = aa_list.nof - 1
    print("Computing local optical flow properties for 2 PCS lines for each image in the image list")
    for k in range(0, stop_idx):
        if k%20==0:
            print(f"Computation ongoing ({k}/{stop_idx})")
        plume_mask = aa_list.get_thresh_mask(0.05)
        plume_props_l1.get_and_append_from_farneback(
            fl, line=PCS1, pix_mask=plume_mask,
            dir_multi_gauss=True)
        plume_props_l2.get_and_append_from_farneback(
            fl, line=PCS2, pix_mask=plume_mask,
            dir_multi_gauss=True)
        aa_list.goto_next()

    fig, ax = subplots(2, 1, figsize=(10, 9))

    plume_props_l1.plot_directions(ax=ax[0],
                                    color=PCS1.color,
                                    label="PCS1")
    plume_props_l2.plot_directions(ax=ax[0], color=PCS2.color,
                                    label="PCS2")

    plume_props_l1.plot_magnitudes(normalised=True, ax=ax[1],
                                    date_fmt="%H:%M:%S", color=PCS1.color,
                                    label="PCS1")
    plume_props_l2.plot_magnitudes(normalised=True, ax=ax[1],
                                    date_fmt="%H:%M:%S", color=PCS2.color,
                                    label="PCS2")
    ax[0].set_xticklabels([])
    
    figs.append(fig)

    # Save the time series as txt
    plume_props_l1.save_txt(SAVE_DIR / "ex09_plumeprops_young_plume.txt")
    plume_props_l2.save_txt(SAVE_DIR / "ex09_plumeprops_aged_plume.txt")

    if SAVEFIGS:
        for k in range(len(figs)):
            outfile = SAVE_DIR / f"ex09_out_{k + 1}.{FORMAT}"
            figs[k].savefig(outfile, format=FORMAT, dpi=DPI)

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        import numpy as np
        res = plume_props_l1
        assert res.roi_id == "young_plume"
        assert np.isnan(res._len_mu_norm).sum() == 5
        assert np.isnan(res._len_sigma_norm).sum() == 5
        assert np.isnan(res._dir_mu).sum() == 5
        assert (np.array(res._dir_sigma) == 180).sum() == 5
        assert (np.array(res._significance) == 0).sum() == 5
        
        npt.assert_allclose(
            actual=[
                np.nanmean(res._len_mu_norm),
                np.nanmean(res._len_sigma_norm),
                np.nanmean(res._dir_mu),
                np.nanmean(res._dir_sigma),
                np.nanmean(res._del_t),
                np.nanmean(res._significance),
                ],
            desired=[
                0.8446354828438595, 
                0.288896828513025, 
                -56.972104109591605, 
                19.873233016577196, 
                4.206971153846154, 
                0.7650539016892137
                ],
            rtol=1e-5)

        res = plume_props_l2
        assert res.roi_id == "old_plume"
        assert np.isnan(res._len_mu_norm).sum() == 5
        assert np.isnan(res._len_sigma_norm).sum() == 5
        assert np.isnan(res._dir_mu).sum() == 5
        assert (np.array(res._dir_sigma) == 180).sum() == 5
        assert (np.array(res._significance) == 0).sum() == 5
        
        npt.assert_allclose(
            actual=[
                np.nanmean(res._len_mu_norm),
                np.nanmean(res._len_sigma_norm),
                np.nanmean(res._dir_mu),
                np.nanmean(res._dir_sigma),
                np.nanmean(res._del_t),
                np.nanmean(res._significance),
                ],
            desired=[
                1.0966500134369748, 
                0.39550772843047144, 
                -93.31145142654839, 
                20.139529500869035, 
                4.206971153846154, 
                0.6430045694337098
                ],
            rtol=1e-5)
        print(f"All tests passed in script: {Path(__file__).name}")
    try:
        if int(options.show) == 1:
            show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()