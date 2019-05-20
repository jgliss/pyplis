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
"""Pyplis example script 12 - Etna emission rate retrieval.

This example import results from the previous examples, for instance the AA
image list including measurement geometry (ex 4), the DOAS calibration
information (which was stored as FITS file, see ex. 6) and the AA sensitivity
correction mask retrieved from the cell calibration and normalised to the
position of the DOAS FOV (ex 7). The emission rates are retrieved for three
different plume velocity retrievals: 1. using the global velocity vector
retrieved from the cross correlation algorithm (ex8), 2. using the raw output
of the optical flow Farneback algorithm (``flow_raw``) and 3. using the
histogram based post analysis of the optical flow field (``flow_histo``).
The analysis is performed using the EmissionRateAnalysis class which basically
checks the AA list and activates ``calib_mode`` (-> images are loaded as
calibrated gas CD images) and loops over all images to retrieve the emission
rates for the 3 velocity modes. Here, emission rates are retrieved along 1
exemplary plume cross section. This can be easily extended by adding additional
PCS lines in the EmissionRateAnalysis class using ``add_pcs_line``.
The results for each velocity mode and for each PCS line are stored within
EmissionRateResults classes.
"""
from __future__ import (absolute_import, division)

from SETTINGS import check_version

from os.path import join, exists
from matplotlib.pyplot import close, show, GridSpec, figure, rc_context
from matplotlib.cm import get_cmap

import pyplis
# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE, LINES

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

rc_context({'font.size': '18'})

# Check script version
check_version()

PCS = LINES[0]

# If false, then only the working environment is initalised
DO_EVAL = True

# Dilution correction
DILCORR = True

# You can specify here if you only want a certain number of images analysed
START_INDEX = 0
STOP_INDEX = None  # 20


# SCRIPT OPTONS
PYRLEVEL = 1
PLUME_VELO_GLOB = 4.29  # m/s
PLUME_VELO_GLOB_ERR = 1.5
# applies multi gauss fit to retrieve local predominant displacement
# direction, if False, then the latter is calculated from 1. and 2. moment
# of histogram (Faster but more sensitive to additional peaks in histogram)
HISTO_ANALYSIS_MULTIGAUSS = True
# molar mass of SO2
MMOL = 64.0638  # g/mol
# minimum required SO2-CD for emission-rate retrieval
CD_MIN = 5e16

# activate background check mode, if True, emission rates are only
# retrieved  for images showing SO2-CDs within specified interval around
# zero in BG reference rectangle LOG_ROI_SKY (see above). This can be
# used to ensure that significant systematic errors are induced in case
# the plume background retrieval failed. The latter could, for instance
# happen, if, for instance a cloud moves through one of the background
# reference areas used to model the background (cf. example script 3)
REF_CHECK_LOWER = -5e16
REF_CHECK_UPPER = 5e16
REF_CHECK_MODE = True

# the following ROI is in the upper right image corner, where no gas occurs in
# the time series. It is used to log mean, min and max for each analysed image
# this information can be used to check, whether the plume background retrieval
# worked well
LOG_ROI_SKY = [530, 30, 600, 100]  # correspond to pyrlevel 1

# RELEVANT DIRECTORIES AND PATHS

# DOAS calibration results from example script 6
CALIB_FILE = join(SAVE_DIR, "ex06_doascalib_aa.fts")

# Scattering extinction coeffcients from example script 11 (stored as txt)
EXT_ON = join(SAVE_DIR, "ex11_ext_scat_on.txt")
EXT_OFF = join(SAVE_DIR, "ex11_ext_scat_off.txt")

# AA sensitivity correction mask retrieved from cell calib in script 7
CORR_MASK_FILE = join(SAVE_DIR, "ex07_aa_corr_mask.fts")

# time series of predominant displacement vector from histogram analysis of
# optical flow field in ROI around the PCS line "young_plume" which is used
# here for the emission rate retrieval. These information is optional, and is
# calculated during the evaluation if not provided
RESULT_PLUMEPROPS_HISTO = join(SAVE_DIR, "ex09_plumeprops_young_plume.txt")

# SCRIPT FUNCTION DEFINITIONS


def plot_and_save_results(ana, line_id="young_plume", date_fmt="%H:%M"):

    # plot colors for different optical flow retrievals
    cmap = get_cmap("Oranges")

    c_optflow_hybrid = cmap(255)
    c_optflow_histo = cmap(175)
    c_optflow_raw = cmap(100)

    fig = figure(figsize=(16, 12))
    gs = GridSpec(4, 1, height_ratios=[.6, .2, .2, .2], hspace=0.05)
    ax3 = fig.add_subplot(gs[3])
    ax0 = fig.add_subplot(gs[0], sharex=ax3)
    ax1 = fig.add_subplot(gs[1], sharex=ax3)
    ax2 = fig.add_subplot(gs[2], sharex=ax3)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    # Get emission rate results for the PCS line
    res0 = ana.get_results(line_id=line_id, velo_mode="glob")
    res1 = ana.get_results(line_id=line_id, velo_mode="flow_raw")
    res2 = ana.get_results(line_id=line_id, velo_mode="flow_histo")
    res3 = ana.get_results(line_id=line_id, velo_mode="flow_hybrid")

    res0.save_txt(join(SAVE_DIR, "ex12_flux_velo_glob.txt"))
    res1.save_txt(join(SAVE_DIR, "ex12_flux_flow_raw.txt"))
    res2.save_txt(join(SAVE_DIR, "ex12_flux_flow_histo.txt"))
    res3.save_txt(join(SAVE_DIR, "ex12_flux_flow_hybrid.txt"))

    # Plot emission rates for the different plume speed retrievals
    res0.plot(yerr=True, date_fmt=date_fmt, ls="-", ax=ax0,
              color="c", ymin=0, alpha_err=0.08)
    res1.plot(yerr=False, ax=ax0, ls="-", color=c_optflow_raw, ymin=0)
    res2.plot(yerr=False, ax=ax0, ls="--", color=c_optflow_histo, ymin=0)
    res3.plot(yerr=True, ax=ax0, lw=3, ls="-", color=c_optflow_hybrid, ymin=0)

    # ax[0].set_title("Retrieved emission rates")
    ax0.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=12)
    ax0.grid()

    # Plot effective velocity retrieved from optical flow histogram analysis
    res3.plot_velo_eff(ax=ax1, date_fmt=date_fmt, color=c_optflow_hybrid)
    # ax[1].set_title("Effective plume speed
    #                  (from optflow histogram analysis)")
    ax1.set_ylim([0, ax1.get_ylim()[1]])

    # Plot time series of predominant plume direction (retrieved from optical
    # flow histogram analysis and stored in object of type LocalPlumeProperties
    # which is part of plumespeed.py module
    ana.pcs_lines[line_id].plume_props.plot_directions(ax=ax2,
                                                       date_fmt=date_fmt,
                                                       color=c_optflow_hybrid)

    ax2.set_ylim([-180, 180])
    pyplis.helpers.rotate_xtick_labels(ax=ax2)
    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    # tight_layout()

    ax3 = ana.plot_bg_roi_vals(ax=ax3, date_fmt="%H:%M")
    # gs.tight_layout(fig, h_pad=0)#0.03)
    gs.update(hspace=0.05, top=0.97, bottom=0.07)
    return fig


# SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    figs = []
    if not exists(CALIB_FILE):
        raise IOError("Calibration file could not be found at specified "
                      "location:\n%s\nPlease run example 6 first")
    if not exists(CORR_MASK_FILE):
        raise IOError("Cannot find AA correction mask, please run example"
                      "script 7 first")

    # convert the retrieval line to the specified pyramid level (script option)
    pcs = PCS.convert(to_pyrlevel=PYRLEVEL)

    # now try to load results of optical flow histogram analysis performed for
    # this line in script no. 9. and assign them to the pcs line. This has the
    # advantage, that  missing velocity vectors (i.e. from images where optical
    # flow analysis failed) can be interpolated. It is, however, not
    # necessarily required to do this in advance. In the latter case the
    # emission rates show gaps at all images, where the optical flow was
    # considered not reliable
    try:
        p = pyplis.LocalPlumeProperties()
        p.load_txt(RESULT_PLUMEPROPS_HISTO)
        p = p.to_pyrlevel(PYRLEVEL)
        fig = p.plot(color="r")
        # p.interpolate()
        # p = p.apply_significance_thresh(0.2).interpolate()
        # p = p.apply_median_filter(3).apply_gauss_filter(2)
        fig = p.plot(date_fmt="%H:%M", fig=fig)

        pcs.plume_props = p
    except BaseException:
        print("Local plume properties could not be loaded and will be "
              "calculated during the emission rate analysis")

    # Load AA list
    # includes viewing direction corrected geometry
    aa_list = prepare_aa_image_list()

    aa_list.pyrlevel = PYRLEVEL

    if DILCORR:
        aa_list.import_ext_coeffs_csv(EXT_ON)
        aa_list.get_off_list().import_ext_coeffs_csv(EXT_OFF)

    # Load DOAS calbration data and FOV information (see example 6)
    doascalib = pyplis.doascalib.DoasCalibData()
    doascalib.load_from_fits(file_path=CALIB_FILE)
    doascalib.fit_calib_data()

    # Load AA corr mask and set in image list(is normalised to DOAS FOV see
    # ex7)
    aa_corr_mask = pyplis.Img(CORR_MASK_FILE)

    aa_list.senscorr_mask = aa_corr_mask

    # set DOAS calibration data in image list
    aa_list.calib_data = doascalib

    ana = pyplis.EmissionRateAnalysis(
        imglist=aa_list,
        bg_roi=LOG_ROI_SKY,
        pcs_lines=pcs,
        velo_glob=PLUME_VELO_GLOB,
        velo_glob_err=PLUME_VELO_GLOB_ERR,
        ref_check_lower_lim=REF_CHECK_LOWER,
        ref_check_upper_lim=REF_CHECK_UPPER,
        velo_dir_multigauss=HISTO_ANALYSIS_MULTIGAUSS,
        senscorr=True,
        dilcorr=DILCORR)

    ana.settings.ref_check_mode = REF_CHECK_MODE

    ana.settings.velo_modes["flow_raw"] = 1
    ana.settings.velo_modes["flow_histo"] = True
    ana.settings.velo_modes["flow_hybrid"] = 1
    ana.settings.min_cd = CD_MIN

    # plot all current PCS lines into current list image (feel free to define
    # and add more PCS lines above)
    ax = ana.plot_pcs_lines(
        vmin=-
        5e18,
        vmax=6e18,
        tit="Dilution corr: %s" %
        DILCORR)
    ax = ana.plot_bg_roi_rect(ax=ax, to_pyrlevel=PYRLEVEL)
    figs.append(ax.figure)

    if not DO_EVAL:
        aa_list.dilcorr_mode = not DILCORR
        aa_list.show_current(
            vmin=-
            5e18,
            vmax=6e18,
            tit="Dilution corr: %s" %
            (not DILCORR))
        # you can check the settings first
        print(ana.settings)
        # check if optical flow works
        ana.imglist.optflow_mode = True
        aa_mask = ana.imglist.get_thresh_mask(CD_MIN)
        ana.imglist.optflow.plot_flow_histograms(line=pcs, pix_mask=aa_mask)

    else:
        ana.run_retrieval(start_index=START_INDEX,
                          stop_index=STOP_INDEX)

        figs.append(plot_and_save_results(ana))
        # the EmissionRateResults class has an informative string
        # representation
        print(ana.get_results("young_plume", "flow_histo"))

    if SAVEFIGS:
        for k in range(len(figs)):
            figs[k].savefig(join(SAVE_DIR, "ex12_out_%d.%s" % (k + 1, FORMAT)),
                            format=FORMAT, dpi=DPI)

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    (options, args) = OPTPARSE.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        from os.path import basename

        npt.assert_array_equal([],
                               [])

        npt.assert_allclose(actual=[],
                            desired=[],
                            rtol=1e-7)
        print("All tests passed in script: %s" % basename(__file__))
    try:
        if int(options.show) == 1:
            show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")
