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
"""Pyplis example script no. 6 - DOAS calibration and FOV search.

Script showing how to work with DOAS calibration data

In this script, a stack of plume AA images from the Etna test data is imported
as well as a time series of DOAS SO2-CDs in the plume retrieved using the
software DOASIS (see directory "spectra" in test data folder for corresponding
analysis details, the folder also contains the RAW data and the jscript code
for analysing the spectra). The DOAS result import is performed using the
Python package ``pydoas``.

Based on these data, position and shape of the DOAS FOV within the camera
uimages is identified using both FOV search methods (IFR and Pearson). The
results of the FOV search are plotted as well as the corresponding calibration
curves retrieved for both FOV parametrisations.

Note
------

In case a MemoryError occurs while determining the AA image stack, then the
stack (3D numpy array) is too large for the RAM. In this case, try
increasing script option PYRLEVEL_ROUGH_SEARCH.

"""
from __future__ import (absolute_import, division)

from SETTINGS import check_version

import pyplis
import pydoas
import numpy.testing as npt
import numpy as np
from datetime import timedelta
from matplotlib.pyplot import close, show, subplots
from os.path import join, exists
from os import remove

# IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, IMG_DIR, OPTPARSE

# IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

# Check script version
check_version()

# SCRIPT OPTONS

# reload and save stack in folder SAVE_DIR, results in increased
# running time due to stack calculation (is automatically switched on if
# the stack is not found at this location)
RELOAD_STACK = False

PYRLEVEL_ROUGH_SEARCH = 2
# Default search settings are at pyramid level 2, the FOV results are upscaled
# to original resolution, if the following option is set 1, then, based on
# the result from pyrlevel=2, another stack is determined at pyrlevel = 0
# (i.e. in full resolution) within a ROI around the center position from the
# search performed at pyrlevel=PYRLEVEL_ROUGH_SEARCH (defined 2 lines below)
DO_FINE_SEARCH = False


# RELEVANT DIRECTORIES AND PATHS

# Directory containing DOAS result files
DOAS_DATA_DIR = join(IMG_DIR, "..", "spectra", "plume_prep", "min10Scans",
                     "ResultFiles")

STACK_PATH = join(SAVE_DIR, "ex06_aa_imgstack.fts")

# SCRIPT FUNCTION DEFINITIONS


def load_doas_results(lt_to_utc_shift=timedelta(-1. / 12)):
    """Specify DOAS data import from DOASIS fit result files.

    In order to perform the DOAS FOV search, as many spectrum datapoints
    as possible are needed. Therefore, only 10 spectra were added (to reduce
    noise) per plume spectrum. The DOAS fit was performed in a wavelength
    range between 314 - 326 nm (fit ID: "f01").
    """
    # This dictionary specifies which information is supposed to be imported
    # from the DOAS fit result files stored in DOAS_DATA_DIR. In the example
    # shown here, only the SO2 fit results are imported from fit scenario
    # with ID "f01" (key of dict). The corresponding value of each key is
    # a list of format ["header_id", ["fit_id1", "fit_id2", ..., "fit_idN"]]
    # specifying the identification string of the species in the result file
    # headers and the second entry is a list specifying all fit scenario IDs
    # from which this species is supposed to be imported (here only f01)
    fit_import_info = {"so2": ["SO2_Hermans_298_air_conv_satCorr1e18",
                               ["f01"]
                               ]}

    # Create a result import setup for the DOAS data based on the import
    # dictionary and the image base directory of the result files ...
    doas_import_setup =\
        pydoas.dataimport.ResultImportSetup(DOAS_DATA_DIR,
                                            result_import_dict=fit_import_info)

    # ... and create a result dataset from that
    doas_dataset = pydoas.analysis.DatasetDoasResults(doas_import_setup)

    # get the SO2 fit results from the dataset. Individual results of certain
    # species can be accessed using the species ID (key in ``fit_import_info``
    # dict) and its fit ID (one of the fit IDs specified for this species, here
    # f01).
    # Note, that the DOAS data was stored using local time, thus they need to
    # be shifted (2h back) to match the camera data time stamps (which are in
    # UTC), otherwise the temporal merging of the two datasets (for the DOAS
    # calibration) does not work
    results_utc = doas_dataset.get_results("so2", "f01").shift(lt_to_utc_shift)
    return results_utc


def make_aa_stack_from_list(aa_list, roi_abs=None, pyrlevel=None,
                            save=True, stack_path=STACK_PATH,
                            save_dir=SAVE_DIR):
    """Get and prepare onband list for aa image mode."""
    # Deactivate auto reload to change some settings (if auto_reload is active
    # list images are reloaded whenever a setting is changed in the list. This
    # can slow down things, thus, if you intend to change a couple of settings
    # you might deactivate auto_reload, adapt the settings and then re-activate
    # auto_reload
    aa_list.auto_reload = False
    if roi_abs is not None:
        aa_list.roi_abs = roi_abs
        aa_list.crop = True
    aa_list.pyrlevel = pyrlevel
    aa_list.auto_reload = True

    # Stack all images in image list at pyrlevel 2 and cropped using specified
    # roi (uncropped if roi_abs=None).
    stack = aa_list.make_stack()
    if save:
        try:
            remove(stack_path)
        except BaseException:
            pass
        stack.save_as_fits(save_dir=save_dir,
                           save_name="ex06_aa_imgstack.fts")
    return stack


def get_stack(reload_stack=RELOAD_STACK, stack_path=STACK_PATH,
              pyrlevel=PYRLEVEL_ROUGH_SEARCH):
    """Load stack data based on current settings."""
    if not exists(stack_path):
        reload_stack = 1

    if not reload_stack:
        stack = pyplis.processing.ImgStack()
        stack.load_stack_fits(stack_path)
        if stack.pyrlevel != pyrlevel:
            reload_stack = True
    aa_list = None
    if reload_stack:
        # import AA image list
        aa_list = prepare_aa_image_list()
        # Try creating stack
        stack = make_aa_stack_from_list(aa_list, pyrlevel=pyrlevel)

    return stack, aa_list


# Test functions used at the end of the script
def test_calib_pears_init(calib):
    calib.fit_calib_data(polyorder=1)
    cc = pyplis.helpers.get_img_maximum(calib.fov.corr_img.img)
    assert cc == (124, 159), cc

    pyrl = calib.fov.pyrlevel
    assert pyrl == 2, pyrl
    res_dict = calib.fov.result_pearson
    npt.assert_allclose([res_dict['rad_rel'],
                         np.max(100 * res_dict['corr_curve'].values)],
                        [3, 95], atol=1)

    fov_ext = calib.fov.pixel_extend(abs_coords=True)
    (fov_x, fov_y) = calib.fov.pixel_position_center(abs_coords=True)

    npt.assert_allclose([fov_ext, fov_x, fov_y],
                        [res_dict['rad_rel'] * 2**pyrl, 636, 496], atol=1)

    npt.assert_allclose(calib.calib_coeffs,
                        [8.58e+18, 2.71e+17], rtol=1e-1)


def test_calib_pears_fine(calib):
    cc = pyplis.helpers.get_img_maximum(calib.fov.corr_img.img)
    npt.assert_allclose((186, 180), cc, atol=1)

    pyrl = calib.fov.pyrlevel
    assert pyrl == 0, pyrl
    res_dict = calib.fov.result_pearson
    npt.assert_allclose([res_dict['rad_rel'],
                         np.max(100 * res_dict['corr_curve'].values)],
                        [6, 95], atol=1)

    fov_ext = calib.fov.pixel_extend(abs_coords=True)
    (fov_x, fov_y) = calib.fov.pixel_position_center(abs_coords=True)

    npt.assert_allclose([fov_ext, fov_x, fov_y],
                        [6, 630, 493], atol=1)

    npt.assert_allclose(calib.calib_coeffs,
                        [8.38e+18, 2.92e+17], rtol=1e-1)


def test_calib_ifr(calib):
    cc = pyplis.helpers.get_img_maximum(calib.fov.corr_img.img)
    npt.assert_allclose((123, 157), cc, atol=1)

    pyrl = calib.fov.pyrlevel
    assert pyrl == 2, pyrl
    npt.assert_allclose(calib.fov.result_ifr['popt'][1:5],
                        [158.6, 122.9, 15.4, 1.5], rtol=1e-1)

    (fov_x, fov_y) = calib.fov.pixel_position_center(abs_coords=True)

    npt.assert_allclose([fov_x, fov_y, calib_ifr.fov.sigma_x_abs,
                         calib_ifr.fov.sigma_y_abs],
                        [635, 492, 61.5, 41.6], atol=2)

    npt.assert_allclose(calib.calib_coeffs,
                        [9.38e+18, 1.75e+17], rtol=1e-1)


# SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    # close all plots
    close("all")

    # import DOAS results
    doas_time_series = load_doas_results()

    # Import script options
    (options, args) = OPTPARSE.parse_args()

    if options.test:
        # if test mode is active, the image stack is always recomputed from
        # scratch and the option DO_FINE_SEARCH is activated, since the tests
        # are based on this. This will lead to an increased computation time
        # Test mode can be activated / deactivated in SETTINGS.py or via
        # the script option --test 1 (on) or --test 0 (off)
        RELOAD_STACK = True
        PYRLEVEL_ROUGH_SEARCH = 2
        DO_FINE_SEARCH = True

    # reload or create the AA image stack based on current script settings
    stack, aa_list = get_stack()

    s = pyplis.doascalib.DoasFOVEngine(stack, doas_time_series)
    calib_pears = s.perform_fov_search(method="pearson")
    calib_ifr = s.perform_fov_search(method="ifr", ifrlbda=4e-3)

    # plot the FOV search results
    ax0 = calib_pears.fov.plot()
    ax1 = calib_ifr.fov.plot()

    calib_pears.fit_calib_data()
    calib_ifr.fit_calib_data()

    fig, ax2 = subplots(1, 1)
    calib_pears.plot(add_label_str="Pearson", color="b", ax=ax2)

    calib_ifr.plot(add_label_str="IFR", color="g", ax=ax2)
    ax2.set_title("Calibration curves Pearson vs. IFR method")
    ax2.grid()
    ax2.set_ylim([0, 1.8e18])
    ax2.set_xlim([0, 0.20])
    ax2.legend(loc=4, fancybox=True, framealpha=0.7, fontsize=11)
    axes = [ax0, ax1, ax2]

    if DO_FINE_SEARCH:
        """Perform FOV search within ROI around result from pearson fov
        search at full resolution (pyrlevel=0)
        """
        if aa_list is None:
            aa_list = prepare_aa_image_list()
        # remember some properties of the current image stack that is stored in
        # the DoasFOVEngine object (i.e. the merged one in low pixel res.)

        num_merge, h, w = s.img_stack.shape
        s_fine = s.run_fov_fine_search(aa_list, doas_time_series,
                                       method="pearson")

        calib_pears_fine = s_fine.calib_data
        calib_pears_fine.plot()
        calib_pears_fine.fov.plot()

    calib_pears.save_as_fits(save_dir=SAVE_DIR,
                             save_name="ex06_doascalib_aa.fts")
    calib_ifr.save_as_fits(save_dir=SAVE_DIR,
                           save_name="ex06_doascalib_aa_ifr_method.fts")

    # you can also change the order of the calibration polynomial and
    # force it to go through the origin
    calib_pears.fit_calib_data(polyorder=2, through_origin=True)
    calib_pears.plot_calib_fun(add_label_str="Pearson (WRONG,\n"
                               "2nd order, through origin)",
                               color="r", ax=ax2)
    ax2.legend(loc=4, fancybox=True, framealpha=0.7, fontsize=11)
    # IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        for k in range(len(axes)):
            ax = axes[k]
            ax.set_title("")
            ax.figure.savefig(join(SAVE_DIR, "ex06_out_%d.%s"
                                   % ((k + 1), FORMAT)),
                              format=FORMAT, dpi=DPI)

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        from os.path import basename

        num, h, w = stack.shape
        num2 = s.img_stack.shape[0]  # stack after fine FOV search
        prep = stack.img_prep

        # check some basic properties of the data used for the different FOV
        # searches
        npt.assert_array_equal(
            [len(doas_time_series), num, num_merge, h, w, stack.pyrlevel,
             prep["darkcorr"] * prep["is_tau"] * prep["is_aa"], num2],
            [120, 209, 88, 256, 336, 2, 1, 209])

        test_calib_pears_init(calib_pears)
        test_calib_ifr(calib_ifr)
        test_calib_pears_fine(calib_pears_fine)

        print("All tests passed in script: %s" % basename(__file__))
    try:
        if int(options.show) == 1:
            show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")
