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
"""Introduction script 2 - The Camera class.

This script introduces the camera class which is of fundamental importance
for image import (e.g. separating on, off, dark background images, etc.) and
also for the data analysis as it includes all relevant detector specifications,
such as number of pixels, pixel size, focal length, etc.

In this script, a newer version of the camera type "ecII" is created manually
in order to illustrate all relevant parameters. The only difference to the
classic ecII camera is, that the filter setup is different.

See also here for more information:

https://pyplis.readthedocs.io/en/latest/tutorials.html#data-import-specifying-
custom-camera-information
"""
import pathlib
from SETTINGS import ARGPARSER
from numpy.testing import assert_array_equal
import os
import tempfile
import pyplis

# ## SCRIPT FUNCTION DEFINITIONS
def create_ecII_cam_new_filters(cam_id: str) -> pyplis.Camera:
    # Start with creating an empty Camera object
    cam = pyplis.setupclasses.Camera(cam_id=cam_id, try_load_from_registry=False)

    # Specify the camera filter setup

    # Create on and off band filters. Obligatory parameters are "type" and
    # "acronym", "type" specifies the filter type ("on" or
    # "off"), "acronym" specifies how to identify this filter in the file
    # name. If "id" is unspecified it will be equal to "type". The parameter
    # "meas_type_acro" is only important if a measurement type (e.g. M -> meas,
    # C -> calib ...) is explicitely specified in the file name.
    # This is not the case for ECII camera but for the HD camera,
    # see specifications in file cam_info.txt for more info.

    on_band = pyplis.utils.Filter(id="on", type="on", acronym="F01",
                                  meas_type_acro="F01", center_wavelength=310)
    off_band = pyplis.utils.Filter(type="off", acronym="F02",
                                   center_wavelength=330)

    # put the two filter into a list and assign to the camera
    filters = [on_band, off_band]

    cam.default_filters = filters
    cam.prepare_filter_setup()

    # Similar to the filter setup, access info for dark and offset images needs
    # to be specified. The ECII typically records 4 different dark images, two
    # recorded at shortest exposure time -> offset signal predominant, one at
    # low and one at high read gain. The other two recorded at longest possible
    # exposure time -> dark current predominant, also at low and high read gain

    offset_low_gain = pyplis.utils.DarkOffsetInfo(id="offset0", type="offset",
                                                  acronym="D0L", read_gain=0)
    offset_high_gain = pyplis.utils.DarkOffsetInfo(id="offset1", type="offset",
                                                   acronym="D0H", read_gain=1)
    dark_low_gain = pyplis.utils.DarkOffsetInfo(id="dark0", type="dark",
                                                acronym="D1L", read_gain=0)
    dark_high_gain = pyplis.utils.DarkOffsetInfo(id="dark1", type="dark",
                                                 acronym="D1H", read_gain=1)

    # put the 4 dark info objects into a list and assign to the camera
    dark_info = [offset_low_gain, offset_high_gain,
                 dark_low_gain, dark_high_gain]

    cam.dark_info = dark_info

    # Now specify further information about the camera

    # camera ID (needs to be unique, i.e. not included in data base, call
    # pyplis.inout.get_all_valid_cam_ids() to check existing IDs)
    cam.cam_id = cam_id

    # image file type
    cam.file_type = "fts"

    # File name delimiter for information extraction
    cam.delim = "_"

    # position of acquisition time (and date) string in file name after
    # splitting with delimiter
    cam.time_info_pos = 3

    # datetime string conversion of acq. time string in file name
    cam.time_info_str = "%Y%m%d%H%M%S%f"

    # position of image filter type acronym in filename
    cam.filter_id_pos = 4

    # position of meas type info
    cam.meas_type_pos = 4

    # Define which dark correction type to use
    # 1: determine a dark image based on image exposure time using a dark img
    # (with long exposure -> dark current predominant) and a dark image with
    # shortest possible exposure (-> detector offset predominant). For more
    # info see function model_dark_image in processing.py module
    # 2: subtraction of a dark image recorded at same exposure time than the
    # actual image
    cam.darkcorr_opt = 1

    # If the file name also includes the exposure time, this can be specified
    # here:
    cam.texp_pos = ""  # the ECII does not...

    # the unit of the exposure time (choose from "s" or "ms")
    cam.texp_unit = ""

    # define the main filter of the camera (this is only important for cameras
    # which include, e.g. several on band filters.). The ID need to be one of
    # the filter IDs specified above
    cam.main_filter_id = "on"

    # camera focal length can be specified here (but does not need to be, in
    # case of the ECII cam, there is no "default" focal length, so this is left
    # empty)
    cam.focal_length = None

    # Detector geometry
    cam.pix_height = 4.65e-6  # pixel height in m
    cam.pix_width = 4.65e-6  # pixel width in m
    cam.pixnum_x = 1344
    cam.pixnum_y = 1024

    cam._init_access_substring_info()

    cam.io_opts = dict(USE_ALL_FILES=False,
                       SEPARATE_FILTERS=True,
                       INCLUDE_SUB_DIRS=True,
                       LINK_OFF_TO_ON=True)

    # Set the custom image import method
    cam.image_import_method = pyplis.custom_image_import.load_ecII_fits
    # That's it...
    return cam

def main():
     
    # Create a test cam_info.txt file for this script in order not to mess with the camera info file shipped with pyplis
    with tempfile.TemporaryDirectory() as temp_data_dir:
        os.environ["PYPLIS_DATADIR"] = temp_data_dir
        cam_info_file_tmp = pathlib.Path(temp_data_dir) / "cam_info.txt"
        cam_info_file_tmp.touch()

        
        available_cam_defs = pyplis.utils.get_cam_ids()
        new_cam_id = "test_cam"
        if new_cam_id in available_cam_defs:
            raise ValueError(f"Camera with ID {new_cam_id} already exists, please choose a different ID")
        
        cam = create_ecII_cam_new_filters(new_cam_id)

        print(cam)

        # you can add the cam to the database (raises error if ID conflict
        # occurs, e.g. if the camera was already added to the database)
        cam.save_as_default(cam_info_file=cam_info_file_tmp)
        
        cam_reload = pyplis.Camera(new_cam_id, cam_info_file=cam_info_file_tmp)
        print(cam_reload)
        
        # ## IMPORTANT STUFF FINISHED - everything below is of minor importance
        # for educational purposes

        options = ARGPARSER.parse_args()
        # apply some tests. This is done only if TESTMODE is active: testmode can
        # be activated globally (see SETTINGS.py) or can also be activated from
        # the command line when executing the script using the option --test 1
        if int(options.test):
            # quick and dirty test

            cam_dict_nominal = {'darkcorr_opt': 1,
                                '_fid_subnum_max': 1,
                                '_fname_access_flags': {'filter_id': False,
                                                        'meas_type': False,
                                                        'start_acq': False,
                                                        'texp': False},
                                '_mtype_subnum_max': 1,
                                '_time_info_subnum': 1,
                                'cam_id': new_cam_id,
                                'delim': '_',
                                'file_type': 'fts',
                                'filter_id_pos': 4,
                                'focal_length': '',
                                'image_import_method':
                                    pyplis.custom_image_import.load_ecII_fits,
                                'main_filter_id': 'on',
                                'meas_type_pos': 4,
                                'pix_height': 4.65e-06,
                                'pix_width': 4.65e-06,
                                'pixnum_x': 1344,
                                'pixnum_y': 1024,
                                'ser_no': 9999,
                                'texp_pos': '',
                                'texp_unit': '',
                                'time_info_pos': 3,
                                'time_info_str': '%Y%m%d%H%M%S%f'}

            from collections import OrderedDict
            geom_data_nominal = OrderedDict([('lon', None),
                                            ('lat', None),
                                            ('altitude', None),
                                            ('azim', None),
                                            ('azim_err', None),
                                            ('elev', None),
                                            ('elev_err', None),
                                            ('alt_offset', 0.0)])

            arr_nominal = list(geom_data_nominal.items())
            arr_nominal.extend(list(cam_dict_nominal.items()))

            arr_vals = list(cam.geom_data.items())
            for k in cam_dict_nominal:
                arr_vals.append((k, cam.__dict__[k]))

            assert_array_equal(arr_nominal, arr_vals)

            print(f"All tests passed in script: {pathlib.Path(__file__).name}")

if __name__ == "__main__":
    main()
