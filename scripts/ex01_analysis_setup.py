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
"""Pyplis example script no. 1 - Analysis setup for example data set.

In this script an example data set, recorded on the 16/9/15 7:06-7:22 at
Mt. Etna is setup. Most of the following example scripts will use the
information specified here.

A typical analysis setup for plume image data contains information about the
camera (e.g. optics, file naming convention, see also ex0_2_camera_setup.py),
gas source, the measurement geometry and the wind conditions. These information
generally needs to be specified by the user before the analysis. The
information is stored within a MeasSetup object which can be used as a basis
for further analysis.
If not all neccessary information is provided, a MeasSetup object will be
created nonetheless but analysis options might be limited.

Such a MeasSetup object can be used as input for Dataset objects which
creates the analysis environment. The main purpose of Dataset classes
is to automatically separate images by their type (e.g. on / off-band
images, dark, offset) and create ImgList classes from that. ImgList
classes typically contain images of one type. The Dataset also links
relevant ImgList to each other, e.g. if the camera has an off-band
filter, and off-band images can be found in the specified directory,
then it is linked to the list containing on-band images. This means,
that the current image index in the off-band list is automatically
updated whenever the index is changed in the on-band list. If
acquisition time is available in the images, then the index is updated
based on the closest acq. time of the off-band images, based on the
current on-band acq. time. If not, it is updated by index (e.g. on-band
contains 100 images and off-band list 50. The current image number in
the on-band list is 50, then the off-band index is set to 25). Also,
dark and offset images lists are linked both to on and off-band image
lists, such that the image can be correccted for dark and offset
automatically on image load (the latter needs to be activated in the
lists using ``darkcorr_mode=True``).

This script shows how to setup a MeasSetup object and create a Dataset object
from it. As example, the first image of the on-band image time series is
displayed.

The Dataset object created here is used in script  ex04_prep_aa_imglist.py
which shows how to create an image list displaying AA images.
"""
from SETTINGS import IMG_DIR, ARGPARSER
import pathlib
import pyplis as pyplis
from datetime import datetime
import matplotlib.pyplot as plt

def create_dataset():
    """Initialize measurement setup and creates dataset from that."""
    start = datetime(2015, 9, 16, 7, 6, 00)
    stop = datetime(2015, 9, 16, 7, 22, 00)
    # Define camera (here the default ecII type is used)
    cam_id = "ecII"

    # the camera filter setup
    filters = [pyplis.utils.Filter(type="on", acronym="F01"),
               pyplis.utils.Filter(type="off", acronym="F02")]

    # camera location and viewing direction (altitude will be retrieved
    # automatically)
    geom_cam = {"lon": 15.1129,
                "lat": 37.73122,
                "elev": 20.0,
                "elev_err": 5.0,
                "azim": 270.0,
                "azim_err": 10.0,
                "alt_offset": 15.0,
                "focal_length": 25e-3}  # altitude offset (above topography)

    # Create camera setup
    # the camera setup includes information about the filename convention in
    # order to identify different image types (e.g. on band, off band, dark..)
    # it furthermore includes information about the detector specifics (e.g.
    # dimension, pixel size, focal length). Measurement specific parameters
    # (e.g. lon, lat, elev, azim) where defined in the dictinary above and
    # can be passed as additional keyword dictionary using **geom_cam
    # Alternatively, they could also be passed directly:

    cam = pyplis.setupclasses.Camera(cam_id, filter_list=filters,**geom_cam)

    # Load default information for Etna. This information is stored in
    # the source_info.txt file of the Pyplis information. You may also access
    # information about any volcano via the available online access to the NOAA
    # database using the method pyplis.inout.get_source_info_online(source_id).

    source = pyplis.setupclasses.Source("etna")

    # Provide wind direction
    wind_info = {"dir": 0.0,
                 "dir_err": 1.0}

    # Create BaseSetup object (which creates the MeasGeometry object)
    stp = pyplis.setupclasses.MeasSetup(
        base_dir=IMG_DIR, 
        start=start, 
        stop=stop, 
        camera=cam,
        source=source,
        wind_info=wind_info
    )
    # Create analysis object (from BaseSetup)
    # The dataset takes care of finding all vali
    return pyplis.Dataset(stp)


# SCRIPT MAIN FUNCTION
def main():
    plt.close("all")
    ds = create_dataset()

    # get on-band image list
    on_list = ds.get_list("on")
    on_list.goto_next()
    off_list = ds.get_list("off")

    # activate dark correction in both lists. Dark and offset image lists are
    # automatically assigned to plume on and off-band image lists on initiation
    # of the dataset object
    on_list.darkcorr_mode = True
    off_list.darkcorr_mode = True

    print("On-band list contains %d images, current image index: %d"
          % (on_list.nof, on_list.cfn))

    img = on_list.current_img()

    # plume distance image retrieved from MeasGeometry class...
    plume_dists = on_list.plume_dists

    # ...these may be overwritten or set manually if desired
    on_list.plume_dists = 10000

    # The same applies for the integration step lengths for emission rate
    # retrievals
    on_list.integration_step_length = 1.8  # m

    img_shift = img.duplicate()

    # images can be shifted using the scipy.ndimage.interpolation.shift method
    # this may be required for image registration in dual camera systems.
    # Whether this is supposed to be done automatically can be specified using
    # the REG_SHIFT_OFF option in a MeasSetup class. It may also be specified
    # directly for your cam in the custom camera definition file cam_info.txt
    # using io_opts:REG_SHIFT_OFF=1 (see e.g. defintion of camera with ID
    # "usgs"). Also, a default registration offset can be defined here using
    #
    img_shift.shift(dx_abs=-30, dy_abs=55)
    img_shift.show(tit="Shifted")
    # Set pixel intensities below 2000 to 0 (method of Img class)
    img.set_val_below_thresh(val=0, threshold=2000)
    # show modified image
    img.show()
    print(str(img))  # image object has an informative string representation

    # IMPORTANT STUFF FINISHED (Below follow tests and display options)

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt

        actual = [plume_dists.mean(), plume_dists.std(),
                  on_list.get_dark_image().mean()]
        npt.assert_allclose(actual=actual,
                            desired=[10909.873427010458, 221.48844132471388,
                                     190.56119],
                            rtol=1e-7)

        npt.assert_array_equal([418, 2, 2368, 1, 1, 0,
                                20150916070600,
                                20150916072200],
                               [on_list.nof + off_list.nof,
                                on_list.current_img().is_darkcorr +
                                off_list.current_img().is_darkcorr,
                                sum(on_list.current_img().shape),
                                on_list.cfn,
                                off_list.cfn,
                                sum(img.img[img.img < 2000]),
                                int(ds.setup.start.strftime("%Y%m%d%H%M%S")),
                                int(ds.setup.stop.strftime("%Y%m%d%H%M%S"))])

        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()
