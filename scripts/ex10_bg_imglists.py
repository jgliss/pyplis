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
"""Pyplis example script no. 10 - Create background image dataset."""
import pyplis
from datetime import datetime
from matplotlib.pyplot import show
from pathlib import Path

# IMPORT GLOBAL SETTINGS
from SETTINGS import IMG_DIR, ARGPARSER

def get_bg_image_lists():
    # start time of sky background image acquisition
    start = datetime(2015, 9, 16, 7, 2, 5)
    
    # stop time of sky background image acquisition
    stop = datetime(2015, 9, 16, 7, 2, 30)

    # Define camera (in the example dataset from Etna, the ECII camera is used)
    cam_id = "ecII"

    # Declare the on and offband camera filters used
    filters = [pyplis.utils.Filter(type="on", acronym="F01"),
               pyplis.utils.Filter(type="off", acronym="F02")]

    # Create Camera instance using the ECII camera type and filters
    cam = pyplis.setupclasses.Camera(cam_id=cam_id, filter_list=filters)

    # Create BaseSetup object (which creates the MeasGeometry object)
    stp = pyplis.setupclasses.MeasSetup(IMG_DIR, start, stop, camera=cam)

    ds = pyplis.dataset.Dataset(stp)
    
    
    on_list, off_list = ds.get_list("on"), ds.get_list("off")
    on_list.darkcorr_mode = True
    off_list.darkcorr_mode = True
    return on_list, off_list

def main():
    on_list, off_list = get_bg_image_lists()
    on_list.show_current()
    off_list.show_current()

    # Import script options
    options = ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt
        npt.assert_allclose(actual=[on_list.current_img().mean(), off_list.current_img().mean()],
                            desired=[2555.4597, 2826.8848],
                            rtol=1e-3)
        print(f"All tests passed in script: {Path(__file__).name}")
    try:
        if int(options.show) == 1:
            show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    # Execute main function
    main()
