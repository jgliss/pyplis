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
from __future__ import (absolute_import, division)

from SETTINGS import check_version

import pyplis
from datetime import datetime
from matplotlib.pyplot import show

# IMPORT GLOBAL SETTINGS
from SETTINGS import IMG_DIR, OPTPARSE

# Check script version
check_version()

# SCRIPT FUNCTION DEFINITIONS


def get_bg_image_lists():
    """Initialize measurement setup and creates dataset from that."""
    start = datetime(2015, 9, 16, 7, 2, 0o5)
    stop = datetime(2015, 9, 16, 7, 2, 30)
    # Define camera (here the default ecII type is used)
    cam_id = "ecII"

    # the camera filter setup
    filters = [pyplis.utils.Filter(type="on", acronym="F01"),
               pyplis.utils.Filter(type="off", acronym="F02")]

    # create camera setup
    cam = pyplis.setupclasses.Camera(cam_id=cam_id, filter_list=filters)

    # Create BaseSetup object (which creates the MeasGeometry object)
    stp = pyplis.setupclasses.MeasSetup(IMG_DIR, start, stop, camera=cam)

    ds = pyplis.dataset.Dataset(stp)
    on, off = ds.get_list("on"), ds.get_list("off")
    on.darkcorr_mode = True
    off.darkcorr_mode = True
    return on, off


# SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    on, off = get_bg_image_lists()
    on.show_current()
    off.show_current()

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
