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
"""Pyplis introduction script 1 - Image import.

This script loads an exemplary image which is part of the pyplis supplied
data. Image data in pyplis is represented by the ``Img`` class, which also
allows for storage of image meta data and keeps track of changes applied to
the image data (e.g. cropping, blurring, dark correction).

The script also illustrates how to manually work with image meta data
encoded in the image file name. The latter can be performed automatically in
pyplis using file name conventions (which can be specified globally, see next
script).
"""
from datetime import datetime
import matplotlib.pyplot as plt
import pathlib
import pyplis

# imports from SETTINGS.py
from SETTINGS import ARGPARSER, SAVE_DIR

# file name of test image stored in data folder
IMG_FILE_NAME = "test_201509160708_F01_335.fts"

IMG_DIR = pathlib.Path(pyplis.__dir__) / "data"

def main():
    plt.close("all")

    img_path = IMG_DIR / IMG_FILE_NAME

    # Create Img object (Img objects can be initiated both with image file
    # paths but also with data in memory in form of a 2D numpy array)
    img = pyplis.image.Img(img_path)

    # log mean of uncropped image for testing mode
    avg = img.mean()

    # The file name of the image includes some image meta information which can
    # be set manually (this is normally done automatically by defining a file
    # name convention, see next script)

    # split filename using delimiter "_"
    spl = IMG_FILE_NAME.split(".")[0].split("_")

    # extract acquisition time and convert to datetime
    acq_time = datetime.strptime(spl[1], "%Y%m%d%H%M")
    # extract and convert exposure time from filename (convert from ms -> s)
    texp = float(spl[3]) / 1000

    img.meta["start_acq"] = acq_time
    img.meta["texp"] = texp
    img.meta["f_num"] = 2.8
    img.meta["focal_length"] = 25e-3

    # add some blurring to the image
    img.add_gaussian_blurring(sigma_final=3)

    # crop the image edges
    roi_crop = [100, 100, 1244, 924]  # [x0, y0, x1, y1]
    img.crop(roi_abs=roi_crop)

    # apply down scaling (gauss pyramid)
    img.to_pyrlevel(2)

    # ## Show image
    img.show()

    img.save_as_fits(SAVE_DIR, "ex0_1_imgsave_test")

    img_reload = pyplis.Img(SAVE_DIR / "ex0_1_imgsave_test.fts")

    # print image information
    print(img)

    # ## IMPORTANT STUFF FINISHED - everything below is of minor importance
    # for educational purposes

    options = ARGPARSER.parse_args()
    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt

        vals = [img.mean(), avg, int(spl[1]), texp,
                img_reload.meta["texp"],
                img_reload.meta["f_num"],
                img_reload.meta["focal_length"]]

        npt.assert_almost_equal([2526.4623422672885,
                                 2413.0870433989026,
                                 201509160708,
                                 0.335,
                                 0.335,
                                 2.8,
                                 0.025],
                                vals, 4)

        print(f"All tests passed in script: {pathlib.Path(__file__).name}")

if __name__ == "__main__":
    main()