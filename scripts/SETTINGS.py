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
from __future__ import (absolute_import, division)

from numpy import subtract
from os.path import join
from warnings import warn
from optparse import OptionParser
from matplotlib import rcParams

from pyplis.inout import find_test_data
from pyplis import __version__, LineOnImage

rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 13})

# if True, some of the results of the scripts are verified
TESTMODE = 1

# the pyplis version for which these scripts correspond to
SCRIPTS_VERSION = "1.4"

SAVEFIGS = 1  # save plots from this script in SAVE_DIR
DPI = 150  # pixel resolution for saving
FORMAT = "png"  # format for saving

SCREENPRINT = 0  # show images on screen when executing script

# Image directory
IMG_DIR = join(find_test_data(), "images")

# Directory where results are stored

SAVE_DIR = join(".", "scripts_out")
# SAVE_DIR = r'D:/Dropbox/TEMP/jgliss_publications/pyplis/graphics/out_code/'

# Emission rate retrieval lines

# ORANGE LINE IN YOUNG PLUME
PCS1 = LineOnImage(345, 350, 450, 195, pyrlevel_def=1,
                   line_id="young_plume", color="#e67300",
                   normal_orientation="left")

# BLUE LINE IN AGED PLUME
PCS2 = LineOnImage(80, 10, 80, 270, pyrlevel_def=1,
                   line_id="old_plume", color="#1a1aff",
                   normal_orientation="left")

LINES = [PCS1, PCS2]

OPTPARSE = OptionParser(usage='')
OPTPARSE.add_option('--show', dest="show", default=SCREENPRINT)
OPTPARSE.add_option('--test', dest="test", default=TESTMODE)
OPTPARSE.add_option('--clear', dest="clear", default=False)


def check_version():
    v_code = [int(x) for x in __version__.split(".")[:2]]
    v_scripts = [int(x) for x in SCRIPTS_VERSION.split(".")[:2]]
    if any(subtract(v_scripts, v_code)) != 0:
        warn("Version conflict between pyplis installation (v%s) "
             "and version of example scripts used (v%s). Please "
             "update your pyplis installation or use the set of example "
             "scripts corresponding to your installation. "
             % (__version__, SCRIPTS_VERSION))
