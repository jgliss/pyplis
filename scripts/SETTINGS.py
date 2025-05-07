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
from numpy import subtract
import pathlib
from warnings import warn
import argparse
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

SCREENPRINT = 0 # show images on screen when executing script

# Image directory
TESTDATA_DIR = find_test_data()
IMG_DIR = pathlib.Path(TESTDATA_DIR) / "images"

# Directory where results are stored
SCRIPTS_DIR = pathlib.Path(__file__).parent
SAVE_DIR = SCRIPTS_DIR / "scripts_out"

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

ARGPARSER = argparse.ArgumentParser()
ARGPARSER.add_argument('--show', dest="show", default=SCREENPRINT)
ARGPARSER.add_argument('--test', dest="test", default=TESTMODE)
ARGPARSER.add_argument('--clear', dest="clear", default=False)
