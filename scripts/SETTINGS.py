# -*- coding: utf-8 -*-
"""
Global settings for example scripts
"""
from pyplis.inout import find_test_data
from pyplis import _LIBDIR
from os.path import join, abspath
from optparse import OptionParser

SAVEFIGS = 1 # save plots from this script in SAVE_DIR
DPI = 150 #pixel resolution for saving
FORMAT = "png" #format for saving

SCREENPRINT = 1 #show images on screen when executing script

# Image directory
IMG_DIR = join(find_test_data(), "images")

# Directory where results are stored

SAVE_DIR = join(".", "scripts_out")
#SAVE_DIR = r'D:/Dropbox/TEMP/jgliss_publications/pyplis/graphics/out_code/'


OPTPARSE = OptionParser(usage='')
OPTPARSE.add_option('--show', dest="show", default=SCREENPRINT)

from matplotlib import rcParams
rcParams.update({'font.size': 13})