# -*- coding: utf-8 -*-
"""
Global settings for example scripts
"""
from piscope.inout import find_test_data
from os.path import join

SAVEFIGS = 1 # save plots from this script in SAVE_DIR
DPI = 300 #pixel resolution for saving
FORMAT = "png" #format for saving

# Image directory
IMG_DIR = join(find_test_data(), "images")

# Directory where results are stored
#SAVE_DIR = join(".", "scripts_out")
SAVE_DIR = r'D:/Dropbox/TEMP/jgliss_publications/piscope/graphics/out_code/'