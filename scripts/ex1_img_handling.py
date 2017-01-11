# -*- coding: utf-8 -*-
"""
piscope example script 1 - Image import
"""
from os.path import join
import piscope
from os import getcwd

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

#fake filename with delimiter _ and encrypted information (see below) 
img_file_name = "test_201509160708_F01_0.3348.fts"

file_path = join(piscope._LIBDIR, "data", img_file_name)

# Create new Camera class
cam = piscope.setup.Camera()
cam.delim = "_"
cam.time_info_pos = 1
cam.time_info_str = "%Y%m%d%H%M" #datetime conversion string
cam.filter_id_pos = 2
cam.texp_pos = 3

### Extract image meta info from the image filename specified
start_acq, filter_id, meas_type, texp, warnings =\
            cam.get_img_meta_from_filename(file_path)

# Print the extracted information
print "Start acq: %s" %start_acq
print "Exposure time [s]: %s"  %texp
print "Filter ID: %s" %filter_id
print "Meas type: %s" %meas_type

img = piscope.image.Img(file_path, start_acq = start_acq, texp = texp)
### Show image
img.show()
print img

