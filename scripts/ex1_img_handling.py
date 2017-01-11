# -*- coding: utf-8 -*-
"""
piscope example script 1 - Image import
"""
from os.path import join
import piscope
from os import getcwd
from os.path import exists
### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")
    
img_file_name = "EC2_1106307_1R02_2015091607080439_F01_Etna.fts"

file_path = join(img_dir, img_file_name)

# This also includes information about what image meta data can be extracted
# from file names
cam = piscope.setup.Camera("ecII")

### Extract image meta info from the image specified above
# In case of the ECII camera naming convention, the acquisition time and a 
# filter acronym is encrypted within the file names. A sub string for defining
# a meas type does not exist in this convention and it is hence set to the
# filter ID position in the file name (i.e. meas_type and filter_id are 
# identical)
start_acq, filter_id, meas_type, texp, warnings =\
            cam.get_img_meta_from_filename(file_path)

# Print the extracted information
print "Start acq: %s" %start_acq
print "Exposure time [s]: %s"  %texp
print "Filter ID: %s" %filter_id
print "Meas type: %s" %meas_type

### Load image object and include the extracted acquisition time in Img meta header
meta_dict = {"start_acq"    : start_acq,
             "texp"         : texp}
             
#img = piscope.image.Img(file_path,  **meta_dict)
img = piscope.image.Img(file_path, start_acq = start_acq, texp = texp)
### Show image
img.show()

