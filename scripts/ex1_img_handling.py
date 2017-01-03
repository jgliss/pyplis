# -*- coding: utf-8 -*-
"""
PISCOPE example script 1 - Camera specification
"""
from os.path import join
import piscope

img_dir = "../test_data/piscope_etna_testdata/images/"
ecII_filename = "EC2_1106307_1R02_2015091607080439_F01_Etnaxxxxxxxxxxxx.fts"

file_path = join(img_dir, ecII_filename)
### specify camera type (ECII)

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

