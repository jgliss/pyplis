# -*- coding: utf-8 -*-
"""
Introduction script 2 - The Camera class

This script introduces the camera class which is of fundamental importance
for image import (e.g. separating on, off, dark background images, etc.) and
also for the data analysis as it includes all relevant detector specifications, 
such as number of pixels, pixel size, focal length, etc. 

In this script, a newer version of the camera type "ecII" is created manually 
in order to illustrate all relevant parameters. The only difference to the 
classic ecII camera is, that the filter setup is different.
"""

import pyplis
### SCRIPT OPTIONS

# Save the new camera as default in database 
SAVE_TO_DATABASE = False

### SCRIPT FUNCTION DEFINITIONS
def create_ecII_cam_new_filters():
    # Start with creating an empty Camera object                  
    cam = pyplis.setupclasses.Camera()
    
    # Specify the camera filter setup

    # create an on and off band filters, obligatory input is param "type" and 
    # acronym", the former specifies the filter type (choose from "on" or 
    # "off"), the acronym specifies, how to identify this filter in the file 
    # name, if id is unspecified it will be set equal the type. Param 
    # meas_type_acro is only of importance if a meas type (e.g. M -> meas, 
    # C -> calib ...) is explicitely specified in the file names (not the case 
    # for ECII camera but for the HD camera, see specifications in file 
    # cam_info.txt for more info)

    on_band = pyplis.utils.Filter(id="on", type="on", acronym="F01",
                                   meas_type_acro="F01", center_wavelength=310)
    off_band = pyplis.utils.Filter(type="off", acronym="F02",
                                    center_wavelength=330)
    
    # put the two filter into a list and assign to the camera
    filters = [on_band, off_band]
    
    cam.default_filters = filters
    cam.prepare_filter_setup()
    
    # Similar to the filter setup, access info for dark and offset images needs
    # to be specified. The ECII typically records 4 different dark images, two 
    # recorded at shortest exposure time -> offset signal predominant, one at 
    # low and one at high read gain. The other two recorded at longest possible 
    #exposure time -> dark current predominant, also at low and high read gain
    
    offset_low_gain = pyplis.utils.DarkOffsetInfo(id="offset0",type="offset",
                                                   acronym="D0L", read_gain=0)
    offset_high_gain = pyplis.utils.DarkOffsetInfo(id="offset1",type="offset",
                                                   acronym="D0H", read_gain=1)
    dark_low_gain = pyplis.utils.DarkOffsetInfo(id="dark0",type="dark",
                                                  acronym="D1L", read_gain=0)
    dark_high_gain = pyplis.utils.DarkOffsetInfo(id="dark1",type="dark",
                                                  acronym="D1H", read_gain=1)
                                                  
    # put the 4 dark info objects into a list and assign to the camera
    dark_info = [offset_low_gain, offset_high_gain,
                 dark_low_gain, dark_high_gain]
    
    cam.dark_info = dark_info
    
    # Now specify further information about the camera
    
    # camera ID (needs to be unique, i.e. not included in data base, call
    # pyplis.inout.get_all_valid_cam_ids() to check existing IDs)
    cam.cam_id = "ecII_new"

    # image file type
    cam.file_type = "fts"

    # File name delimiter for information extraction
    cam.delim = "_"

    # position of acquisition time (and date) string in file name after 
    # splitting with delimiter
    cam.time_info_pos = 3

    # datetime string conversion of acq. time string in file name
    cam.time_info_str = "%Y%m%d%H%M%S%f"

    # position of image filter type acronym in filename 
    cam.filter_id_pos = 4
    
    # position of meas type info
    cam.meas_type_pos = 4

    # Define which dark correction type to use
    # 1: determine a dark image based on image exposure time using a dark img
    # (with long exposure -> dark current predominant) and a dark image with
    # shortest possible exposure (-> detector offset predominant). For more 
    # info see function model_dark_image in processing.py module
    # 2: subtraction of a dark image recorded at same exposure time than the 
    # actual image
    cam.DARK_CORR_OPT = 1
    
    # If the file name also includes the exposure time, this can be specified 
    # here:
    cam.texp_pos = "" #the ECII does not...

    # the unit of the exposure time (choose from "s" or "ms")
    cam.texp_unit = ""
    
    # define the main filter of the camera (this is only important for cameras
    # which include, e.g. several on band filters.). The ID need to be one of
    # the filter IDs specified above
    cam.main_filter_id = "on"
    
    # camera focal length can be specified here (but does not need to be, in 
    # case of the ECII cam, there is no "default" focal length, so this is left
    # empty)
    cam.focal_length = ""
    
    # Detector geometry
    cam.pix_height = 4.65e-6 # pixel height in m
    cam.pix_width = 4.65e-6 # pixel width in m
    cam.pixnum_x = 1344
    cam.pixnum_y = 1024
    
    cam._init_access_substring_info()
    
    # That's it... 
    return cam

### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    
    cam = create_ecII_cam_new_filters()
    
    print cam
    
    if SAVE_TO_DATABASE:
        # you can add the cam to the database (raises error if ID 
        # conflict occurs, e.g. if the camera was already added to the database)
        cam.save_as_default()
        
    
