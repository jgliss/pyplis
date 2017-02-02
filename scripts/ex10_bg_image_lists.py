# -*- coding: utf-8 -*-
"""
piscope example script no. 10 - background image dataset
"""

import piscope
from os.path import join
from datetime import datetime

### RELEVANT DIRECTORIES AND PATHS

# Image directory
IMG_DIR = join(piscope.inout.find_test_data(), "images")

### SCRIPT FUNCTION DEFINITIONS
def get_bg_image_lists():
    """Initiates measurement setup and creates dataset from that"""
    start = datetime(2015, 9, 16, 7, 2, 05)
    stop = datetime(2015, 9, 16, 7, 2, 30)
    ### Define camera (here the default ecII type is used)
    cam_id = "ecII"
    
    #the camera filter setup
    filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
              piscope.utils.Filter(type = "off", acronym = "F02")]
    
    #create camera setup
    cam = piscope.setupclasses.Camera(cam_id=cam_id, filter_list=filters)

    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setupclasses.MeasSetup(IMG_DIR, start, stop, camera=cam)
    
    ds = piscope.dataset.Dataset(stp)
    on, off = ds.get_list("on"), ds.get_list("off")
    on.darkcorr_mode = True
    off.darkcorr_mode = True
    return on, off

### SCRIPT MAIN FUNCTION   
if __name__ == "__main__":
    on, off = get_bg_image_lists()
    on.show_current()
    off.show_current()
    