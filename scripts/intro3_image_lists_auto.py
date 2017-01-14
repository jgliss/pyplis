# -*- coding: utf-8 -*-
"""
piscope example script 3 - Automatic creation of image lists
    
Steps:
    
    1. Sets image base path and initiate working environment
    #. Open GUI
    
"""
import piscope
from os.path import join
from os import getcwd

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

camId = "ecII"

#the camera filter setup 
filters = [piscope.utils.Filter(type = "on", acronym = "F01"),
           piscope.utils.Filter(type = "off", acronym = "F02")]

#Now put this stuff into the camera setup (which will afterwards be filled with 
#some more specific information related to the measurement)                   
cam = piscope.setup.Camera(cam_id = camId, filter_list = filters)

#now throw all this stuff into the BaseSetup object
stp = piscope.setup.MeasSetup(img_dir, camera = cam)

# Create a Dataset which creates separate ImgLists for all types (dark,
# offset, etc.)
ds = piscope.dataset.Dataset(stp)

l = ds.get_list("on")
l.roi = [100, 100, 1300, 900]
l.dark_corr_mode = 1
l.crop = 1
l.pyrlevel = 2
l.add_gaussian_blurring(1)
l.goto_img(100)
l.show_current()

