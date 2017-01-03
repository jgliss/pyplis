# -*- coding: utf-8 -*-
"""
PISCOPE example script

Automatic cell from cell calibration time window

User todo's:

    1. Set ``img_dir`` to a local folder which contains image files
    
Steps:
    
    1. Sets image base path and initiate working environment
    #. Open GUI
    
"""
import piscope

img_dir = "../test_data/piscope_etna_testdata/images/"

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

