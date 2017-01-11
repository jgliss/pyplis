# -*- coding: utf-8 -*-
"""
piscope example script no. 1

This script illustrates how to specify a measurement setup and create a 
dataset from that. 
"""
import piscope as piscope
from datetime import datetime

from os.path import join
from os import getcwd

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

def create_dataset():
    """Initiates measurement setup and creates dataset from that"""
    #time stamps of plume image data
    start = datetime(2015,9,16,7,6,00)
    stop = datetime(2015,9,16,7,22,00)
    
    ### Define camera (here the default ecII type is used)
    cam_id = "ecII"
    
    #the camera filter setup
    filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
              piscope.utils.Filter(type = "off", acronym = "F02")]
    
    #camera location and viewing direction (altitude will be retrieved automatically)                    
    geom_cam = {"lon"           :   15.1129,
                "lat"           :   37.73122,
                "elev"          :   15.0,
                "elevErr"       :   5.0,
                "azim"          :   274.0,
                "azimErr"       :   10.0,
                "alt_offset"    :   15.0} #altitude offset (above topography)
    
    #create camera setup
    cam = piscope.setup.Camera(cam_id = cam_id, geom_data = geom_cam,\
                filter_list = filters, focal_length = 25.0)
    
    ### Load default information for Etna
    source = piscope.setup.Source("etna") 
    
    #### Provide wind direction
    wind_info= {"dir"     : 0.0,
                "dir_err"  : 15.0}


    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setup.MeasSetup(img_dir, start, stop, camera=cam,\
                        source = source, wind_info = wind_info)
    
    ### Create analysis object (from BaseSetup)
    return piscope.dataset.Dataset(stp)


    
if __name__ == "__main__":
    ds = create_dataset()
    ds.get_list("on").show_current()
    
    