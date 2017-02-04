# -*- coding: utf-8 -*-
"""
piscope example script no. 1

This script illustrates how to specify a measurement setup and create a 
dataset from that. 
"""
import piscope as piscope
from datetime import datetime
from matplotlib.pyplot import show


### IMPORT GLOBAL SETTINGS
from SETTINGS import IMG_DIR, OPTPARSE

### SCRIPT FUNCTION DEFINITIONS
def create_dataset():
    """Initiates measurement setup and creates dataset from that"""
    start = datetime(2015, 9, 16, 7, 6, 00)
    stop = datetime(2015, 9, 16, 7, 22, 00)
    ### Define camera (here the default ecII type is used)
    cam_id = "ecII"
    
    #the camera filter setup
    filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
              piscope.utils.Filter(type = "off", acronym = "F02")]
    
    #camera location and viewing direction (altitude will be retrieved automatically)                    
    geom_cam = {"lon"           :   15.1129,
                "lat"           :   37.73122,
                "elev"          :   20.0,
                "elev_err"      :   5.0,
                "azim"          :   270.0,
                "azim_err"      :   10.0,
                "alt_offset"    :   15.0,
                "focal_length"  :   25e-3} #altitude offset (above topography)
    
    #Create camera setup
    #the camera setup includes information about the filename convention in
    #order to identify different image types (e.g. on band, off band, dark..)
    #it furthermore includes information about the detector specifics (e.g.
    #dimension, pixel size, focal length). Measurement specific parameters
    #(e.g. lon, lat, elev, azim) where defined in the dictinary above and 
    #can be passed as additional keyword dictionary using **geom_cam 
    #Alternatively, they could also be passed directly, e.g.:
    
    #cam = piscope.setup.Camera(cam_id, filter_list=filters, lon=15.1129,
    #                           lat=37.73122)

    cam = piscope.setupclasses.Camera(cam_id, filter_list=filters, **geom_cam)
    
    ### Load default information for Etna
    source = piscope.setupclasses.Source("etna") 
    
    #### Provide wind direction
    wind_info= {"dir"     : 0.0,
                "dir_err"  : 15.0}


    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setupclasses.MeasSetup(IMG_DIR, start, stop, camera=cam,
                                         source=source, wind_info=wind_info)
    
    ### Create analysis object (from BaseSetup)
    # The dataset takes care of finding all vali
    return piscope.dataset.Dataset(stp)


### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    ds = create_dataset()
    img = ds.get_list("on").current_img()
    print str(img) #the image object has an informative string representation
    
    
    ### IMPORTANT STUFF FINISHED
    
    # Display images or not (nothing to understand here...)
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            img.show()
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"
    
    
