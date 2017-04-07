# -*- coding: utf-8 -*-
"""
pyplis example script no. 1 - Analysis setup for example data set

In this script an example data set, recorded on the 16/9/15 7:06-7:22 at 
Mt. Etna is setup. Most of the following example scripts will work on this data 
set. 

A typical analysis setup for plume image data contains information about the
camera (e.g. optics, file naming convention, see also ex0_2_camera_setup.py), 
gas source, the measurement geometry and the wind conditions. These information 
is needs to be generally provided by the user before the analysis. The 
information is stored within a MeasSetup object which can be used as a basis
for further analysis.  
If not all neccessary information is entered, a MeasSetup object will be 
created nonetheless but analysis options might be limited.

Such a MeasSetup object can be used as input for Dataset objects which creates
the analysis environment (i.e. separating files by image type). 

This script shows how to setup a MeasSetup object and create a Dataset object
from it. As example, the first image of the on-band image time series is 
displayed.

The Dataset object created here is also used in script 
ex04_prep_aa_imglist.py which shows how to create an image list displaying
AA images.

"""
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

import pyplis as pyplis
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
    filters= [pyplis.utils.Filter(type="on", acronym="F01"),
              pyplis.utils.Filter(type="off", acronym="F02")]
    
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
    
    #cam = pyplis.setup.Camera(cam_id, filter_list=filters, lon=15.1129,
    #                           lat=37.73122)

    cam = pyplis.setupclasses.Camera(cam_id, filter_list=filters, **geom_cam)
    
    ### Load default information for Etna
    source = pyplis.setupclasses.Source("etna")
    
    #### Provide wind direction
    wind_info= {"dir"     : 0.0,
                "dir_err"  : 1.0}
                
#                "dir_err"  : 15.0}


    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = pyplis.setupclasses.MeasSetup(IMG_DIR, start, stop, camera=cam,
                                         source=source, wind_info=wind_info)
    
    ### Create analysis object (from BaseSetup)
    # The dataset takes care of finding all vali
    return pyplis.dataset.Dataset(stp)


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
    
    
