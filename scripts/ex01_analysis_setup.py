# -*- coding: utf-8 -*-
"""
pyplis example script no. 1 - Analysis setup for example data set

In this script an example data set, recorded on the 16/9/15 7:06-7:22 at 
Mt. Etna is setup. Most of the following example scripts will use the information 
specified here.

A typical analysis setup for plume image data contains information about the
camera (e.g. optics, file naming convention, see also ex0_2_camera_setup.py), 
gas source, the measurement geometry and the wind conditions. These information 
generally needs to be specified by the user before the analysis. The 
information is stored within a MeasSetup object which can be used as a basis
for further analysis.  
If not all neccessary information is provided, a MeasSetup object will be 
created nonetheless but analysis options might be limited.

Such a MeasSetup object can be used as input for Dataset objects which creates
the analysis environment. The main purpose of Dataset classes is to automatically 
separate images by their type (e.g. on / off-band images, dark, offset) and create ImgList
classes from that. ImgList classes typically contain images of one type. The Dataset 
also links relevant ImgList to each other, e.g. if the camera has an off-band filter, and 
off-band images can be found in the specified directory, then it is linked to the list
containing on-band images. This means, that the current image index in the off-band list
is automatically updated whenever the index is changed in the on-band list. If acquisition
time is available in the images, then the index is updated based on the closest acq. time
of the off-band images, based on the current on-band acq. time. If not, it is updated by 
index (e.g. on-band contains 100 images and off-band list 50. The current image number in
the on-band list is 50, then the off-band index is set to 25). Also, dark and offset images 
lists are linked both to on and off-band image lists, such that the image can be correccted
for dark and offset automatically on image load (the latter needs to be activated in the 
lists using ``darkcorr_mode=True``).

This script shows how to setup a MeasSetup object and create a Dataset object
from it. As example, the first image of the on-band image time series is 
displayed.

The Dataset object created here is used in script  ex04_prep_aa_imglist.py which shows 
how to create an image list displaying AA images.
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
    
    #Set pixel intensities below 2000 to 0 (method of Img class)
    img.set_val_below_thresh(val=0, threshold=2000)
    #show modified image
    img.show()
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
    
    
