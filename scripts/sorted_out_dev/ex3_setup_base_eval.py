# -*- coding: utf-8 -*-
"""
PISCOPE example script XX

Set up plume dataset and cell calib dataset, perform automatic cell detection
and separate cell images from background images. Fit background to cell time
stamps and retrieve calibration polynomial. Set background image in plume data
set (i.e. the image lists) and set plume free reference areas for the background
modelling. 
    
"""
import piscope
from datetime import datetime

### Set path where all images are located
imgPath = "../data/piscope_etna_testdata/images/"

### Setup camera

# The camera is an ECII which has a default implementation
camId = "ecII"
# The camera filter setup is different from the ECII default setup and is
# therefore defined explicitely
filters= [piscope.Utils.Filter(type = "on", acronym = "F01", cWL = 310),
          piscope.Utils.Filter(type = "off", acronym = "F02", cWL = 330)]

# create camera setup, this includes the filename convention for image 
# separation
cam = piscope.Setup.CameraSetup(camId = camId, filterList = filters)

#now throw all this stuff into the BaseSetup object
stp = piscope.Setup.BaseSetup(imgPath, camSetup = cam)

### Calibration time stamps
startCalib = datetime(2015,9,16,6,59,00)
stopCalib = datetime(2015,9,16,7,15,00)
