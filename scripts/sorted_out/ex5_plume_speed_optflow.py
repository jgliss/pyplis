# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 19:55:45 2016

@author: jg
"""

"""
PISCOPE example script

Sript showing how to work with cell calibration data
"""

import piscope 
import pydoas
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.close("all")


### Set image base path
imgPath = "../data/piscope_etna_testdata/images/"


### Set plume background image 
# this is the same image which is also used for example script NO
# demonstrating the plume background routines
pBgImg = imgPath + 'EC2_1106307_1R02_2015091607022602_F01_Etnaxxxxxxxxxxxx.fts'

### Set path to folder containing DOAS result files
doasDataPath = "../data/piscope_etna_testdata/spectra/plume_prep/min10Scans/ResultFiles/"

### Plume data time stamps
start = datetime(2015, 9, 16, 7, 6, 00)
stop  = datetime(2015, 9, 16, 7, 22, 00)

### Specify the camera
# default camera ID
camId = "ecII"

# The camera filter setup is different from the ECII default setup and is
# therefore defined explicitely
filters= [piscope.Utils.Filter(type = "on", acronym = "F01", cWL = 310),
          piscope.Utils.Filter(type = "off", acronym = "F02", cWL = 330)]

### create camera setup, this includes the filename convention for image separation
cam = piscope.Setup.CameraSetup(camId = camId, filterList = filters)

### Create base setup for data import
stp = piscope.Setup.BaseSetup(imgPath, start, stop, camSetup = cam)

### Now load plume data into dataset
plumeData = piscope.Datasets.PlumeData(stp)

### Prepare on band plume image list and 
onList = plumeData.get_list("on")
onList.activate_dark_corr()
onList.add_gaussian_blurring(1)
# load background image and correct for dark and offset
bg = piscope.Img(pBgImg)
bg.subtract_dark_image(onList.get_dark_image())
bg.add_gaussian_blurring(1)

onList.set_bg_image(bg)
onList.bgModel.guess_missing_settings(onList.current_img())
onList.bgModel.corrMode = 6
#onList.bgModel.get_tau_image(onList.current_img(), bg)

onList.activate_tau_mode()
onList.bgModel.plot_tau_result()