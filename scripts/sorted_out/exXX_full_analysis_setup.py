# -*- coding: utf-8 -*-
"""
PYGASCAM SETUP MASK 

Type: Basic setup (without spectral calibration)
    
version: 0.1.0.dev1

This script can be used as mask to create your own evaluation setup for camera
data. The setup will be saved as binary setup file in ../data/TEMP/ folder.

In the unchanged version, the script is intended to create the evaluation setup
for the test data from Guallatiri volcano, Chile, 11/2014, which is provided
<<<INSERT LINK TO BITBUCKET>>> 

Data usage information: <<<some legally correct info, that the data is only 
allowed to be used for test purposes of the :mod:`piscope` library and not 
to be published etc...>>>
"""

import piscope as piscope
from datetime import datetime
from bunch import Bunch

onLinux=0

"""Define important paths

.. note::

    Download test dataset from webpage and save it on your local machine, then 
    insert this path as datapath here. For creating the setup file it is not 
    important that this path actually contains imagery information, but for 
    ex2 (which uses the output of this script) it is.
    
"""
datapath='../data/guallatiri_ecII_test_dataset/' #the path with the test images
savepath='../data/TEMP/' #the path were stuff is saved

"""
SPECIFYING THE CAMERA 
---------------------

Main library object: :class:`piscope.Setup.CameraSetup`
"""
camId="ecII"
camSerNo=1106307
#the camera filter setup
filterSetup     =   {"On"  :   piscope.Utils.Filter("On","F01",310),
                     "Off" :   piscope.Utils.Filter("Off","F03",330)}

#important geometrical and geo-coordinate info                      
opticsCam ={"focalLength"    :   50.0}
geomCam = {"lon"     :   -69.213911,
          "lat"     :   -18.444883,
          "altitude":   4243.0,
          "elev"    :   8.0,
          "elevErr" :   1.0,
          "azim"    :   84.0,
          "azimErr" :   3.0}

#Now put this stuff into the camera setup (which will afterwards be filled with 
#some more specific information related to the measurement)                   
camSetup=piscope.Setup.CameraSetup(camId,camSerNo,filterSetup,"On", geomCam, opticsCam)

"""
DEFINING THE EMISSION SOURCE
----------------------------

Main library object: :class:`piscope.Setup.CameraSetup`
"""
sourceName="Guallatiri"
guallatiriInfo = {"lon"     : -69.090369,
                  "lat"     : -18.423672,
                  "altitude": 6071.0}

#put this stuff in the corresponding source class                  
sourceSetup=piscope.Setup.GasEmitter(sourceName, guallatiriInfo)

#Provide a rough estimate of the wind velocity (this will actually be updated
#during the evaluation using optical flow) and an as accurate as possible
#estimate of the direction (because there is no tool yet which provides a
#reasonably good enough guess of the direction at the given time, location and
#altitude).
"""
PROVIDE SOME WIND METEOROLOGY INFO
----------------------------------

For this, right now, a specific class does not exists, this info is stored
in the :class:`piscope.Setup.BaseSetup`object which creates the 
:class:`piscope.Utils.MeasGeometry` used to determine plume distances
etc. for the measurement setup.
"""
windDefaultInfo= {"dir"     : 320,
                  "dirErr"  : 15.0,
                  "vel"     : 4.43,
                  "velErr"  : 1.0}

"""
NOW DEFINE TIME STAMPS AND PUT IT ALL TOGETHER IN BASESETUP
-----------------------------------------------------------

Main library object: :class:`piscope.Setup.BaseSetup`

The provided test data includes 3 different types of data with different time
stamps

    1. 12:19 - 12:21 (Plume image data)
    2. 12:23 (4 dark images)
    3. 15:25 - 15:28 (cell calibration and background images)
    
On and offband images were taken consecutively in this exemplary dataset.

.. note::

    The main object for the whole data evaluation is the 
    :class:`piscope.DataSets.DataSet` and it is normally initiated with the
    :class:`piscope.Setup.BaseSetup` object we're about to create. The
    :class:`piscope.DataSets.DataSet` object is mainly defined by the image 
    base path (specified at beginning of this script) and a start and 
    stop time (specified below), further the camera used (defined above, 
    specifying details about file reading, how to distinguish different image
    types - e.g. onband, offband, dark... -) and basic geometry information (
    also specified above). When initialised, the :class:`piscope.DataSets.DataSet`
    object checks if dark and offset images are available for the specified time
    span (which will be defined in a second) and if no dark (offset) images
    available in this time period, it searches the closest set of dark / offset 
    images considering all files in the base folder. In general, how to deal with dark
    and offset images and how to perform the dark correction needs to be specified 
    for individual cameras (:class:`piscope.Setup.CameraSetup`) and is
    defined in the subclass :class:`piscope.Utils.CameraBaseInfo`. In 
    the current version, two camera standards are included in the library:
    
        1. ECII camera (NILU, Norway)
        2. HDcam (Camera from Heidelberg group)
        
    Feel free to contact the author (jg@nilu.no) if you need help including your
    own file / CameraConvention and if not, look at the files provided and 
    the :class:`piscope.Utils.CameraBaseInfo` class (for "ecII" standard)
    and write your own default. 
    
    One more remark:
    If the camera is unspecified (e.g. if a :class:`piscope.DataSets.DataSet`
    object is created without input) then the image filetype is unspecified so, 
    all images in the basePath, which can be read as image will be loaded. 
    Furthermore, dark image correction will not be possible, for details regarding
    initiation of image lists in :class:`piscope.DataSets.DataSet` have a look
    at the :class:`piscope.ImgLists.SortedList` object, which is creating all
    relevant :class:`piscope.ImgLists.MyImgListStatic` and 
    :class:`piscope.ImgLists.MyImgList` objects. The latter objects are central
    for anything you might do and are also taking care of the dark image subtraction
    (which will be deactivated, when the camera type is unspecified)
"""
#start and stop time of measurement data
start=datetime(2014,11,22,12,18,00)
stop=datetime(2014,11,22,12,22,00)

#now we throw all this stuff into the BaseSetup object (which will also be 
#updated later with some more information)
baseSetup=piscope.Setup.BaseSetup(datapath, start, stop, camSetup,\
                                                sourceSetup, windDefaultInfo)

baseSetup.set_save_path(savepath)
#: create :class:`piscope.Utils.MeasGeometry` object
baseSetup.create_meas_geometry()
"""
DEFINE IMPORTANT FORMS FOR DATA ANALYSIS
----------------------------------------

1.  Rectangles: Two rectangles (scale, ygradient) are defined here and they are used
    to model the background images for a given plume image based on the assumption
    that the image area defined by these rectangles is gas free. Normally, the 
    background needs to be modelled (and especially in the case of the test dataset
    provided here) since the background images are normally taken at a different 
    time and viewing direction of the camera (blue sky images) which then have 
    different radiances. Central object for background image modelling is 
    :class:`piscope.Processing.BackgroundModel` which has different modi::
        
            settings=Bunch({"scale"    :   1,
                            "ygradient":   0,
                            "polyMode" :   0})
    
    i.      "scale": scale background using "scale" rectangle.
            The image area defined by the "scale" rectangle is used to scale the 
            raw background image to the actual plume image (such that the difference 
            in intensity mean in this rectangle is 0)
            
    ii.     Performe "ygradient" correction to "scaled" bg image using
            "ygradient" rectangle.
            The "ygradient" rectangle can be used if i. was performed to correct
            for differences in the vertical gradient of the background radiance by
            comparing measured intensities in the ygradient rectangles after scaling
            was applied. Therefore it is important to keep a certain distance 
            in pixel row between the two different rectangles
            
    iii.    Fit a polynomial to radiance time series in plume images to correct
            for short term disturbances (e.g. gas entering the rect or a cloud
            is moving thorugh) 
            "polyMode" is a great tool and should be used, but more needs
            to be written on that.. and this happens later ;)
            
            .. todo::
            
                Write more about background polynomial mode
            
2.  icalines:  Lines used for flux analysis
"""
rects=Bunch()
rects.scale=[[4,20],[44,60]] #format [[x0,y0],[x1,y1]]
rects.ygradient=[[4,550],[44,590]]
#rects.ygradient=[[4,688],[44,728]]

pcslines=Bunch()
pcslines.plume=[[278,507],[368,623]]
#icalines.rest=[[628,780],[977,1013]]

#Some information about the measurement geometry
formsSetup=piscope.Setup.FormsSetup(rectDict=rects, pcsLineDict=pcslines)
#write the forms information into the baseSetup
baseSetup.set_forms(formsSetup)

"""
The base setup for the data evaluation is now completed, this object could 
be used as input for a :class:`DataSet` object like::

    ds=piscope.DataSets.DataSet(baseSetup)
    
and could for instance be viewed using::

    ds.open_in_gui()
    
to get a feeling for the data and/or to see whether the rectangles and lines 
(defined above) are drawn at the right position or for instance 
to optimise the optical flow input parameters for these data.

"""

"""

In the following, everything related to Cell calibration and background image
information will be set up. In this case, the imagery base path is the same
as for the above volcanic data and is therefore also used to create
the cell calibration and background :class:`DataSet` objects.

.. note::

    Actually in the default scheme it is assumed that the cell calibration was
    performed in a gas-free and cloudless region. If this holds, the data defined
    by the time interval for cell calibration also includes background images
    which will be sorted out automatically as will be the setup of the 
    corresponding :class:`DataSetBackground` object. Therefore, there are no
    further definitions for background imagery data necessary in this script.
"""
#define everything related to cell calibration (and background images)
startCalib=datetime(2014,11,22,15,24,00)
stopCalib=datetime(2014,11,22,15,29,00)

calibCells= {'a41'  :   [1.06e18,1.45e17],
             'a53'  :   [4.85e17,7.06e16],
             'a57'  :   [1.81e18,2.62e17]}
#==============================================================================
# calibCells= {'a41'  :   [9.66e17,1.45e17],
#              'a53'  :   [4.47e17,7.06e16],
#              'a57'  :   [1.62e18,2.62e17]}
#==============================================================================
#And in analogy to the baseSetup, create a cellCalibSetup object (the camera
#is the same)
cellCalibSetup=piscope.Setup.CellCalibSetup(calibCells,datapath,startCalib,\
                                               stopCalib, camSetup)             
 

"""
Now put all the stuff together in the :class:`EvalSetup` class
"""
evalSetup=piscope.Setup.EvalSetup(baseSetup) 
evalSetup.set_cellcalib_setup(cellCalibSetup)
evalSetup.create_folder_structure()
evalSetup.save()

if __name__ == "__main__":
    print "Blaaa"
#==============================================================================
#     ds=piscope.Datasets.Dataset(baseSetup)
#     ds1=piscope.Calibration.CellCalib(cellCalibSetup)
#     ds1.find_and_assign_cells_all_filter_lists()
#     ds2=piscope.Datasets.BackgroundData()
#     ds2.get_bgfiles_from_calib_data_set(ds1)   
#==============================================================================
    eval=piscope.Evaluation.Evaluation(evalSetup)
    #eveval.prepare_cell_calib_and_background()
    eval.init_bg_models()
    #eval.open_in_gui()

