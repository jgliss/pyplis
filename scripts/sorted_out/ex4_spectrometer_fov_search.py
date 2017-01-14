# -*- coding: utf-8 -*-
"""
PYGASCAM SETUP MASK 

Type: Basic setup (without spectral calibration)

v0.1

This script can be used as mask to create your own evaluation setup for camera
data. 

In the unchanged version, the script is intended to create the evaluation setup
for the test data from Guallatiri volcano, Chile, 11/2014, which is provided
<<<INSERT LINK TO BITBUCKET>>> 

Data usage information: <<<some legally correct info, that the data is only 
allowed to be used for test purposes of the :mod:`pygascam` library and not 
to be published etc...>>>
"""

import piscope
from datetime import datetime

onLinux=0

"""Define the base path (all paths below are relative to this)
"""
if not onLinux:
    base=r'D:/'
else:
    base='/media/jg/Data/'

"""A guideline for setting up the evaluation of a SO2 camera dataset
"""


"""
DEFINING THE CAMERASETUP
------------------------

STARTING WITH DEFINING THE CAMERA TYPE AND THE FILTER SETUP 
"""
camId="ecII"
camSerNo=1106307
#the camera filter setup
filterSetup     =  [piscope.Utils.Filter("on","F01",310),piscope.Utils.Filter("off","F03",330)]

#important geometrical and geo-coordinate info                      
opticsCam={"focalLength"    :   50.0}
geomCam= {"lon"     :   -69.213911,
          "lat"     :   -18.444883,
          "altitude":   4243.0,
          "elev"    :   8.99,
          "elevErr" :   0.15,
          "azim"    :   82.61,
          "azimErr" :   1.0}

camSetup=piscope.Setup.CameraSetup(camId,camSerNo,filterSetup,"On", geomCam, opticsCam)

"""
DEFINING THE BASESETUP
----------------------

NOW WE PROVIDE INFORMATION ABOUT THE SOURCE, THE METEOROLOGY 
AND THE MEASUREMENT GEOMETRY
"""
sourceName="Guallatiri"

#put this stuff in the corresponding source class                  
sourceSetup=piscope.Setup.GasEmitter(sourceName)

"""
NOW DEFINE TIME STAMP OF PLUME DATA AND THE IMAGERY BASEPATH
------------------------------------------------------------

Further define a saveBasePath which is used to store results
"""
datapath=base + 'Jonas/Research/Data/Campaigns/201411_Chile/DATA_RAW_CURRENT_WRK/ECII_1106307/images/326/'
savepath=base + 'Jonas/Research/Data/Campaigns/201411_Chile/DATA_ANALYSIS/Guallatiri/AnalysisResults/cameras/ec2_1106307/Analysis/'
#start and stop time of measurement data
start=datetime(2014,11,22,14,47,00)
stop=datetime(2014,11,22,14,58,54)

#now we throw all this stuff into the BaseSetup object (which will also be 
#updated later with some more information)
baseSetup=piscope.Setup.BaseSetup(datapath, start, stop, camSetup, sourceSetup)

baseSetup.set_save_path(savepath)
#: create :class:`piscope.Utils.MeasGeometry` object
baseSetup.create_meas_geometry()

"""
Define lines and rectangles used for the data evaluation and put them into
the FormsSetup object
"""
rects = {}
rects["scale"]=[[4,20],[44,60]]
rects["ygradient"]=[[12,690],[32,720]]
#rects.ygradient=[[1250,930],[1290,970]]
#rects.ygradient=[[4,688],[44,728]]
formsSetup=piscope.Setup.FormsSetup(rectDict = rects)
#write the forms information into the baseSetup
baseSetup.set_forms(formsSetup)

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
SPECTRAL CALIBRATION AND LOADING SPECTRAL RESULTS

Setup everything related to load spectral results (need to be saved according
to DOASIS resultfile standars, for details see :mod:`SpectralAnalyis`)
"""
pSpecsAvantes=(base + 'Jonas/Research/Data/Campaigns/201411_Chile/DATA_ANALYSIS/'
    'Guallatiri/Avantes_DOAS/min20Scans/Eval20151208_141121_141122/ResultFiles/')


#Define the import of spectral results
#what fit results should be imported
fitImportDict={'so2':['SO2_Hermans_298_air_conv', ['f02','fx02']],
               'o3' :['O3_Serdyuchenko(2014)_223K',['f03','fx03']]}

#Spectral results setup for Avantes data
avantesRes = piscope.Setup.DoasResultSetup(pSpecsAvantes,start,stop,\
    resultImportDict = fitImportDict, resType="doasis", deviceId="avantes")

"""
Now put all the stuff together in the :class:`SO2CamEvalSetup` class
"""
evalSetup = piscope.Setup.EvalSetup(baseSetup) 
evalSetup.set_cellcalib_setup(cellCalibSetup)
evalSetup.add_doas_results_setup(avantesRes)
reload = 1
if reload:
    ds = piscope.Evaluation.Analysis(evalSetup)
    ds.update_img_prep_settings(darkcorr = 1,blurring = 2)
    ds.prepare_cell_calib_and_background()
    ds.init_bg_models()
    ds.load_doas_results("avantes")
    
    l = ds.dataSet.get_list("on")
    ds.bgModels.on.settings["ygradient"] = 1
    so2 = ds.doasResults.avantes.get_results("so2")
    s = piscope.DoasFov.SearchFOVSpectrometer(so2, l)
    res_i, res_f = s.perform_fov_search(1,4)
    res_i1, res_f1 = s.perform_fov_search(1,4, mergeType = "interpolation")

#==============================================================================
# p=r'D:/Dropbox/Python27/jgliss/modules/piscope/_private/out/20160830_hsihler_fovSearchData/'
# 
# res_i.specData.values.dump(p + "specDataVector")
# res_i.stack.stack.dump(p + "tauImgStack")
# res_i.stack.times.dump(p + "acqTimes")
# res_i.corrIm.dump(p + "corrIm")
#==============================================================================
    
#==============================================================================
#     s.make_img_stack(1,4)
#     xPos0, yPos0, corrIm0, specData0 = s.find_viewing_direction()
#     res0 = s.find_fov_radius(s.stack, xPos0, yPos0, specData0, 1)
#     
#     xPos1, yPos1, corrIm1, specData1 = s.find_viewing_direction("binning")
#     res1 = s.find_fov_radius(s.stack, xPos1, yPos1, specData1)
#==============================================================================
import matplotlib.pyplot as plt
plt.close("all")
fig, axes = plt.subplots(2,1)
im0 = axes[0].imshow(corrIm0)
tit0="Method: intepolation, num=%s, pos max corr x=%s,y=%s" %(len(specData0),xPos0, yPos0)
axes[0].set_title(tit0)
fig.colorbar(im0, ax = axes[0])
im1 = axes[1].imshow(corrIm1)
tit1="Method: binning, num=%s, pos max corr x=%s,y=%s" %(len(specData1),xPos1,yPos1)
axes[1].set_title(tit1)
fig.colorbar(im1, ax = axes[1])
#==============================================================================
#     
#     l = ds.dataSet.get_list("On")
#     m = l.get_mean_value()
#==============================================================================
#==============================================================================
# 
# so2 = ds.doasResults.avantes.get_results("so2")*10**(-17)
# df = pd.concat([pd.Series(m.values, m.times), so2], axis = 1).interpolate("cubic").dropna()
#     
#==============================================================================
    
    #ds.open_in_gui()
