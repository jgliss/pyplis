# -*- coding: utf-8 -*-
"""
PISCOPE example script

Sript showing how to work with cell calibration data
"""

import piscope 
import pydoas
from datetime import datetime, timedelta
from matplotlib.pyplot import close
from os.path import join, exists
from os import getcwd, remove

close("all")
my_dat = r'D:/repos/tests'
stack_path = join(my_dat,\
        "piscope_imgstack_id_aa_20150916_0706_0721.fts")


reload_stack = 0
if not exists(stack_path):
    reload_stack =1
#==============================================================================
# if not exists(stack_path):
#     raise IOError("Stack obj not found")
#==============================================================================

    
### Set save directory for figures
save_path = join(getcwd(), "..", "scripts_out")

# test data path
test_data_path = piscope.inout.find_test_data()

# Image base path
img_dir = join(test_data_path, "images")

# Path containing DOAS result files
doas_data_path = join(test_data_path, "spectra", "plume_prep", "min10Scans",\
    "ResultFiles")

### Set plume background image 
# this is the same image which is also used for example script NO
# demonstrating the plume background routines
path_bg_on = join(img_dir, 
                  'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')

path_bg_off = join(img_dir, 
                  'EC2_1106307_1R02_2015091607022820_F02_Etna.fts')


### Plume data time stamps
start = datetime(2015, 9, 16, 7, 6, 00)
#stop  = datetime(2015, 9, 16, 7, 7, 00)
stop  = datetime(2015, 9, 16, 7, 22, 00)
### Set image base path

#

### Specify the camera
# default camera ID
cam_id = "ecII"

# The camera filter setup is different from the ECII default setup and is
# therefore defined explicitely
filters= [piscope.utils.Filter(type = "on", acronym = "F01"),
          piscope.utils.Filter(type = "off", acronym = "F02")]

### create camera setup, this includes the filename convention for image separation
cam = piscope.setup.Camera(cam_id = cam_id, filter_list = filters)

### Create base setup for data import
stp = piscope.setup.MeasSetup(img_dir, start, stop, camera = cam)

### Now load plume data into dataset
plume_data = piscope.dataset.Dataset(stp)

### Specify DOAS data import from DOASIS fit result files
# In order to perform the DOAS FOV search, as much spectrum datapoints 
# as possible are needed. Therefore, we only added 10 scans per plume
# spectrum. In this case (and because the spectrometer was not temperature 
# stabilised, the DOAS fit in the recommended wavelength range (~ 314 - 326, 
# here "f01") might not exceed the S/N ratio for low SO2 CDs. Thus, to have a 
# quality check of the fit performance, SO2 was therefore also fitted in a lower 
# wavelength region ("f02" : ~ 309 - 323 nm), both datasets are imported here
# and are plotted against each other below, showing that the correlation is 
# good (i.e. f01 is trustworthy) and that the SO2 CDs are too small

fit_import_info = {"so2" : ["SO2_Hermans_298_air_conv_satCorr1e18", 
                                                        ["f01"]]}

doas_import_setup = pydoas.dataimport.ResultImportSetup(doas_data_path,\
                                    result_import_dict = fit_import_info)
### Import the DOAS fit results
doas_dataset = pydoas.analysis.DatasetDoasResults(doas_import_setup)

### get fit results from standard so2 fit (f01)
doas_res_std = doas_dataset.get_results("so2", "f01").shift(timedelta(-1./12))

if reload_stack:
    ### Prepare on band plume image list and 
    on_list = plume_data.get_list("on")
    on_list.activate_dark_corr()
    
    # load background image and correct for dark and offset
    bg_on = piscope.Img(path_bg_on)
    bg_on.subtract_dark_image(on_list.get_dark_image())
    
    on_list.set_bg_image(bg_on)
    on_list.bg_model.guess_missing_settings(on_list.current_img())
    on_list.bg_model.CORR_MODE = 6
    
    off_list = plume_data.get_list("off")
    bg_off = piscope.Img(path_bg_off)
    bg_off.subtract_dark_image(off_list.get_dark_image())
    off_list.set_bg_image(bg_off)
    off_list.activate_dark_corr()
    
    on_list.pyrlevel = 2
    on_list.aa_mode = True
        
    stack = on_list.make_stack()
    try:
        remove(stack_path)
    except:
        pass    
    stack.save_as_fits(save_dir=my_dat)  
else:
    stack = piscope.processing.ImgStack()
    stack.load_stack_fits(stack_path)
    #stack, hdu = load_stack_fits(stack, stack_path)
s = piscope.doascalib.DoasFOVEngine(stack, doas_res_std, pearson_max_radius = 10)
calib1 = s.perform_fov_search(method = "pearson")
#fov2 = s.perform_fov_search(method = "ifr", ifr_lambda = 8e-2)
#fov2 = s.perform_fov_search(method = "ifr", ifr_lambda = 1e-4)
calib2= s.perform_fov_search(method = "ifr", ifr_lambda = 2e-3)

poly1 = calib1.fit_calib_polynomial()
ax1 = calib1.plot()
poly2, ax2 = calib2.fit_calib_polynomial()
ax2 = calib2.plot()
ax3 = calib1.fov.plot()
ax4 = calib2.fov.plot()
