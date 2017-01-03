# -*- coding: utf-8 -*-
"""
PISCOPE example script

Sript showing how to work with cell calibration data
"""

import piscope 
import numpy as np
import pydoas
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from os.path import join
from os import getcwd

reload_data = 1
### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

img_path = "../../data/piscope_etna_testdata/images/"

### Set plume background image 
# this is the same image which is also used for example script NO
# demonstrating the plume background routines
path_bg_on = join(img_path, 
                  'EC2_1106307_1R02_2015091607022602_F01_Etnaxxxxxxxxxxxx.fts')

path_bg_off = join(img_path, 
                  'EC2_1106307_1R02_2015091607022820_F02_Etnaxxxxxxxxxxxx.fts')

### Set path to folder containing DOAS result files
doas_data_path = ("../../data/piscope_etna_testdata/spectra/"
                    "plume_prep/min10Scans/ResultFiles/")

### Plume data time stamps
start = datetime(2015, 9, 16, 7, 6, 00)
stop  = datetime(2015, 9, 16, 7, 14, 00)
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

if reload_data:

    ### create camera setup, this includes the filename convention for image separation
    cam = piscope.setup.Camera(cam_id = cam_id, filter_list = filters)
    
    ### Create base setup for data import
    stp = piscope.setup.MeasSetup(img_path, start, stop, camera = cam)
    
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
    
    stack = on_list.make_stack(stack_id = "aa_stack")
    
    s= piscope.doasfov.DoasFOVEngine(stack, doas_res_std)
    #fov_0 = s.perform_fov_search(method = "pearson")
    fov_1 = s.perform_fov_search(method = "ifr", ifr_lambda = 2e-3)
#==============================================================================
#     s.merge_data()
#     corr_img_p = s.det_correlation_image()
#     res = s.get_fov_shape()
#     
#     corr_img_ifr = s.det_correlation_image(search_type="ifr", ifr_lambda = 5e-2)
#     
#     plt.imshow(corr_img_p)
#     plt.show()
#     plt.imshow(corr_img_ifr)
#     plt.show()
#     
#==============================================================================
    


#==============================================================================
# 
# hdu = fits.open(join(save_dir, 'test.fts'))
# h = hdu[0].header 
# loaded_stack = hdu[0].data.astype(np.float64)
# #self.img=hdu[0].data.astype(np.float16)
# hdu.close()
#==============================================================================
#==============================================================================
# 
# stack, bad_indices = stack.det_stack_time_average(doas_res_std.start_acq,\
#                                                       doas_res_std.stop_acq)
# doas_series = doas_res_std.drop(doas_res_std.index[bad_indices])
# 
# search_engine = piscope.doasfov.DoasFOVEngine(stack, doas_series)
#==============================================================================
#corrImStd = piscope.doasfov.search_correlation(stack, doas_vec_std)
