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

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

### Set image base path
img_path = "../data/piscope_etna_testdata/images/"

### Set plume background image 
# this is the same image which is also used for example script NO
# demonstrating the plume background routines
path_bg_on = join(img_path, 
                  'EC2_1106307_1R02_2015091607022602_F01_Etnaxxxxxxxxxxxx.fts')

path_bg_off = join(img_path, 
                  'EC2_1106307_1R02_2015091607022820_F02_Etnaxxxxxxxxxxxx.fts')

### Set path to folder containing DOAS result files
doas_data_path = ("../data/piscope_etna_testdata/spectra/"
                    "plume_prep/min10Scans/ResultFiles/")

def do_doas_calibration(pyrlevel_stack = 3):
    ### Plume data time stamps
    start = datetime(2015, 9, 16, 7, 6, 00)
    #stop  = datetime(2015, 9, 16, 7, 12, 00)
    stop  = datetime(2015, 9, 16, 7, 22, 00)
    
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
                                                            ["f01", "f02"]]}
    
    doas_import_setup = pydoas.dataimport.ResultImportSetup(doas_data_path,\
                                        result_import_dict = fit_import_info)
    ### Import the DOAS fit results
    doas_dataset = pydoas.analysis.DatasetDoasResults(doas_import_setup)
    
    ### Make a scatter plot of both SO
    ax = doas_dataset.scatter_plot("so2", "f01", "so2", "f02")
    ax.set_xlabel("SO2 CDs [cm-2] (314 - 326 nm)")
    ax.set_ylabel("SO2 CDs [cm-2] (309 - 323 nm)")
    
    ax.figure.savefig(join(save_path, "ex6_1_so2_cds_scatter.png"))
    
    ### get fit results from standard so2 fit (f01)
    
    doas_res_std = doas_dataset.get_results("so2", "f01")
    doas_res_low = doas_dataset.get_results("so2", "f02")
    i, f = doas_res_std.start_times - timedelta(1./12),\
                        doas_res_std.stop_times - timedelta(1./12)
    
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
    
    on_list.aa_mode = True
    
    fig = on_list.bg_model.plot_tau_result(on_list.current_img())
    fig.savefig(join(save_path, "ex6_2_first_aa_image.png"))
    
    stack = on_list.make_stack(pyrlevel = pyrlevel_stack)
    
    stack, bad_indices = stack.det_stack_time_average(i, f)
    doas_vec_std = doas_res_std.drop(doas_res_std.index[bad_indices])
    doas_vec_low = doas_res_low.drop(doas_res_low.index[bad_indices])
    
    corrImStd = piscope.doasfov.search_correlation(stack, doas_vec_std)
    #corrImLow = piscope.doasfov.search_correlation(stack, doas_vec_low)
    
    lbda0 = 1e-6
    lbda1 = 5e-2
    (lsmrOf, lsmrImStd_lbda0) = piscope.doasfov.IFRlsmr(doas_vec_std.values,\
                                                            stack.stack, lbda0)
    (lsmrOf, lsmrImStd_lbda1) = piscope.doasfov.IFRlsmr(doas_vec_std.values,\
                                                            stack.stack, lbda1)
    (lsmrOf, lsmrImLow_lbda0) = piscope.doasfov.IFRlsmr(doas_vec_low.values,\
                                                            stack.stack, lbda0)
    (lsmrOf, lsmrImLow_lbda1) = piscope.doasfov.IFRlsmr(doas_vec_low.values,\
                                                            stack.stack, lbda1)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
#==============================================================================
#     #fig.set_size_inches(fig.get_size_inches()*np.array([2,1]))
#     ax[0, 0].imshow(corrImLow)#, cmap=plt.cm.jet)
#     ax[0, 0].set_title("DOAS (309 - 323 nm), pearson method")
#     ax[0, 1].imshow(lsmrImLow_lbda1)#, cmap=plt.cm.jet)
#     ax[0, 1].set_title(r"DOAS (309 - 323 nm) , LSMR ($\lambda$ = %.1e" %lbda1)
#     ax[0, 2].imshow(lsmrImLow_lbda0)#, cmap=plt.cm.jet)
#     ax[0, 2].set_title(r"DOAS (309 - 323 nm) , LSMR ($\lambda$ = %.1e" %lbda0)
#==============================================================================
    
    ax[0].imshow(corrImStd)
    ax[0].set_title("DOAS (314 - 326 nm) , pearson method")
    ax[1].imshow(lsmrImLow_lbda1)#, cmap=plt.cm.jet)
    ax[1].set_title(r"DOAS (314 - 326 nm) , LSMR ($\lambda$ = %.1e)" %lbda1)
    ax[2].imshow(lsmrImLow_lbda0)#, cmap=plt.cm.jet)
    ax[2].set_title(r"DOAS (314 - 326 nm) , LSMR ($\lambda$ = %.1e)" %lbda0)
    
    fig.savefig(join(save_path, "ex6_3_corrimgs.png"))
    ym, xm = piscope.doasfov.get_img_maximum(corrImStd)
    
    radius, corr_curve, tau_data, spec_data = piscope.doasfov.find_fov_radius(\
                                            stack, doas_vec_std, xm, ym)
    poly = np.poly1d(np.polyfit(tau_data, spec_data, 1))
    
    print "Radius %s" %radius
    
    fig, ax = plt.subplots(1,1)
    ax.plot(tau_data, spec_data, " b*", label = "calib data")
    ax.plot(tau_data, poly(tau_data), "r-", label = "poly")
    ax.legend()
    fig.savefig(join(save_path, "ex6_4_calib_poly.png"))
    #f = open(join(save_path, "ex4_doas_calib_coeffs.txt", "w"))
    return poly, tau_data, spec_data
    
if __name__ == "__main__":   
    poly, tau_dat, spec_dat = do_doas_calibration()

    

#==============================================================================
# if saveData:
#     dump(stack, open(join(saveResultsPath, "tauStack.obj"), 'wb'))
#     dump(doas_vec_low, open(join(saveResultsPath, "doas_datasetLowFit.obj"), 'wb'))
#     dump(doas_vec_std, open(join(saveResultsPath, "doas_datasetStdFit.obj"), 'wb'))
# 
#==============================================================================
