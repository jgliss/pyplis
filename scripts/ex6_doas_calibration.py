# -*- coding: utf-8 -*-
"""
piscope example script no. 6 - DOAS calibration and FOV search

Sript showing how to work with cell calibration data
"""

import piscope 
import pydoas
from datetime import timedelta
from matplotlib.pyplot import close
from os.path import join, exists
from os import remove

from ex1_measurement_setup_plume_data import create_dataset
from ex4_prepare_aa_imglist import prepare_aa_image_list, test_data_path,\
                                                                save_path
                                                                
### SCRIPT OPTONS

#reload and save stack in folder my_dat (see below), results in increased
#running time due to stack calculation
RELOAD_STACK = 0

#Default search settings are at pyramid level 2, the FOV results are upscaled
#to original resolution, if the following option is set 1, then, based on 
#the result from pyrlevel=2, another stack is determined at pyrlevel = 0 
#(i.e. in full resolution) within ROI around the center position from 
#pyrlevel=2
DO_FINE_SEARCH = 0

my_dat = r'D:/repos/tests' #location of image stack
stack_path = join(my_dat, "piscope_imgstack_id_aa_20150916_0706_0721.fts")
        
piscope.inout.set_test_data_path("D:\\repos") #test data location

reload_stack = 0 #re create stack from image list
if not exists(stack_path):
    reload_stack =1

# Path containing DOAS result files
doas_data_path = join(test_data_path, "spectra", "plume_prep", "min10Scans",\
    "ResultFiles")
    
# Image base path
img_dir = join(test_data_path, "images")
### Set plume background images for on and off
# this is the same image which is also used for example script NO
# demonstrating the plume background routines
path_bg_on = join(img_dir, 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
path_bg_off = join(img_dir, 'EC2_1106307_1R02_2015091607022820_F02_Etna.fts')
                  
def load_doas_results():
    """ Specify DOAS data import from DOASIS fit result files
    In order to perform the DOAS FOV search, as much spectrum datapoints 
    as possible are needed. Therefore, we only added 10 scans per plume
    spectrum. In this case (and because the spectrometer was not temperature 
    stabilised, the DOAS fit in the recommended wavelength range (~ 314 - 326, 
    here "f01") might not exceed the S/N ratio for low SO2 CDs. Thus, to have a 
    quality check of the fit performance, SO2 was therefore also fitted in a lower 
    wavelength region ("f02" : ~ 309 - 323 nm), both datasets are imported here
    and are plotted against each other below, showing that the correlation is 
    good (i.e. f01 is trustworthy) and that the SO2 CDs are too small
    """
    fit_import_info = {"so2" : ["SO2_Hermans_298_air_conv_satCorr1e18", 
                                                            ["f01"]]}
    
    doas_import_setup = pydoas.dataimport.ResultImportSetup(doas_data_path,\
                                        result_import_dict = fit_import_info)
    ### Import the DOAS fit results
    doas_dataset = pydoas.analysis.DatasetDoasResults(doas_import_setup)
    
    ### get fit results from standard so2 fit (f01)
    return doas_dataset.get_results("so2", "f01").shift(timedelta(-1./12))

def make_aa_stack_from_list(aa_list, roi_abs = None, pyrlevel = 2,\
                                                        save = True):
    """Get and prepare onband list for aa image mode"""
    aa_list.auto_reload = False
    if roi_abs is not None:
        aa_list.roi_abs = roi_abs
        aa_list.crop=True
    aa_list.pyrlevel = pyrlevel
    aa_list.auto_reload = True
    
    stack = aa_list.make_stack()
    if save:
        try:
            remove(stack_path)
        except:
            pass    
        stack.save_as_fits(save_dir = my_dat)  
    return stack


if __name__ == "__main__":
    close("all")
    aa_list = None
    if RELOAD_STACK:
        dataset = create_dataset()
        aa_list = prepare_aa_image_list(dataset, path_bg_on, path_bg_off)
        stack = make_aa_stack_from_list(aa_list)
    else:
        stack = piscope.processing.ImgStack()
        stack.load_stack_fits(stack_path)
        
    doas_time_series = load_doas_results()
    s = piscope.doascalib.DoasFOVEngine(stack, doas_time_series,\
                                                pearson_max_radius = 10)
    calib_pears = s.perform_fov_search(method = "pearson")
    calib_ifr= s.perform_fov_search(method = "ifr", ifr_lambda = 2e-3)
    
    axes = [] #used to store axes objects from plots (for saving)
    calib_pears.fit_calib_polynomial()
    axes.append(calib_pears.plot())
    axes.append(calib_pears.fov.plot())
    
    calib_ifr.fit_calib_polynomial()
    axes.append(calib_ifr.plot())
    axes.append(calib_ifr.fov.plot())
    
    #now get position in absolute coordinates and perform a fov search within
    #ROI around result from pearson fov search at full resolution (pyrlevel=0)
    if aa_list is None:
        dataset = create_dataset()
        aa_list = prepare_aa_image_list(dataset, path_bg_on, path_bg_off)
    extend = calib_pears.fov.pixel_extend(abs_coords=True)
    pos_x, pos_y = calib_pears.fov.pixel_position_center(abs_coords=True)
    if DO_FINE_SEARCH:
        del stack # make space for new stack
        #create ROI around center position of FOV
        roi = [ pos_x - 5*extend, pos_y - 5*extend,\
                pos_x + 5*extend, pos_y + 5*extend]
                
        stack = make_aa_stack_from_list(aa_list, roi_abs=roi, pyrlevel=0, save = 0)
        s = piscope.doascalib.DoasFOVEngine(stack, doas_time_series,\
                                                    pearson_max_radius = 30)
        calib_pears_fine = s.perform_fov_search(method = "pearson")
        calib_pears_fine.fit_calib_polynomial()
        axes.append(calib_pears_fine.plot())
        axes.append(calib_pears_fine.fov.plot())
    for k in range(len(axes)):
        axes[k].figure.savefig(join(save_path, "ex6_out_%d.png" %k))
 
    
    remove(join(my_dat, "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts"))
    calib_pears.save_as_fits(save_dir = my_dat)