# -*- coding: utf-8 -*-
"""
piscope example script no. 6 - DOAS calibration and FOV search

Sript showing how to work with cell calibration data
"""

import piscope 
import pydoas
from datetime import timedelta
from matplotlib.pyplot import close, show
from os.path import join, exists
from os import remove

from ex4_prepare_aa_imglist import prepare_aa_image_list, img_dir, save_path
                                                                
### SCRIPT OPTONS

#reload and save stack in folder my_dat (see below), results in increased
#running time due to stack calculation
RELOAD_STACK = 0
stack_path = join(save_path, "piscope_imgstack_id_aa_20150916_0706_0721.fts")

#Default search settings are at pyramid level 2, the FOV results are upscaled
#to original resolution, if the following option is set 1, then, based on 
#the result from pyrlevel=2, another stack is determined at pyrlevel = 0 
#(i.e. in full resolution) within ROI around the center position from 
#pyrlevel=2
DO_FINE_SEARCH = 0

if not exists(stack_path):
    RELOAD_STACK = 1

# Path containing DOAS result files
doas_data_path = join(img_dir, "..", "spectra", "plume_prep", "min10Scans",\
    "ResultFiles")
                      
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
        aa_list.crop = True
    aa_list.pyrlevel = pyrlevel
    aa_list.auto_reload = True
    
    stack = aa_list.make_stack()
    if save:
        try:
            remove(stack_path)
        except:
            pass    
        stack.save_as_fits(save_dir = save_path)  
    return stack


if __name__ == "__main__":
    close("all")
    aa_list = None
    if RELOAD_STACK:
        aa_list = prepare_aa_image_list()
        stack = make_aa_stack_from_list(aa_list)
    else:
        stack = piscope.processing.ImgStack()
        stack.load_stack_fits(stack_path)
    proc = 1
    if proc:
        doas_time_series = load_doas_results()
        s = piscope.doascalib.DoasFOVEngine(stack, doas_time_series, maxrad = 10)
        calib_pears = s.perform_fov_search(method = "pearson")
        calib_ifr= s.perform_fov_search(method = "ifr", ifrlbda = 2e-3)
        
        axes = [] #used to store axes objects from plots (for saving)
        calib_pears.fit_calib_polynomial()
        axes.append(calib_pears.plot())
        axes[-1].set_title("Calibration curve Pearson method")
        axes.append(calib_pears.fov.plot())
        
        calib_ifr.fit_calib_polynomial()
        axes.append(calib_ifr.plot())
        axes[-1].set_title("Calibration curve IFR method")
        axes.append(calib_ifr.fov.plot())
    
        if DO_FINE_SEARCH:    
            """Get position in absolute coordinates and perform a fov search within
            ROI around result from pearson fov search at full resolution 
            (pyrlevel=0)
            """
            if aa_list is None:
                aa_list = prepare_aa_image_list()
            extend = calib_pears.fov.pixel_extend(abs_coords=True)
            pos_x, pos_y = calib_pears.fov.pixel_position_center(abs_coords=True)
    
            del stack # make space for new stack
            #create ROI around center position of FOV
            roi = [ pos_x - 5*extend, pos_y - 5*extend,\
                    pos_x + 5*extend, pos_y + 5*extend]
                    
            stack = make_aa_stack_from_list(aa_list, roi_abs=roi, pyrlevel=0, 
                                            save=0)
            s = piscope.doascalib.DoasFOVEngine(stack, doas_time_series,\
                                                        pearson_max_radius = 30)
            calib_pears_fine = s.perform_fov_search(method = "pearson")
            calib_pears_fine.fit_calib_polynomial()
            axes.append(calib_pears_fine.plot())
            axes.append(calib_pears_fine.fov.plot())
        for k in range(len(axes)):
            axes[k].figure.savefig(join(save_path, "ex6_out_%d.png" %k))
        try:
            remove(join(save_path, 
                        "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts"))
        except:
            pass
        calib_pears.save_as_fits(save_dir = save_path)
        show()
