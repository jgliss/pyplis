# -*- coding: utf-8 -*-
"""
piscope example script no. 6 - DOAS calibration and FOV search

Sript showing how to work with cell calibration data
"""

import piscope 
import pydoas
from datetime import timedelta
from matplotlib.pyplot import close, show, subplots
from os.path import join, exists
from os import remove

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex4_prepare_aa_imglist import prepare_aa_image_list

### SCRIPT OPTONS  
SAVEFIGS = 1 # save plots from this script in SAVE_DIR

#reload and save stack in folder SAVE_DIR, results in increased
#running time due to stack calculation (is automatically switched on if
#the stack is not found at this location)
RELOAD_STACK = 0

#Default search settings are at pyramid level 2, the FOV results are upscaled
#to original resolution, if the following option is set 1, then, based on 
#the result from pyrlevel=2, another stack is determined at pyrlevel = 0 
#(i.e. in full resolution) within ROI around the center position from 
#pyrlevel=2
DO_FINE_SEARCH = 0

### RELEVANT DIRECTORIES AND PATHS

# Image directory
IMG_DIR = join(piscope.inout.find_test_data(), "images")

# Directory where results are stored
SAVE_DIR = join(".", "scripts_out")

# Directory containing DOAS result files
DOAS_DATA_DIR = join(IMG_DIR, "..", "spectra", "plume_prep", "min10Scans",
                     "ResultFiles")                                                                

STACK_PATH = join(SAVE_DIR, "piscope_imgstack_id_aa_20150916_0706_0721.fts")

### SCRIPT FUNCTION DEFINITIONS                    
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
    
    doas_import_setup = pydoas.dataimport.ResultImportSetup(DOAS_DATA_DIR,\
                                        result_import_dict = fit_import_info)
    ### Import the DOAS fit results
    doas_dataset = pydoas.analysis.DatasetDoasResults(doas_import_setup)
    
    ### get fit results from standard so2 fit (f01)
    # they were recorded in LT, so shift them to UTC (2h back) 
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
            remove(STACK_PATH)
        except:
            pass    
        stack.save_as_fits(save_dir = SAVE_DIR)  
    return stack


### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    
    if not exists(STACK_PATH):
        RELOAD_STACK = 1
        
    aa_list = None
    if RELOAD_STACK:
        aa_list = prepare_aa_image_list()
        stack = make_aa_stack_from_list(aa_list)
    else:
        stack = piscope.processing.ImgStack()
        stack.load_stack_fits(STACK_PATH)

    
    doas_time_series = load_doas_results()
    s = piscope.doascalib.DoasFOVEngine(stack, doas_time_series, maxrad = 10)
    calib_pears = s.perform_fov_search(method = "pearson")
    calib_ifr= s.perform_fov_search(method = "ifr", ifrlbda = 2e-3)
    
    axes = [] #used to store axes objects from plots (for saving)
    
    fig, ax = subplots(1,1)
    calib_pears.fit_calib_polynomial()
    axes.append(calib_pears.plot(add_label_str="Pearson", color="b", ax=ax))
    axes.append(calib_pears.fov.plot())
    
    calib_ifr.fit_calib_polynomial()
    
    calib_ifr.plot(add_label_str="ifr", color="g", ax=ax)
    ax.set_title("Calibration curves Pearson vs. IFR method")
    ax.grid()
    ax.set_ylim([0, 1.8e18])
    ax.set_xlim([0, 0.20])
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

    if SAVEFIGS:
        for k in range(len(axes)):
            axes[k].figure.savefig(join(SAVE_DIR, "ex6_out_%d.png" %k))
            
    try:
        remove(join(SAVE_DIR, 
                    "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts"))
    except:
        pass
    calib_pears.save_as_fits(save_dir = SAVE_DIR)
    show()
