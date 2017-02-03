# -*- coding: utf-8 -*-
"""
piscope example script no. 4 - Prepare AA image list (from onband list)

Sript showing how to work in AA mode using ImgList object
"""

import piscope 
from matplotlib.pyplot import close
from os.path import join

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, IMG_DIR

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex01_measurement_setup_plume_data import create_dataset
from ex02_measurement_geometry import correct_viewing_direction


### SCRIPT FUNCTION DEFINITIONS        
def prepare_aa_image_list(bg_corr_mode = 6):
    """Get and prepare onband list for aa image mode
    
    The relevant gas free areas for background image modelling are set 
    automatically (see also ex. 3 for details)
    
    :return: - on list in AA mode    
    """

    dataset = create_dataset()
    geom, _ = correct_viewing_direction(dataset.meas_geometry, False)
    
    ### Set plume background images for on and off
    # this is the same image which is also used for example script NO
    # demonstrating the plume background routines
    path_bg_on = join(IMG_DIR, 
                      'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
    path_bg_off = join(IMG_DIR, 
                       'EC2_1106307_1R02_2015091607022820_F02_Etna.fts')
    
    ### Get on and off lists and activate dark correction
    lst = dataset.get_list("on")
    lst.activate_darkcorr() #same as lst.darkcorr_mode = 1
    
    off_list = dataset.get_list("off")
    off_list.activate_darkcorr()

    # Prepare on and offband background images
    bg_on = piscope.Img(path_bg_on)
    bg_on.subtract_dark_image(lst.get_dark_image())
    
    bg_off = piscope.Img(path_bg_off)
    bg_off.subtract_dark_image(off_list.get_dark_image())
    
    #set the background images within the lists
    lst.set_bg_img(bg_on)
    off_list.set_bg_img(bg_off)
    
    # automatically set gas free areas
    lst.bg_model.guess_missing_settings(lst.current_img())
    #Now update some of the information from the automatically set sky ref 
    #areas    
    lst.bg_model.xgrad_line_startcol = 20
    lst.bg_model.xgrad_line_rownum = 25
    off_list.bg_model.xgrad_line_startcol = 20
    off_list.bg_model.xgrad_line_rownum = 25
    
    #set background modelling mode
    lst.bg_model.CORR_MODE = bg_corr_mode
    off_list.bg_model.CORR_MODE = bg_corr_mode
    
    lst.aa_mode = True # activate AA mode 
    
    lst.meas_geometry = geom
    return lst
    
### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    from matplotlib.pyplot import show
    from time import time
    
    close("all")
    aa_list = prepare_aa_image_list()
    
    aa_list.show_current()
    
    t0=time()
    #Deactivate auto reload while changing some settings (else, after each
    #of the following operations the images are reloaded and edited, which)
    aa_list.auto_reload = False
    aa_list.goto_img(50)
    
    aa_list.add_gaussian_blurring(1)
    #aa_list.pyrlevel = 2
    aa_list.roi_abs = [300, 300, 1120, 1000]
    aa_list.crop = True
    #now reactive image reload in list (loads image no. 50 with all changes
    #that where set in the previous lines)
    aa_list.auto_reload = True
    ax = aa_list.show_current()
    print "Elapsed time: %s s" %(time() - t0)
    
    aa_list.crop = False
    ax1 = aa_list.bg_model.plot_sky_reference_areas(aa_list.current_img())
    fig = aa_list.bg_model.plot_tau_result(aa_list.current_img())
    
    if SAVEFIGS:
        ax.figure.savefig(join(SAVE_DIR, "ex04_out_1.%s" %FORMAT), format=FORMAT,
                    dpi=DPI)

    show()