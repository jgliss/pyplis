# -*- coding: utf-8 -*-
"""
piscope example script no. 4 - Prepare AA image list (from onband list)

Sript showing how to work in AA mode using ImgList object
"""

import piscope 
from matplotlib.pyplot import close
from os.path import join

#import relevant paths and methods from example one
from ex1_measurement_setup_plume_data import create_dataset, save_path, img_dir

### Set plume background images for on and off
# this is the same image which is also used for example script NO
# demonstrating the plume background routines
path_bg_on = join(img_dir, 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
path_bg_off = join(img_dir, 'EC2_1106307_1R02_2015091607022820_F02_Etna.fts')

def prepare_aa_image_list(dataset, bg_path_on, bg_path_off, bg_corr_mode = 6):
    """Get and prepare onband list for aa image mode
    
    The relevant gas free areas for background image modelling are set 
    automatically (see also ex. 3 for details)
    
    :return: - on list in AA mode    
    """
    ### Get on and off lists and activate dark correction
    on_list = dataset.get_list("on")
    on_list.activate_darkcorr() #same as on_list.darkcorr_mode = 1
    
    off_list = dataset.get_list("off")
    off_list.activate_darkcorr()

    # Prepare on and offband background images
    bg_on = piscope.Img(path_bg_on)
    bg_on.subtract_dark_image(on_list.get_dark_image())
    
    bg_off = piscope.Img(path_bg_off)
    bg_off.subtract_dark_image(off_list.get_dark_image())
    
    #set the background images within the lists
    on_list.set_bg_img(bg_on)
    off_list.set_bg_img(bg_off)
    
    # automatically set gas free areas
    on_list.bg_model.guess_missing_settings(on_list.current_img())
    #Now update some of the information from the automatically set sky ref 
    #areas    
    on_list.bg_model.xgrad_line_startcol = 20
    on_list.bg_model.xgrad_line_rownum = 25
    off_list.bg_model.xgrad_line_startcol = 20
    off_list.bg_model.xgrad_line_rownum = 25
    
    #set background modelling mode
    on_list.bg_model.CORR_MODE = bg_corr_mode
    off_list.bg_model.CORR_MODE = bg_corr_mode
    
    on_list.aa_mode = True # activate AA mode 
    
    return on_list

    #stack, hdu = load_stack_fits(stack, stack_path)


if __name__ == "__main__":
    from matplotlib.pyplot import show
    from time import time
    
    
    close("all")
    dataset = create_dataset()
    aa_list = prepare_aa_image_list(dataset, path_bg_on, path_bg_off)
    
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
    aa_list.bg_model.plot_sky_reference_areas(aa_list.current_img())
    aa_list.bg_model.plot_tau_result(aa_list.current_img())
    show()
    ax.figure.savefig(join(save_path, "ex4_out_1.png"))