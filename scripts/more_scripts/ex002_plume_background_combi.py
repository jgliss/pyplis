# -*- coding: utf-8 -*-
"""
example script 002 - Combine background correction modes

This script illustrates how to efficiently apply the plume background analysis
in case no sky reference image is available.

"""
from os.path import join
from matplotlib.pyplot import close
import numpy as np
import sys
import pyplis

from matplotlib import rcParams
rcParams.update({'font.size': 15})

sys.path.append(join(".."))

from ex01_analysis_setup import create_dataset

### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    close("all")
    ds = create_dataset()
    
    on = ds.get_list("on")
    on.darkcorr_mode = True
    on.gaussian_blurring = 2
    
    # Find and plot sky reference areas
    on.bg_model.guess_missing_settings(on.current_img())
    ax = on.bg_model.plot_sky_reference_areas(on.current_img())
    
    # this is a beta version
    kernel = np.ones((90, 90), dtype=np.uint8)   
    mask = on.prepare_bg_fit_mask(dilation=True, dilate_kernel= kernel,
                                  optflow_blur=0, optflow_median=10, 
                                  i_min=1500, i_max=2600, plot_masks=True)
    mask = pyplis.Img(mask)
    ax2 = mask.show(tit="Input mask for surface fit")
    on.set_bg_img_from_polyfit(mask.img)
    ax3 = on.bg_img.show(tit="Surface fit result")
    
    on.bg_model.CORR_MODE = 5
    on.tau_mode=True
    on.show_current()
            
            