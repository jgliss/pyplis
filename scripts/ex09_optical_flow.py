# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - optical flow analysis
"""
from matplotlib.pyplot import close, show
from os.path import join
import piscope

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prepare_aa_imglist import prepare_aa_image_list

### SCRIPT OPTONS  
PYRLEVEL = 1
BLUR = 0
ROI_FLOW = [615, 350, 1230, 790]

### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    # Prepare aa image list (see example 4)
    aa_list = prepare_aa_image_list()
    
    # the aa image list includes the measurement geometry, get pixel
    # distance image where pixel values correspond to step widths in the plume, 
    # obviously, the distance values depend on the downscaling factor, which
    # is calculated from the analysis pyramid level (PYRLEVEL)
    dist_img, _, _ = aa_list.meas_geometry.get_all_pix_to_pix_dists(
                                            pyrlevel=PYRLEVEL)
    # set the pyramid level in the list
    aa_list.pyrlevel = PYRLEVEL
    # add some blurring.. or not (if BLUR = 0)
    aa_list.add_gaussian_blurring(BLUR)
    
    # Access to the optical flow module in the image list. If optflow_mode is 
    # active in the list, then, whenever the list index changes (e.g. using
    # list.next_img(), or list.goto_img(100)), the optical flow field is 
    # calculated between the current list image and the next one
    fl = aa_list.optflow 
    #(! note: fl is only a pointer, i.e. the "=" is not making a copy of the 
    # object, meaning, that whenever something changes in "fl", it also does
    # in "aa_list.optflow")
    
    # Set the region of interest in the optical flow module. The ROI only 
    # applies to the region used for post analysis of the flow field, the flow
    # field itself is calculated for the whole image (should become clear from
    # the plots)
    fl.roi_abs = ROI_FLOW
    
    # Now activate optical flow calculation in list (this slows down the 
    # speed of the analysis, since the optical flow calculation is 
    # comparatively slow
    aa_list.optflow_mode = 1
    
    # Plots the flow field
    ax0 = fl.draw_flow(1)[0]
    
    mask = fl.prepare_intensity_condition_mask(lower_val=0.05)
    count, bins, angles, fit3 = fl.flow_orientation_histo(cond_mask_flat=mask)
    count, bins, angles, fit4 = fl.flow_length_histo(cond_mask_flat = mask)
    
    #plot the fit results 
    ax1 = fit3.plot_result(add_single_gaussians = True)[0]
    ax2 = fit4.plot_result(add_single_gaussians = True)[0]
    
    #Show an image containing plume speed magnitudes (ignoring direction)
    velo_img = piscope.Img(fl.to_plume_speed(dist_img))
    velo_img.show(vmin = 0, vmax = 7, cmap = "Greens",
                  tit = "Optical flow plume velocities",
                  zlabel ="Plume velo [m/s]")
    
    plume_params = piscope.plumespeed.LocalPlumeProperties()
    #mask = fl.prepare_intensity_condition_mask(lower_val = 0.10)
    plume_params.get_and_append_from_farneback(fl, cond_mask_flat = mask)  
    
    v = plume_params.len_mu[-1] * dist_img.mean() / fl.del_t
    fig = fl.plot_flow_histograms()
    fig.suptitle("v = %.2f m/s" %(v))
    
    if SAVEFIGS:
        ax0.figure.savefig(join(SAVE_DIR, "ex09_out_1.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        ax1.figure.savefig(join(SAVE_DIR, "ex09_out_2.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        ax2.figure.savefig(join(SAVE_DIR, "ex09_out_3.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        fig.savefig(join(SAVE_DIR, "ex09_out_4.png"))
    
    show() 
        
        
        
        
