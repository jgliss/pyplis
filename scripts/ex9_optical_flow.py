# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - optical flow analysis
"""
from matplotlib.pyplot import close, show

import piscope
from ex1_measurement_setup_plume_data import create_dataset, save_path, join
from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
    path_bg_off

# Options    
SAVEFIGS = 0
PYRLEVEL = 1
BLUR = 0
ROI_FLOW = [615, 350, 1230, 790]

close("all")
if __name__ == "__main__":
    ds = create_dataset()
    dist_img, _, plume_dist_img = ds.meas_geometry.get_all_pix_to_pix_dists(
                                            pyrlevel=PYRLEVEL)
    #pix_len = dist_img.mean()
    aa_list = prepare_aa_image_list(ds, path_bg_on, path_bg_off)
    aa_list.pyrlevel = PYRLEVEL
    aa_list.add_gaussian_blurring(BLUR)
    fl = aa_list.optflow
    fl.roi_abs = ROI_FLOW
    
    aa_list.optflow_mode = 1
    
    ax0 = fl.draw_flow(1)[0]
    
    
    mask = fl.prepare_intensity_condition_mask(lower_val = 0.05)
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
    
    #The following function can be used to perform a his
    dir_mu, dir_sigma, len_mu, len_sigma, cond = fl.get_main_flow_field_params(\
                                                    cond_mask_flat = mask)
    #the return parameter "cond" is a flattend condition mask which was used
    #to retrieve the main flow field parameters within the ROI, it may be used
    #to visualise which vectors were used for the statistics
    
    #first reshape to ROI sub image
    shape = fl.get_flow_in_roi().shape[:2]
    stat_mask = piscope.Img(cond.reshape(shape))
    
    
    v = len_mu * dist_img.mean() / fl.del_t
    fig = fl.plot_flow_histograms()
    fig.suptitle("v = %.2f m/s" %(v))
    
    if SAVEFIGS:
        ax0.figure.savefig(join(save_path, "ex9_out_1.png"))
        ax1.figure.savefig(join(save_path, "ex9_out_2.png"))
        ax2.figure.savefig(join(save_path, "ex9_out_3.png"))
        fig.savefig(join(save_path, "ex9_out_5.png"))
    
    show() 
        
        
        
        
