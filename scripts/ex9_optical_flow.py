# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - optical flow analysis
"""
from matplotlib.pyplot import close, show

from ex1_measurement_setup_plume_data import create_dataset, save_path, join
from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
    path_bg_off
    
SAVEFIGS = 1
FIT_ANALYSIS = 0

pyrlevel = 2
blur = 0

close("all")

roi_flow = [615, 350, 1230, 790]

ds = create_dataset()
dist_img, plume_dist_img = ds.meas_geometry.get_all_pix_to_pix_dists()
pix_len = dist_img.mean()
aa_list = prepare_aa_image_list(ds, path_bg_on, path_bg_off)
aa_list.pyrlevel = pyrlevel
aa_list.add_gaussian_blurring(blur)
fl = aa_list.optflow
fl.roi_abs = roi_flow

aa_list.optflow_mode = 1
print aa_list.optflow.settings

ax0 = aa_list.show_current()

ax1 = fl.draw_flow(1)[0]

mask = fl.prepare_intensity_condition_mask(lower_val = 0.1)
count, bins, angles, fit3 = fl.flow_orientation_histo(cond_mask_flat = mask)
count, bins, angles, fit4 = fl.flow_length_histo(cond_mask_flat = mask)
#count, bins, angles, fit2 = fl.flow_length_histo()

ax2 = fit3.plot_result()[0]
ax3 = fit4.plot_result()[0]

mu_or, sigma_or,_ = fit3.analyse_fit_result()
mu_len, sigma_len,_ = fit4.analyse_fit_result()

good_lens, good_angles = fl.get_main_flow_field_params()
v = mu_len * 2**pyrlevel * pix_len / fl.del_t

fig = fl.plot_flow_histograms()
fig.suptitle("v = %.2f m/s" %(v))

if SAVEFIGS:
    ax0.figure.savefig(join(save_path, "ex9_out_1.png"))
    ax1.figure.savefig(join(save_path, "ex9_out_2.png"))
    ax2.figure.savefig(join(save_path, "ex9_out_3.png"))
    ax3.figure.savefig(join(save_path, "ex9_out_4.png"))
    fig.savefig(join(save_path, "ex9_out_5.png"))

if FIT_ANALYSIS:
    count, bins, angles, fit1 = fl.flow_orientation_histo()
    count, bins, angles, fit2 = fl.flow_length_histo()
    
    fit1.plot_result()
    fit2.plot_result()
    
    
    mask = fl.prepare_intensity_condition_mask(lower_val = 0.15)
    count, bins, angles, fit5 = fl.flow_orientation_histo(cond_mask_flat = mask)
    count, bins, angles, fit6 = fl.flow_length_histo(cond_mask_flat = mask)
    #count, bins, angles, fit2 = fl.flow_length_histo()
    
    fit5.plot_result()
    fit6.plot_result()
show() 
    
    
    
    
