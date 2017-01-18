# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - optical flow analysis
"""
from matplotlib.pyplot import close

from ex1_measurement_setup_plume_data import create_dataset
from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
    path_bg_off

FIT_ANALYSIS = False

close("all")

roi_flow = [615, 350, 1230, 790]

ds = create_dataset()
aa_list = prepare_aa_image_list(ds, path_bg_on, path_bg_off)
aa_list.pyrlevel = 1

fl = aa_list.opt_flow
fl.roi_abs = roi_flow

aa_list.opt_flow_mode = 1
print aa_list.opt_flow.settings

aa_list.show_current()

ax2 = fl.draw_flow(1)

mask = fl.prepare_intensity_condition_mask(lower_val = 0.1)
count, bins, angles, fit3 = fl.flow_orientation_histo(cond_mask_flat = mask)
count, bins, angles, fit4 = fl.flow_length_histo(cond_mask_flat = mask)
#count, bins, angles, fit2 = fl.flow_length_histo()

fit3.plot_result()
fit4.plot_result()

good_lens, good_angles = fl.get_main_flow_field_params()

fl.plot_flow_histograms()
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
    
    
    
    
    
