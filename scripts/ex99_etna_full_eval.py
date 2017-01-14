# -*- coding: utf-8 -*-
"""
Etna data analysis script

@author: jg
"""

import piscope
from datetime import datetime
from os.path import join
from os import getcwd
import matplotlib.pyplot as plt

from ex5_automatic_cell_and_bg_detection import perform_auto_cell_calib
from ex6_doas_calibration import do_doas_calibration

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

### Directory containing images
img_path = '../test_data/piscope_etna_testdata/images/'

bg_file_on = join(img_path, 
                  'EC2_1106307_1R02_2015091607022602_F01_Etnaxxxxxxxxxxxx.fts')
bg_file_off = join(img_path, 
                  'EC2_1106307_1R02_2015091607022820_F02_Etnaxxxxxxxxxxxx.fts')

cam_id = "ecII"

def create_environment(plot_geom = True):
    ### Define exemplary plume cross section line
    #pcs1     =   [400, 655, 860, 200]
    pcs1     =   [500, 655, 960, 200]
    pcs2     =   [400, 655, 860, 200]
    
    pcs_line1 = piscope.processing.LineOnImage(*pcs1, id = "pcs1")
    pcs_line2 = piscope.processing.LineOnImage(*pcs2, id = "pcs2")
    
    filters = [piscope.utils.Filter(type = "on", acronym = "F01",\
                                                    center_wavelength = 310),
               piscope.utils.Filter(type = "off", acronym = "F02",\
                                                   center_wavelength = 330)]
    
    #camera location and viewing direction (altitude will be retrieved automatically)                    
    geom_cam = {"lon"     :   15.1129,
               "lat"     :   37.73122,
               "elev"    :   15.0,
               "elev_err" :   5.0,
               "azim"    :   274.0,
               "azim_err" :   10.0}
    
    #Camera height in m with respect to topographic altitude  at site
    #(We were standing on the roof of a building, guessed 20m)
    camZOffset = 20 #25 
    #create camera setup
    cam = piscope.setup.Camera(cam_id = cam_id, geom_data = geom_cam,\
                filter_list = filters, focal_length = 25.0e-3)
    
    ### Load default information for Etna
    source = piscope.setup.Source("etna") 
    
    #### Provide wind direction
    windInfo= {"dir"     : 0.0,
               "dir_err"  : 15.0}
    
    
    ### Define start and stop time of measurement data
    start = datetime(2015,9,16,7,6,00)
    stop = datetime(2015,9,16,7,22,00)
    #stop = datetime(2015,9,16,7,10,00)
    
    ### +++++++++++++++++++
    ### END OF INPUT AREA
    ### +++++++++++++++++++
    
    ### Create BaseSetup object (which creates the MeasGeometry object)
    stp = piscope.setup.MeasSetup(img_path, start, stop, cam,\
                                                    source, windInfo)
    
    plume_dataset = piscope.dataset.Dataset(stp)
    
    if plot_geom:
        geom = stp.meas_geometry
        map = geom.draw_map_2d()
        map.ax.figure.savefig(join(save_path, "ex99_full_eval_map.png"))
    
        return plume_dataset, pcs_line1, pcs_line2
 
def prepare_aa_img_list(plume_dataset):
    on_list = plume_dataset.get_list("on")
    off_list = plume_dataset.get_list("off")
    
    on_list.activate_dark_corr()
    off_list.activate_dark_corr()
    
    on_list.add_gaussian_blurring(1)
    off_list.add_gaussian_blurring(1)

    on_list.link_imglist(off_list)
    
    #Load background images
    bg_on = piscope.image.Img(bg_file_on)
    bg_off = piscope.image.Img(bg_file_off)
    
    dark_raw = on_list.dark_lists["0"]["list"].current_img()
    offset_raw = on_list.offset_lists["0"]["list"].current_img()
    
    darkbg_on = piscope.processing.model_dark_image(bg_on, dark_raw, offset_raw)
    darkbg_off = piscope.processing.model_dark_image(bg_off, dark_raw, offset_raw)
    
    bg_on.subtract_dark_image(darkbg_on)
    bg_off.subtract_dark_image(darkbg_off)

    on_list.set_bg_image(bg_on)
    off_list.set_bg_image(bg_off)

    bg_model_on = on_list.bg_model
    bg_model_on.guess_missing_settings(on_list.current_img())
    bg_model_on.CORR_MODE = 6
    
    bg_model_off = off_list.bg_model
    bg_model_off.guess_missing_settings(off_list.current_img())
    bg_model_off.CORR_MODE = 6
    
    tau_est = bg_model_on.get_tau_image(on_list.current_img(), on_list.bg_img)
    bg_model_on.plot_tau_result(tau_est)
    on_list.activate_aa_mode()
    return on_list
    
def load_cell_and_doas_calib():    
    doas_poly, doas_tau_dat, doas_spec_dat = do_doas_calibration()
    cell_calib = perform_auto_cell_calib()
    return doas_poly, doas_tau_dat, doas_spec_dat, cell_calib

        
#==============================================================================
# def get_transmission_sensitivity_mask(cell_calib, stack_id = "aa",\
#                             cell_index = 0, polyorder = 3, pyrlevel = 0):
#     stack = cell_calib.tau_stacks[stack_id]
#     cell_tau_img = stack.stack[:, :, cell_index]
#     cell_cd = stack.add_data[cell_index]
#     
#     fit = piscope.fitting.PolySurfaceFit(cell_tau_img, polyorder = polyorder,\
#                                              pyrlevel = pyrlevel)
#     mask = piscope.image.Img(fit.model / fit.model.min())
#     mask.pyr_up(pyrlevel)
#     return mask, cell_tau_img, cell_cd
# 
#     mask, cell_tau_img, cell_cd
#     
# mask, cell_tau_img, cell_cd = get_transmission_sensitivity_mask(cell_calib)
# fig, axes = plt.subplots(1,3, figsize = (6, 20))
# 
# im0 = axes[0].imshow(cell_tau_img)
# fig.colorabar(im0, ax = axes[0])
# axes[0].set_title("Cell CD = %s" )
# axes[1].imshow(fit.model)
# axes[2].imshow(cell_tau_img - fit.model)
#==============================================================================

#fig.savefig(join(save_path, "ex99_sensitivity_cell.png"))


#==============================================================================
# for k in range(on_list.numberOfFiles):
#     on = on_list.current_img()
#     off = off_list.current_img()
#     aa = (on - off)
# 
#==============================================================================
if __name__ == "__main__":
    import numpy as np
    from pandas import Series
    from scipy.signal import correlate
    from scipy.ndimage.filters import gaussian_filter1d
    
    dataset, pcs_line1, pcs_line2 = create_environment()
    aa_list = prepare_aa_img_list(dataset)
    
    #doas_calib_coeffs, cell_calib = load_cell_and_doas_calib()
    #doas_poly, doas_tau_dat, doas_spec_dat = do_doas_calibration()
    
    geometry = dataset.meas_geometry
    dist_img, plume_dist_img = geometry.get_all_pix_to_pix_dists
    
    fig, ax = plt.subplots(1,2)
    disp_l = ax[0].imshow(dist_img, cmap = "gray")
    fig.colorbar(disp_l, ax = ax[0])
    disp_r = ax[1].imshow(plume_dist_img, cmap = "gray")
    fig.colorbar(disp_l, ax = ax[1])
    
    dists1 = pcs_line1.get_line_profile(dist_img)
    dists2 = pcs_line2.get_line_profile(dist_img)
    
    icas1 = []
    icas2 = []
    
    times, _ = aa_list.get_img_meta_all_filenames()
    
#==============================================================================
#     for k in range(aa_list.nof):
#         p1 = doas_poly(pcs_line1.get_line_profile(aa_list.current_img().img))*100**2
#         p2 = doas_poly(pcs_line2.get_line_profile(aa_list.current_img().img))*100**2
#         icas1.append(sum(p1 * dists1))
#         icas2.append(sum(p2 * dists2))
#         aa_list.next_img()
#         
#     s1 = Series(icas1, times).resample("1S").mean().interpolate().dropna()
#     s2 = Series(icas2, times).resample("1S").mean().interpolate().dropna()
#     
#     num_tot = len(s1)
#     start = int(num_tot / 20.0)
#     stop = num_tot - start 
#     s1_cut = gaussian_filter1d(s1[start:stop], 6) 
#     s2_cut = gaussian_filter1d(s2[start:stop], 6) 
# #==============================================================================
# #     xcorr = correlate(s1_cut, s2_cut)
# #     num = len(s1_cut)
# #     num_range = np.arange(1 - num, num)
# # 
# #     recovered_index_shift = num_range[xcorr.argmax()]
# # 
# #     print "Index shift: %d" % (recovered_index_shift)
# #==============================================================================
#     
#     fig, ax = plt.subplots(1,3, figsize=(18,5))
#     aa_list.goto_img(10)
#     ax[0].imshow(aa_list.current_img().img, cmap = "gray")
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023, 0])
#     ax[0].plot([pcs_line1.start[0], pcs_line1.stop[0]], [pcs_line1.start[1],pcs_line1.stop[1]],\
#             'go-')
#     ax[0].plot([pcs_line2.start[0], pcs_line2.stop[0]], [pcs_line2.start[1],pcs_line2.stop[1]],\
#             'ro--')
# #==============================================================================
# #     pcs_line1.plot_line_on_grid()
# #==============================================================================
#     ax[1].plot(s1_cut, "-g", label = "ICA1")
#     ax[1].plot(s2_cut, "--r", label = "ICA2")
#     
# #==============================================================================
# #     ax[1].plot(num_range, xcorr, label = "Cross corr")
# #==============================================================================
#     
#     
#     from scipy.stats.stats import pearsonr
#     coeffs = []
#     for k in range(100):
#         coeffs.append(pearsonr(np.roll(s1_cut, k), s2_cut)[0])
#     ax[2].plot(coeffs)
#     ax[2].set_xlabel("Shift")
#     ax[2].set_ylabel("Correlation coeff")
#     
#     
#     del_t = np.argmax(coeffs) #already in seconds, since data was resampled to 1s
#     
#     delX = 100 #horizontal distance between the two lines (HARD CODED FOR NOW)
#     dist_pix = np.sin(pcs_line1.get_kappa()) * delX
#     
#     dist_in_m = dist_pix * dists1.mean()
#     
#     plumespeed = dist_in_m / del_t
#     fig.suptitle("Retrieved gas speed: %s m/s" %plumespeed)
#     
#     fig.savefig(join(save_path, "cross_corr_test.png"))
#     
#     mmol = 64 #g/mol
#     na = 6.022 * 10**23 #n/mol
#     
#     flux = np.asarray(icas1) * plumespeed * mmol / na / 1000.0
#     
#     fig, ax = plt.subplots(1,1)
#     ax.plot(times, flux, "--x", label = "Emission rate")
#     ax.set_ylabel("SO2 emission rate [kg/s]")
#     fig.savefig(join(save_path, "ex99_emission_rate_etna.png"))
#==============================================================================
        
    
    
