# -*- coding: utf-8 -*-
"""
piSCOPE example script 3: Plume background analysis 

This example script introduces features related to plume background modelling
and tau image calculations.
"""
import numpy as np
from os.path import join
from os import getcwd
import piscope
import matplotlib.pyplot as plt

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

###OPTIONS
USE_AUTO_SETTINGS = False # If this is True, then sky reference areas are set in auto mode

plume_file = join(img_dir, 'EC2_1106307_1R02_2015091607065477_F01_Etna.fts')
bg_file = join(img_dir, 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
offset_file = join(img_dir, 'EC2_1106307_1R02_2015091607064723_D0L_Etna.fts')
dark_file = join(img_dir, 'EC2_1106307_1R02_2015091607064865_D1L_Etna.fts')


test_case = 1 # DELETE THIS BEFORE PUBLISHING

### Set options for background retrieval using poly surface fit
mask_thresh = 2600 #threshold to init mask for bg surface fit
vign_corr = 1
polyorder = 1
pyrlevel = 4


### Create background modellin object
m = piscope.plumebackground.PlumeBackgroundModel()

### Define default gas free areas in plume image
w, h = 40, 40
scale   =   [1280, 20 , 1280 + w, 20 + h]
xgrad   =   [20, 20, 20 + w, 20 + h]
ygrad   =   [1280, 660, 1280 + w, 660 + h]
#ygrad   =   

### Define exemplary plume cross section line
pcs     =   [420, 655, 860, 200]
### Define coordinates of horizontal and vertical profile lines 
hor_line_rownum = 40
vert_line_colnum = 1300

### Load the image objects and peform dark correction
plume, bg = piscope.Img(plume_file), piscope.Img(bg_file)
dark, offset = piscope.Img(dark_file), piscope.Img(offset_file)


# Model dark image for tExp of plume image
dark_plume = piscope.processing.model_dark_image(plume, dark, offset)
# Model dark image for tExp of background image
darkBg = piscope.processing.model_dark_image(bg, dark, offset)

plume.subtract_dark_image(dark_plume)
bg.subtract_dark_image(darkBg)
### Blur the images (sigma = 1)
plume.add_gaussian_blurring(1)
bg.add_gaussian_blurring(1)

### Create vignetting correction mask from background image 
vign = bg.img / bg.img.max() #NOTE: potentially includes y and x gradients
plume_corr = plume.img / vign

### Set the defined sky reference areas within the model engine
m.update(scale_rect = scale, ygrad_rect = ygrad, xgrad_rect = xgrad,\
    ygrad_line_colnum = vert_line_colnum, ygrad_line_startrow = 10,\
    ygrad_line_stoprow = 700, ygrad_line_polyorder = 2, xgrad_line_rownum =\
    hor_line_rownum, xgrad_line_startcol = 20, xgrad_line_stopcol = 1323)
    
### If you are lazy...
# (i.e. you dont want to define all these reference areas), then you could also
# use the auto search function, a comparison is plotted here:
auto_params = piscope.plumebackground.find_sky_reference_areas(plume)
current_params = m.sky_ref_areas_to_dict()
fig, axes = plt.subplots(1, 2, figsize = (16, 6))
piscope.plumebackground.plot_sky_reference_areas(plume, current_params,\
                                                            ax = axes[0])
axes[0].set_title("Manually set parameters")
piscope.plumebackground.plot_sky_reference_areas(plume, auto_params,\
                                                            ax = axes[1])
axes[1].set_title("Automatically set parameters")

if USE_AUTO_SETTINGS:
    m.update(**auto_params)

mask = np.ones(plume_corr.shape, dtype = np.float32)   
mask[plume_corr < mask_thresh] = 0
### Fourth Method: retrieve tau image using poly surface fit
tau0 = m.get_tau_image(piscope.Img(plume_corr), CORR_MODE = 0,\
                surface_fit_mask = mask, surface_fit_polyorder = 1)
fig0 = m.plot_tau_result(tau0, pcs = pcs)
### First method: scale background image to plume image in "scale" rect
tau1 = m.get_tau_image(plume, bg, CORR_MODE = 1)
fig1 = m.plot_tau_result(tau1, pcs = pcs)
### Second method: Linear correction for radiance differences based on two 
### rectangles (scale, ygrad)
tau2 = m.get_tau_image(plume, bg, CORR_MODE = 4)
fig2 = m.plot_tau_result(tau2, pcs = pcs)

### Third method: 2nd order polynomial fit along vertical profile line
### For this method, determine tau on tau off and AA image
tau3 = m.get_tau_image(plume, bg, CORR_MODE = 6)
fig3 = m.plot_tau_result(tau3, pcs = pcs)

### Plot PCS profiles for all 4 methods

pcs_line = piscope.processing.LineOnImage(*pcs, id = "pcs")
fig4, ax1 = plt.subplots(1,1)
p0 = pcs_line.get_line_profile(tau0.img)
p1 = pcs_line.get_line_profile(tau1.img)
p2 = pcs_line.get_line_profile(tau2.img)
p3 = pcs_line.get_line_profile(tau3.img)
num = len(p0)

phi0 = sum(p0)/num
phi1 = sum(p1)/num
phi2 = sum(p2)/num
phi3 = sum(p3)/num
ax1.plot(p0, "-", label = r"Mode 0: $\phi=%.3f$" %(phi0))
ax1.plot(p1, "-", label = r"Mode 1: $\phi=%.3f$" %(phi1))
ax1.plot(p2, "-", label = r"Mode 2: $\phi=%.3f$" %(phi2))
ax1.plot(p3, "-", label = r"Mode 6: $\phi=%.3f$" %(phi3))


ax1.grid()
ax1.set_ylabel(r"$\tau$", fontsize=20)
ax1.set_xlim([0,632])
ax1.set_xticklabels([])
ax1.set_xlabel("PCS", fontsize=16)
ax1.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=12)
if test_case:
    plt.close("all")
    m=piscope.plumebackground.PlumeBackgroundModel()
    m.guess_missing_settings(plume)
    
    tau0 = np.log(bg.img / plume.img)
    fig0 = m.plot_tau_result(piscope.image.Img(tau0))
    fig0.suptitle("INIT")
    
    tau1 = piscope.plumebackground.corr_tau_curvature_hor_two_rects(tau0,\
                    m.scale_rect, m.xgrad_rect)
    fig1 = m.plot_tau_result(piscope.image.Img(tau1))
    fig1.suptitle("HOR CORR")
    
    tau2 = piscope.plumebackground.corr_tau_curvature_vert_two_rects(tau1,\
                                             m.scale_rect, m.ygrad_rect)
    
    fig2 = m.plot_tau_result(piscope.image.Img(tau2))
    fig2.suptitle("HOR + VERT CORR")
    #==============================================================================
    # tau_new = piscope.plumebackground.corr_tau_curvature_hor_two_rects(tau2.img,\
    #     m.scale_rect, m.xgrad_rect)
    #==============================================================================
plt.show()
fig.savefig(join(save_path, "ex7_bg_corr_ref_sects.png"))
fig4.savefig(join(save_path, "ex7_bg_corr_ICAs.png"))
fig0.savefig(join(save_path, "ex7_bg_corr_0.png"))
fig1.savefig(join(save_path, "ex7_bg_corr_1.png"))
fig2.savefig(join(save_path, "ex7_bg_corr_2.png"))
fig3.savefig(join(save_path, "ex7_bg_corr_3.png"))




