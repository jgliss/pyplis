# -*- coding: utf-8 -*-
"""
piscope example script no. 3: Plume background analysis

This example script introduces features related to plume background modelling
and tau image calculations.    
"""
import numpy as np
from os.path import join
import piscope
import matplotlib.pyplot as plt

### SCRIPT OPTONS  
SAVEFIGS = 1 # save plots from this script in SAVE_DIR

# If this is True, then sky reference areas are set in auto mode
USE_AUTO_SETTINGS = False 

#intensity threshold to init mask for bg surface fit
POLYFIT_2D__MASK_THRESH = 2600 

# Choose the background correction modes you want to use

BG_CORR_MODES = [0, #2D poly surface fit (without sky radiance image)
                 1, #Scaling of sky radiance image
                 4, #Scaling + linear gradient correction in x and y direction
                 6] #Scaling + quadr. gradient correction in x and y direction

### RELEVANT DIRECTORIES AND PATHS

# Image directory
IMG_DIR = join(piscope.inout.find_test_data(), "images")

# Directory where results are stored
SAVE_DIR = join(".", "scripts_out")

# Relevant file paths
PLUME_FILE = join(IMG_DIR, 'EC2_1106307_1R02_2015091607065477_F01_Etna.fts')
BG_FILE = join(IMG_DIR, 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
OFFSET_FILE = join(IMG_DIR, 'EC2_1106307_1R02_2015091607064723_D0L_Etna.fts')
DARK_FILE = join(IMG_DIR, 'EC2_1106307_1R02_2015091607064865_D1L_Etna.fts')

### SCRIPT FUNCTION DEFINITIONS        
def init_background_model():
    """Create background model and define relevant sky reference areas"""
    ### Create background modelling object
    m = piscope.plumebackground.PlumeBackgroundModel()
    
    ### Define default gas free areas in plume image
    w, h = 40, 40 #width/height of rectangles
    
    m.scale_rect =   [1280, 20 , 1280 + w, 20 + h]
    m.xgrad_rect =   [20, 20, 20 + w, 20 + h]
    m.ygrad_rect =   [1280, 660, 1280 + w, 660 + h]
                                              
    ### Define coordinates of horizontal and vertical profile lines 

    #row number of profile line for horizontal corretions in the sky gradient... 
    m.xgrad_line_rownum = 40 
    # ... and start / stop columns for the corrections
    m.xgrad_line_startcol = 20
    m.xgrad_line_stopcol =1323
    
    #col number of profile line for vertical corretions in the sky gradient... 
    m.ygrad_line_colnum = 1300
    # ... and start / stop rows for the corrections
    m.ygrad_line_startrow = 10
    m.ygrad_line_stoprow = 700
    # Order of polyonmial fit applied for the gradient correction 
    m.ygrad_line_polyorder = 2
    
    return m
    
def load_and_prepare_images():
    """Load images defined above and prepare them for the background analysis
    
    Returns
    -------
    
        - Img, plume image
        - Img, plume image vignetting corrected
        - Img, sky radiance image
        
    """    
    ### Load the image objects and peform dark correction
    plume, bg = piscope.Img(PLUME_FILE), piscope.Img(BG_FILE)
    dark, offset = piscope.Img(DARK_FILE), piscope.Img(OFFSET_FILE)
    
    # Model dark image for tExp of plume image
    dark_plume = piscope.processing.model_dark_image(plume, dark, offset)
    # Model dark image for tExp of background image
    dark_bg = piscope.processing.model_dark_image(bg, dark, offset)
    
    plume.subtract_dark_image(dark_plume)
    bg.subtract_dark_image(dark_bg)
    ### Blur the images (sigma = 1)
    plume.add_gaussian_blurring(1)
    bg.add_gaussian_blurring(1)
    
    ### Create vignetting correction mask from background image 
    vign = bg.img / bg.img.max() #NOTE: potentially includes y and x gradients
    plume_vigncorr = piscope.Img(plume.img / vign)
    return plume, plume_vigncorr, bg

def autosettings_vs_manual_settings(bg_model):
    """Perform automatic retrieval of sky reference areas
    If you are lazy... (i.e. you dont want to define all these reference areas), 
    then you could also use the auto search function, a comparison is plotted 
    here
    """
    auto_params = piscope.plumebackground.find_sky_reference_areas(plume)
    current_params = bg_model.sky_ref_areas_to_dict()
                                                                
    fig, axes = plt.subplots(1, 2, figsize = (16, 6))                                                                
    axes[0].set_title("Manually set parameters")
    piscope.plumebackground.plot_sky_reference_areas(plume, current_params,
                                                     ax=axes[0])
    piscope.plumebackground.plot_sky_reference_areas(plume, auto_params,
                                                     ax=axes[1])
    axes[1].set_title("Automatically set parameters")
    
    return auto_params, fig

def plot_pcs_profiles_4_tau_images(tau0, tau1, tau2, tau3, pcs_line):
    ### Plot PCS profiles for all 4 methods
    fig, ax = plt.subplots(1,1)
    tau_imgs = [tau0, tau1, tau2, tau3]
    
    for k in range(4):
        img = tau_imgs[k]
        profile = pcs_line.get_line_profile(img)
        ax.plot(profile, "-", label=r"Mode %d: $\phi=%.3f$"
            %(BG_CORR_MODES[k], np.mean(profile)))
    
    ax.grid()
    ax.set_ylabel(r"$\tau$", fontsize=20)
    ax.set_xlim([0,632])
    ax.set_xticklabels([])
    ax.set_xlabel("PCS", fontsize=16)
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=12)
    return fig

### SCRIPT MAIN FUNCTION   
if __name__=="__main__":
    plt.close("all")

    ### Create a background model with relevant sky reference areas
    bg_model = init_background_model()
    
    ### Define exemplary plume cross section line
    pcs_line = piscope.processing.LineOnImage(x0=420,
                                              y0=655, 
                                              x1=860,
                                              y1=200,
                                              line_id = "pcs")
    
    plume, plume_vigncorr, bg = load_and_prepare_images()

    auto_params, fig0 = autosettings_vs_manual_settings(bg_model)   
    
    #Script option
    if USE_AUTO_SETTINGS:
        bg_model.update(**auto_params)
    
    ### Model 4 exemplary tau images    
    
    # list to store figures of tau plotted tau images
    _tau_figs = [] 
    
    #mask for corr mode 0 (i.e. 2D polyfit)
    mask = np.ones(plume_vigncorr.img.shape, dtype = np.float32)   
    mask[plume_vigncorr.img < POLYFIT_2D__MASK_THRESH] = 0
    
    
    ### First method: retrieve tau image using poly surface fit
    tau0 = bg_model.get_tau_image(plume_vigncorr,
                                  CORR_MODE = BG_CORR_MODES[0],
                                  surface_fit_mask = mask,
                                  surface_fit_polyorder = 1)
    
    #Plot the result and append the figure to _tau_figs                                 
    _tau_figs.append(bg_model.plot_tau_result(tau0, pcs = pcs_line.to_list()))
    
    ### Second method: scale background image to plume image in "scale" rect
    tau1 = bg_model.get_tau_image(plume, bg, CORR_MODE = BG_CORR_MODES[1])
    _tau_figs.append(bg_model.plot_tau_result(tau1, pcs = pcs_line.to_list()))
    
    ### Third method: Linear correction for radiance differences based on two 
    ### rectangles (scale, ygrad)
    tau2 = bg_model.get_tau_image(plume, bg, CORR_MODE = BG_CORR_MODES[2])
    _tau_figs.append(bg_model.plot_tau_result(tau2, pcs = pcs_line.to_list()))
    
    ### 4th method: 2nd order polynomial fit along vertical profile line
    ### For this method, determine tau on tau off and AA image
    tau3 = bg_model.get_tau_image(plume, bg, CORR_MODE = BG_CORR_MODES[3])
    _tau_figs.append(bg_model.plot_tau_result(tau3, pcs = pcs_line.to_list()))
    
    fig6 = plot_pcs_profiles_4_tau_images(tau0, tau1, tau2, tau3, pcs_line)
    
    if SAVEFIGS:
        fig0.savefig(join(SAVE_DIR, "ex3_out_1.png"))
        for k in range(len(_tau_figs)):
            _tau_figs[k].savefig(join(SAVE_DIR, "ex3_out_%d.png" %(k+2)))
        
        fig6.savefig(join(SAVE_DIR, "ex3_out_6.png"))
    
    plt.show()