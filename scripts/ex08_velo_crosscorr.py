# -*- coding: utf-8 -*-
"""
pyplis example script no. 8 - Plume velocity retrieval using cross correlation
"""
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

from matplotlib.pyplot import close, show, subplots
from os.path import join

from pyplis.plumespeed import VeloCrossCorrEngine

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE, LINES

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

### SCRIPT OPTONS  
OFFSET_PIXNUM = 40 # distance in pixels between two lines used for cross corr analysis
RELOAD = 0 #reload AA profile images for PCS lines 

# start / stop indices of considered images in image list (only relevant if PCS profiles are
# reloaded, i.e. Opt RELOAD=True)
START_IDX = 10
STOP_IDX = 200

# PCS line for which velocity is supposed to retrieved
PCS = LINES[0] # orange "young_plume" line

# Color of PCS offset line used to perform cross correlation analysis (relevant for illustration)
COLOR_PCS_OFFS = "c"
  
### RELEVANT DIRECTORIES AND PATHS
# the time series of PCS profiles for both lines are stored as
# ProfileTimeSeriesImg objects using the following names. The images 
# contain the PCS profiles (y-axis) for each image in the list (y-axis)
# and have thus dimension MXN where M denotes the pixel number of the lines
# and N denotes the total number of images from which the profiles are 
# extracted. The images will be stored in SAVE_DIR after this script is run
# once. After that, re-running the script and applying the cross-correlation
# analysis will be much faster, since the profiles are imported from the 
# two precomupted images and do not need to be extracted by looping over
# the image list.
PCS_PROFILES_PIC_NAME = "ex08_ica_tseries_pcs.fts"
OFFSET_PROFILES_PIC_NAME = "ex08_ica_tseries_offset.fts"
    

### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    axes=[]
    #prepare the AA image list (see ex4)
    aa_list = prepare_aa_image_list()
    aa_list.pyrlevel=1
    
    cc = VeloCrossCorrEngine(aa_list, PCS)
    cc.create_parallel_pcs_offset(offset_pix=40,
                                  color=COLOR_PCS_OFFS,
                                  linestyle="--")
    
    fig, ax = subplots(1,2, figsize=(20,6))
    axes.append(cc.plot_pcs_lines(ax=ax[0]))
    try:
        cc.load_pcs_profile_img(join(SAVE_DIR, PCS_PROFILES_PIC_NAME),
                                line_id="pcs")
        cc.load_pcs_profile_img(join(SAVE_DIR, OFFSET_PROFILES_PIC_NAME), 
                                line_id="pcs_offset")
    except:
        cc.get_pcs_tseries_from_list(start_idx=START_IDX,
                                     stop_idx=STOP_IDX)
        cc.save_pcs_profile_images(save_dir=SAVE_DIR, 
                                   fname1=PCS_PROFILES_PIC_NAME,
                                   fname2=OFFSET_PROFILES_PIC_NAME)
    
    # the run method of the high level VeloCrossCorrEngine class is 
    # basically a wrapper method for the low-level find_signal_correlation
    # function which is part of the plumespeed.py module. Before calling
    # the latter, the ICA time-series are extracted from the two 
    # ProfileTimeSeriesImg objects which were computed above from the 
    # ImgList class containing AA images, and which are stored as FITS
    # files for fast re-computing of this script. The following run 
    # command passes valid input parameters to the find_signal_correlation
    # method.
    velo = cc.run(cut_border_idx=10, 
                  reg_grid_tres=100, 
                  freq_unit="L", 
                  sigma_smooth=2,
                  plot=0)
    
    cc.plot_ica_tseries_overlay(ax=ax[1])
    axes.append(cc.plot_corrcoeff_tseries())
    
    print "Retrieved plume velocity of v = %.2f m/s" %velo
     
# =============================================================================
#     print ("PCS (load): %s\nPCS (orig): %s\nOffset (load): %s\nOffset (Orig): %s" 
#     %(cc.pcs, L0, cc.pcs_offset, L1))
# =============================================================================
    ### IMPORTANT STUFF FINISHED    
    if SAVEFIGS:
        for k in range(len(axes)):
            axes[k].figure.savefig(join(SAVE_DIR, "ex08_out_%d.%s"
                                        %((k+1), FORMAT)),
                                        format=FORMAT, dpi=DPI)
             
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"