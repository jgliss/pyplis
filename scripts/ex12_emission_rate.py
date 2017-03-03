# -*- coding: utf-8 -*-
"""
Example script 12 - Etna emission rate retrieval

This example import results from the previous examples, for instance the AA 
image list including measurement geometry (ex 4), the DOAS calibration 
information (which was stored as FITS file, see ex. 6) and the AA sensitivity
correction mask retrieved from the cell calibration and normalised to the 
position of the DOAS FOV (ex 7). The emission rates are retrieved for three 
different plume velocity retrievals: 1. using the global velocity vector 
retrieved from the cross correlation algorithm (ex8), 2. using the raw output 
of the optical flow Farneback algorithm (``farneback_raw``) and 3. using the 
histogram based post analysis of the optical flow field (``farneback_histo``).
The analysis is performed using the EmissionRateAnalysis class which basically 
checks the AA list and activates ``calib_mode`` (-> images are loaded as 
calibrated gas CD images) and loops over all images to retrieve the emission 
rates for the 3 velocity modes. Here, emission rates are retrieved along 1 
exemplary plume cross section. This can be easily extended by adding additional
PCS lines in the EmissionRateAnalysis class using ``add_pcs_line``. 
The results for each velocity mode and for each PCS line are stored within 
EmissionRateResults classes.
"""

import pyplis
from os.path import join, exists
from matplotlib.pyplot import close, subplots, tight_layout, show

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

# use PCS line defined for cross correlation analysis for the exemplary
# emission rate retrieval
from ex08_velo_crosscorr import PCS

### SCRIPT OPTONS  
PYRLEVEL = 1
PLUME_VEL_GLOB = 4.14 #m/s
MMOL = 64.0638 #g/mol
CD_MIN = 2.5e17
DO_EVAL = 1

#the following ROI is in the upper right image corner, where no gas occurs in
#the time series. It is used to log mean, min and max for each analysed image
#this information can be used to check, whether the plume background retrieval
#worked well
LOG_ROI_SKY = [530, 30, 600, 100] #correspond to pyrlevel 1

### RELEVANT DIRECTORIES AND PATHS

CALIB_FILE = join(SAVE_DIR,
                  "pyplis_doascalib_id_aa_avg_20150916_0706_0721.fts")

CORR_MASK_FILE = join(SAVE_DIR, "aa_corr_mask.fts")

### SCRIPT FUNCTION DEFINITIONS        
def plot_results(ana, line_id = "1. PCS"):
    fig, ax = subplots(2, 1, figsize = (7, 9), sharex = True)
    
    #Get emission rate results for the PCS line 
    res0 = ana.get_results(line_id=line_id, velo_mode="glob")
    res1 = ana.get_results(line_id=line_id, velo_mode="farneback_raw")
    res2 = ana.get_results(line_id=line_id, velo_mode="farneback_histo")
    
    #Plot emission rates for the different plume speed retrievals
    res0.plot(yerr=False, ax=ax[0], color="r")
    res1.plot(yerr=False, ax=ax[0], color="b")
    res2.plot(yerr=True, ax=ax[0], color="g")
    #ax[0].set_title("Retrieved emission rates")
    ax[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=12)
    
    #Plot effective velocity retrieved from optical flow histogram analysis    
    res2.plot_velo_eff(ax=ax[1], color="g")
    #ax[1].set_title("Effective plume speed (from optflow histogram analysis)")
    ax[1].set_ylim([0, ax[1].get_ylim()[1]])
    ax2 = ax[1].twinx()
    #Plot time series of predominant plume direction (retrieved from optical
    #flow histogram analysis and stored in object of type LocalPlumeProperties
    #which is part of plumespeed.py module
    ana.plume_properties[line_id].plot_directions(ax=ax2, color="#ff9900")

    ax2.set_ylim([-180, 180])
    pyplis.helpers.rotate_xtick_labels(ax=ax[1])
    tight_layout()
    
    fig2, ax2 = subplots(1,1)
    ax2 = ana.plot_bg_roi_vals(ax = ax2)
    ax2.set_title("SO2 CD time series in scale_rect")
    return ax, ax2
    
### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    close("all")
    
    if not exists(CALIB_FILE):
        raise IOError("Calibration file could not be found at specified location:\n"
            "%s\nYou might need to run example 6 first")
        
    ### Load AA list
    aa_list = prepare_aa_image_list() #includes viewing direction corrected geometry
    aa_list.pyrlevel = PYRLEVEL
    
    ### Load DOAS calbration data and FOV information (see example 6)
    doascalib = pyplis.doascalib.DoasCalibData()
    doascalib.load_from_fits(file_path=CALIB_FILE)
    doascalib.fit_calib_polynomial()
    
    #Load AA corr mask and set in image list(is normalised to DOAS FOV see ex7)
    aa_corr_mask = pyplis.Img(CORR_MASK_FILE)
    aa_list.aa_corr_mask = aa_corr_mask
    
    #set DOAS calibration data in image list
    aa_list.calib_data = doascalib
    
    pcs = PCS.convert(pyrlevel=PYRLEVEL)
                                             
    ana = pyplis.fluxcalc.EmissionRateAnalysis(aa_list, pcs,
                                                velo_glob=PLUME_VEL_GLOB,
                                                bg_roi=LOG_ROI_SKY)
    
    ana.settings.velo_modes["farneback_raw"] = True
    ana.settings.velo_modes["farneback_histo"] = True
    ana.settings.min_cd = CD_MIN
    if not DO_EVAL:
        #you can check the settings first
        print ana.settings 
        #plot all current PCS lines into current list image (feel free to define
        #and add more PCS lines above)
        ax = ana.plot_pcs_lines() 
        #check if optical flow works
        ana.imglist.optflow_mode = True
        ana.imglist.optflow.plot_flow_histograms()
    else:
        ana.calc_emission_rate(start_index=1)
        
        ax0, ax1 = plot_results(ana)
        
        if SAVEFIGS:
            ax0[0].figure.savefig(join(SAVE_DIR, "ex12_out_1.%s" %FORMAT),
                               format=FORMAT, dpi=DPI)
            ax1.figure.savefig(join(SAVE_DIR, "ex12_out_2.%s" %FORMAT),
                               format=FORMAT, dpi=DPI)
                               
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"