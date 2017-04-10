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
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

import pyplis
from os.path import join, exists
from matplotlib.pyplot import close, show, GridSpec, figure

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list

# use PCS line defined for cross correlation analysis for the exemplary
# emission rate retrieval
from ex08_velo_crosscorr import PCS

### SCRIPT OPTONS  
PYRLEVEL = 1
PLUME_VELO_GLOB = 4.14 #m/s
PLUME_VELO_GLOB_ERR = 1.5
MMOL = 64.0638 #g/mol
CD_MIN = 2.5e17

START_INDEX = 1
STOP_INDEX = None
DO_EVAL = 1

REF_CHECK_LOWER = -5e16
REF_CHECK_UPPER = 5e16
REF_CHECK_MODE = True

#the following ROI is in the upper right image corner, where no gas occurs in
#the time series. It is used to log mean, min and max for each analysed image
#this information can be used to check, whether the plume background retrieval
#worked well
LOG_ROI_SKY = [530, 30, 600, 100] #correspond to pyrlevel 1

### RELEVANT DIRECTORIES AND PATHS

CALIB_FILE = join(SAVE_DIR, "ex06_doascalib_aa.fts")

CORR_MASK_FILE = join(SAVE_DIR, "ex07_aa_corr_mask.fts")

### SCRIPT FUNCTION DEFINITIONS        
def plot_and_save_results(ana, line_id="1. PCS", date_fmt="%H:%M"):
    fig = figure(figsize=(10,9))
    gs = GridSpec(4, 1, height_ratios = [.6, .2, .2, .2], hspace=0.05)
    ax3 = fig.add_subplot(gs[3]) 
    ax0 = fig.add_subplot(gs[0], sharex=ax3) 
    ax1 = fig.add_subplot(gs[1], sharex=ax3) 
    ax2 = fig.add_subplot(gs[2], sharex=ax3) 
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    
    
    #Get emission rate results for the PCS line 
    res0 = ana.get_results(line_id=line_id, velo_mode="glob")
    res1 = ana.get_results(line_id=line_id, velo_mode="farneback_raw")
    res2 = ana.get_results(line_id=line_id, velo_mode="farneback_histo")
    res3 = ana.get_results(line_id=line_id, velo_mode="farneback_hybrid")
    
    res0.save_txt(join(SAVE_DIR, "ex12_flux_velo_glob.txt"))
    res1.save_txt(join(SAVE_DIR, "ex12_flux_farneback_raw.txt"))
    res2.save_txt(join(SAVE_DIR, "ex12_flux_farneback_histo.txt"))
    res3.save_txt(join(SAVE_DIR, "ex12_flux_farneback_hybrid.txt"))
    
    #Plot emission rates for the different plume speed retrievals
    res0.plot(yerr=True, date_fmt=date_fmt, ls="--", marker="x", ax=ax0, 
              color="#e67300", ymin=0, alpha_err=0.05)
    res1.plot(yerr=False, ax=ax0, ls="none", marker="x", color="b", ymin=0)
    res2.plot(yerr=False, ax=ax0, ls="-", color="#ff00ff", ymin=0)
    res3.plot(yerr=True, ax=ax0, lw=2, color="g", ymin=0)
    
    #ax[0].set_title("Retrieved emission rates")
    ax0.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=12)
    ax0.grid()
    
    #Plot effective velocity retrieved from optical flow histogram analysis    
    res3.plot_velo_eff(ax=ax1, date_fmt=date_fmt, color="g")
    #ax[1].set_title("Effective plume speed (from optflow histogram analysis)")
    ax1.set_ylim([0, ax1.get_ylim()[1]])

    #Plot time series of predominant plume direction (retrieved from optical
    #flow histogram analysis and stored in object of type LocalPlumeProperties
    #which is part of plumespeed.py module
    ana.pcs_lines[line_id].plume_props.plot_directions(ax=ax2, 
                                                       date_fmt=date_fmt,
                                                       color="g")

    ax2.set_ylim([-180, 180])
    pyplis.helpers.rotate_xtick_labels(ax=ax2)
    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    #tight_layout()
    
    ax3 = ana.plot_bg_roi_vals(ax=ax3, date_fmt="%H:%M")
    #gs.tight_layout(fig, h_pad=0)#0.03)
    gs.update(hspace=0.05, top=0.97, bottom=0.07)
    return fig
    
### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    close("all")
    figs = []
    if not exists(CALIB_FILE):
        raise IOError("Calibration file could not be found at specified "
            "location:\n%s\nPlease run example 6 first")
    if not exists(CORR_MASK_FILE):
        raise IOError("Cannot find AA correction mask, please run example script"
            "7 first")  
            
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
    
    pcs = PCS.convert(to_pyrlevel=PYRLEVEL)
                                             
    ana = pyplis.fluxcalc.EmissionRateAnalysis(imglist=aa_list, 
                                               bg_roi=LOG_ROI_SKY,
                                               pcs_lines=pcs,
                                               velo_glob=PLUME_VELO_GLOB,
                                               velo_glob_err=PLUME_VELO_GLOB_ERR,
                                               ref_check_lower_lim=REF_CHECK_LOWER,
                                               ref_check_upper_lim=REF_CHECK_UPPER)
    ana.settings.ref_check_mode = REF_CHECK_MODE
    
    ana.settings.velo_modes["farneback_raw"] = True
    ana.settings.velo_modes["farneback_histo"] = True
    ana.settings.velo_modes["farneback_hybrid"] = True
    ana.settings.min_cd = CD_MIN
    if not DO_EVAL:
        #you can check the settings first
        print ana.settings 
        #plot all current PCS lines into current list image (feel free to define
        #and add more PCS lines above)
        ax = ana.plot_pcs_lines() 
        #check if optical flow works
        ana.imglist.optflow_mode = True
        aa_mask = ana.imglist.get_thresh_mask(CD_MIN)
        ana.imglist.optflow.plot_flow_histograms(line=pcs, pix_mask=aa_mask)
    else:
        ana.calc_emission_rate(start_index=START_INDEX, 
                               stop_index=STOP_INDEX)
        
        figs.append(plot_and_save_results(ana))
        
        # the EmissionRateResults class has an informative string representation
        print ana.get_results("1. PCS", "farneback_histo")
        
        if SAVEFIGS:
            for k in range(len(figs)):
                figs[k].savefig(join(SAVE_DIR, "ex12_out_%d.%s" %(k+1,FORMAT)),
                                format=FORMAT, dpi=DPI)
                               
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"