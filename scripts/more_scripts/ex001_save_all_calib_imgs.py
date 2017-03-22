# -*- coding: utf-8 -*-
"""
example script 001 - Save plots of all SO2 CD images of test dataset

This script prepares an AA image list for calibration (using DOAS calib
data) and loops over the list to save all SO2 CD images
"""
from os.path import join, exists
from os import mkdir
import sys
import pyplis

from matplotlib import rcParams
rcParams.update({'font.size': 15})

sys.path.append(join(".."))

### Plot settings
CDMIN =-1.8e18
CDMAX = 1.8e18
DPI = 150 
FORMAT = "png"

### OPTIONS
CROP_IDX = 3

TEST = 0 #save only first image in time series
BLURRING = 1 #Gaussian blurring applied to images
PYRLEVEL = 1 #Scale space level (Gauss pyramid)

### Relevant paths
SAVE_DIR = join(pyplis._LIBDIR, "..", "scripts", "scripts_out", "all_cd_imgs")

if not exists(SAVE_DIR):
    mkdir(SAVE_DIR)

### Imports from other example scripts
from ex04_prep_aa_imglist import prepare_aa_image_list 

CALIB_FILE = join("..","scripts_out",
                  "pyplis_doascalib_id_aa_avg_20150916_0706_0721.fts")
if not exists(CALIB_FILE):
    raise IOError("path to CALIB_FILE not found %s" %CALIB_FILE)
    

    
CORR_MASK_FILE = join("..","scripts_out", "aa_corr_mask.fts")
if not exists(CORR_MASK_FILE):
    raise IOError("path to CORR_MASK_FILE not found %s" %CORR_MASK_FILE)
    
### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    from matplotlib.pyplot import close
    ### Load AA list
    aa_list = prepare_aa_image_list() 
    aa_list.add_gaussian_blurring(BLURRING)
    aa_list.pyrlevel = PYRLEVEL
    
    ### Load DOAS calbration data and FOV information (see example 6)
    doascalib = pyplis.doascalib.DoasCalibData()
    doascalib.load_from_fits(file_path=CALIB_FILE)
    doascalib.fit_calib_polynomial()
    
    #Load AA corr mask and set in image list(is normalised to DOAS FOV see ex7)
    aa_corr_mask = pyplis.Img(CORR_MASK_FILE)
    aa_list.aa_corr_mask = aa_corr_mask
    aa_list.sensitivity_corr_mode = True
    
    #set DOAS calibration data in image list
    aa_list.calib_data = doascalib
    aa_list.calib_mode = True
    
    aa_list.goto_img(CROP_IDX)
    
    ax = aa_list.current_img().show(zlabel=r"$S_{SO2}$ [cm$^{-2}$]", 
                                  vmin=CDMIN, vmax=CDMAX)
    if TEST:
        ax.set_title(aa_list.current_time_str(), fontsize=18)
        ax.figure.savefig(join(SAVE_DIR, "TEST.%s" %FORMAT),format=FORMAT, 
                          dpi=DPI)
    else:
        
        for k in range(1, aa_list.nof - 2*CROP_IDX):
            ax.set_title(aa_list.current_time_str(), fontsize=18)
            ax.figure.savefig(join(SAVE_DIR, "%d.%s" %(k, FORMAT)),
                              format=FORMAT, dpi=DPI)
            close("all")
            aa_list.next_img()
            ax = aa_list.current_img().show(zlabel=r"$S_{SO2}$ [cm$^{-2}$]", 
                                  vmin=CDMIN, vmax=CDMAX)
            