# -*- coding: utf-8 -*-
"""
pyplis example script no. 7 - Combined DOAS and cell calibration
"""
import pyplis
from os.path import join, exists
import numpy as np
from matplotlib.pyplot import close, subplots, show
from matplotlib.patches import Circle

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex05_cell_calib_auto import perform_auto_cell_calib
from ex04_prep_aa_imglist import prepare_aa_image_list

### RELEVANT DIRECTORIES AND PATHS

#fits file containing DOAS calibration information (from ex6)
CALIB_FILE = join(SAVE_DIR,
                  "pyplis_doascalib_id_aa_avg_20150916_0706_0721.fts")
                  
### SCRIPT FUNCTION DEFINITIONS    
def draw_doas_fov(fov_x, fov_y, fov_extend, ax):
    # add FOV position to plot of examplary AA image
    c = Circle((fov_x, fov_y), fov_extend, ec = "k", fc = "lime", alpha = .5)
    ax.add_artist(c)
    ax.text(fov_x, (fov_y - fov_extend*1.3), "DOAS FOV")
    ax.set_xlim([0, 1343]), ax.set_ylim([1023, 0])
    return ax

def prepare_sensitivity_corr_masks_cells(cellcalib, doasfov):
    so2_cds = cellcalib.cell_gas_cds
    masks = {}
    for cd in so2_cds:
        mask, _ = cellcalib.get_sensitivity_corr_mask(doas_fov=doasfov,
                                                      cell_cd=cd,
                                                      surface_fit_pyrlevel=2)
        masks[cd] = mask
    return masks
   
def plot_pcs_comparison(aa_init, aa_imgs_corr, pcs1, pcs2):
    fig, axes = subplots(1,2, figsize=(18, 6))
    p10 = pcs1.get_line_profile(aa_init.img)
    p20 = pcs2.get_line_profile(aa_init.img)
    
    num = len(p10)
    
    axes[0].set_title("Line %s" %pcs1.line_id)
    axes[1].set_title("Line %s" %pcs2.line_id)
    
    axes[0].plot(p10, "-", label = r"Init $\phi=%.3f$" %(sum(p10)/num))
    axes[1].plot(p20, "-", label = r"Init $\phi=%.3f$" %(sum(p20)/num))
    
    for cd, aa_corr in aa_imgs_corr.iteritems():
        p1 = pcs1.get_line_profile(aa_corr.img)
        p2 = pcs2.get_line_profile(aa_corr.img)
        
        axes[0].plot(p1, "-", label = r"Cell CD: %.2e $\phi=%.3f$"\
                                                %(cd, sum(p1)/num))
        axes[1].plot(p2, "-", label = r"Cell CD: %.2e $\phi=%.3f$"\
                                                %(cd, sum(p2)/num))
                                                    
    axes[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
    axes[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
    axes[0].grid()
    axes[1].grid()
    return fig, axes

### SCRIPT MAIN FUNCTION    
if __name__ == "__main__":
    close("all")
    
    if not exists(CALIB_FILE):
        raise IOError("Calibration file could not be found at specified "
            "location:\n %s\nYou might need to run example 6 first")

    ### Load AA list
    aa_list = prepare_aa_image_list()
    aa_list.add_gaussian_blurring(2)
    
    ### Load DOAS calbration data and FOV information (see example 6)
    doascalib = pyplis.doascalib.DoasCalibData()
    doascalib.load_from_fits(file_path=CALIB_FILE)
    doascalib.fit_calib_polynomial()
    
    ### Get DOAS FOV parameters in absolute coordinates
    fov_x, fov_y = doascalib.fov.pixel_position_center(abs_coords=True)
    fov_extend = doascalib.fov.pixel_extend(abs_coords=True)
    
    ### Load cell calibration (see example 5)
    cellcalib = perform_auto_cell_calib().calib_data["aa"]
    
    ### Define lines on image for plume profiles
    pcs1 = pyplis.processing.LineOnImage(620, 700, 940, 280,\
                                                        line_id = "center")
    pcs2 = pyplis.processing.LineOnImage(40, 40, 40, 600,\
                                                        line_id = "edge")

    ### Plot DOAS calibration polynomial
    ax0 = doascalib.plot()
    ax0 = cellcalib.plot(pos_x_abs=fov_x, pos_y_abs=fov_y,
                         radius_abs=fov_extend, ax=ax0)
                                                            
    ### Show current AA image from image list
    aa_init = aa_list.current_img()
    ax = aa_init.show()
    
    # plot the two lines into the exemplary AA image
    pcs1.plot_line_on_grid(ax = ax, c="r")
    pcs2.plot_line_on_grid(ax = ax, c="g")
    ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
    ax = draw_doas_fov(fov_x, fov_y, fov_extend, ax=ax)

    masks = prepare_sensitivity_corr_masks_cells(cellcalib, doascalib.fov)        
    aa_imgs_corr = {}    
    for cd, mask in masks.iteritems():        
        aa_imgs_corr[cd] = pyplis.Img(aa_init.img / mask)
    
    fig, _ = plot_pcs_comparison(aa_init, aa_imgs_corr, pcs1, pcs2) 
    
    ### IMPORTANT STUFF FINISHED    
    
    if SAVEFIGS:
        ax0.figure.savefig(join(SAVE_DIR, "ex07_out_1.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        ax.figure.savefig(join(SAVE_DIR, "ex07_out_2.%s" %FORMAT),
                          format=FORMAT, dpi=DPI)
        fig.savefig(join(SAVE_DIR, "ex07_out_3.%s" %FORMAT), format=FORMAT,
                    dpi=DPI)
    
    #Save the sensitivity correction mask from the cell with the lowest SO2 CD
    so2min = np.min(masks.keys())
    mask = pyplis.Img(masks[so2min])
    mask.save_as_fits(SAVE_DIR, "aa_corr_mask")
    
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"


