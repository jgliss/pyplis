# -*- coding: utf-8 -*-
"""
pyplis example script no. 9 - Plume velocity retrieval using Farneback optical 
flow algorithm
"""
from matplotlib.pyplot import close, show, subplots
from os.path import join, basename
import pyplis

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prep_aa_imglist import prepare_aa_image_list
SCRIPT_ID = basename(__file__).split("_")[0]

### SCRIPT OPTONS  

# perform histogram analysis for all images in time series
HISTO_ANALYSIS_ALL = 1
HISTO_ANALYSIS_START_IDX = 1
HISTO_ANALYSIS_STOP_IDX = 207#10

#Gauss pyramid leve
PYRLEVEL = 1
BLUR = 0
ROI_CONTRAST = [0, 0, 1344, 730]
MIN_AA = 0.05

PCS1 = pyplis.LineOnImage(345, 350, 450, 195, pyrlevel_def=1, 
                          line_id="young_plume", color="g")
PCS2 = pyplis.LineOnImage(80, 10, 80, 270, pyrlevel_def=1, 
                          line_id="old_plume", color="r")
    
LINES = [PCS1, PCS2]
   
def analyse_and_plot(lst, lines):
    mask = lst.get_thresh_mask(MIN_AA)
    fl=lst.optflow
    fig, ax = subplots(1,3, figsize=(22,6))
    fl.plot(ax=ax[0])#, in_roi=True)
    for line in lines:
        m = mask * line.get_rotated_roi_mask(fl.flow.shape[:2])
        line.plot_line_on_grid(ax=ax[0], include_roi_rot=1)
        #try:
        _, mu, sigma = fl.plot_orientation_histo(pix_mask=m, 
                                                 apply_fit=True, ax=ax[1], 
                                                 color=line.color)
        low, high = mu-sigma, mu+sigma
        fl.plot_length_histo(pix_mask=m, apply_fit=False, ax=ax[2], 
                             dir_low=low, dir_high=high, color=line.color)
        
    ax[1].set_title("Orientation histograms")
    ax[1].set_xlabel(r"$\Theta\,[^{\circ}]$", fontsize=14)
    ax[1].set_ylabel(r"Count", fontsize=14)
    ax[2].set_title("Length histograms")
    ax[2].set_xlabel(r"Magnitude [pix]", fontsize=14)
    ax[2].set_ylabel(r"Count", fontsize=14)
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])
    fig.tight_layout()
    return fig

    
### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    figs = []
    # Prepare aa image list (see example 4)
    aa_list = prepare_aa_image_list()
    
    # the aa image list includes the measurement geometry, get pixel
    # distance image where pixel values correspond to step widths in the plume, 
    # obviously, the distance values depend on the downscaling factor, which
    # is calculated from the analysis pyramid level (PYRLEVEL)
    dist_img, _, _ = aa_list.meas_geometry.get_all_pix_to_pix_dists(
                                            pyrlevel=PYRLEVEL)
    # set the pyramid level in the list
    aa_list.pyrlevel = PYRLEVEL
    # add some blurring.. or not (if BLUR = 0)
    aa_list.add_gaussian_blurring(BLUR)
    
    # Access to the optical flow module in the image list. If optflow_mode is 
    # active in the list, then, whenever the list index changes (e.g. using
    # list.next_img(), or list.goto_img(100)), the optical flow field is 
    # calculated between the current list image and the next one
    fl = aa_list.optflow 
    #(! note: fl is only a pointer, i.e. the "=" is not making a copy of the 
    # object, meaning, that whenever something changes in "fl", it also does
    # in "aa_list.optflow")
    
    # Now activate optical flow calculation in list (this slows down the 
    # speed of the analysis, since the optical flow calculation is 
    # comparatively slow    
    s = aa_list.optflow.settings
    s.hist_dir_gnum_max = 10
    s.hist_dir_binres = 10
    s.roi_rad = ROI_CONTRAST
    

    aa_list.optflow_mode = True
    
    plume_mask = pyplis.Img(aa_list.get_thresh_mask(MIN_AA))
    plume_mask.show(tit="AA threshold mask")
    
    figs.append(analyse_and_plot(aa_list, LINES))
        
    figs.append(fl.plot_flow_histograms(PCS1, plume_mask.img))
    figs.append(fl.plot_flow_histograms(PCS2, plume_mask.img))
    
    #Show an image containing plume speed magnitudes (ignoring direction)
    velo_img = pyplis.Img(fl.to_plume_speed(dist_img))
    velo_img.show(vmin=0, vmax = 10, cmap = "Greens",
                  tit = "Optical flow plume velocities",
                  zlabel ="Plume velo [m/s]")
    
    # Create two objects used to store time series information about the 
    # retrieved plume properties
    plume_props_l1 = pyplis.plumespeed.LocalPlumeProperties(PCS1.line_id)
    plume_props_l2 = pyplis.plumespeed.LocalPlumeProperties(PCS2.line_id)
    
    if HISTO_ANALYSIS_ALL:
        for k in range(HISTO_ANALYSIS_START_IDX, HISTO_ANALYSIS_STOP_IDX):
            plume_mask = aa_list.get_thresh_mask(MIN_AA)
            plume_props_l1.get_and_append_from_farneback(fl, line=PCS1,
                                                         pix_mask=plume_mask)
            plume_props_l2.get_and_append_from_farneback(fl, line=PCS2,
                                                         pix_mask=plume_mask)
            aa_list.next_img()
            
        fig, ax = subplots(2, 1, figsize=(10, 9))
    
        plume_props_l1.plot_directions(ax=ax[0], 
                                       color=PCS1.color, 
                                       label="PCS1")
        plume_props_l2.plot_directions(ax=ax[0], color=PCS2.color, 
                                       label="PCS2")
       
        plume_props_l1.plot_magnitudes(normalised=True, ax=ax[1], 
                                      date_fmt="%H:%M:%S", color=PCS1.color, 
                                      label="PCS1")
        plume_props_l2.plot_magnitudes(normalised=True, ax=ax[1], 
                                      date_fmt="%H:%M:%S", color=PCS2.color, 
                                      label="PCS2") 
        ax[0].set_xticklabels([])
        ax[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=14) 
        ax[0].set_title("Movement direction")
        ax[1].set_title("Displacement length")
        figs.append(fig)
        
    if SAVEFIGS:
        for k in range(len(figs)):
            figs[k].savefig(join(SAVE_DIR, "%s_out_%d.%s" 
                            %(SCRIPT_ID, (k+1), FORMAT)),
                            format=FORMAT, dpi=DPI)
    
    # Save the time series as txt
    plume_props_l1.save_txt(SAVE_DIR)
    plume_props_l2.save_txt(SAVE_DIR)

    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"
        
        
        
        
