# -*- coding: utf-8 -*-
"""
pyplis example script no. 8 - Plume velocity retrieval using cross correlation
"""
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

from matplotlib.pyplot import close, show, subplots
from matplotlib.dates import DateFormatter
from os.path import join
import numpy as np

from pyplis.processing import ProfileTimeSeriesImg
from pyplis.plumespeed import find_signal_correlation

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
# the time series of PCS profiles for both lines are stored as profile image
# this accelarates re-execution of this script
FIRST_ICA_TSERIES = join(SAVE_DIR, "ex08_1st_ica_tseries.fts")
SECOND_ICA_TSERIES = join(SAVE_DIR, "ex08_2nd_ica_tseries.fts")

### SCRIPT FUNCTION DEFINITIONS
def init_lines(pcs=PCS, offset_pixnum=OFFSET_PIXNUM, color_offs=COLOR_PCS_OFFS):
    """Map PCS line to analysis pyramide level 0 and create offset line used for analysis"""
    #convert line to pyrlevel 0
    pcs = pcs.convert(to_pyrlevel=0)
    pcs.linestyle="--"
    # create a second line at 40 pixels distance to the first line 
    # (in direction of its normal vector)
    pcs_offs = pcs.offset(pixel_num=offset_pixnum)    
    pcs_offs.line_id = "Offset line"
    pcs_offs.color = color_offs
    return (pcs, pcs_offs)
    
def reload_profile_tseries_from_aa_list(aa_list, pcs1, pcs2, dist_img,
                                        start_idx=START_IDX, stop_idx=STOP_IDX):
    """Reload profile time series pictures from AA img list
    
    Loop over all images in AA image list and extract AA profiles for each 
    of the 2 provided plume cross section lines. The profiles are written into
    a :class:`ProfileTimeSeriesImg` which can be stored as FITS file 
    """
    # go to first image in list
    aa_list.goto_img(start_idx)
        
    if stop_idx is None:
        stop_idx = aa_list.nof
        
    num = stop_idx - start_idx
    if not num > 20:
        raise ValueError("Please set start / stop indices such that at least 20 images are used")
    # get the number of datapoints of the first profile line (the second one
    # has the same length in this case, since it was created from the first 
    # one using pcs1.offset(pixel_num=40), see above..
    profile_len = len(pcs1.get_line_profile(dist_img.img))
    
    # now create two empty 2D numpy arrays with height == profile_len and
    # width == number of images in aa_list
    profiles1 = np.empty((profile_len, num), dtype = np.float32)
    profiles2 = np.empty((profile_len, num), dtype = np.float32)
    
    # for each of the 2 lines, extract pixel to pixel distances from the 
    # provided dist_img (comes from measurement geometry) which is required 
    # in order to perform integration along the profiles 
    dists_pcs1 = pcs1.get_line_profile(dist_img.img) #pix to pix dists line 1
    dists_pcs2 = pcs2.get_line_profile(dist_img.img) #pix to pix dists line 2

    # loop over all images in list, extract profiles and write in the 
    # corresponding column of the profile picture
    times=[]
    for k in range(num):
        if k % 25 == 0:
            print "Loading AA profiles from list: %d (%d)" %(k, num)
        aa = aa_list.current_img().img
        profiles1[:, k] = pcs1.get_line_profile(aa)
        profiles2[:, k] = pcs2.get_line_profile(aa)
        times.append(aa_list.current_time())
        aa_list.next_img()
     
    #mutiply pix to pix dists to the AA profiles in the 2 images
    profiles1 = profiles1 * dists_pcs1.reshape((len(dists_pcs1), 1)) 
    profiles2 = profiles2 * dists_pcs2.reshape((len(dists_pcs2), 1)) 
    
    # Get dictionary containing image preparation information
    img_prep = aa_list.current_img().edit_log
    
    # now create 2 ProfileTimeSeriesImg objects from the 2 just determined
    # images and include meta information (e.g. time stamp vector, image
    # preparation information)
    prof_pic1 = ProfileTimeSeriesImg(profiles1, time_stamps=times,
                                     img_id=pcs1.line_id,
                                     profile_info_dict=pcs1.to_dict(),
                                     **img_prep) 
                                     
    prof_pic2 = ProfileTimeSeriesImg(profiles2, time_stamps=times,
                                     img_id=pcs2.line_id, 
                                     profile_info_dict=pcs2.to_dict(),
                                     **img_prep)
                                            

    # save the two profile pics (these files are used in the main function
    # of this script in case they exist and option RELOAD = 0)
    prof_pic1.save_as_fits(SAVE_DIR, "ex08_1st_ica_tseries.fts")
    prof_pic2.save_as_fits(SAVE_DIR, "ex08_2nd_ica_tseries.fts")
    return prof_pic1, prof_pic2

def load_profile_tseries_from_fits(fits_path1, fits_path2):
    """Load profile time series pictures from fits file
    
    :param str fits_path1: path of first line
    :param str fits_path2: path of second line
    """
    prof_pic1 = ProfileTimeSeriesImg()
    prof_pic1.load_fits(fits_path1)
    prof_pic2 = ProfileTimeSeriesImg()
    prof_pic2.load_fits(fits_path2)
    return prof_pic1, prof_pic2
    
def apply_cross_correlation(prof_pic1, prof_pic2, dist_img, **kwargs):
    """Applies correlation search algorighm to ICA time series of both lines
    
    :param prof_pic1: first profile time series picture
    :param prof_pic2: second profile time series picture
    """
    # Integrate the profiles for each image (y axis in profile images)
    icas1 = np.sum(prof_pic1.img, axis=0)
    icas2 = np.sum(prof_pic2.img, axis=0)
    times = prof_pic1.time_stamps
    
    res = find_signal_correlation(icas1, icas2, times, **kwargs)
    lag = res[0]
    
    #Average pix-to-pix distances for both lines
    pix_dist_avg_line1 = pcs1.get_line_profile(dist_img.img).mean()
    pix_dist_avg_line2 = pcs2.get_line_profile(dist_img.img).mean()
    
    #Take the mean of those to determine distance between both lines in m
    pix_dist_avg = np.mean([pix_dist_avg_line1, pix_dist_avg_line2])
    
    #the lines are 40 pixels apart from each other, this yields the velocity
    v = 40 * pix_dist_avg / lag 
    return v, res

def plot_result(crosscorr_res, pcs1, pcs2, example_img):
    from matplotlib.pyplot import rc_context
    rc_context({'font.size':'18'})
    fig, ax = subplots(1,2, figsize=(18,6))
    ### Plot image with lines in it
    ax0 = example_img.show(ax=ax[0], zlabel=r"$\tau_{AA}$",
                           zlabel_size=20)
    ax0.set_title("")
    #plot the two PCS lines (with normal vector) into AA image
    pcs1.plot_line_on_grid(ax=ax0, include_normal=True)    
    pcs2.plot_line_on_grid(ax=ax0, include_normal=True)
    ax0.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=14) 
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    
    lag, coeffs, tseries1, tseries2, tseries_shift, _ = crosscorr_res
    
    ax2=ax[1]
    
    tseries1.index.to_pydatetime()    
    #plot original ICA time series along pcs 1
    tseries1.plot(ax=ax2, style="--", color=pcs1.color, 
                  label="%s (original)" %pcs1.line_id)
    # plot shifted time series along pcs 1 and apply light fill
    tseries_shift.plot(ax=ax2, style="-", color=pcs1.color, 
                       label="%s (shift %.1f s)" %(PCS.line_id, lag))
    ax2.fill_between(tseries_shift.index, tseries_shift.values, 
                     color=pcs1.color, alpha=0.05)
    
    tseries2.plot(ax=ax2, style="-", color=COLOR_PCS_OFFS, label=pcs2.line_id)
    
    ax2.set_ylim([40, 100])
    ax2.set_ylabel("ICA [m]")
    ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    #ax[0,].set_title("Original time series", fontsize = 10)
    ax2.grid()
    ax2.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=12) 
    
    fig.tight_layout()
    ### Plot correclation signal
    fig1, ax1 = subplots(1,1)
    x = np.arange(0, len(coeffs), 1) * lag / np.argmax(coeffs)
    ax1.plot(x, coeffs, "-r")
    ax1.set_xlabel(r"$\Delta$t [s]")
    ax1.grid()
    #ax[1].set_xlabel("Shift")
    ax1.set_ylabel("Pearson correlation")
    ax1.set_xlim([0, 300])
    
    return [ax0, ax1]
    
    
### SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    
    #prepare the AA image list (see ex4)
    aa_list = prepare_aa_image_list()
    
    #get pixel distance image
    dist_img, _, _ = aa_list.meas_geometry.get_all_pix_to_pix_dists()

    #draw current AA image 
    img = aa_list.current_img()
    img.add_gaussian_blurring(1)
    
    #Create two PCS lines for cross correlation analysis
    pcs1, pcs2 = init_lines(PCS)
    
    if not RELOAD:
        try:
            prof_pic1, prof_pic2 =\
            load_profile_tseries_from_fits(FIRST_ICA_TSERIES,
                                           SECOND_ICA_TSERIES)
        except:
            RELOAD = 1
    if RELOAD:
        prof_pic1, prof_pic2 = reload_profile_tseries_from_aa_list(aa_list,
                                                                   pcs1, pcs2,
                                                                   dist_img)

    # Apply cross correlation analysis between the signals retrieved
    # between the two cross section lines which are stored as profile images           
    v, result = apply_cross_correlation(prof_pic1, prof_pic2, dist_img,
                                        cut_border_idx=10, 
                                        reg_grid_tres=100, 
                                        freq_unit="L", sigma_smooth=2,
                                        plot=0) 
    
    axes = plot_result(result, pcs1, pcs2, img)
    print "Retrieved plume velocity of v = %.2f m/s" %v
    
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
    