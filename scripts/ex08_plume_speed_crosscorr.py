# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - plume speed via cross correlation
"""
from matplotlib.pyplot import close, show
from os.path import join
import numpy as np

from piscope.processing import LineOnImage, ProfileTimeSeriesImg
from piscope.plumespeed import find_signal_correlation

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI

### IMPORTS FROM OTHER EXAMPLE SCRIPTS
from ex04_prepare_aa_imglist import prepare_aa_image_list

### SCRIPT OPTONS  
RELOAD = 0 #reload AA profile images for PCS lines 

### RELEVANT DIRECTORIES AND PATHS

FIRST_ICA_TSERIES = join(SAVE_DIR, "first_ica_tseries.fts")
SECOND_ICA_TSERIES = join(SAVE_DIR, "second_ica_tseries.fts")

### SCRIPT FUNCTION DEFINITIONS
def create_pcs_lines():
    #Create PCS line
    pcs1 = LineOnImage(150, 180, 215, 75, normal_orientation = "left",\
    pyrlevel = 2, line_id = "l1")
    #these coordinates correspond to a cross section originally defined 
    #from a downscaled plume image at pyrlevel=2, therefore
    #convert line to pyrlevel 0
    pcs1 = pcs1.convert(pyrlevel=0)
    # create a second line at 40 pixels distance to the first line 
    # (in direction of its normal vector)
    pcs2 = pcs1.offset(pixel_num=40)    
    return pcs1, pcs2
    
def reload_profile_tseries_from_aa_list(aa_list, pcs1, pcs2, dist_img):
    """Reload profile time series pictures from AA img list
    
    Loop over all images in AA image list and extract AA profiles for each 
    of the 2 provided plume cross section lines. The profiles are written into
    a :class:`ProfileTimeSeriesImg` which can be stored as FITS file 
    """
    # go to first image in list
    aa_list.goto_img(0)
        
    # get the number of datapoints of the first profile line (the second one
    # has the same length in this case, since it was created from the first 
    # one using pcs1.offset(pixel_num=40), see above..
    profile_len = len(pcs1.get_line_profile(dist_img.img))
    
    # now create two empty 2D numpy arrays with height == profile_len and
    # width == number of images in aa_list
    profiles1 = np.empty((profile_len, aa_list.nof), dtype = np.float32)
    profiles2 = np.empty((profile_len, aa_list.nof), dtype = np.float32)
    
    # for each of the 2 lines, extract pixel to pixel distances from the 
    # provided dist_img (comes from measurement geometry) which is required 
    # in order to perform integration along the profiles 
    dists_pcs1 = pcs1.get_line_profile(dist_img.img) #pix to pix dists line 1
    dists_pcs2 = pcs2.get_line_profile(dist_img.img) #pix to pix dists line 2

    # loop over all images in list, extract profiles and write in the 
    # corresponding column of the profile picture
    for k in range(aa_list.nof):
        print "Loading AA profiles from list: %d (%d)" %(k, aa_list.nof)
        aa = aa_list.current_img().img
        profiles1[:, k] = pcs1.get_line_profile(aa)
        profiles2[:, k] = pcs2.get_line_profile(aa)
        
        aa_list.next_img()
     
    #mutiply pix to pix dists to the AA profiles in the 2 images
    profiles1 = profiles1 * dists_pcs1.reshape((len(dists_pcs1), 1)) 
    profiles2 = profiles2 * dists_pcs2.reshape((len(dists_pcs2), 1)) 
    
    # extract acquisition time stamps from image list 
    times = aa_list.acq_times
    
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
    prof_pic1.save_as_fits(SAVE_DIR, "first_ica_tseries.fts")
    prof_pic2.save_as_fits(SAVE_DIR, "second_ica_tseries.fts")
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
    icas1 = np.sum(prof_pic1.img, axis = 0)
    icas2 = np.sum(prof_pic2.img, axis = 0)
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
    ax0 = img.show()
    ax0.set_title("Profile lines for cross correlation analysis")
    
    #Create two PCS lines for cross correlation analysis
    pcs1, pcs2 = create_pcs_lines()
    
    #plot the two PCS lines (with normal vector) into AA image
    pcs1.plot_line_on_grid(ax = ax0, include_normal=True, c = "b")    
    pcs2.plot_line_on_grid(ax = ax0, include_normal=True, c = "g")
    ax0.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
    
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
                                        cut_border_idx=10, reg_grid_tres=100, 
                                        freq_unit="L", sigma_smooth=2,
                                        plot=True) 
    ax1 = result[-1][0]
    tit = "Retrieved plume velocity of v = %.2f m/s" %v
    ax1.figure.suptitle(tit)
    print tit
    
    if SAVEFIGS:
        ax0.figure.savefig(join(SAVE_DIR, "ex08_out_1.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        ax1.figure.savefig(join(SAVE_DIR, "ex08_out_2.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
                           
    show()
    