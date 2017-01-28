# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - plume speed via cross correlation
"""
from matplotlib.pyplot import close, show
from os.path import join
import numpy as np

from piscope.processing import LineOnImage, ProfileTimeSeriesImg
from piscope.plumespeed import find_signal_correlation

from ex2_measurement_geometry import create_dataset, correct_viewing_direction
from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
        path_bg_off, save_path

p1 = join(save_path, "first_ica_tseries.fts")
p2 = join(save_path, "second_ica_tseries.fts")

RELOAD = 0

def create_pcs_lines():
    #these coordinates correspond to a cross section defined 
    #at pyrlevel=2
    pcs1 = LineOnImage(150, 180, 215, 75, normal_orientation = "left",\
    pyrlevel = 2, line_id = "l1")
    
    #convert line to pyrlevel 0
    pcs1 = pcs1.convert(pyrlevel=0)
    pcs2 = pcs1.offset(pixel_num = 40)    
    return pcs1, pcs2
    
def reload_profile_tseries_from_aa_list(aa_list, pcs1, pcs2):
    """Reload profile time series pictures from AA img list
    
    .. note::
    
        This takes some time
    """
    aa_list.goto_img(0)
        
    profile_len = len(pcs1.get_line_profile(dist_img.img))
    profiles1 = np.empty((profile_len, aa_list.nof), dtype = np.float32)
    profiles2 = np.empty((profile_len, aa_list.nof), dtype = np.float32)
    d1 = pcs1.get_line_profile(dist_img.img) #pix to pix dists line 1
    d2 = pcs2.get_line_profile(dist_img.img) #pix to pix dists line 2
    #loop over all images in list
    for k in range(aa_list.nof):
        print k
        aa = aa_list.current_img().img
        profiles1[:, k] = pcs1.get_line_profile(aa)
        profiles2[:, k] = pcs2.get_line_profile(aa)
        
        aa_list.next_img()
        
    times = aa_list.acq_times
    prof_pic1 = ProfileTimeSeriesImg(profiles1, time_stamps=times,\
            img_id = pcs1.line_id, profile_info_dict=pcs1.to_dict(),\
                                        **aa_list.current_img().edit_log)
    prof_pic2 = ProfileTimeSeriesImg(profiles2, time_stamps=times, img_id =\
                            pcs2.line_id, profile_info_dict=pcs2.to_dict(),\
                                            **aa_list.current_img().edit_log)
                                            
    #mutiply pix to pix dists to the AA profiles in the image
    prof_pic1.img = prof_pic1.img * d1.reshape((len(d1), 1)) 
    prof_pic2.img = prof_pic2.img * d2.reshape((len(d2), 1)) 
    
    prof_pic1.save_as_fits(save_path, "first_ica_tseries.fts")
    prof_pic2.save_as_fits(save_path, "second_ica_tseries.fts")
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

if __name__ == "__main__":
    close("all")
    ds = create_dataset()
    geom, basemap = correct_viewing_direction(ds.meas_geometry)
    
    #get pix-to-pix and plume distance image
    dist_img, plume_dist_img = geom.get_all_pix_to_pix_dists()
    
    #prepare the AA image list (see ex4)
    aa_list = prepare_aa_image_list(ds, path_bg_on, path_bg_off)
    
    #draw current AA image 
    img = aa_list.current_img()
    img.add_gaussian_blurring(1)
    ax = img.show()
    
    #Create two PCS lines for cross correlation analysis
    pcs1, pcs2 = create_pcs_lines()
    
    #plot the two PCS lines (with normal vector) into AA image
    pcs1.plot_line_on_grid(ax = ax, include_normal=True, c = "b")    
    pcs2.plot_line_on_grid(ax = ax, include_normal=True, c = "g")
    ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
    
    if not RELOAD:
        try:
            prof_pic1, prof_pic2 = load_profile_tseries_from_fits(p1, p2)
        except:
            RELOAD = 1
    if RELOAD:
        prof_pic1, prof_pic2 = reload_profile_tseries_from_aa_list(\
            aa_list, pcs1, pcs2)
    
    # Apply cross correlation analysis to the two            
    v, res = apply_cross_correlation(prof_pic1, prof_pic2, dist_img,\
            cut_border_idx = 10, reg_grid_tres = 100, freq_unit = "L",\
                                            sigma_smooth = 2, plot = True) 
    ax = res[-1][0]
    tit = "Retrieved plume velocity of v = %.2f m/s" %v
    ax.figure.suptitle(tit)
    print tit
    show()
    