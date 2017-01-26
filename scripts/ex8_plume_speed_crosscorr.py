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
    """
    icas1 = np.sum(prof_pic1.img, axis = 0)
    icas2 = np.sum(prof_pic2.img, axis = 0)
    times = prof_pic1.time_stamps
    #lag, coeffs, s1, s2, max_coeff_signal, ax = 
    res = find_signal_correlation(icas1, icas2, times, **kwargs)
    lag = res[0]
    pix_dist_avg = np.mean([pcs1.get_line_profile(dist_img.img).mean(),\
                            pcs2.get_line_profile(dist_img.img).mean()])
    v = 40 * pix_dist_avg / lag 
    return v, res

if __name__ == "__main__":
    close("all")
    ds = create_dataset()
    geom, m = correct_viewing_direction(ds.meas_geometry)
    
    dist_img, plume_dist_img = geom.get_all_pix_to_pix_dists()
    
    aa_list = prepare_aa_image_list(ds, path_bg_on, path_bg_off)
    
    ax = aa_list.show_current()
    
    pcs1, pcs2 = create_pcs_lines()
    
    pcs1.plot_line_on_grid(ax = ax, include_normal=True)    
    pcs2.plot_line_on_grid(ax = ax, include_normal=True)
    h, w = aa_list.current_img().shape
    ax.set_xlim([0, w - 1])
    ax.set_ylim([h -1, 0])
    ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
    if not RELOAD:
        try:
            prof_pic1, prof_pic2 = load_profile_tseries_from_fits(p1, p2)
        except:
            RELOAD = 1
    if RELOAD:
        prof_pic1, prof_pic2 = reload_profile_tseries_from_aa_list(\
            aa_list, pcs1, pcs2)
            
    v0, res0 = apply_cross_correlation(prof_pic1, prof_pic2, dist_img,\
            cut_border_idx = 10, reg_grid_tres = 100, freq_unit = "L",\
                                            sigma_smooth = 2, plot = True) 
    v1, res1 = apply_cross_correlation(prof_pic1, prof_pic2, dist_img,\
                                            sigma_smooth = 0, plot = True) 
    res0[-1][0].figure.suptitle("Retrieved plume velocity v = %.2f m/s" %v0)
    res1[-1][0].figure.suptitle("Retrieved plume velocity v = %.2f m/s" %v1)
    show()
    

#==============================================================================
# print "Avg. plume velocity = %.2f m / s"\
#             %(pcs_offset * dist_img.mean() / lag)
#==============================================================================
