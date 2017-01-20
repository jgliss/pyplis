# -*- coding: utf-8 -*-
"""
piscope example script no. 8 - optical flow analysis
"""
from matplotlib.pyplot import close, show, subplots
from scipy.signal import correlate
from piscope import Img
import numpy as np
from scipy.ndimage.filters import gaussian_filter    

from piscope.processing import LineOnImage
from piscope.plumespeed import determine_ica_cross_correlation

RELOAD = 1
    
sigma = 3
first_idx = 30
last_idx = 190
shift = 40
close("all")

if RELOAD:
    from ex1_measurement_setup_plume_data import create_dataset
    from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
        path_bg_off

    
    ds = create_dataset()
    
    dist_img, plume_dist_img = ds.meas_geometry.get_all_pix_to_pix_dists()
    
    aa_list = prepare_aa_image_list(ds, path_bg_on, path_bg_off)
    
    
    aa_list.pyrlevel = 0
    ax = aa_list.show_current()
    
    pcs1 = LineOnImage(150, 180, 215, 75, normal_orientation = "left",\
        pyrlevel = 2, line_id="l1")

    pcs1 = pcs1.convert(pyrlevel=0)
    pcs1.plot_line_on_grid(ax = ax)
    
    pcs2 = pcs1.offset(pixel_num=shift)
    pcs2.plot_line_on_grid(ax = ax)
    ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
    
    ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)  
    show()
    
    #dist_im_prep = Img(dist_img).pyr_down(2).img
    dist_im_prep = dist_img
    aa_list.goto_img(0)
    
    icas1 = []
    icas2 = []
    for k in range(aa_list.nof):
        print k
        aa = aa_list.current_img().img
        d1 = pcs1.get_line_profile(dist_im_prep)
        p1 = pcs1.get_line_profile(aa)
        icas1.append(np.sum(p1*d1))
        d2 = pcs2.get_line_profile(dist_im_prep)
        p2 = pcs2.get_line_profile(aa)
        icas2.append(np.sum(p2*d2))
        aa_list.next_img()
    times = aa_list.acq_times


fig, axes = subplots(1,2, figsize = (18,6))
ax = axes[0]
icas1 = np.asarray(icas1)
icas2 = np.asarray(icas2)
x = np.arange(first_idx, last_idx, 1)
ax.plot(icas1, "--b", label = "ICAs %s" %pcs1.line_id)
ax.plot(icas2, "--g", label = "ICAs %s" %pcs2.line_id)

dat1= gaussian_filter(icas1, sigma)[first_idx: last_idx]
dat2= gaussian_filter(icas2, sigma)[first_idx: last_idx]

ax.plot(x, dat1, "-b", label = "%s smooth" %pcs1.line_id)
ax.plot(x, dat2, "-g", label = "%s smooth" %pcs2.line_id)
ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)         

xcorr = correlate(dat1, dat2)
num = len(dat1)

num_range = np.arange(1 - num, num)
axes[1].plot(num_range, xcorr)
axes[1].grid()
axes[1].set_title("Cross corr spectrum")
recovered_index_shift = num_range[xcorr.argmax()]
fig.suptitle("RESULT SCIPY FUNC crosscorr: failed...")
print "Index shift: %d" % (recovered_index_shift)

lag, coeffs, s1, s2, max_coeff_signal, ax = determine_ica_cross_correlation(\
    icas1, icas2, times, cut_border = 10, sigma_smooth = 2, plot = True)
    
print "Avg. plume velocity = %.2f m / s" %(shift * dist_im_prep.mean() / lag)
#==============================================================================
# delts = np.asarray([delt.total_seconds() for delt in (times[1:] - times[:-1])])
# 
# delt_str = "%dS" %(np.ceil(delts.mean()) - 1)
# s1 = Series(icas1, times).resample(delt_str).interpolate().dropna()
# s2 = Series(icas2, times).resample(delt_str).interpolate().dropna()
# num_tot = len(s1)
#     
# s1 = gaussian_filter(s1[first_idx:last_idx], sigma) 
# s2 = gaussian_filter(s2[first_idx:last_idx], sigma) 
# 
# fig, ax = subplots(1,1)
# 
# coeffs = []
# for k in range(len(s1)):
#     coeffs.append(pearsonr(np.roll(s1, k), s2)[0])
# ax.plot(coeffs)
# ax.set_xlabel(["Delta t [s]"])
# ax.set_xlabel("Shift")
# ax.set_ylabel("Correlation coeff")
# 
# lag = np.argmax(coeffs) #already in seconds, since data was resampled to 1s
# 
# 
#==============================================================================
