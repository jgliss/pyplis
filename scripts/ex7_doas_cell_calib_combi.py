# -*- coding: utf-8 -*-
"""
piscope example script no. 7 - Combined DOAS and cell calibration
"""

import piscope 
from os.path import join, exists
from matplotlib.pyplot import close, subplots, show
from matplotlib.patches import Circle

from ex1_measurement_setup_plume_data import create_dataset, save_path
from ex5_auto_cellcalib import perform_auto_cell_calib
from ex4_prepare_aa_imglist import prepare_aa_image_list, path_bg_on,\
                                                                path_bg_off
from ex6_doas_calibration import my_dat #folder where FOV fits is saved

close("all")
calib_file = join(my_dat, "piscope_doascalib_id_aa_avg_20150916_0706_0721.fts")

if not exists(calib_file):
    raise IOError("Calibration file could not be found at specified location:\n"
        "%s\nYou might need to run example 6 first")

### Load AA list
dataset = create_dataset()
aa_list = prepare_aa_image_list(dataset, path_bg_on, path_bg_off)

aa_list.add_gaussian_blurring(2)

### Load DOAS calbration data and FOV information (see example 6)
doas = piscope.doascalib.DoasCalibData()
doas.load_from_fits(file_path=calib_file)
doas.fit_calib_polynomial()

### Get DOAS FOV parameters in absolute coordinates
fov_x, fov_y = doas.fov.pixel_position_center(abs_coords=True)
fov_extend = doas.fov.pixel_extend(abs_coords=True)

### Load cell calibration (see example 5)
cell = perform_auto_cell_calib()

### Define lines on image for plume profiles
l1 = piscope.processing.LineOnImage(620, 700, 940, 280,\
                                                    line_id = "center")
l2 = piscope.processing.LineOnImage(40, 40, 40, 600,\
                                                    line_id = "img_edge")
### Plot DOAS calibration polynomial
ax0 = doas.plot()
ax0 = cell.plot_calib_curve("aa", pos_x_abs= fov_x, pos_y_abs= fov_y, \
                                        radius_abs=fov_extend, ax = ax0)
                                                        
### Show current AA image from image list
aa0 = aa_list.current_img()
ax = aa0.show()

# plot the two lines into the exemplary AA image
l1.plot_line_on_grid(ax = ax, c="r")
l2.plot_line_on_grid(ax = ax, c="g")

# add FOV position to plot of examplary AA image
c = Circle((fov_x, fov_y), fov_extend, ec = "lime", fc = "none", label = "DOAS FOV")
ax.add_artist(c)
ax.set_xlim([0, 1343]), ax.set_ylim([1023, 0])
ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)

fig0, axes = subplots(1,2, figsize=(18, 6))
p10 = l1.get_line_profile(aa0.img)
p20 = l2.get_line_profile(aa0.img)

num = len(p10)

axes[0].set_title("Line %s" %l1.line_id)
axes[1].set_title("Line %s" %l2.line_id)

axes[0].plot(p10, "-", label = r"Init $\phi=%.3f$" %(sum(p10)/num))
axes[1].plot(p20, "-", label = r"Init $\phi=%.3f$" %(sum(p20)/num))

axes[0].grid()
axes[1].grid()


#get cell AA stack in original image resolution
cell_aa_stack = cell.tau_stacks["aa"].pyr_up(cell.tau_stacks["aa"].pyrlevel)

fov_mask = cell_aa_stack.make_circular_access_mask(fov_x, fov_y, fov_extend)

### Now get correction masks for each cell AA image apply correction and plot

fig1, axes2 = subplots(2,3, figsize = (18, 7))

def prepare_mask(stack, idx, fov_mask):
    im = stack.stack[idx]
    cd =stack.add_data[idx] #SO2 CD of first cell
    im = piscope.optimisation.PolySurfaceFit(im, pyrlevel = 2).model
    mean = (im * fov_mask).sum() / fov_mask.sum()
    mask = im / mean
    return mask, cd

masks, cds, aa_imgs_corr = [], [], []
vmin_mask, vmax_mask = 0.8, 1.6
vmin_aa, vmax_aa = -0.18, 0.18
for k in range(cell_aa_stack.num_of_imgs):
    mask, cd = prepare_mask(cell_aa_stack, k, fov_mask)
    masks.append(mask), cds.append(cd)
    aa_img = piscope.Img(aa0.img / mask)
    aa_imgs_corr.append(aa_img)


    disp1 = axes2[0, k].imshow(mask, cmap = "gray", vmin = vmin_mask,\
        vmax = vmax_mask)
    fig1.colorbar(disp1, ax = axes2[0, k])
    axes2[0, k].set_title("Correction mask (2D poly fit)\nSO2 CD: %.2e [cm-2]"\
            %cd, fontsize=12)
    aa_img.edit_log["is_tau"] = True
    aa_img.show_img(ax = axes2[1, k], vmin = vmin_aa, vmax = vmax_aa)
    
    p1 = l1.get_line_profile(aa_img.img)
    p2 = l2.get_line_profile(aa_img.img)
    
    axes[0].plot(p1, "-", label = r"Cell CD: %.2e $\phi=%.3f$"\
                                                %(cd, sum(p1)/num))
    axes[1].plot(p2, "-", label = r"Cell CD: %.2e $\phi=%.3f$"\
                                                %(cd, sum(p2)/num))
                                                
axes[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
axes[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)

show()
ax0.figure.savefig(join(save_path, "ex7_out_1.png"))
ax.figure.savefig(join(save_path, "ex7_out_2.png"))
fig0.savefig(join(save_path, "ex7_out_3.png"))
fig1.savefig(join(save_path, "ex7_out_4.png"))                                                                
