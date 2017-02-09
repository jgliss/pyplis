# -*- coding: utf-8 -*-
"""
piscope introduction script 3 - manual creation of image lists    
"""

import piscope
from matplotlib.pyplot import subplots, close

from os.path import join, isfile
from os import getcwd, listdir

close("all")
### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

# Image base path
img_dir = join(piscope.inout.find_test_data(), "images")

### OPTIONS
MAKE_STACK = False

offset_file = join(img_dir, "EC2_1106307_1R02_2015091607064723_D0L_Etna.fts")
dark_file = join(img_dir, "EC2_1106307_1R02_2015091607064865_D1L_Etna.fts")

### Now get all images in the image path which are FITS files (actually all)
all_paths = [join(img_dir, f) for f in listdir(img_dir) if\
                        isfile(join(img_dir, f)) and f.endswith("fts")]
                        
### Now put them all into an image list 
# Note that the files are not separated by filter type, or dark and offset, etc
# so the list simply contains all images
all_imgs = piscope.imagelists.ImgList(all_paths, list_id = "all")

### Perform manual separation into onband and offband list from file names
on_list, rest = all_imgs.separate_by_substr_filename("F01", 4, "_")
off_list, rest = rest.separate_by_substr_filename("F02", 4, "_")

### Create master dark image and set this in on band image list
dark_img = piscope.image.Img(dark_file)
offset_img = piscope.image.Img(offset_file)

on_list.add_master_dark_image(dark_img)
on_list.add_master_offset_image(offset_img)

on_list.goto_img(100)
img_0 = on_list.current_img().img
### Change image preparation settings
on_list.roi_abs = [100, 100, 1300, 900]
on_list.crop = 1
on_list.pyrlevel = 2
on_list.set_darkcorr_mode(1)
on_list.add_gaussian_blurring(1)

img = on_list.current_img()

ax = on_list.show_current()
ax.set_title("Cropped and size reduced image")
ax.figure.savefig(join(save_path, "ex2_out_1.png"))

#on_list.roi
### Load all images into a stack 
# (note that they are size reduced by factor 8)
if MAKE_STACK:
    stack = on_list.make_stack()
    fig, ax2 = subplots(1, 2, figsize = (16,5))
    series, access_mask = stack.get_time_series(\
                        pos_x = 200, pos_y = 100, radius = 10)
    series.plot(style = " x", ax = ax2[0])
    ax2[0].set_title("Intensity time series of all onband images "
        "(piscope testdata)\nRetrieved at", fontsize = 10)
    ax2[1].imshow(img.img, cmap = "gray")
    ax2[1].imshow(access_mask.astype(float), alpha = 0.3)
    ax2[1].set_title("Example image and access mask")
    fig.savefig(join(save_path, "ex2_out_2.png"))
    





