# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 15:04:08 2016

@author: jg
"""

import piscope
from os.path import join, isfile
from os import listdir
from os import getcwd

### OPTIONS
MAKE_STACK = False

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

### Set path where all images are located
img_dir = "../test_data/piscope_etna_testdata/images/"

offset_file = join(img_dir, "EC2_1106307_1R02_2015091607064723_D0L_Etnaxxxxxxxxxxxx.fts")
dark_file = join(img_dir, "EC2_1106307_1R02_2015091607064865_D1L_Etnaxxxxxxxxxxxx.fts")

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

### Change image preparation settings
on_list.roi = [100, 100, 1300, 900]
on_list.crop = 1
on_list.pyrlevel = 2
on_list.set_dark_corr_mode(1)
on_list.add_gaussian_blurring(1)
on_list.goto_img(100)
on_list.show_current()
#on_list.roi
### Load all images into a stack 
# (note that they are size reduced by factor 8)
if MAKE_STACK:
    stack = on_list.make_stack(pyrlevel = 2)
    
    print stack.shape
    
    series = stack.get_time_series(pos_x=200, pos_y=100, radius =10)
    ax = series.plot()
    
    ax.figure.savefig(join(save_path, "ex2_stack_tseries_on_all.png"))
    





