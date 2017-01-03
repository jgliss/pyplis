# -*- coding: utf-8 -*-
"""
NOT FINISHED

@author: jg
"""

import piscope
from os.path import join, isfile
from os import listdir

### Set path where all images are located
img_dir = "../data/piscope_etna_testdata/images/"

darkImgPath = join(imgDir, "EC2_1106307_1R02_2015091606472819_D0L_Etnaxxxxxxxxxxxx.fts")
offsetImgPath = join(imgDir, "EC2_1106307_1R02_2015091606472962_D1L_Etnaxxxxxxxxxxxx.fts")

### Now get all images in the image path which are FITS files (actually all)
allPaths = [join(imgDir, f) for f in listdir(imgDir) if\
                        isfile(join(imgDir, f)) and f.endswith("fts")]
                                                
### Now put them all into an image list 
# Note that the files are not separated by filter type, or dark and offset, etc
# so the list simply contains all images
                        
imgList = piscope.ImageLists.ImgList(allPaths, id = "all")

### Create a master dark image and set this in image list

darkImg = piscope.Image.Img(darkImgPath)
offsetImg = piscope.Image.Img(offsetImgPath)

### Get the current image object
img = imgList.current_img()
