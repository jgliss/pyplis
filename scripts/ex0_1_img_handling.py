# -*- coding: utf-8 -*-
"""
piscope intorduction script 1 - Image import
"""
from os.path import join
import piscope
from os import getcwd

### Set save directory for figures
save_path = join(getcwd(), "scripts_out")

#fake filename with delimiter _ and encrypted information (see below) 
img_file_name = "test_201509160708_F01_0.3348.fts"

file_path = join(piscope._LIBDIR, "data", img_file_name)

img = piscope.image.Img(file_path)
### Show image
img.show()
print img

