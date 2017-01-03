# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 09:15:51 2016

@author: jg
"""
from piscope.geometry import MeasGeometry
from piscope import Img
from piscope.imagelists import ImgList

class DilutionCorrection(object):
    
    def __init__(self, img_like, meas_geom, vign_mask = None, bg_img = None,\
                                                    bg_settings = {}, **lines):
                                                        
        if not any([isinstance(img_like, x) for x in [Img, ImgList]]):
            raise TypeError("Invalid input type: need ImgList or Img object")
        if not isinstance(meas_geom, MeasGeometry):
            raise TypeError("Invalid input type: need MeasGeometry")
        