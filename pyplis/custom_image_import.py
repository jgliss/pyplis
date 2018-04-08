# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliß (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Custom image load methods for different camera standards

.. note::

  This file can be used to include cusotmised image import method. Please 
  re-install pyplis after defining your customised import method here. 
  The method requires the following input / output:
  
    1. Input: ``str``, file_path ->  full file path of the image
    2. Optional input: dict, dictionary specifying image meta information 
        (e.g. extracted from file name before image load)
    3. Two return parameters
    
      1. ``ndarray``, the image data (2D numpy array)
      2. ``dict``, additional meta information (is required as return value
        , if no meta data is imported from your custom method, then simply 
        return an empty dictionary. Please also make sure to use valid 
        pyplis image meta data keys (listed below)
      
Valid keys for import of image meta information:

'start_acq', 'stop_acq', 'texp', 'focal_length', 'pix_width', 'pix_height', 
'bit_depth', 'f_num', 'read_gain', 'filter', 'path', 'file_name', 'file_type', 
'device_id', 'ser_no'
      
"""
from __future__ import division
from matplotlib.pyplot import imread
from numpy import swapaxes, flipud, asarray, rot90, float32
from warnings import warn
from astropy.io import fits
from cv2 import resize
from os.path import basename
from datetime import datetime
from .helpers import matlab_datenum_to_datetime

from re import sub

try:
    from PIL.Image import open as open_pil
except:
    pass

def load_ecII_fits(file_path, meta={}, **kwargs):
    """Load NILU ECII camera FITS file and import meta information"""
    hdu = fits.open(file_path)
    ec2header = hdu[0].header 
    img = hdu[0].data
    hdu.close()
    gain_info = {"LOW"  :   0,"HIGH" :   1}
    meta["texp"] = float(ec2header['EXP'])*10**-6        #unit s
    meta["bit_depth"] = 12
    meta["device_id"] = 'ECII'        
    meta["file_type"] = 'fts'
    meta["start_acq"] = datetime.strptime(ec2header['STIME'],\
                                                    '%Y-%m-%d %H:%M:%S.%f')
    meta["stop_acq"] = datetime.strptime(ec2header['ETIME'],\
                                                    '%Y-%m-%d %H:%M:%S.%f')
    meta["read_gain"] = gain_info[ec2header['GAIN']]
    meta["pix_width"] = meta["pix_height"] = 4.65e-6 #m
    meta.update(ec2header)                
    return (img, meta)

def load_hd_custom(file_path, meta={}, **kwargs):
    """Load image from HD custom camera
    
    The camera specs can be found in 
    `Kern et al. 2015, Intercomparison of SO2 camera systems for imaging 
    volcanic gas plumes <http://www.sciencedirect.com/science/article/pii/
    S0377027314002662#>`__
    
    Images recorded with this camera type are stored as .tiff files and are
    
    :param file_path: image file path 
    :param dict meta: optional, meta info dictionary to which additional 
        meta information is suppose to be appended
    :return: 
        - ndarray, image data
        - dict, dictionary containing meta information
    
    """
    im = imread(file_path, 2)#[1::, 1::]
    img = flipud(swapaxes(resize(im, (1024, 1024)), 0, 1))
    try:
        f = sub('.tiff', '.txt', file_path)
        file = open(f)
        spl = file.read().split('\n')
        spl2 = spl[0].split("_")
        try:
            meta["texp"] = float(spl[1].split("Exposure Time: ")[1])/1000.0   
        except:
            meta["texp"] = float(spl[1].split("Exposure Time: ")[1].\
                replace(",","."))
        meta["start_acq"] = datetime.strptime(spl2[0] + spl2[1],
                                                   '%Y%m%d%H%M%S%f') 
    except:
        raise
        warn("Failed to read image meta data from text file (cam_id: hd)")                                         
    return (img, meta)                                                
                                                
def load_hd_new(file_path, meta={}, **kwargs):
    """Load new format from Heidelberg group
    
    This format contains IPTC information
    
    :param file_path: image file path 
    :param dict meta: optional, meta info dictionary to which additional meta
        information is supposed to be appended
    :return: 
        - ndarray, image data
        - dict, dictionary containing meta information
    """
    im = open_pil(file_path)
    #img = asarray(im)[::-1, 0::] #flip
    img = rot90(rot90(asarray(im)))
    meta["texp"] = float(im.tag_v2[270].split(" ")[0].split("s")[0])
    meta["start_acq"] = datetime.strptime("_".join(basename(file_path)
                            .split("_")[:3]), "%Y%m%d_%H%M%S_%f")
    
    return (img, meta)

def load_usgs_multifits(file_path, meta={}):
    img = None
    if not "filter_id" in meta:
        warn("filter_id (i.e. on or off) in input arg meta not specified."
             "Using default filer_id=on")
        meta["filter_id"] = "on"
    try:
        f = fits.open(file_path)
        idx = 2 if meta["filter_id"] == "off" else 1
        hdu = f[idx]
        h = hdu.header

        try:
            meta["start_acq"] = matlab_datenum_to_datetime(h["DATETIME"])
            meta["texp"] = h["EXPTIME"] * h["NUMEXP"] / 1000
            meta["bit_depth"] = h["BITDEPTH"]
        except:
            warn("Failed to import image specific meta information from image "
                 "HDU")
        h = f[0].header
        try:
            meta["lon"] = h["LON"]
            meta["lat"] = h["LAT"]
            meta["altitude"] = h["ALT"]
            meta["elev"] = h["ELEVANGL"]
            meta["azim"] = h["AZMTANGL"]
        except:
            warn("Failed to import camera specific meta information from "
                 "primary HDU of FITS file...")
        img = hdu.data
        f.close()
    except Exception as e:
        raise IOError("Failed to import image data using custom method\n"
                      "Error message: %s" %repr(e))
    return (img, meta)
    
def load_usgs_multifits_uncompr(file_path, meta={}):
    img = None
    if not "filter_id" in meta:
        warn("filter_id (i.e. on or off) in input arg meta not specified."
             "Using default filer_id=on")
        meta["filter_id"] = "on"
    try:
        f = fits.open(file_path)
        idx = 1 if meta["filter_id"] == "off" else 0
        hdu = f[idx]
        h = hdu.header

        try:
            meta["start_acq"] = matlab_datenum_to_datetime(h["DATETIME"])
            meta["texp"] = h["EXPTIME"] * h["NUMEXP"] / 1000
            meta["bit_depth"] = h["BITDEPTH"]
        except:
            warn("Failed to import image specific meta information from image "
                 "HDU")
        h = f[0].header
        try:
            meta["lon"] = h["LON"]
            meta["lat"] = h["LAT"]
            meta["altitude"] = h["ALT"]
            meta["elev"] = h["ELEVANGL"]
            meta["azim"] = h["AZMTANGL"]
        except:
            warn("Failed to import camera specific meta information from "
                 "primary HDU of FITS file...")
        img = hdu.data
        f.close()
    except Exception as e:
        raise IOError("Failed to import image data using custom method\n"
                      "Error message: %s" %repr(e))
    return (img, meta)