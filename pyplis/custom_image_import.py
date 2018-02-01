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
from matplotlib.pyplot import imread
from numpy import swapaxes, flipud, asarray, rot90
from warnings import warn
from cv2 import resize
from os.path import basename
from datetime import datetime, timedelta
from re import sub
from astropy.io import fits

try:
    from PIL.Image import open as open_pil
except:
    pass


def load_hd_custom(file_path, meta={}):
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
                                                
def load_hd_new(file_path, meta={}):
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

def _read_binary_timestamp(timestamp):
    """ Converts the an array of pixel as given by the pco camware software to
    a valid datetime 

    Parameters
    ----------
    timestamp : array
        array containg 14 pixel which code as the following
        0 pixel 1 image counter (MSB) (00 … 99)
        1 pixel 2 image counter (00 … 99)
        2 pixel 3 image counter (00 … 99)
        3 pixel 4 image counter (LSB) (00 … 99)
        4 pixel 5 year (MSB) (20)
        5 pixel 6 year (LSB) (03 … 99)
        6 pixel 7 month (01 … 12) 
        7 pixel 8 day (01 ... 31)
        8 pixel 9 h (00 … 23)
        9 pixel 10 min (00 … 59)
        10 pixel 11 s (00 … 59)
        11 pixel 12 μs * 10000 (00 … 99)
        12 pixel 13 μs * 100 (00 … 99)
        13 pixel 14 μs (00 … 90)
            
    Returns
    -------
    datetime.datetime
        3-element tuple containing
    """
    try:
        values = [10 * (timestamp[0,j] >> 4) +  timestamp[0,j] - ((timestamp[0,j] >> 4) << 4) for j in range(14)]
    except:
        try:
            values = [10 * (timestamp[j] >> 4) +  timestamp[j] - ((timestamp[j] >> 4) << 4) for j in range(14)]
        except:
            print('Theres an error in the conversion of the binary timestamp.')
    year = int(values[4]*100 + values[5])
    microsecond = int(values[11]*10000 + values[12]*100 + values[13])
    endtime = datetime(year, values[6], values[7], values[8], values[9],
                       values[10], microsecond)
    return endtime

def load_comtessa(file_path, meta={}):
    """ Load image from a comtessa fits file (several images in one file)
    Meta data is available only inside the header.
    
    Note
    ----
    The comtessa *.fits files have several timestamps: 1) Filename --> minute 
    in which the image was saved. 2) Meta information in the image header -->
    computer time when the image was saved. 3) First 14 pixel contain binary
    timestamp --> time when exposure was finished. Here nr 3) is saved as
    meta['stop_acq'] and nr 2) as meta['custom1']. meta['start_acq'] is
    calculated from meta['stop_acq'] and meta['texp'].
    
    Parameters
    ----------
    file_path : string
        image file path
    meta: dictionary
        optional, meta info dictionary to which additional meta
        information is appended. The image index should be provided with key 
        "img_idx".
        
    Returns 
    -------
    ndarray
        image data
    dict
        dictionary containing meta information 

    """ 
    hdulist = fits.open(file_path)
    try:
        img_hdu = meta['img_idx']
    except:
        img_hdu = 0
        meta['img_idx'] = 0
        warn("Loading of comtessa fits file without providing the image index "
             "of desired image within the file. Image index was set to 0. "
             "Provide the image index via the meta = {'img_idx':0} keyword.")
    # Load the image
    image = hdulist[img_hdu].data
    # read and replace binary time stamp
    endtime = _read_binary_timestamp(image)
    image[0,0:14] = image[1,0:14]
    # load meta data
    imageHeader = hdulist[img_hdu].header
    meta.update({"start_acq"    : endtime - timedelta(microseconds=int(imageHeader['EXP'])*1000),
                "stop_acq"      : endtime,
                "texp"          : float(imageHeader['EXP']) / 1000., # in seconds
                "temperature"   : float(imageHeader['TCAM']),
                "ser_no"        : imageHeader['SERNO'],
                "custom1"       : float(imageHeader['GAIN']),
                "custom_dt"     :  datetime.strptime(imageHeader['ENDTIME'],
                                                     '%Y.%m.%dZ%H:%M:%S.%f') })
    return (image, meta) 

if __name__ == "__main__":
    from os.path import join
    import matplotlib.pyplot as plt
    
    plt.close("all")
    
    ### TODO: Add these example fotos to an internal example dataset
    base_dir = r"D:/OneDrive - Universitetet i Oslo/python27/my_data/piscope/"
    base_dir2 = r"C:/Users/asd/documents/pyplis_branch/"
    '''
    # load and display HD custom image
    p_hd = join(base_dir, "HD_cam_raw_data/23/06353343_M_B0_19.tiff")

    fig, ax = plt.subplots(1,1)
    im, meta = load_hd_custom(p_hd)
    ax.imshow(im)
    print meta
    
    # load and display HD new image
    p_hdnew = join(base_dir, "HD_newcam_juan_test_data/20161208_145201_785_M_B.tif")
    fig1, ax1 = plt.subplots(1,1)
    im1, meta1 = load_hd_new(p_hdnew)
    ax1.imshow(im1)
    print meta1
    '''
    # load and display a Comtessa image
    path_comtessa = join(base_dir2, "pyplis", "data", "20170720Z0800_UV3A.fits")
    meta_comtessa = {'img_idx' : 30}
    
    fig_c, ax_c = plt.subplots(1,1)
    img_c, meta_c = load_comtessa(path_comtessa, meta_comtessa)
    ax_c.imshow(img_c)
    print meta_c
