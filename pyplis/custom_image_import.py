# -*- coding: utf-8 -*-
"""
Custom image load methods for different camera standards

.. note::

  This file can be used to include cusotmised image import method. Please re-install pyplis after defining your customised import method here. The method requires the following input / output:
  
    1. Input: ``str``, file_path ->  full file path of the image
    2. Optional input: dict, dictionary specifying image meta information (e.g. extracted from file name before image load)
    3. Two return parameters
    
      1. ``ndarray``, the image data (2D numpy array)
      2. ``dict``, additional meta information (is required as return value, if no meta data is imported from your custom method, then simply return an empty dictionary. Please also make sure to use valid pyplis image meta data keys (listed below)
      
Valid keys for import of image meta information:

'start_acq', 'stop_acq', 'texp', 'focal_length', 'pix_width', 'pix_height', 
'bit_depth', 'f_num', 'read_gain', 'filter', 'path', 'file_name', 'file_type', 
'device_id', 'ser_no'
      
"""
from matplotlib.pyplot import imread
from numpy import swapaxes, flipud, asarray
from warnings import warn
from cv2 import resize
from os.path import basename
from datetime import datetime
from re import sub

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
    :param dict meta: optional, meta info dictionary to which additional meta
        information is suppose to be appended
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
    img = asarray(im)[::-1, 0::] #flip
    meta["texp"] = float(im.tag_v2[270].split(" ")[0].split("s")[0])
    meta["start_acq"] = datetime.strptime("_".join(basename(file_path)
                            .split("_")[:3]), "%Y%m%d_%H%M%S_%f")

    return (img, meta)


from astropy.io import fits
def load_comtessa(file_path, meta={}):
        """ Load an image
        Don't apply any image processing!
         Check out if this could be put inro camera.import method
         """           
        hdulist = fits.open(file_path)
        img_hdu = meta['img_idx']
        image = hdulist[img_hdu].data
        # load meta data
        imageHeader = hdulist[img_hdu].header
        imageMeta = {"start_acq"    : datetime.strptime(imageHeader['ENDTIME'],
                                                        '%Y.%m.%dZ%H:%M:%S.%f'),
                    "texp"         : int(imageHeader['EXP']),
                    "temperature"  : int(imageHeader['TCAM']),
                    "img_idx"       : meta['img_idx']}

        # replace binary time stamp
        image[0,0:14] = image[1,0:14]            
        #Define pyplis image
        return (image, imageMeta) 

if __name__ == "__main__":
    from os.path import join
    import matplotlib.pyplot as plt
    
    plt.close("all")
    
    base_dir = r"D:/OneDrive - Universitetet i Oslo/python27/my_data/piscope/"
    
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

   