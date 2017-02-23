# -*- coding: utf-8 -*-
"""
Custom image load methods for different camera standards
"""
from matplotlib.pyplot import imread
from numpy import swapaxes, flipud, asarray
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
    img = flipud(swapaxes(resize(im, (512, 512)), 0, 1))
    f = sub('.tiff', '.txt', file_path)
    file = open(f)
    spl = file.read().split('\n')
    spl2 = spl[0].split("_")

    meta["texp"] = float(spl[1].split("Exposure Time: ")[1])    
    meta["start_acq"] = datetime.strptime(spl2[0] + spl2[1],
                                               '%Y%m%d%H%M%S%f') 
                                                
    return (img, meta)                                                
                                                
def load_hd_new(file_path, meta={}):
    """Load new format from Heidelberg group
    
    This format contains IPTC information
    
    :param file_path: image file path 
    :param dict meta: optional, meta info dictionary to which additional meta
        information is suppose to be appended
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
    