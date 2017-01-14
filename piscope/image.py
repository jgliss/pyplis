# -*- coding: utf-8 -*-
from astropy.io import fits
from matplotlib import gridspec
import matplotlib.cm as cmaps
from matplotlib.pyplot import imread, figure, tight_layout
from numpy import ndarray, argmax, histogram, float32, uint, nan, linspace,\
                                                swapaxes, flipud, isnan
from os.path import abspath, splitext, basename, exists
from datetime import datetime
from re import sub
from decimal import Decimal
from cv2 import pyrDown,resize, pyrUp
from scipy.ndimage.filters import gaussian_filter, median_filter
from traceback import format_exc
from collections import OrderedDict as od
from copy import deepcopy

from .inout import get_all_valid_cam_ids
from .helpers import shifted_color_map, bytescale, map_roi, check_roi
from .exceptions import ImgMetaError

class Img(object):
    """ Image base class
    
    Implementation of image object for :mod:`piscope` library. Images are
    represented as :class:`numpy.ndarray` objects and the image data is 
    stored in the attribute ``self.img``.
    
    Supported file formats include those supported by the Python Imaging 
    Library (see `here <http://pillow.readthedocs.io/en/3.4.x/handbook/
    image-file-formats.html#image-file-formats>`_) and the `FITS format 
    <http://docs.astropy.org/en/stable/io/fits/>`_. Img objects can also be 
    created from numpy arrays directly.
    
    The object includes several loading routines and basic image editing. 
    Image meta information can be provided on creation of this instance by
    providing valid meta keys and the corresponding values, i.e.::
        
        png_image_file = "C:/Test/my_img_file.png"
        acq_time = datetime(2016, 10, 10, 13, 15, 12) #10/10/2016, 13:15:12
        exposure_time = 0.75 #s
        img = Img(png_image_file, start_acq = acq_time, texp = exposure_time)
        
    Meta information is stored in the dictionary ``self.meta`` and can be 
    printed using :func:`print_meta`. The two most important image meta 
    parameters are the acquisition time (``img.meta["start_acq"]``) and the
    exposure time (``img.meta["texp"]``). These two parameters have class own
    access methods (:func:`start_acq` and :func:`texp`).
    
    The class provides several image editing routines, of which the most 
    important ones (within this library) are (please see documentation of the 
    individual functions for more information):
        
        1. :func:`subtract_dark_image` (subtract a dark image)
        #. :func:`correct_dark_offset` (Correct for dark and offset. Models 
                a dark image based on one dark and one offset image using the 
                exposure time of this image, then uses 1. for subtraction)
        #. :func:`crop` (crop image within region of interest)
        #. :func:`apply_median_filter` (median filtering of image)
        #. :func:`add_gaussian_blurring` (Add blurring to image taking into 
                account current blurring amount)
        #. :func:`apply_gaussian_blurring` (applies gaussian filter to image)
        #. :func:`pyr_down` (reduce image size using gaussian pyramide)
        #. :func:`pyr_up` (increase image size using gaussian pyramide)
                
    All image editing steps performed using these functions are logged in the 
    dictionary ``self.edit_log``, it is therefore recommended to use the class
    own methods for these image editing routines (and not apply them manually 
    to the image data, e.g. by using ``cv2.pyrDown(img.img)`` for resizing or 
    ``img.img = img.img[y0:y1, x0:x1]`` for cropping a ROI ``[x0, x1, y0, y1]``
    ) in order to keep track of the changes applied.
    
    The default data accuracy is 32 bit floating point and can be changed
    on initiation (see :func:`__init__`).
        
    """
    _FITSEXT = [".fits", ".fit", ".fts"]
    
    def __init__(self, input = None, cam_id = "", dtype = float32,\
                                                         **meta_info):
        """Class initialisation
        
        :param input: if input is valid (e.g. file path to an image type which
            can be read or numpy array) it is loaded
        :param str cam_id (""): ID of camera used. This can be important if 
            a certain camera type has its own import routine within this class
        :param dtype: datatype for image data (float32)
        """
        if isinstance(input, Img):
            return input
            
        self.img = None #: the actual image data
        self.dtype = dtype
        
        self.cam_id = cam_id
        
        #Log of applied edit operations
        self.edit_log = od([  ("darkcorr"  ,   0), # boolean
                              ("blurring"  ,   0), # int (width of kernel)
                              ("median"    ,   0), # int (size of filter)
                              ("crop"      ,   0), # boolean
                              ("8bit"      ,   0), # boolean
                              ("pyrlevel"  ,   0), # int (pyramide level)
                              ("is_tau"    ,   0), # boolean
                              ("others"    ,   0)])# boolean 
        
        self._roi_abs = [0, 0, 9999, 9999] #will be set on image load
        
        self.meta = od([("start_acq"     ,   datetime(1900, 1, 1)),#datetime(1900, 1, 1)),
                        ("stop_acq"      ,   datetime(1900, 1, 1)),#datetime(1900, 1, 1)),
                        ("texp"          ,   0.0), # exposure time [s]
                        ("focal_length"  ,   nan), # lense focal length [mm]
                        ("pix_width"     ,   nan), # horizontal pix pitch
                        ("pix_height"    ,   nan), # vert. pix pitch
                        ("bit_depth"     ,   nan), # pix bit depth
                        ("f_num"         ,   nan), # f number of lense
                        ("read_gain"     ,   0),   # boolean (on / off)
                        ("filter"        ,   ""),
                        ("path"          ,   ""),
                        ("file_name"     ,   ""),
                        ("file_type"     ,   ""),
                        ("device_id"     ,   ""),
                        ("ser_no"        ,   "")])
                        
        
        if input is not None:                              
            self.load_input(input)
          
        for k, v in meta_info.iteritems():
            if self.meta.has_key(k):
                func = type(self.meta[k])
                try:                
                    self.meta[k] = func(v)
                except:
                    print ("Failed to read image meta info %s, i.e. convert "
                        "%s into type %s" %(k, v, func))
            elif self.edit_log.has_key(k):
                self.edit_log[k] = v
        try:
            self.set_roi_whole_image()
        except:
            pass
        
    def set_data(self, input):
        """Try load input"""
        try:
            self.load_input(input)
        except Exception as e:
            print repr(e)
            
    def load_input(self, input):
        """Try to load input as numpy array and additional meta data"""
        try:
            if any([isinstance(input, x) for x in [str, unicode]]) and\
                                                                exists(input):
                self.load_file(input)
            
            elif isinstance(input, ndarray):
                self.img = input
        except:
            raise IOError("Image data could not be imported:\nError msg: %s"\
                                        %format_exc())
    
    def make_histogram(self):
        """Make histogram of current image"""
        if isnan(self.meta["bit_depth"]):
            print ("Error in " + self.__str__() + ".make_histogram\n "
                "No MetaData available => BitDepth could not be retrieved. "
                "Using 100 bins and img min/max range instead")
            hist, bins = histogram(self.img, 100)
            return hist, bins
        #print "Determining Histogram"
        hist, bins = histogram(self.img, 2**(self.meta["bit_depth"]),\
                                        [0,2**(self.meta["bit_depth"])])
        return hist, bins
            
    def get_brightness_range(self):
        """Analyses the Histogram to retrieve a suited brightness range (i.e. 
        for diplaying the image)            
        """
        hist, bins = self.make_histogram()
        thresh = hist.max() * 0.03
        rad_low = bins[argmax(hist > thresh)]
        rad_high = bins[len(hist) - argmax(hist[::-1]>thresh)-1]
        return rad_low, rad_high, hist, bins
    
    def crop(self, roi_abs = [0, 0, 9999, 9999], new_img = False):
        """Cut subimage specified by rectangular ROI
        
        :param list roi_abs: region of interest (i.e. ``[x0, y0, x1, y1]``)
            in ABSOLUTE image coordinates. The ROI is automatically converted 
            with respect to current pyrlevel.
            
        :returns Img: sub image object
        """
        if self.edit_log["crop"]:
            raise AttributeError("Could not crop image in ROI, image "
                " was already cropped...")
        self.roi_abs = roi_abs
        print self.meta["start_acq"]
        roi = self.roi #self.roi is @property method and takes care of ROI conv
        sub = self.img[roi[1]:roi[3], roi[0]:roi[2]] 
        im = self
        if new_img:
            im = self.duplicate()
#        im._roi_abs = roi
        im.edit_log["crop"] = 1
        im.img = sub
        return im
    
    @property
    def pyrlevel(self):
        """Returns current gauss pyramid level (stored in ``self.edit_log``)"""
        return self.edit_log["pyrlevel"]
    
    @property 
    def roi(self):
        """Returns current roi (in consideration of current pyrlevel)"""
        roi_sub = map_roi(self._roi_abs, self.edit_log["pyrlevel"])
        print ("Current roi in Img (in abs coords): %s, mapped to pyrlevel: "
            "%s" %(self._roi_abs, roi_sub))
        return roi_sub
    
    @property
    def roi_abs(self):
        """Get / set current ROI in absolute image coordinates
        
        .. note::
        
            use :func:`roi` to get ROI for current pyrlevel
        """
        return self._roi_abs
        
    @roi_abs.setter
    def roi_abs(self, val):
        """Updates current ROI"""
        if check_roi(val):
            self._roi_abs = val
            
    def correct_dark_offset(self, dark, offset):
        """Perform dark frame subtraction, 3 different modi possible
        
        :param Img dark: dark image object (dark with long(est) exposure time)
        :param Img offset: offset image (dark with short(est) exposure time)
        :return Img: modelled dark image 
        
        Uses :func:`model_dark_image` (in :mod:`Processing`) to model a dark 
        image based on the exposure time of this image object. This is then
        subtracted from the current image. 
        
        .. note:: 
        
            This algorithm works only, if no other image processing operations
            were applied to the input image beforehand, i.e. if 
            :func:`modified` returns False.
            
        """
        from piscope.processing import model_dark_image
        if self.modified:
            print ("Dark correction not possible, it was either already "
            "performed, the image was already modified")
            return 
            
        dark = model_dark_image(self, dark, offset)                
        self.subtract_dark_image(dark)
        
        return dark
    
    def subtract_dark_image(self, dark):
        """Subtracts a dark (+offset) image and updates ``self.edit_log``
        
        :param Img dark: dark image data
        
        Simple image subtraction without any modifications of input image
        """
        try:
            self.img = self.img - dark
        except:
            self.img = self.img - dark.img
            
        self.edit_log["darkcorr"] = 1
        
    def set_roi_whole_image(self):
        """Set current ROI to whole image area based on shape of image data"""
        h, w = self.img.shape[:2]
    
        self._roi_abs = [0, 0, w * 2**self.pyrlevel, h * 2**self.pyrlevel]     
    
    def apply_median_filter(self, size_final = 3):
        """Apply a median filter to 
        
        :param tuple shape (3,3): size of the filter        
        """
        diff = int(size_final - self.edit_log["median"])
        if diff > 0:
            self.img = median_filter(self.img, diff)
            self.edit_log["median"] += diff

        
    def add_gaussian_blurring(self, sigma_final = 1):
        """Add blurring to image
        
        :param int sigma_final: the final width of gauss blurring kernel
        """
        diff = int(sigma_final - self.edit_log["blurring"])
        if diff > 0:
            self.apply_gaussian_blurring(diff)
                
    def apply_gaussian_blurring(self, sigma, **kwargs):
        """Add gaussian blurring using :class:`scipy.ndimage.filters.gaussian_filter`
        
        :param int sigma: amount of blurring
        """
        self.img = gaussian_filter(self.img, sigma, **kwargs)
        self.edit_log["blurring"] += sigma   
        
    def pyr_down(self, steps = 0):
        """Reduce the image size using gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrDown` 
        """
        if not steps:
            return
        #print "Reducing image size, pyrlevel %s" %steps
        for i in range(steps):
            self.img = pyrDown(self.img)
        self.edit_log["pyrlevel"] += steps
        return self
    
    def pyr_up(self, steps):
        """Increasing the image size using gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrUp` 
        """
        for i in range(steps):
            self.img = pyrUp(self.img)
        self.edit_log["pyrlevel"] -= steps  
        self.edit_log["others"] = 1
        return self
    
    def _to_8bit_int(self, current_bit_depth = None, new_img = True):
        """Convert image to 8 bit representation and return new image object
        
        :returns array 
        .. note::
        
            1. leaves this image unchanged
            #. if the bit_depth is unknown or unspecified in ``self.meta``, then
            
        """
        if current_bit_depth == None:
            current_bit_depth = self.meta["bit_depth"]
            if isnan(current_bit_depth):
                raise ValueError("Image cannot be converted into 8 bit repr."
                    "Plese insert the bit depth of the current image")

        sc = bytescale(self.img, cmin = 0, cmax = 2**(current_bit_depth) - 1)

        if new_img:
            img = self.duplicate()
        else:
            img = self
            self.edit_log["8bit"] = 1
        img.meta["bit_depth"] = 8
        img.img = sc
        return img
            
    """
    HELPER FUNCTIONS ETC
    """
    def print_meta(self):
        """Print current image meta information"""
        for key, val in self.meta.iteritems():
            print "%s: %s\n" %(key, val)
        
    def make_info_header_str(self):
        """Make header string for image (using image meta information)"""
        try:
            return ("Acq.: %s, texp: %.2f s, rgain %s\n"
                    "pyrlevel: %d, roi_abs: %s" %(self.meta["start_acq"].\
                    strftime('%d/%m/%Y %H:%M:%S'), self.meta["texp"],\
                    self.meta["read_gain"], self.pyrlevel, self.roi_abs)) 
        except Exception as e:
            print repr(e)
            return self.meta["file_name"]
        
    def duplicate(self):
        """Duplicate this image"""
        #print self.meta["file_name") + ' successfully duplicated'
        return deepcopy(self)
    
    def min(self):
        return self.img.min()
    
    def max(self):
        return self.img.max()
        
    def _valid_cam_id(self):
        """Checks if current cam ID is one of the piscope default IDs"""
        if self.cam_id.lower() in get_all_valid_cam_ids():
            return True
        return False
        
    def meta(self, meta_key):
        """Returns current meta data for input key"""
        return self.meta[meta_key]
    
    """DECORATORS"""    
    @property
    def start_acq(self):
        """Get image acquisition time
        
        :returns: acquisition time if available (i.e. it deviates from the
            default 1/1/1900), else, raises ImgMetaError
        """
        if self.meta["start_acq"] == datetime(1900, 1, 1):
            raise ImgMetaError("Image acquisition time not set")
        return self.meta["start_acq"]
    
    @property
    def stop_acq(self):
        """Returns stop time of acquisition (if available)"""
        return self.meta["stop_acq"]
    
    @property
    def texp(self):
        """Get image acquisition time
        
        :returns: acquisition time if available (i.e. it deviates from the
            default 1/1/1900), else, raises ImgMetaError
        """
        if self.meta["texp"] == 0.0:
            raise ImgMetaError("Image exposure time not set")
        return self.meta["texp"]
        
    @property
    def gain(self):
        """Returns read gain value from meta info"""
        gain = self.meta["read_gain"]
        if not gain in [1, 0]:
            raise Exception("Invalid gain value in Img: %s " %gain)
        return gain
    
    @property
    def shape(self):
        """Return shape of image data"""
        return self.img.shape
    
    @property    
    def pyr_up_factor(self):
        """Factor to convert coordinates at current pyramid level into 
        original size coordinates
        """
        return 2 ** self.edit_log["pyrlevel"]
        
    @property
    def is_tau(self):
        """Returns boolean whether image is a tau image or not"""
        return self.edit_log["is_tau"]
        
    @property
    def is_gray(self):
        """Checks if image is gray image"""
        if self.img.ndim == 2:
            return True
        elif self.img.ndim == 3:
            return False
        else:
            raise Exception("Unexpected image dimension %s..." %self.img.ndim)
            
    @property
    def modified(self):
        """Check if this image was already modified"""
        if sum(self.edit_log.values()) > 0:
            return 1
        return 0
     
    """I/O""" 
    def load_hd_custom(self, file_path):
        """Load HD custom file and meta information"""
        im = imread(file_path, 2)#[1::, 1::]
        self.img = flipud(swapaxes(resize(im, (512, 512)),0,1)).astype(\
                                                                self.dtype)
        try:
            f = sub('.tiff', '.txt', file_path) #open the *.txt instead    
            file = open(f)
            spl = file.read().split('\n')
            print spl[1].split("Exposure Time: ")[1]
            self.meta["texp"] = float(spl[1].split("Exposure Time: ")[1])
            spl2 = spl[0].split("_")
            self.meta["start_acq"] = datetime.strptime(spl2[0] + spl2[1],\
                                                            '%Y%m%d%H%M%S%f') 
            #self.meta["read_gain"] = 0
        except:
            print "Error loading image meta info for HD custom cam"
        
    def load_fits(self, file_path):
        """Import a FITS file 
        
        `Fits info <http://docs.astropy.org/en/stable/io/fits/>`_
        """
        #t0 = time()
        hdu = fits.open(file_path)
        h = hdu[0].header 
        self.img = hdu[0].data.astype(self.dtype)
        #self.img=hdu[0].data.astype(np.float16)
        hdu.close()
        try:
            if h["CAMTYPE"] == 'EC2':
                self.cam_id = "ecII"
                self.import_ec2_header(h)
        except:
            pass
            
    def load_file(self, file_path):
        """Try to import file specified by input path"""
        ext = splitext(file_path)[-1]
        if ext in self._FITSEXT:
            self.load_fits(file_path)
        elif self.cam_id == "hdcam":
            self.load_hd_custom(file_path)
        else:
            self.img = imread(file_path).astype(self.dtype)
        self.meta["path"] = abspath(file_path)
        self.meta["file_name"] = basename(file_path)
        self.meta["file_type"] = ext
        
    def import_ec2_header(self, ec2header):
        """Import image meta info for ECII camera type from FITS file header"""
        gain_info = {"LOW"  :   0,
                     "HIGH" :   1}
                     
        self.meta["texp"] = float(ec2header['EXP'])*10**-6        #unit s
        self.meta["bit_depth"] = 12
        self.meta["device_id"] = 'ECII'        
        self.meta["file_type"] = '.fts'
        self.meta["start_acq"] = datetime.strptime(ec2header['STIME'],\
                                                    '%Y-%m-%d %H:%M:%S.%f')
        self.meta["stop_acq"] = datetime.strptime(ec2header['ETIME'],\
                                                    '%Y-%m-%d %H:%M:%S.%f')
        self.meta["read_gain"] = gain_info[ec2header['GAIN']]
        self.meta["pix_width"] = self.meta["pix_height"] = 4.65e-6 #m
    
    """PLOTTING AND VISUALSATION FUNCTIONS"""  
    def get_cmap(self):
        """Determine and return default cmap for current image"""
        if self.is_tau:
            return shifted_color_map(self.min(), self.max(), cmaps.RdBu)
        return cmaps.gray
        
    def show(self, **kwargs):
        """Plot image"""
        return self.show_img(**kwargs)

    def show_img(self, **kwargs):
        """Show image using plt.imshow"""
        if not "cmap" in kwargs.keys():
            kwargs["cmap"] = self.get_cmap()
        fig = figure(facecolor = 'w', edgecolor = 'none', figsize = (12,7))  
        ax = fig.add_subplot(111)
        im = ax.imshow(self.img, **kwargs)
        fig.colorbar(im, ax = ax)
        ax.set_title(self.make_info_header_str(), fontsize = 9)
        tight_layout()
        return ax
        
    def show_img_with_histo(self, **kwargs):
        """Show image using plt.imshow"""
        if not "cmap" in kwargs.keys():
            kwargs["cmap"] = self.get_cmap()
        fig = figure(figsize = (13, 5), dpi = 80, facecolor = 'w',\
                                                        edgecolor = 'k')  
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 

        ax = fig.add_subplot(gs[0])
        im = ax.imshow(self.img, **kwargs)
        fig.colorbar(im, ax = ax)
        ax.set_title(self.make_info_header_str(), fontsize = 9)
        ax2 = fig.add_subplot(gs[1])
        self.show_histogram(ax2)
        tight_layout()
        return ax
            
    def show_histogram(self, ax = None):
        """Plot histogram of current image
        
        .. todo::
        
            Needs more edit (i.e. better representation of labels)
            
        """
        hist, bins = self.make_histogram()
        if ax is None:
            fig = figure(self.meta["file_name"])
            ax = fig.add_subplot(121)
        change_labels=0
        try:
            i, f = 0, 2**(self.meta["bit_depth"]) - 1
            change_labels = 1
        except:
            l, h = self.img.min(), self.img.max()
            i, f = l - abs(l) * 0.2, h + abs(h) * 0.2
        print i, f
        ax.fill_between(linspace(i, f, len(hist)), hist, 0,\
                                        color = '#0000FF', alpha = 0.5)
        ax.set_xlim([i, f])
        
        ax.set_ylabel('Counts')
        ax.set_xlabel('Intensity')
        ax.set_title("Histogram", fontsize = 12)
        if change_labels:
            xticks = ax.get_xticks()
            ax.set_xticklabels(uint(xticks), rotation = 35, ha = "right")
            labels = ax.get_xticklabels()
            newlabels = []
            for k in range(len(labels)):
                newlabels.append('%.1E' % Decimal(labels[k].get_text()))
            ax.set_xticklabels(newlabels)
        ax.grid()
    
    def info(self):
        """Image info (prints string representation)"""
        print self.__str__
        
    """MAGIC METHODS"""
    def __str__(self):
        """String representation"""
        s = "piscope Img\n--------------\n\n"
        s += "Shape: %s\n" %str(self.shape)
        s += "Min / Max intensity: %s - %s\n" %(self.min(), self.max())
        s += "\nMeta information\n-----------------------------\n"
        for k, v in self.meta.iteritems():
            s += "%s: %s\n" %(k, v)
        s += "\nEdit log\n-----------------------------\n"
        for k, v in self.edit_log.iteritems():
            s += "%s: %s\n" %(k, v)
        return s
            
    def __call__(self):
        """Return image numpy array on call"""
        return self.img
        
    def __add__(self, img_obj):
        """Add another image object
        
        :param Img img_obj: object to be added
        :return: new image object
        """
        try:
            return Img(self.img + img_obj.img)
            
        except:
            raise TypeError("Could not add target image of type %s"
                                                                %type(img_obj))
            
    def __sub__(self, img_obj):
        """Subtract another image object
        
        :param Img img_obj: object to be subtracted
        :return: new image object
        """
        try:
            return Img(self.img - img_obj.img)            
        except:
            raise TypeError("Could not subtract target image of type %s"
                                                                %type(img_obj))
    
    def __mul__(self, img_obj):
        """Multiply another image object
        
        :param Img img_obj: object to be multiplied
        :return: new image object
        """
        try:
            return Img(self.img * img_obj.img)
        except:
            raise TypeError("Could not subtract target image of type %s"
                                                                %type(img_obj))

    def __div__(self, img_obj):
        """Divide another image object
        
        :param Img img_obj: object to be multiplied
        :return: new image object
        """
        try:
            return Img(self.img / img_obj.img)
        except:
            raise TypeError("Could not subtract target image of type %s"
                                                                %type(img_obj))