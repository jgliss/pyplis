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

"""Module containing classes representing image data and corresponding
processing features. The image base class :class:`Img` is a powerful object for 
image data, containing I/O routines for many data formats, processing classes 
and keeping track on changes applied to the images. 
The actual image data is stored as numpy array in the :attr:`img` of an 
instance of the :class:`Img` object.

The :class:`ProfileTimeSeriesImg` class is used to store and process time 
series of pixel profiles (e.g. along a :class:`LineOnImage`). These are, for 
instance used when performing a plume velocity cross-correlation analysis 
(where the optimal lag between a time-series of two plume intersection lines is 
searched, for details see :class:`pyplis.plumespeed.VeloCrossCorrEngine`).
"""
#from __future__ import division
from astropy.io import fits
from matplotlib import gridspec
import matplotlib.cm as cmaps
from matplotlib.pyplot import imread, figure, tight_layout
from numpy import (ndarray, argmax, histogram, uint, nan, linspace, isnan, 
                   uint8, float32, finfo, ones, invert, log, ogrid, asarray)
from json import loads, dumps
from os.path import abspath, splitext, basename, exists, join, isdir, dirname
from os import remove
from warnings import warn
from datetime import datetime
from decimal import Decimal
from cv2 import pyrDown, pyrUp, addWeighted, dilate, erode
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.interpolation import shift
from collections import OrderedDict as od
from copy import deepcopy

from .glob import DEFAULT_ROI
from .helpers import bytescale, map_roi, check_roi
from .exceptions import ImgMetaError
from .optimisation import PolySurfaceFit
from .utils import LineOnImage

class Img(object):
    """ Image base class
    
    Implementation of image object for :mod:`pyplis` library. The image data is
    represented as :class:`numpy.ndarray` objects and the is stored in the 
    attribute :attr:`self.img`.
    
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
    
    def __init__(self, input=None, import_method=None, dtype=float32,
                 **meta_info):
        """Class initialisation
        
        :param input: if input is valid (e.g. file path to an image type which
            can be read or numpy array) it is loaded
        :param function import_method: custom image load method, must return
            tuple containing image data (2D ndarray) and dictionary containing
            meta information (can be empty if read routine does not import 
            any meta information)
        :param dtype: datatype for image data (float32)
        :param **meta_info: keyword args specifying meta data
        """
        if isinstance(input, Img):
            meta_info = input.edit_log
            meta_info.update(input.meta)
            input = input.img
            
        self._img = None #: the actual image data (use method `img` for access)
        self.dtype = dtype
        self.vign_mask = None
        
        # custom data import method (optional on class initialisation)
        self.import_method = import_method
        
        #Log of applied edit operations
        self.edit_log = od([  ("darkcorr"   ,   False), # boolean
                              ("blurring"   ,   0), # int (width of kernel)
                              ("median"     ,   0), # int (size of filter)
                              ("crop"       ,   False), # boolean
                              ("8bit"       ,   False), # boolean
                              ("pyrlevel"   ,   0), # int (pyramide level)
                              ("is_tau"     ,   False), # boolean
                              ("is_aa"      ,   False), # boolean
                              ("vigncorr"   ,   False), # boolean (vignette corrected)
                              ("senscorr"   ,   False), # boolean (correction for sensitivity changes due to filter shifts)
                              ("dilcorr"    ,   False), # light dilution corrected
                              ("gascalib"   ,   False), # image is gas CD image
                              ("is_bin"     ,   False),
                              ("is_inv"     ,   False),
                              ("others"     ,   False),
                              ])# boolean 
        
        self._roi_abs = DEFAULT_ROI #will be set on image load
        
        self._header_raw = {}
        self.meta = od([("start_acq"     ,   datetime(1900, 1, 1)),#datetime(1900, 1, 1)),
                        ("stop_acq"      ,   datetime(1900, 1, 1)),#datetime(1900, 1, 1)),
                        ("texp"          ,   float(0.0)), # exposure time [s]
                        ("focal_length"  ,   nan), # lense focal length [mm]
                        ("pix_width"     ,   nan), # horizontal pix pitch
                        ("pix_height"    ,   nan), # vert. pix pitch
                        ("bit_depth"     ,   nan), # pix bit depth
                        ("f_num"         ,   nan), # f number of lense
                        ("read_gain"     ,   0),   # boolean (on / off)
                        ("lon"           ,   nan),   #longitude (dec deg)
                        ("lat"           ,   nan),   #latitude (dec. deg)
                        ("altitude"      ,   nan),    # in m
                        ("altitude_offs" ,   nan),    # offset in altitude above topography
                        ("elev"         ,    nan),
                        ("elev_err"     ,    nan),
                        ("azim"         ,    nan),
                        ("azim_err"     ,    nan),
                        ("filter_id"     ,   ""),
                        ("path"          ,   ""),
                        ("file_name"     ,   ""),
                        ("file_type"     ,   ""),
                        ("device_id"     ,   ""),
                        ("ser_no"        ,   ""),
                        ("wvlngth"       ,   nan)])
                      
        try:
            data, meta_info = import_method(input, meta_info) 
            input = data
            #meta_info.update(add_meta)
        except:
            pass
          
        for k, v in meta_info.iteritems():
            if self.meta.has_key(k) and isinstance(v, type(self.meta[k])):
                self.meta[k] = v
            elif self.edit_log.has_key(k):
                self.edit_log[k] = v
    
        if input is not None:                              
            self.load_input(input)
        try:
            self.set_roi_whole_image()
        except:
            pass
    
    @property
    def img(self):
        """Get / set image data"""
        return self._img
    
    @img.setter
    def img(self, val):
        """Setter for image data"""
        self._img = val.astype(self.dtype)
        
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
    def xy_aspect(self):
        """Aspect ratio (delx / dely)"""
        s = self.shape[:2]
        return s[1] / float(s[0]) 
    
    @property    
    def pyr_up_factor(self):
        """Factor to convert coordinates at current pyramid level into 
        original size coordinates
        """
        return 2 ** self.edit_log["pyrlevel"]
    
    @property
    def is_darkcorr(self):
        """Boolean specifying whether image is dark corrected"""
        return self.edit_log["darkcorr"]
    
    @property
    def is_tau(self):
        """Returns boolean whether image is a tau image or not"""
        return self.edit_log["is_tau"]
        
    @property
    def is_aa(self):
        """Returns boolean whether current image is AA image"""
        return self.edit_log["is_aa"]
    
    @property
    def is_calibrated(self):
        """Flag for image calibration status"""
        return self.edit_log["gascalib"]
        
    @property
    def is_vignetting_corrected(self):
        """Boolean stating whether image is vignetting corrected or not"""
        return self.edit_log["vigncorr"]
    
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
    def is_binary(self):
        """Attribute specifying whether image is binary image"""
        return self.edit_log["is_bin"]
    
    @property
    def is_inverted(self):
        """Flag specifying whether image was inverted or not"""
        return self.edit_log["is_inv"]
    
    @property
    def is_vigncorr(self):
        """Bool specifying whether or not image is vignetting corrected"""
        return bool(self.edit_log["vigncorr"])
    
    @property
    def is_cropped(self):
        """Boolean specifying whether image is cropped"""
        return bool(self.edit_log["crop"])
    
    @property
    def is_resized(self):
        """Boolean specifying whether image pyramid level unequals 0"""
        return False if self.pyrlevel == 0 else True
    
    @property
    def modified(self):
        """Check if this image was already modified"""
        if sum(self.edit_log.values()) > 0:
            return 1
        return 0
    
    @property
    def pyrlevel(self):
        """Returns current gauss pyramid level (stored in ``self.edit_log``)"""
        return self.edit_log["pyrlevel"]
    
    @property 
    def roi(self):
        """Returns current roi (in consideration of current pyrlevel)"""
        roi_sub = map_roi(self._roi_abs, self.edit_log["pyrlevel"])
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
            
    def set_data(self, input):
        """Try load input"""
        try:
            self.load_input(input)
        except Exception as e:
            print repr(e)
    
    def reload(self):
        """Try reload from file"""
        file_path = self.meta["path"]
        if not exists(file_path):
            warn("Image reload failed, no valid filepath set in meta info")
        else:
            self.__init__(input=file_path)
        
    def load_input(self, input):
        """Try to load input as numpy array and additional meta data"""
        try:
            if any([isinstance(input, x) for x in [str, unicode]]) and\
                                                                exists(input):
                self.load_file(input)
            
            elif isinstance(input, ndarray):
                self.img = input.astype(self.dtype)
            else:
                raise
        except:
            raise IOError("Image data could not be imported, invalid input: %s"
                        %(input))
    
    def make_histogram(self):
        """Make histogram of current image"""
        if isnan(self.meta["bit_depth"]):
            print ("Error in " + self.__str__() + ".make_histogram\n "
                "No MetaData available => BitDepth could not be retrieved. "
                "Using 100 bins and img min/max range instead")
            hist, bins = histogram(self.img, 100)
            return hist, bins
        #print "Determining Histogram"
        hist, bins = histogram(self.img, 2**(self.meta["bit_depth"]),
                               [0, 2**(self.meta["bit_depth"])])
        return hist, bins
            
    def get_brightness_range(self):
        """Analyses the Histogram to retrieve a suited brightness range
                
        Note
        ----
        Currently not in use (was originally used for App)
        
        """
        hist, bins = self.make_histogram()
        thresh = hist.max() * 0.03
        rad_low = bins[argmax(hist > thresh)]
        rad_high = bins[len(hist) - argmax(hist[::-1]>thresh)-1]
        return rad_low, rad_high, hist, bins
    
    def avg_in_roi(self, mask=None, roi_rect=None, pos_x=None, pos_y=None, 
                   radius=1):
        """Get mean value in an ROI
        
        The ROI can be specified either by providing a mask, an rectangular
        ROI, or x and y position and a specific radius. The input is dealt with
        in the specified order, i.e. if :param:`mask` is valid, none of the 
        other input parameters is tested.
        
        Parameters
        ----------
        mask : :obj:`ndarray` or :obj:`Img`
            convolution mask (e.g. DOAS FOV mask).
        roi_rect : list
            rectangular ROI ``[x0, y0, x1, y1]`` specifying upper left and 
            lower right corners of region
        pos_x : int
            detector x-position
        pos_y : int
            detector y-position
        radius : int
            radius of ROI
        
        Raises
        ------
        TypeError 
            if none of the provided input works
        
        Returns
        -------
        float
            mean value within specified ROI
        """
        try:
            mask.to_pyrlevel(self.pyrlevel)
            mask = mask.img
        except:
            pass
        try:
            data_conv = (self.img * mask.astype(float32))
        except:
            try:
                return self.img[roi_rect[1]:roi_rect[3], 
                                roi_rect[0]:roi_rect[2]].mean() 
                
            except:
                if radius == 1:
                    return self.img[pos_y, pos_x]
                else:
                    h, w = self.shape
                    y, x = ogrid[:h, :w]
                    mask = (x - pos_x)**2 + (y - pos_y)**2 < radius**2
                    data_conv = (self.img * mask.astype(float32))
             
        return data_conv.sum() / mask.sum()
# =============================================================================
#         except:
#             raise TypeError("Invalid input, failed to retrieve mean in ROI")
#             
# =============================================================================
       
    def crop(self, roi_abs=DEFAULT_ROI, new_img=False):
        """Cut subimage specified by rectangular ROI
        
        :param list roi_abs: region of interest (i.e. ``[x0, y0, x1, y1]``)
            in ABSOLUTE image coordinates. The ROI is automatically converted 
            with respect to current pyrlevel
        :param bool new_img: creates and returns a new image object and leaves 
            this one uncropped        
        :return:
            - Img, cropped image
        """
        if self.edit_log["crop"]:
            warn("Cropping image that was already cropped...")
        self.roi_abs = roi_abs #updates current roi_abs setting
        roi = self.roi #self.roi is @property method and takes care of ROI conv
        sub = self.img[roi[1]:roi[3], roi[0]:roi[2]] 
        im = self
        if new_img:
            im = self.duplicate()
#        im._roi_abs = roi
        im.edit_log["crop"] = True
        im.img = sub
        return im
    
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
        from pyplis.processing import model_dark_image
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
            corr = self.img - dark
        except:
            corr = self.img - dark.img
        corr[corr <= 0] = finfo(float32).eps
        self.img = corr
        self.edit_log["darkcorr"] = True
    
    def correct_vignetting(self, mask, new_state=True):
        """Apply vignetting correction
        
        Performs either of the following operations::
        
            self.img * mask     (if input param ``new_state=False``)
            self.img / mask     (if input param ``new_state=True``)
            
        :param ndarray mask: vignetting correction mask
        :param bool reverse: if False, the inverse correction is applied (img
            needs to be corrected)
        """
        try:
            mask = mask.img
        except:
            pass
        if new_state == self.edit_log["vigncorr"]:
            return self
        try:
            if self.edit_log["vigncorr"]: #then, new_state is 0, i.e. want uncorrected image
                self.img = self.img * mask
            else: #then, new_state is 1, i.e. want corrected image
                self.img = self.img / mask
        except Exception as e:
            raise type(e), type(e)(e.message + "\nPlease check vignetting mask")  
        self.edit_log["vigncorr"] = new_state
        self.vign_mask = mask
        return self
        
    def set_roi_whole_image(self):
        """Set current ROI to whole image area based on shape of image data"""
        h, w = self.img.shape[:2]
    
        self._roi_abs = [0, 0, w * 2**self.pyrlevel, h * 2**self.pyrlevel]     
    
    def apply_median_filter(self, size_final=3):
        """Apply a median filter to 
        
        :param tuple shape (3,3): size of the filter        
        """
        diff = int(size_final - self.edit_log["median"])
        if diff > 0:
            self.img = median_filter(self.img, diff)
            self.edit_log["median"] += diff
        return self
        
    def add_gaussian_blurring(self, sigma_final=1):
        """Add blurring to image
        
        :param int sigma_final: the final width of gauss blurring kernel
        """
        diff = int(sigma_final - self.edit_log["blurring"])
        if diff > 0:
            self.apply_gaussian_blurring(diff)
        return self
                
    def apply_gaussian_blurring(self, sigma, **kwargs):
        """Add gaussian blurring 
        
        Uses :class:`scipy.ndimage.filters.gaussian_filter`
        
        :param int sigma: amount of blurring
        """
        self.img = gaussian_filter(self.img, sigma, **kwargs)
        self.edit_log["blurring"] += sigma   
    
    def to_binary(self, threshold=None, new_img=False):
        """Convert image to binary image using threshold
        
        Note
        ----
        
        The changes are applied to this image object
        
        Parameters
        ----------
        threshold : float
            threshold, if None, use mean value of image data
            
        Returns
        -------
        Img
            binary image
        """
        if threshold is None:
            threshold = self.mean()
        mask = (self.img > threshold).astype(uint8)
        if new_img:
            return Img(mask, dtype=uint8, is_bin=True)
        self.img = (self.img > threshold).astype(uint8)
        self.edit_log["is_bin"] = True
        return self
        
    
        
    def invert(self):
        """Invert image
        
        Note
        ----
        
        Does not yet work for tau images
        
        Returns
        -------
        Img
            inverted image object
        
        """
        if self.is_tau:
            raise NotImplementedError("Tau images can not yet be inverted")
        elif self.is_binary:
            inv = ~self.img/255
            self.img = (inv).astype(uint8)
            
            return self
        else:
            if not self.is_8bit:
                self._to_8bit_int(new_img=False)
            self.img = invert(self.img)
        self.edit_log["is_inv"] = not self.edit_log["is_inv"]
        return self
            
    def dilate(self, kernel=ones((9,9), dtype=uint8)):
        """Apply morphological transformation Dilation to image
        
        Uses :func:`cv2.dilate` for dilation. The method requires specification
        of a smoothing kernel, if unspecified, a 9x9 neighbourhood is used
        
        Note
        ----
        
        This operation can only be performed to binary images, use 
        :func:`to_binary` if applicable.
        
        Parameters
        ----------
        kernel : array
            kernel used for :func:`cv2.dilate`, default is 9x9 kernel
        
        Returns
        -------
        Img 
            dilated binary image
        """
        if not self.is_binary:
            raise AttributeError("Img needs to be binary, use method to_binary")

        self.img = dilate(self.img, kernel=kernel)
        self.edit_log["others"] = True
        return self
    
    def erode(self, kernel=ones((9,9), dtype=uint8)):
        """Apply morphological transformation Erosion to image
        
        Uses :func:`cv2.erode` to apply erosion. The method requires 
        specification of a kernel, if unspecified, a 9x9 neighbourhood is used
        
        Note
        ----
        
        This operation can only be performed to binary images, use 
        :func:`to_binary` if applicable.
        
        Parameters
        ----------
        kernel : array
            kernel used for :func:`cv2.dilate`, default is 9x9 kernel
        
        Returns
        -------
        Img 
            dilated binary image
        """
        if not self.is_binary:
            raise AttributeError("Img needs to be binary, use method to_binary")

        self.img = erode(self.img, kernel=kernel)
        self.edit_log["others"] = True
        return self
    
    def fit_2d_poly(self, mask=None, polyorder=3, pyrlevel=4, **kwargs):
        """Fit 2D surface poly to data
        
        Parameters
        ----------
        mask : array
            mask specifying pixels considered for the fit (if None, then all 
            pixels of the image data are considered
        polyorder : int
            order of polynomial for fit (default=3)
        pyrlevel : int
            level of Gauss pyramid at which the fit is performed (relative to
            Gauss pyramid level of input data)
        **kwargs :
            additional optional keyword args passed to :class:`PolySurfaceFit`
        
        Returns
        -------
        Img
            new image object corresponding to fit results
        """
        if mask is not None:
            try:
                if not mask.shape == self.shape:
                    warn("Shape of input mask does not match image shape, "
                        "trying to update pyrlevel in mask")
                    try:
                        mask.to_pyrlevel(self.pyrlevel)
                        if not mask.shape == self.shape:
                            raise Exception
                    except:
                        raise Exception
            except:
                warn("Failed to match shapes of input mask and image data, "
                     "using all pixels for fit")
                mask = None
                                    
        fit = PolySurfaceFit(self.img, mask, polyorder, pyrlevel)
        try:
            if fit.model.shape == self.shape:
                print "Fit successful"
                return Img(fit.model)
            raise Exception
        except:
            raise Exception("Poly surface fit failed in Img object")
    
    def to_tau(self, bg, new_img=True):
        """Convert into tau image
        
        Converts this image into a tau image using a provided input 
        background image (which is used without any modifications).
        
        Note
        ----
        By default, creates and returns new instance of :class:`Img` object
        (i.e. this object remains unchanged if not other specified using 
         :param:`new_img`)
        
        Parameters
        ----------
        bg : Img
            background image used to determin tau image (REMAINS UNCHANGED, NO
            MODELLING PERFORMED HERE)
        new_img : bool
            boolean specifying whether this object remains unchanged
        
        Returns
        -------
        Img 
            new Img object containing tau image data 
            (this object remains unchanged)
        """
        tau = self
        if new_img:
            tau = self.duplicate()
        if isinstance(bg, Img):
            bg = bg.img
        
        r = bg / tau.img
        r[r <= 0] = finfo(float).eps
        tau.img = log(r)
        tau.edit_log["is_tau"] = True
        return tau
        
    def to_pyrlevel(self, final_state=0):
        """Down / upscale image to a given pyramid level"""
        steps = final_state - self.edit_log["pyrlevel"]
        if steps > 0:
            return self.pyr_down(steps)
        elif steps < 0:
            return self.pyr_up(-steps)
        return self
     
    def pyr_down(self, steps=0):
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
        self.edit_log["others"] = True
        return self
    
    def bytescale(self, cmin=None, cmax=None, high=255, low=0):
        """Convert image to 8 bit integer values
        
        :param float cmin: minimum intensity for mapping, if None, the current 
            ``self.min()`` is used.
        :param float cmax: maximum intensity for mapping, if None, the current 
            ``self.max()`` is used.
        :param int high: mapping value of cmax
        :param int low: mapping value of cmin
        """
        img = deepcopy(self)
        img.img = bytescale(self.img, cmin, cmax, high, low)
        return img
        
    def _to_8bit_int(self, current_bit_depth=None, new_img=True):
        """Convert image to 8 bit representation and return new image object
        
        :returns array 
        .. note::
        
            1. leaves this image unchanged
            #. if the bit_depth is unknown or unspecified in ``self.meta``, then
            
        """
        if current_bit_depth == None:
            current_bit_depth = self.meta["bit_depth"]
            
        if isnan(current_bit_depth):
            cmax = None
        else:
            cmax = 2**(current_bit_depth) - 1

        sc = bytescale(self.img, cmin = 0, cmax = cmax)

        if new_img:
            img = self.duplicate()
        else:
            img = self
            self.edit_log["8bit"] = True
        img.meta["bit_depth"] = 8
        img.img = sc
        return img
    
    def is_8bit(self):
        """Flag specifying whether image is 8 bit"""
        if self.meta["bit_depth"] == 8:
            return True
        return False
        
    def print_meta(self):
        """Print current image meta information"""
        for key, val in self.meta.iteritems():
            print "%s: %s\n" %(key, val)
        
    def make_info_header_str(self):
        """Make header string for image (using image meta information)"""
        try:
            return ("Acq.: %s, texp: %.2f s, rgain %s\n"
                    "pyrlevel: %d, roi_abs: %s" %(self.meta["start_acq"].\
                    strftime('%H:%M:%S'), self.meta["texp"],\
                    self.meta["read_gain"], self.pyrlevel, self.roi_abs)) 
        except Exception as e:
            print repr(e)
            return self.meta["file_name"]
        
    def duplicate(self):
        """Duplicate this image"""
        #print self.meta["file_name") + ' successfully duplicated'
        return deepcopy(self)
    
    def normalise(self, blur=1):
        """Normalise this image"""
        new = self.duplicate()
        if self.edit_log["blurring"] == 0 and blur != 0:
            new.add_gaussian_blurring(blur)
            new.img = new.img / new.img.max()
        return new
        
    def mean(self):
        """Returns mean value of current image data"""
        return self.img.mean()
        
    def sum(self):
        """The sum of all pixel values"""
        return self.img.sum()
    
    def std(self):
        """Returns standard deviation of current image data"""
        return self.img.std()
        
    def min(self):
        """Returns minimum value of current image data"""
        return self.img.min()
    
    def max(self):
        """Returns maximum value of current image data"""
        return self.img.max()
    
    def set_val_below_thresh(self, val, threshold):
        """Sets value in all pixels with intensities below threshold
        
        Note
        ----
        Modifies this Img object
        
        Parameters
        ----------
        val : float
            new value for all pixels below the input threshold
        threshold : float
            considered intensity threshold
        """
        mask = self.img < threshold
        self.img[mask] = val
        self.edit_log["others"] = True
    
    def set_val_above_thresh(self, val, threshold):
        """Sets value in all pixels with intensities above threshold
        
        Note
        ----
        Modifies this Img object
        
        Parameters
        ----------
        val : float
            new value for all pixels above the input threshold
        threshold : float
            considered intensity threshold
        """
        mask = self.img > threshold
        self.img[mask] = val
        self.edit_log["others"] = True
        
    def blend_other(self, other, fac=0.5):
        """Blends another image to this and returns new Img object
        
        Uses cv2 :func:`addWeighted` method"
        
        :param float fac: percentage blend factor (between 0 and 1)
        """
        if not 0 < fac < 1:
            raise ValueError("Invalid input valued for fac: %.2f ... "
                "must be between 0 and 1")
        try:
            other = other.img
        except:
            pass
        if any([x < 0 for x in [self.img.min(), other.min()]]):
            raise ValueError("Could not blend images, has one of the input "
                "images has negative values, you might remap the value (e.g. "
                "using _to_8bit_int method)")
        im = addWeighted(self.img, 1-fac, other, fac, 0)
        return Img(im)
        
    def meta(self, meta_key):
        """Returns current meta data for input key"""
        return self.meta[meta_key]

    def load_file(self, file_path):
        """Try to import file specified by input path"""
        ext = splitext(file_path)[-1]
        try:
            self.load_fits(file_path)
        except:
            self.img = imread(file_path).astype(self.dtype)
        self.meta["path"] = abspath(file_path)
        self.meta["file_name"] = basename(file_path)
        self.meta["file_type"] = ext
    
    def load_fits(self, file_path):
        """Import a FITS file 
        
        This import method assumes, that data and corresponding meta-info is 
        stored in the first HDU of the FITS file (index = 0).
        
        `Fits info <http://docs.astropy.org/en/stable/io/fits/>`_
        """
        hdu = fits.open(file_path)
        head = hdu[0].header 
        self._header_raw = head
        self.img = hdu[0].data.astype(self.dtype)
        hdu.close()
        try:
            if head["CAMTYPE"] == 'EC2':
                self.import_ec2_header(head)
        except:
            pass
        editkeys = self.edit_log.keys()
        metakeys = self.meta.keys()
        for key, val in head.iteritems():
            k = key.lower()
            if k in editkeys:
                self.edit_log[k] = val
            elif k in metakeys:
                try:
                    self.meta[k] = datetime.strptime(val, "%Y%m%d%H%M%S%f")
                except:
                    if val == "nan":
                        val = nan
                    self.meta[k] = val
        try:
            self._roi_abs = hdu[1].data["roi_abs"]
        except:
            pass
        try:
            self.vign_mask = hdu[2].data
            print "Fits file includes vignetting mask"
        except:
            pass
    
    def _prep_meta_dict_fits(self):
        """Prepares current meta-information for storage in FITS header"""
        d = od()
        for k, v in self.meta.iteritems():
            try:
                d[k] = v.strftime("%Y%m%d%H%M%S%f")
            except:
                try:
                    if isnan(v):
                        v = "nan"
                except:
                    pass
                d[k] = v
        return d
    
    def save_as_fits(self, save_dir=None, save_name=None):
        """Save this image as FITS file
        
        Parameters
        ----------
        save_dir : str
            optional, if None (default), then the current working directory is
            used
        save_name : str
            opional, if None (default), try to use file name of this object
            (if set) or use default name
        
        Returns
        -------
        str 
            name of saved file
        """
        save_dir = abspath(save_dir)
        if not isdir(save_dir): #save_dir is a file path
            save_name = basename(save_dir)
            save_dir = dirname(save_dir)
        if save_name is None:
            if self.meta["file_name"] == "":
                save_name = "pyplis_img_name_undefined.fts"
            else:
                save_name = self.meta["file_name"].split(".")[0] + ".fts"
        else:
            save_name = save_name.split(".")[0] + ".fts"
        
        hdu = fits.PrimaryHDU()
        hdu.data = self._img
        hdu.header.update(self.edit_log)
        hdu.header.update(self._header_raw)
        hdu.header.update(self._prep_meta_dict_fits())
        hdu.header.append()
    
        roi_abs = fits.BinTableHDU.from_columns([fits.Column(name = "roi_abs",\
                                format = "I", array = self._roi_abs)])
        hdulist = fits.HDUList([hdu, roi_abs])
        if isinstance(self.vign_mask, ndarray):
            hdulist.append(fits.ImageHDU(data = self.vign_mask.astype(uint8)))
        path = join(save_dir, save_name)
        if exists(path):
            print "Image already exists at %s and will be overwritten" %path
            remove(path)
        hdulist.writeto(path)
        return save_name
        
    def import_ec2_header(self, ec2header):
        """Import image meta info for ECII camera type from FITS file header"""
        gain_info = {"LOW"  :   0,
                     "HIGH" :   1}
                     
         
        self.meta["texp"] = float(ec2header['EXP'])*10**-6        #unit s
        self.meta["bit_depth"] = 12
        self.meta["device_id"] = 'ECII'        
        self.meta["file_type"] = 'fts'
        self.meta["start_acq"] = datetime.strptime(ec2header['STIME'],\
                                                    '%Y-%m-%d %H:%M:%S.%f')
        self.meta["stop_acq"] = datetime.strptime(ec2header['ETIME'],\
                                                    '%Y-%m-%d %H:%M:%S.%f')
        self.meta["read_gain"] = gain_info[ec2header['GAIN']]
        self.meta["pix_width"] = self.meta["pix_height"] = 4.65e-6 #m
    
    """PLOTTING AND VISUALSATION FUNCTIONS"""  
    def get_cmap(self, vmin=None, vmax=None, **kwargs):
        """Determine and return default cmap for current image"""
        if self.is_tau or self.is_aa:
            return cmaps.viridis
# =============================================================================
#             if vmin is None:
#                 vmin = self.min()
#             if vmax is None:
#                 vmax = self.max()
#             return shifted_color_map(vmin, vmax, cmaps.RdBu)
# =============================================================================
        return cmaps.gray
        
    def show(self, zlabel=None, tit=None,**kwargs):
        """Plot image"""
        return self.show_img(zlabel,tit,**kwargs)

    def show_img(self, zlabel=None, tit=None, cbar=True, ax=None,
                 zlabel_size=18, **kwargs):
        """Show image using matplotlib method imshow"""
        if not "cmap" in kwargs.keys():
            kwargs["cmap"] = self.get_cmap(**kwargs)
        new_ax = False
        try:
            fig = ax.figure
            ax = ax
        except:
            fig = figure(facecolor='w', edgecolor='none', figsize=(12,7))  
            ax = fig.add_subplot(111)
            new_ax = True
        
        im = ax.imshow(self.img, **kwargs)
        if cbar:
            cb = fig.colorbar(im, ax=ax)
            if isinstance(zlabel, str):
                cb.set_label(zlabel, fontsize=zlabel_size)
        if not isinstance(tit, str):
            tit = self.make_info_header_str()
        ax.set_title(tit, fontsize=14)
        if new_ax:
            tight_layout()
        return ax
        
    def show_img_with_histo(self, **kwargs):
        """Show image using plt.imshow"""
        if not "cmap" in kwargs.keys():
            kwargs["cmap"] = self.get_cmap()
        fig = figure(figsize = (13, 5), dpi=80, 
                     facecolor='w', edgecolor='k')  
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 

        ax = fig.add_subplot(gs[0])
        im = ax.imshow(self.img, **kwargs)
        fig.colorbar(im, ax = ax)
        ax.set_title(self.make_info_header_str(), fontsize = 9)
        ax2 = fig.add_subplot(gs[1])
        self.show_histogram(ax2)
        tight_layout()
        return ax
    
    def apply_registration_shift(self, dx_abs=0.0, dy_abs=0.0):
        """Applies constant image registration shift to this object
        
        Parameters
        ----------
        dx_abs : float
            shift in x-direction
        dy_abs : float
            shift in y-direction
        """
        dx_rel, dy_rel = dx_abs / 2**self.pyrlevel, dy_abs / 2**self.pyrlevel
        print("Shifting", dx_rel, dy_rel)
        self.img = shift(self.img, shift=(dy_rel, dx_rel))
            
    def show_histogram(self, ax=None):
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
        #print i, f
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
        print self.__str__()
        
    """MAGIC METHODS"""
    def __str__(self):
        """String representation"""
        s = "\n-----------\npyplis Img\n-----------\n\n"
        s += "Min / Max intensity: %s - %s\n" %(self.min(), self.max())
        s += "Mean intensity: %s\n" %(self.img.mean())
        s += "Shape: %s\n" %str(self.shape)
        s += "ROI (abs. coords): %s\n" %self.roi_abs
        s += "\nMeta information\n-------------------\n"
        for k, v in self.meta.iteritems():
            s += "%s: %s\n" %(k, v)
        s += "\nEdit log\n-----------\n"
        for k, v in self.edit_log.iteritems():
            s += "%s: %s\n" %(k, v)
        return s
            
    def __call__(self):
        """Return image numpy array on call"""
        return self.img
        
    def __add__(self, val):
        """Add another image object
        
        :param Img img_obj: object to be added
        :return: new image object
        """
        try:
            im = self.duplicate()
            im.img = self.img + val.img
            return im
        except:
            try:
                im = self.duplicate()
                im.img = self.img + val
                return im
            except:
                raise TypeError("Could not add value %s to image" %type(val))
        
            
    def __sub__(self, val):
        """Subtract another image object
        
        :param Img img_obj: object to be subtracted
        :return: new image object
        """
        try:
            im = self.duplicate()
            im.img = self.img - val.img
            return im
        except:
            try:
                im = self.duplicate()
                im.img = self.img - val
                return im
            except:
                raise TypeError("Could not subtract value %s from image" 
                                                                %type(val))
    
    def __mul__(self, val):
        """Multiply another image object
        
        :param Img img_obj: object to be multiplied
        :return: new image object
        """
        try:
            im = self.duplicate()
            im.img = self.img * val.img
            return im
        except:
            try:
                im = self.duplicate()
                im.img = self.img * val
                return im
            except:
                raise TypeError("Could not multilply image with value %s" 
                                                                %type(val))

    def __div__(self, val):
        """Divide another image object
        
        :param Img img_obj: object to be multiplied
        :return: new image object
        """
        try:
            im = self.duplicate()
            im.img = self.img / val.img
            return im
        except:
            try:
                im = self.duplicate()
                im.img = self.img / val
                return im
            except:
                raise TypeError("Could not divide image with value %s" 
                                                                %type(val))

def model_dark_image(texp, dark, offset):
    """Model a dark image for input image based on dark and offset images
    
    Determine a modified dark image (D_mod) from the current dark and 
    offset images. The dark image is determined based on the image 
    exposure time of the image object to be corrected (t_exp,I). 
    D_mod represents dark and offset signal for this image object and 
    is then subtracted from the image data.

    Formula for modified dark image:

    .. math::

        D_{mod} = O + \\frac{(D - O)*t_{exp,I}}{(t_{exp, D}-t_{exp, O})} 
        
    :param Img img: the image for which dark and offset is modelled
    :param Img dark: dark image object (dark with long(est) exposure time)
    :param Img offset: offset image (dark with short(est) exposure time)
    :returns: - :class:`Img`, modelled dark image 
 
    """
    if not all([x.meta["texp"] > 0.0 for x in [dark, offset]]):
        raise ImgMetaError("Could not model dark image, invalid value for "
            "exposure time encountered for at least one of the input images")
    if any([x.modified for x in [dark, offset]]):
        warn("Images used for modelling dark image are modified")

    dark_img = (offset.img + (dark.img - offset.img) * texp/
                             (dark.meta["texp"] - offset.meta["texp"]))

    return Img(dark_img, texp=texp, pyrlevel=offset.pyrlevel)
                
class ProfileTimeSeriesImg(Img):
    """Image representing time series of line profiles
    
    The y axis of the profile image corresponds to the actual profiles 
    (retrieved from the individual images) and the x axis corresponds to the 
    image time axis (i.e. the individual frames). Time stamps (mapping of 
    x indices) can also be stored in this object.
    
    Example usage is, for instance to represent ICA time series retrieved
    along a profile (e.g. using :class:`LineOnImage`) for plume speed cross 
    correlation
    """
    def __init__(self, img_data=None, time_stamps=asarray([]), img_id="",
                 dtype=float32, profile_info_dict={}, **meta_info):
        self.img_id = img_id
        self.time_stamps = asarray(time_stamps)
        self.profile_info = {}
        if isinstance(profile_info_dict, dict):
            self.profile_info = profile_info_dict
        #Initiate object as Img object
        super(ProfileTimeSeriesImg, self).__init__(input=img_data,
                                                   dtype=dtype, **meta_info)
                                                
    @property
    def img(self):
        """Get / set image data"""
        return self._img
    
    @img.setter
    def img(self, val):
        """Setter for image data"""
        if not isinstance(val, ndarray) or val.ndim != 2:
            raise ValueError("Could not set image data, need 2 dimensional"
                " numpy array as input")
        self._img = val
        num = val.shape[1]
        if not len(self.time_stamps) == num:
            self.time_stamps = asarray([datetime(1900,1,1)] * num)
        
    def _format_check(self):
        """Checks if current data is of right format"""
        if not all([isinstance(x, ndarray) for x in [self._img,\
                                                self.time_stamps]]):
            raise TypeError("self.img and self.time_stamps must be numpy "
                "arrays")
        if not len(self.time_stamps) == self.shape[1]:
            raise ValueError("Mismatch in array lengths")

    @property
    def start(self):
        """Returns first datetime from ``self.time_stamps``"""
        try:
            return self.time_stamps[0]
        except:
            print "no time information available, return 1/1/1900"
            return datetime(1900,1,1)
    
    @property
    def stop(self):
        """Returns first datetime from ``self.time_stamps``"""
        try:
            return self.time_stamps[-1]
        except:
            print "no time information available, return 1/1/1900"
            return datetime(1900,1,1)
              
    def save_as_fits(self, save_dir=None, save_name=None,
                     overwrite_existing=True):
        """Save stack as FITS file
        
        Parameters
        ----------
        save_dir : str
            directory where image is stored (can also be full file path, then
            parameter ``save_name`` is not considered)
        save_name : str
            name of file
        overwrite_existing : bool
            if True, an existing file with the same name will be overwritten
        """
        self._format_check()
        save_dir = abspath(save_dir) #returns abspath of current wkdir if None
        if not isdir(save_dir): #save_dir is a file path
            save_name = basename(save_dir)
            save_dir = dirname(save_dir)
        if save_name is None:
            save_name = "pyplis_profile_tseries_id_%s_%s_%s_%s.fts"\
                %(self.img_id, self.start.strftime("%Y%m%d"),\
                self.start.strftime("%H%M"), self.stop.strftime("%H%M"))
        else:
            save_name = save_name.split(".")[0] + ".fts"
    
        hdu = fits.PrimaryHDU()
        time_strings = [x.strftime("%Y%m%d%H%M%S%f") for x in self.time_stamps]
        col1 = fits.Column(name = "time_stamps", format = "25A", array =\
            time_strings)
    
        cols = fits.ColDefs([col1])
        arrays = fits.BinTableHDU.from_columns(cols)
        
        hdu.data = self._img
        hdu.header.update(self.edit_log)
        hdu.header["img_id"] = self.img_id
        for key, val in self.profile_info.iteritems():
            if key == "_roi_abs_def":
                try:
                    hdu.header["_roi_abs_def"] = dumps(val)
                except:
                    warn("Failed to write roi_abs_def")
            else:
                hdu.header[key] = val
    
        hdu.header.append()
        hdulist = fits.HDUList([hdu, arrays])
        path = join(save_dir, save_name)
        if exists(path):
            try:
                print "Image already exists at %s and will be overwritten" %path
                remove(path)
            except:
                warn("Failed to delete existing file...")
        try:
            hdulist.writeto(path, clobber=overwrite_existing)
        except:
            warn("Failed to save FITS File (check previous warnings)")
    
    def _profile_dict_keys(self, profile_type="LineOnImage"):
        """Returns profile dictionary keys for input profile type"""
        d = {"LineOnImage"  :   LineOnImage().to_dict().keys()}
        return d[profile_type]
        
    def load_fits(self, file_path, profile_type="LineOnImage"):
        """Load stack object (fits)
        
        :param str file_path: file path of fits image
        """
        
        if not exists(file_path):
            raise IOError("Img could not be loaded, path %s does not "
                          "exist" %file_path)
        hdu = fits.open(file_path)
        self.img = asarray(hdu[0].data)
        prep = Img().edit_log
        try:
            profile_keys = self._profile_dict_keys(profile_type)
        except:
            profile_keys = []
            print "Failed to load profile info dictionary"
        
        for key, val in hdu[0].header.iteritems():
            k = key.lower()
            if k in prep.keys():
                self.edit_log[k] = val
            elif k in profile_keys:
                if k == "_roi_abs_def":
                    self.profile_info[k] = loads(val)
                else:
                    self.profile_info[k] = val
        self.img_id = hdu[0].header["img_id"]
        
        try:
            self.time_stamps = asarray([datetime.strptime(x, "%Y%m%d%H%M%S%f")\
                for x in hdu[1].data["time_stamps"]])
        except:
            print "Failed to import time stamps"
        self._format_check()
            