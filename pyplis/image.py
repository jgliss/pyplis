# -*- coding: utf-8 -*-
"""Image base module"""
from astropy.io import fits
from matplotlib import gridspec
import matplotlib.cm as cmaps
from matplotlib.pyplot import imread, figure, tight_layout
from numpy import ndarray, argmax, histogram, uint, nan, linspace,\
    isnan, uint8, float32, finfo, ones, invert, log
from os.path import abspath, splitext, basename, exists, join, isdir, dirname
from os import remove
from warnings import warn
from datetime import datetime
from decimal import Decimal
from cv2 import pyrDown, pyrUp, addWeighted, dilate
from scipy.ndimage.filters import gaussian_filter, median_filter
from collections import OrderedDict as od
from copy import deepcopy

from .helpers import shifted_color_map, bytescale, map_roi, check_roi
from .exceptions import ImgMetaError
from .optimisation import PolySurfaceFit

class Img(object):
    """ Image base class
    
    Implementation of image object for :mod:`pyplis` library. Images are
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
            return input
            
        self._img = None #: the actual image data
        self.dtype = dtype
        self.vign_mask = None
        
        # custom data import method (optional on class initialisation)
        self.import_method = import_method
        
        #Log of applied edit operations
        self.edit_log = od([  ("darkcorr"   ,   0), # boolean
                              ("blurring"   ,   0), # int (width of kernel)
                              ("median"     ,   0), # int (size of filter)
                              ("crop"       ,   0), # boolean
                              ("8bit"       ,   0), # boolean
                              ("pyrlevel"   ,   0), # int (pyramide level)
                              ("is_tau"     ,   0), # boolean
                              ("is_aa"      ,   0), # boolean
                              ("vigncorr"   ,   0), # boolean (vignette corrected)
                              ("senscorr"   ,   0), # boolean (correction for sensitivity changes due to filter shifts)
                              ("dilcorr"    ,   0), # light dilution corrected
                              ("gascalib"   ,   0), # image is gas CD image
                              ("is_bin"     ,   0),
                              ("is_inv"     ,   0),
                              ("others"     ,   0),
                              ])# boolean 
        
        self._roi_abs = [0, 0, 9999, 9999] #will be set on image load
        
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
                        ("filter"        ,   ""),
                        ("path"          ,   ""),
                        ("file_name"     ,   ""),
                        ("file_type"     ,   ""),
                        ("device_id"     ,   ""),
                        ("ser_no"        ,   ""),
                        ("temperature"   ,   float(0.0)),   # temperature in Celsius
                        ("img_idx"       ,   int(0)),    # index in multi-img file formats as *.fits
                        ("custom1"       ,   nan),  # 3 dummy attributes which can be used for custom defined parameters
                        ("custom2"       ,   nan),
                        ("custom3"       ,   nan)])
                        
        try:
            data, meta_info = import_method(input, meta_info) 
            #print meta_info
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
        self._img = val
    
    @property
    def is_vigncorr(self):
        """Bool specifying whether or not image is vignetting corrected"""
        return bool(self.edit_log["vigncorr"])
        
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
    
    def crop(self, roi_abs=[0, 0, 9999, 9999], new_img=False):
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
        from pyplis.processing import model_dark_image
        if self.modified:
            print ("Dark correction not possible, it was either already "
            "performed, the image was already modified")
            return 
            
        dark = model_dark_image(self, dark, offset)                
        self.subtract_dark_image(dark)
        
        return dark
    
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
        self.edit_log["darkcorr"] = 1
        
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
    
    def to_binary(self, threshold=None):
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
            the binary image
        """
        if threshold is None:
            threshold = self.mean()
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
            
    def dilate(self, kernel=None):
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
            kernel used for :func:`cv2.dilate`, if None a 9x9 array is used::
            
                kernel = np.ones((9,9), dtype=np.uint8)
        
        Returns
        -------
        Img 
            dilated binary image
        """
        if not self.is_binary:
            raise AttributeError("Img needs to be binary, use method to_binary")
        if kernel is None:
            kernel = ones((9,9), dtype=uint8)
        self.img = dilate(self.img, kernel=kernel)
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
    
    def to_tau(self, bg):
        """Convert into tau image
        
        Converts this image into a tau image using a provided input 
        background image (which is used without any modifications).
        
        Parameters
        ----------
        bg : Img
            background image used to determin tau image (REMAINS UNCHANGED, NO
            MODELLING PERFORMED HERE)
        
        Returns
        -------
        Img 
            new Img object containing tau image data 
            (this object remains unchanged)
        """
        tau = self.duplicate()
        if isinstance(bg, Img):
            bg = bg.img
        
        r = bg / tau.img
        r[r <= 0] = finfo(float).eps
        tau.img = log(r)
        tau.edit_log["is_tau"] = True
        return tau
        
    def to_pyrlevel(self, final_state=0):
        """Down / upscale image to a given pyramide level"""
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
        self.edit_log["others"] = 1
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
            self.edit_log["8bit"] = 1
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
    def is_tau(self):
        """Returns boolean whether image is a tau image or not"""
        return self.edit_log["is_tau"]
        
    @property
    def is_aa(self):
        """Returns boolean whether current image is AA image"""
        return self.edit_log["is_aa"]
        
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
    def modified(self):
        """Check if this image was already modified"""
        if sum(self.edit_log.values()) > 0:
            return 1
        return 0

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
    

        
#==============================================================================
#         except:
#             self.img = imread(file_path).astype(self.dtype)
#==============================================================================
            
    def load_fits(self, file_path):
        """Import a FITS file 
        
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
            if vmin is None:
                vmin = self.min()
            if vmax is None:
                vmax = self.max()
            return shifted_color_map(vmin, vmax, cmaps.RdBu)
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
            