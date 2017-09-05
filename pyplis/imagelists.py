# -*- coding: utf-8 -*-
"""
Image list objects of pyplis library

.. todo::

    1. Update indices in linked lists and linked dark / offset lists whenever
       the attribute :attr:`files` is changed in an image list (e.g. in
       :func:`clear`, :func:`pop`)
"""
from numpy import asarray, zeros, argmin, arange, ndarray, float32, ceil,\
    isnan, logical_or, ones, uint8, finfo, exp
from datetime import timedelta, datetime, date
#from bunch import Bunch
from pandas import Series, DataFrame
from matplotlib.pyplot import figure, draw, subplots, ion, ioff, close
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
from warnings import warn
from os.path import exists, abspath, dirname, join, basename
from os import mkdir
from collections import OrderedDict as od
from cv2 import dilate

from traceback import format_exc

from .image import Img
from .inout import load_img_dummy
from .exceptions import ImgMetaError
from .setupclasses import Camera
from .geometry import MeasGeometry
from .processing import ImgStack, PixelMeanTimeSeries, LineOnImage,\
                                                            model_dark_image
from .optimisation import PolySurfaceFit                                                    
from .plumebackground import PlumeBackgroundModel
from .plumespeed import OptflowFarneback, LocalPlumeProperties
from .helpers import check_roi, map_roi, _print_list, closest_index,exponent

class BaseImgList(object):
    """Basic image list object
    
    Basic class for image list objects providing indexing and image loading 
    functionality
    
    In this class, only the current image is loaded at a time while
    :class:`ImgList` loads current and next image whenever the index is 
    changed (e.g. required for :attr:`optflow_mode`)
    
    This object and all objects inheriting from this are fundamentally based 
    on a list of image file paths, which are dynamically loaded and processed
    during usage.
    
    Parameters
    ----------
    files : list
        list with image file paths
    list_id : str
        a string used to identify this list (e.g. "second_onband")
    list_type : str
        type of images in list (please use "on" or "off")
    camera : Camera
        camera specifications
    init : bool
        if True, list will be initiated and files loaded (given that image 
        files are provided on input)
    **img_prep_settings
        additional keyword args specifying image preparation settings applied
        on image load        
    """
    def __init__(self, files=[], list_id=None, list_type=None,
                 camera=None, geometry=None, init=True, **img_prep_settings):
        
        #this list will be filled with filepaths
        self.files = []
        #id of this list
        self.list_id = list_id
        self.list_type = list_type
        
        self.filter = None #can be used to store filter information
        self._meas_geometry = None
        
        
        self.set_camera(camera)
        
        self.edit_active = True
        #the following dictionary contains settings for image preparation
        #applied on image load
        self.img_prep = {"blurring"     :   0, #width of gauss filter
                         "median"       :   0, #width of median filter
                         "crop"         :   False,
                         "pyrlevel"     :   0, #int, gauss pyramide level
                         "8bit"         :   0} #to 8bit 
        
        self._roi_abs = [0, 0, 9999, 9999] #in original img resolution
        self._auto_reload = True
        
        self._list_modes = {} #init for :class:`ImgList` object
        
        self.vign_mask = None #a vignetting correction mask can be stored here
        self.loaded_images = {"this"  :    None}
        #used to store the img edit state on load
        self._load_edit = {}

        self.index = 0
        self.next_index = 0
        self.prev_index = 0
        
        # update image preparation settings (if applicable)
        for key, val in img_prep_settings.iteritems():
            if self.img_prep.has_key(key):
                self.img_prep[key] = val
                
        if bool(files):
            self.add_files(files)
        
        if isinstance(geometry, MeasGeometry):
            self.meas_geometry = geometry
            
        if self.data_available and init:
            self.load()
            
    """ATTRIBUTES / DECORATORS"""
    @property
    def start(self):
        """Acquisistion time of first image"""
        try:
            return self.start_acq[0]
        except IndexError:
            raise IndexError("No data available")
        
    @property
    def stop(self):
        """Start acqusition time of last image"""
        try:
            return self.start_acq[-1]
        except IndexError:
            raise IndexError("No data available")
        
    @property
    def this(self):
        """Current image"""
        return self.current_img()         
        
    @property
    def meas_geometry(self):
        """Measurement geometry"""
        return self._meas_geometry
    
    @meas_geometry.setter
    def meas_geometry(self, val):
        if not isinstance(val, MeasGeometry):
            raise TypeError("Could not set meas_geometry, need MeasGeometry "
                "object")
        self._meas_geometry = val
            
    @property
    def auto_reload(self):
        """Activate / deactivate automatic reload of images"""
        return self._auto_reload
        
    @auto_reload.setter
    def auto_reload(self, val):
        self._auto_reload = val
        if bool(val):
            print "Reloading images..."
            self.load()
          
    @property
    def crop(self):
        """Activate / deactivate crop mode"""
        return self.img_prep["crop"]
    
    @crop.setter
    def crop(self, value):
        """Set crop"""
        self.img_prep["crop"] = bool(value)
        self.load()
    
    @property
    def pyrlevel(self):
        """Current Gauss pyramid level
        
        Note
        ----
        images are reloaded on change
        """
        return self.img_prep["pyrlevel"]
    
    @pyrlevel.setter
    def pyrlevel(self, value):
        self.img_prep["pyrlevel"] = int(value)
        self.load()
    
    @property
    def gaussian_blurring(self):
        """Current blurring level
        
        Note
        ----
        images are reloaded on change        
        """
        return self.img_prep["blurring"]
    
    @gaussian_blurring.setter
    def gaussian_blurring(self, val):
        if val < 0:
            raise ValueError("Negative smoothing kernel does not make sense..")
        elif val > 10:
            warn("Activate gaussian blurring with kernel size exceeding 10, "
                "this might significantly slow down things..")
        self.img_prep["blurring"] = val
        self.load()
        
    @property
    def roi(self):
        """Current ROI (in relative coordinates)
        
        The ROI is returned with respect to the current :attr:`pyrlevel`
        """
        return map_roi(self._roi_abs, self.pyrlevel)
    
    @roi.setter
    def roi(self):
        raise AttributeError("Please use roi_abs to set the current ROI in "
            "absolute image coordinates. :func:`roi` is used to access the "
            "current ROI for the actual pyramide level.")
            
    @property 
    def roi_abs(self):
        """Current roi in absolute detector coords (cf. :attr:`roi`)
        """
        #return map_roi(self._roi_abs, self.img_prep["pyrlevel"])
        return self._roi_abs
        
    @roi_abs.setter
    def roi_abs(self, val):
        if check_roi(val):
            self._roi_abs = val
            self.load()
            
    @property
    def cfn(self):
        """Current index (file number in ``files``)"""
        return self.index
        
    @property
    def nof(self):
        """Number of files in this list"""
        return len(self.files)
        
    @property
    def last_index(self):
        """Index of last image"""
        return len(self.files) - 1
        
    @property
    def data_available(self):
        """Wrapper for :func:`has_files`"""
        return self.has_files()
        
    @property
    def has_images(self):
        """Wrapper for :func:`has_files`"""
        return self.has_files()
    
    @property
    def img_mode(self):
        """Checks and returs current img mode (tau, aa, raw)
        
        This function is overwritten in :class:`ImgList` where more states
        are allowed. It is, for instance used in :func:`make_stack`.
        
        Returns
        -------
        str
            string "raw" (:class:`BaseImgList` does not support tau or aa image 
            determination)
        """
        return "raw"
    
    @property
    def start_acq(self):
        """Array containing all image acq. time stamps of this list
        
        Note
        ----
        The time stamps are extracted from the file names
        """
        ts = self.get_img_meta_all_filenames()[0]
        return ts
    
    @property
    def acq_times(self):
        """Wrapper (old name) for attribute start_acq"""
        warn("Old name of attribute start_acq: still works, but annoys you "
            "with this warning")
        return self.start_acq
        
    def activate_edit(self, val=True):
        """Activate / deactivate image edit mode
        
        If inactive, images will be loaded raw without any editing or 
        further calculations (e.g. determination of optical flow, or updates of
        linked image lists). Images will be reloaded.
        
        Parameters
        ----------
        val : bool
            new mode
        """
        if val == self.edit_active:
            return
        self.edit_active = val
        self.load()
        
    def has_files(self):
        """Returns boolean whether or not images are available in list"""
        return bool(self.nof)
    
    def plume_dist_access(self):
        """Checks if measurement geometry is available"""
        if not isinstance(self.meas_geometry, MeasGeometry):
            return False
        try:
            _, _, plume_dist_img = self.meas_geometry.get_all_pix_to_pix_dists()  
            print "Plume distances available, dist_avg = %.2f" %plume_dist_img.mean()
        except:
            return False
            
    def update_img_prep(self, **settings):
        """Update image preparation settings and reload
        
        Parameters
        ----------
        **settings
            key word args specifying settings to be updated (see keys of
            ``img_prep`` dictionary)
        """
        for key, val in settings.iteritems():
            if self.img_prep.has_key(key) and\
                        isinstance(val, type(self.img_prep[key])):
                self.img_prep[key] = val
        try:
            self.load()
        except IndexError:
            pass
        
    def clear(self):
        """Empty this list (i.e. :attr:`files`)"""
        self.files = []
    
    def separate_by_substr_filename(self, sub_str, sub_str_pos, delim="_"):
        """Separate this list by filename specifications
        
        The function checks all current filenames, and keeps those, which have
        a certain sub string at a certain position in the file name after 
        splitting using a provided delimiter. All other files are added to a 
        new image list which is returned.
        
        Parameters
        ----------
        sub_str : str
            string identification used to identify the image type which is 
            supposed to be kept within this list object
        sub_str_pos : int
            position of sub string after filename was split (using input 
            param delim)
        delim : str  
            filename delimiter, defaults to "_"
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`ImgList`, list contains images matching the requirement
            - :obj:`ImgList`, list containing all other images
        """
        match = []
        rest = []
        for p in self.files:
            spl = basename(p).split(".")[0].split(delim)
            if spl[sub_str_pos] == sub_str:
                match.append(p)
            else:
                rest.append(p)
        
        lst_match = ImgList(match, list_id="match", camera=self.camera)
        lst_rest = ImgList(rest, list_id="rest", camera=self.camera)
        return (lst_match, lst_rest)
        
    def add_files(self, file_list):
        """Add images to this list
    
        Parameters
        ----------
        file_list : list
            list with file paths
            
        Returns
        -------
        bool
            success / failed
        """
        if isinstance(file_list, str):
            file_list = [file_list]
        if not isinstance(file_list, list):
            print ("Error: file paths could not be added to image list,"
                " wrong input type %s" %type(file_list))
            return False
        
        self.files.extend(file_list)
        self.init_filelist()
        return True
        
    def init_filelist(self, num=0):
        """Initiate the filelist
        
        Sets current list index and resets loaded images 
        
        Parameters
        ----------
        num : int
            desired image index, defaults to 0
        """
        self.index = num
                   
        for key, val in self.loaded_images.iteritems():
            self.loaded_images[key] = None
        
        if self.nof > 0:
            print "\nInit ImgList %s" %self.list_id
            print "-------------------------"
            print "Number of files: " + str(self.nof)
            print "-----------------------------------------"
        
    def set_dummy(self):
        """Load dummy image"""
        dummy = Img(load_img_dummy())
        for key in self.loaded_images:
            self.loaded_images[key] = dummy
            
    def set_camera(self, camera=None, cam_id=None):
        """Set the current camera
        
        Two options:
            
            1. set :obj:`Camera` directly
            #. provide one of the default camera IDs (e.g. "ecII", "hdcam")
        
        Parameters
        ----------
        camera : Camera
            the camera used
        cam_id : str
            one of the default cameras (use 
            :func:`pyplis.inout.get_all_valid_cam_ids` to get the default 
            camera IDs)
            
        """
        if not isinstance(camera, Camera):
            camera = Camera(cam_id)
        self.camera = camera
    
    def reset_img_prep(self):
        """Init image pre-edit settings"""
        self.img_prep = dict.fromkeys(self.img_prep, 0)
        self._roi_abs = [0, 0, 9999, 9999]
        if self.nof > 0:
            self.load()
    
    def get_img_meta_from_filename(self, file_path):
        """Loads and prepares img meta input dict for Img object
        
        Parameters
        ----------
        file_path : str 
            file path of image
        
        Returns
        -------
        dict
            dictionary containing retrieved values for ``start_acq`` and 
            ``texp``
        """
        info = self.camera.get_img_meta_from_filename(file_path)
        return {"start_acq" : info[0], "texp": info[3]}
    
    
        
    def get_img_meta_all_filenames(self):   
        """Try to load acquisition and exposure times from filenames
        
        Note
        ----
        Only works if relevant information is specified in ``self.camera`` and
        can be accessed from the file names, missing 
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - list, list containing all retrieved acq. time stamps
            - list, containing all retrieved exposure times

        """
        times, texps = asarray([None] * self.nof), asarray([None] * self.nof) 
        
        for k in range(self.nof):
            try:
                info = self.camera.get_img_meta_from_filename(self.files[k])
                times[k] = info[0]
                texps[k] = info[3]
            except:
                pass
        if times[0].date() == date(1900,1,1):
            d = self.this.meta["start_acq"].date()
            warn("Warning accessing acq. time stamps from file names in "
                "ImgList: date information could not be accessed, using "
                "date of currently loaded image meta info: %s" %d)
            times = asarray([datetime(d.year, d.month, d.day, x.hour, x.minute, 
                              x.second, x.microsecond) for x in times])
            
        return times, texps

    def assign_indices_linked_list(self, lst):
        """Create a look up table for fast indexing between image lists
        
        Parameters
        ----------
        lst : BaseImgList
            image list supposed to be linked
            
        Returns
        -------
        array
            array contining linked indices
        """
        idx_array = zeros(self.nof, dtype = int)
        times, _ = self.get_img_meta_all_filenames()
        times_lst, _ = lst.get_img_meta_all_filenames()
        if lst.nof == 1:
            warn("Other list contains only one file, assign all indices to "
                 "the corresponding image")
        elif (any([x is None for x in times]) or 
              any([x is None for x in times_lst])):
            warn("Image acquisition times could not be accessed from file"
                " names, assigning by indices")
            lst_idx = arange(lst.nof)
            for k in range(self.nof):
                idx_array[k] = abs(k - lst_idx).argmin()
        else:
            for k in range(self.nof):
                idx = abs(times[k] - times_lst).argmin()
                idx_array[k] = idx
        
        return idx_array
    
    def same_preedit_settings(self, settings_dict):
        """Compare input settings dictionary with self.img_prep 
        
        Parameters
        ----------
        **settings_dict
            keyword args specifying settings to be compared
        
        Returns
        -------
        bool
            False if not the same, True else
        """
        sd = self.img_prep
        for key, val in settings_dict.iteritems():
            if sd.has_key(key):
                if not sd[key] == val:
                    return False
        return True
    
    def make_stack(self, stack_id=None, pyrlevel=None, roi_abs=None,
                   start_idx=0, stop_idx=None, ref_check_roi_abs=None,
                   ref_check_min_val=None, ref_check_max_val=None,
                   dtype=float32):
        """Stack all images in this list 
        
        The stacking is performed using the current image preparation
        settings (blurring, dark correction etc). Only stack ROI and pyrlevel
        can be set explicitely.
        
        Note
        ----
        In case of ``MemoryError`` try stacking less images (specifying 
        start / stop index) or reduce the size setting a different Gauss
        pyramid level
        
        Parameters
        ----------
        stack_id : :obj:`str`, optional
            identification string of the image stack
        pyrlevel : :obj:`int`, optional
            Gauss pyramid level of stack
        roi_abs : list
            build stack of images cropped in ROI
        start_idx : int
            index of first considered image, defaults to 0
        stop_idx : :obj:`int`, optional
            index of last considered image (if None, the last image in this 
            list is used), defaults to last index
        ref_check_roi_abs : :obj:`list`, optional
            rectangular area specifying a reference area which can be specified
            in combination with the following 2 parameters in order to include
            only images in the stack that are within a certain intensity range
            within this ROI (Note that this ROI needs to be specified in
            absolute coordinate, i.e. corresponding to pyrlevel 0).
        ref_check_min_val : :obj:`float`, optional
            if attribute ``roi_ref_check`` is a valid ROI, then only images 
            are included in the stack that exceed the specified intensity 
            value (can e.g. be optical density or minimum gas CD in calib
            mode)
        ref_check_max_val : :obj:`float`, optional
            if attribute ``roi_ref_check`` is a valid ROI, then only images 
            are included in the stack that are smaller than the specified 
            intensity value (can e.g. be optical density or minimum gas CD in 
            calib mode)    
        dtype 
            data type of stack
            
        Returns
        -------
        ImgStack
            result stack
        
        
        """
        self.activate_edit()
        if stop_idx is None:
            stop_idx = self.nof
            
        num = stop_idx - start_idx
        #remember last image shape settings
        _roi = deepcopy(self._roi_abs)
        _pyrlevel = deepcopy(self.pyrlevel)
        _crop = self.crop
        
        self.auto_reload = False
        if pyrlevel is not None and pyrlevel != _pyrlevel:
            print ("Changing image list pyrlevel from %d to %d"\
                                            %(_pyrlevel, pyrlevel))
            self.pyrlevel = pyrlevel
        if check_roi(roi_abs):
            print "Activate cropping in ROI %s (absolute coordinates)" %roi_abs
            self.roi_abs = roi_abs
            self.crop = True

        if stack_id is None:
            stack_id = self.list_id + "_" + self.img_mode           
            
        if stack_id in ["raw", "tau"]:
            stack_id = "%s_%s" %(self.list_id, stack_id)
            
        #create a new settings object for stack preparation
        self.goto_img(start_idx)
        self.auto_reload = True
        h, w = self.current_img().shape
        stack = ImgStack(h, w, num, dtype, stack_id, camera=self.camera,
                         img_prep=self.current_img().edit_log)
        lid = self.list_id
        ref_check = True
        if not check_roi(ref_check_roi_abs):
            ref_check = False
        try:
            ref_check_min_val = float(ref_check_min_val)
        except:
            ref_check = False
        try:
            ref_check_max_val = float(ref_check_max_val)
        except:
            ref_check = False
        exp = int(10**exponent(num)/4.0)
        if not exp:
            exp = 1
        for k in range(num):
            if k % exp == 0:
                print ("Building img-stack from list %s, progress: (%s | %s)" 
                       %(lid, k, num-1))
            img = self.loaded_images["this"]
            #print im.meta["start_acq"]
            append = True
            if ref_check:
                sub_val = img.crop(roi_abs=ref_check_roi_abs, new_img=1).mean()
                if not ref_check_min_val <= sub_val <= ref_check_max_val:
                    print("Exclude image no. %d from stack, got value=%.2f in "
                        "ref check ROI (out of specified range)" %(k, sub_val))
                append = False
            if append:
                stack.append_img(img.img, img.meta["start_acq"], 
                                 img.meta["texp"])
            self.next_img()  
        stack.start_acq = asarray(stack.start_acq)
        stack.texps = asarray(stack.texps)
        stack.roi_abs = self._roi_abs
        
        print ("Img stack calculation finished, rolling back to intial list"
            "state:\npyrlevel: %d\ncrop modus: %s\nroi (abs coords): %s "
            %(_pyrlevel, _crop, _roi))
        self.auto_reload = False
        self.pyrlevel = _pyrlevel
        self.crop = _crop
        self.roi_abs = _roi
        self.auto_reload = True
        if not sum(stack._access_mask) > 0:
            raise ValueError("Failed to build stack, stack is empty...")
        return stack
    
    def get_mean_img(self, start_idx=0, stop_idx=None):
        """Determines average image from list images
        
        Parameters
        ----------
        start_idx : int
            index of first considered image
        stop_idx : int
            index of last considered image (if None, the last image in this 
            list is used)
            
        Returns
        -------
        Img 
            average image 
        """
        cfn = self.cfn
        self.goto_img(start_idx)
        if stop_idx is None or stop_idx > self.nof:
            print "Setting stop_idx to last list index"
            stop_idx = self.nof
        img = Img(zeros(self.current_img().shape))
        img.edit_log = self.current_img().edit_log
        img.meta["start_acq"] = self.current_time()
        added = 0
        texps = []
        for k in range(start_idx, stop_idx):
            try:
                cim = self.current_img()
                img.img += cim.img
                try:
                    texps.append(cim.texp)
                except:
                    pass
                self.next_img()
                added += 1
            except:
                warn("Failed to add image at index %d" %k)
        img.img = img.img / added
        img.meta["stop_acq"] = self.current_time()
        if len(texps) == added:
            img.meta["texp"] = asarray(texps).mean()
        self.goto_img(cfn)
        return img
    
    def get_mean_tseries_rects(self, *rois):
        """Similar to :func:`get_mean_value` but for multiple rects
        
        Parameters
        ----------
        *rois
            non keyword args specifying rectangles for data access
        
        Returns
        -------
        tuple
            N-element tuple containing :class:`PixelMeanTimeSeries` objects 
            (one for each ROI specified on input)
        """
        if not self.data_available:
            raise IndexError("No images available in ImgList object")
        dat = []
        num_rois = len(rois)
        for roi in rois:
            dat.append([[],[],[],[]])
        cfn = self.cfn
        num = self.nof
        
        self.goto_img(0)
        lid = self.list_id
        pnum = int(10**exponent(num) / 4.0)
        for k in range(num):
            try:
                if k % pnum == 0:    
                    print ("Calc pixel mean t-series in list %s (%d | %d)" 
                                                    %(lid,(k+1),num))
            except:
                pass
            img = self.loaded_images["this"]
            for i in range(num_rois):
                roi = rois[i]
                d=dat[i]
                d[0].append(img.meta["texp"])
                d[1].append(img.meta["start_acq"])
                sub = img.img[roi[1]:roi[3],roi[0]:roi[2]]
                d[2].append(sub.mean())
                d[3].append(sub.std())
            
            self.next_img()
        
        self.goto_img(cfn)
        means = []
        for i in range(num_rois):
            d = dat[i]
            mean = PixelMeanTimeSeries(d[2], d[1], d[3], d[0], rois[i],
                                       img.edit_log)
            means.append(mean)
        return means
        
    def get_mean_value(self, roi=[0, 0, 9999, 9999], apply_img_prep=True):
        """Determine pixel mean value time series in ROI
        
        Determines the mean pixel value (and standard deviation) for all images 
        in this list. Default ROI is the whole image and can be set via
        input param roi, image preparation can be turned on or off.
        
        Parameters
        ----------
        roi : list
            rectangular region of interest ``[x0, y0, x1, y1]``, 
            defaults to [0, 0, 9999, 9999] (i.e. whole image)
        apply_img_prep : bool
            if True, img preparation is performed as specified in 
            ``self.img_prep`` dictionary, defaults to True
            
        Returns
        -------
        PixelMeanTimeSeries
            time series of retrieved values
        """
        if not self.data_available:
            raise IndexError("No images available in ImgList object")
        #settings = deepcopy(self.img_prep)
        self.activate_edit(apply_img_prep)
        cfn = self.cfn
        num = self.nof
        vals, stds, texps, acq_times = [],[],[],[]
        self.goto_img(0)
        lid=self.list_id
        pnum = int(10**exponent(num) / 4.0)
        for k in range(num):
            try:
                if k % pnum == 0:
                    print ("Calc pixel mean t-series in list %s (%d | %d)" 
                                                %(lid,(k+1),num))
            except:
                pass
            img = self.loaded_images["this"]
            texps.append(img.meta["texp"])
            acq_times.append(img.meta["start_acq"])
            sub = img.img[roi[1]:roi[3],roi[0]:roi[2]]
            vals.append(sub.mean())
            stds.append(sub.std())
            
            self.next_img()
        
        self.goto_img(cfn)

        return PixelMeanTimeSeries(vals, acq_times, stds, texps, roi,
                                   img.edit_log)
        
    def current_edit(self):
        """Returns :attr:`edit_log` of current image"""
        return self.current_img().edit_log
        
    def edit_info(self):
        """Print the current image preparation settings"""
        d = self.current_img().edit_log
        print("\nImgList %s, image edit info\n----------------------------" 
            %self.list_id)
        for key, val in d.iteritems():
            print "%s: %s" %(key, val)
        
    def _make_header(self):
        """Make header for current image (based on image meta information)"""
        try:
            im = self.current_img()
            if not isinstance(im, Img):
                raise Exception("Current image not accessible in ImgList...")

            s = ("%s (Img %s of %s), read_gain %s, texp %.2f s"
                %(self.current_time().strftime('%d/%m/%Y %H:%M:%S'),\
                        self.index + 1, self.nof, im.meta["read_gain"],\
                                                        im.meta["texp"]))
            return s
            
        except Exception as e:
            print repr(e)
            return "Creating img header failed...(Do you see the img Dummy??)"
            
    def update_prev_next_index(self):
        """Get and set the filenumbers of the previous and next image"""
        if self.index == 0:
            self.prev_index = self.nof - 1
            self.next_index = 1
        elif self.index == (self.nof - 1):
            self.prev_index = self.nof - 2
            self.next_index = 0
        else:
            self.prev_index = self.index - 1
            self.next_index = self.index + 1
    """
    Image loading functions 
    """
    def _merge_edit_states(self):
        """Merges the current list edit state with the image state on load
        
        Todo
        ----
        
        This function needs revision and should in parts be moved to 
        :class:`ImgList`.
        
        """
        onload = self._load_edit
        if onload["blurring"] > 0:
            print "Image was already blurred (on load)"
            if self.img_prep["blurring"] < onload["blurring"]:
                self.img_prep["blurring"] = onload["blurring"]
        if onload["median"] > 0:
            print "Image was already median filtered (on load)"
            if self.img_prep["median"] < onload["media"]:
                self.img_prep["median"] = onload["median"]
        if onload["darkcorr"]:
            print "Image was already dark corrected (on load)"
            self._list_modes["darkcorr"] = True
            
        if onload["vigncorr"]:
            self._list_modes["vigncorr"]
            
    def load(self):
        """Load current image
        
        Try to load the current file ``self.files[self.cfn]`` and if remove the 
        file from the list if the import fails
        Raises
        ------
        
        Returns
        -------
        bool
            if True, image was loaded, if False not        
        """
        if not self._auto_reload:
            print ("Automatic image reload deactivated in image list %s"\
                                                                %self.list_id)
            return False        
        try:
            img_file = self.files[self.index]
            img = Img(img_file,
                      import_method=self.camera.image_import_method,
                      **self.get_img_meta_from_filename(img_file))
            self.loaded_images["this"] = img
            self._load_edit.update(img.edit_log)
            
            if img.vign_mask is not None:
                self.vign_mask = img.vign_mask
            
            self.update_prev_next_index()
            self._apply_edit("this")
                    
        except IOError:
            warn("Invalid file encountered at list index %s, file will"
                " be removed from list" %self.index)
            self.pop()
            if self.nof == 0:
                raise IndexError("No filepaths left in image list...")
            self.load()
            
        except IndexError:
            try:
                self.init_filelist()
                self.load()
            except:
                raise IndexError("Could not load image in list %s: file list "
                    " is empty" %(self.list_id))
        return True
    
    def pop(self, idx=None):
        """Remove one file from this list"""
        if idx == None:
            idx = self.index
        self.files.pop(idx)
        
    def load_next(self):
        """Load next image in list"""
        if self.nof < 2:
            return
        self.index = self.next_index
        self.load()
        
    def load_prev(self):  
        """Load previous image in list"""
        if self.nof < 2:
            return
        self.index = self.prev_index
        self.load()
        
    """
    Functions related to image editing and edit management
    """ 
    def add_gaussian_blurring(self, sigma=1):
        """Increase amount of gaussian blurring on image load
        
        :param int sigma (1): Add width gaussian blurring kernel
        """
        self.img_prep["blurring"] += sigma
        self.load()
        
    def _apply_edit(self, key):
        """Applies the current image edit settings to image
        
        :param str key: image id (e.g. this)            
        """
        if not self.edit_active:
            print ("Edit not active in img_list " + self.list_id + ": no image "
                "preparation will be performed")
            return
        img = self.loaded_images[key]
        img.pyr_down(self.img_prep["pyrlevel"])
        if self.img_prep["crop"]:
            img.crop(self.roi_abs)
        img.add_gaussian_blurring(self.img_prep["blurring"])
        img.apply_median_filter(self.img_prep["median"])
        if self.img_prep["8bit"]:
            img._to_8bit_int(new_im = False)
        self.loaded_images[key] = img
        
    def cam_id(self):
        """Get the current camera ID (if camera is available)"""
        return self.camera.cam_id

            
    def current_time(self):
        """Get the acquisition time of the current image from image meta data
        
        Raises
        ------
        IndexError
            if list does not contain images
        
        Returns
        -------
        datetime
            start acquisition time of currently loaded image
            
        """
        return self.current_img().meta["start_acq"]
        
    def current_time_str(self, format='%H:%M:%S'):
        """Returns a string of the current acq time"""
        return self.current_img().meta["start_acq"].strftime(format)
        
    def current_img(self, key="this"):
        """Get the current image object
        
        Parameters
        -----------
        key : str
            this" or "next"
        
        Returns
        -------
        Img
            currently loaded image in list
        """
        img = self.loaded_images[key]
        if not isinstance(img, Img):
            self.load()
            img = self.loaded_images[key]
        return img
        
    def show_current(self, **kwargs):
        """Show the current image"""
        return self.current_img().show(**kwargs)
                 
    def goto_img(self, num):
        """Go to a specific image
        
        :param int num: file number index of the desired image
        
        """
        #print "Go to img number %s in img list %s" %(num, self.list_id)
        self.index = num
        self.load()
        return self.loaded_images["this"]
        
    def next_img(self):
        """Go to next image 
        
        Calls :func:`load_next` 
        """
        self.load_next()
        return self.loaded_images["this"]
            
    def prev_img(self):
        """Go to previous image
        
        Calls :func:`load_prev`
        """
        self.load_prev()
        return self.loaded_images["this"]
    
    def append(self, file_path):
        """Append image file to list
        
        :param str file_path: valid file path
        """
        if not exists(file_path):
            raise IOError("Image file path does not exist %s" %file_path)
        
        self.files.append(file_path)
        
    def _first_file(self):
        """get first file path of image list"""
        try:
            return self.files[0]
        except IndexError:
            print "Filelist empty..."
        except:
            raise 
    
    def _last_file(self):
        """get last file path of image list"""
        try:
            return self.files[self.nof - 1]
        except IndexError:
            print "Filelist empty..."
        except:
            raise 
    
#==============================================================================
#     """GUI features
#     """
#     def open_in_imageviewer(self):
#         from .gui.ImgViewer import ImgViewer
#         app = QApplication(argv)
#         widget = ImgViewer(self.list_id, self)
#         widget.show()
#         exit(app.exec_())        
#==============================================================================
        
    """
    Plotting etc
    """
    def plot_mean_value(self, roi=[0, 0, 9999, 9999], yerr=False, ax=None):
        """Plot mean value of image time series
        
        Parameters
        ----------
        roi : list
            rectangular ROI in which mean is determined (default is
            ``[0, 0, 9999, 9999]``, i.e. whole image)
        yerr : bool
            include errorbars (std), defaults to False
        ax : :obj:`Axes`, optional
            matplotlib axes object
            
        Returns
        -------
        Axes
            matplotlib axes object
        """
        if ax is None:
            fig = figure()#figsize=(16, 6))
            ax = fig.add_subplot(1, 1, 1)

        mean = self.get_mean_value()
        ax = mean.plot(yerr=yerr, ax=ax)
        return ax
    
    def plot_tseries_vert_profile(self, pos_x, start_y=0, stop_y=None,
                                  step_size=0.1, blur=4):
        """Plot the temporal evolution of a line profile
        
        Parameters
        ----------
        pos_x : int 
            number of pixel column
        start_y : int
            Start row of profile (y coordinate, default: 10)
        stop_y : int
            Stop row of profile (is set to rownum - 10pix if input is None)
        step_size : float
            stretch between different line profiles of the evolution (0.1)
        blur : int
            blurring of individual profiles (4)
            
        Returns
        -------
        Figure
            figure containing result plot
            
        """
        cfn = deepcopy(self.index)
        self.goto_img(0)
        name = "vertAtCol" + str(pos_x)
        h, w = self.get_img_shape()
        h_rel = float(h) / w
        width = 18
        height = int(9 * h_rel)
        if stop_y is None:
            stop_y = h - 10
        l = LineOnImage(pos_x, start_y, pos_x, stop_y, name)
        fig = figure(figsize=(width, height))
        #fig,axes=plt.subplots(1,2,sharey=True,figsize=(width,height))
        cidx = 0
        img_arr = self.loaded_images["this"].img
        rad = gaussian_filter(l.get_line_profile(img_arr), blur)
        del_x = int((rad.max() - rad.min()) * step_size)
        y_arr = arange(start_y, stop_y, 1)
        ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.8])
        times = self.get_img_meta_all_filenames()[0]
        if any([x == None for x in times]):
            raise ValueError("Cannot access all image acq. times")
        idx = []
        idx.append(cidx)
        for k in range(1, self.nof):
            rad = rad - rad.min() + cidx
            ax1.plot(rad, y_arr,"-b")        
            img_arr = self.next_img().img
            rad = gaussian_filter(l.get_line_profile(img_arr),blur)
            cidx = cidx + del_x
            idx.append(cidx)
        idx = asarray(idx)
        ax1.set_ylim([0, h])
        ax1.invert_yaxis()
        draw()
        new_labels=[]
    #==============================================================================
    #     labelNums=[int(a.get_text()) for a in ax1.get_xticklabels()]
    #     print labelNums
    #==============================================================================
        ticks=ax1.get_xticklabels()
        new_labels.append("")
        for k in range(1, len(ticks)-1):
            tick = ticks[k]
            index = argmin(abs(idx - int(tick.get_text())))
            new_labels.append(times[index].strftime("%H:%M:%S"))
        new_labels.append("")
        ax1.set_xticklabels(new_labels)
        ax1.grid()
        self.goto_img(cfn)
        ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.8])
        l.plot_line_on_grid(self.loaded_images["this"].img,ax = ax2)
        ax2.set_title(self.loaded_images["this"].meta["start_acq"].strftime(\
            "%d.%m.%Y %H:%M:%S"))
        return fig

    """
    Magic methods
    """  
    def __str__(self):
        """String representation of image list"""
        s = "\npyplis ImgList\n----------------------------------\n"
        s += "ID: %s\nType: %s\n" %(self.list_id, self.list_type)
        s += "Number of files (imgs): %s\n\n" %self.nof
        s += "Current image prep settings\n.................................\n"
        if not self.has_files():
            return s
        try:
            for k, v in self.current_img().edit_log.iteritems():
                s += "%s: %s\n" %(k, v)
            if self.crop is True:
                s += "Cropped in ROI\t[x0, y0, x1, y1]:\n"
                s += "  Absolute coords:\t%s\n" %self.roi_abs 
                s += "  @pyrlevel %d:\t%s\n" %(self.pyrlevel, self.roi)
        except:
            s += "FATAL: Image access failed, msg\n: %s" %format_exc()
        return s
        
    def __call__(self, num = 0):
        """Change current file number, load and return image
        
        :param int num: file number
        """
        return self.goto_img(num)
            
    def __getitem__(self, name):
        """Get item method"""
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        for k,v in self.__dict__.iteritems():
            try:
                if v.has_key(name):
                    return v[name]
            except:
                pass

class DarkImgList(BaseImgList):
    """A :class:`BaseImgList`object only extended by read_gain value"""
    def __init__(self, files=[], list_id=None, list_type=None, read_gain=0,
                 camera=None, init=True):
        
        super(DarkImgList, self).__init__(files, list_id, list_type, camera,
                                          init=False)
        self.read_gain = read_gain
        if init:
            self.add_files(files)
            
class ImgList(BaseImgList):
    """Image list object with expanded functionality (cf. :class:`BaseImgList`)
    
    Additional features:
        
            1. Optical flow determination
            #. Linking of lists (e.g. on and offband lists)
            #. Dark and offset image correction
            #. Plume background modelling and tau image determination
            #. Methods for dilution correction
            #. Automatic vignetting correction
            #. Assignment of calibration data and automatic image calibration
            
    Parameters
    ----------
    files : list
        list containing image file paths, defaults to ``[]`` (i.e. empty list)
    list_id : :obj:`str`, optional
        string ID of this list, defaults to None
    list_type : :obj:`str`, optional
        string specifying type of image data in this list (e.g. on, off)
    camera : :obj:`Camera`, optional
        camera specifications, defaults to None
    geometry : :obj:`MeasGeometry`, optional
        measurement geometry
    init : bool
        if True, the first two images in list ``files`` are loaded
    
    """
    def __init__(self, files=[], list_id=None, list_type=None, camera=None,
                 geometry=None, init=True):

        super(ImgList, self).__init__(files, list_id, list_type, camera, 
                                      geometry, init=False)
                                      
        self.loaded_images.update({"next": None})
    
        #: List modes (currently only tau) are flags for different list states
        #: and need to be activated / deactivated using the corresponding
        #: method (e.g. :func:`activate_tau_mode`) to be changed, dont change
        #: them directly via this private dictionary
        self._list_modes.update({"darkcorr"  :  0,
                                 "optflow"   :  0,
                                 "vigncorr"  :  0,
                                 "tau"       :  0,
                                 "aa"        :  0,
                                 "senscorr"  :  0,
                                 "gascalib"  :  0})
                                 
        self.dil_corr_thresh = {"on"    :   0.0,
                                "off"   :   0.0,
                                "aa"    :   0.0}
        self._ext_coeffs = None
        
        self._bg_imgs = [None, None] #sets bg images
        self._bg_list_id = None #ID of linked background list
        self._which_bg = "img" #change using :attr:`which_bg` either, img or list
        
        self.bg_model = PlumeBackgroundModel()
        
        self._aa_corr_mask = None
        self._calib_data = None
        # these two images can be set manually, if desired
        self.master_dark = None
        self.master_offset = None
        
        # These dicitonaries contain lists with dark and offset images 
        self.dark_lists = od()
        self.offset_lists = od()
        self._dark_corr_opt = self.camera.DARK_CORR_OPT
        
        # Dark images will be updated every 10 minutes (i.e. before an image is
        # dark and offset corrected it will be checked if the currently loaded
        # images match the time interval (+-10 min) of this image and if not
        # a new one will be searched).
        self.update_dark_ival = 10 #mins
        self.time_last_dark_check = datetime(1900, 1, 1)                      
        
        """
        Additional variables
        """
        #: Other image lists can be linked to this and are automatically updated
        self.linked_lists = {}
        #: this dict (linked_indices) is filled in :func:`link_imglist` to 
        #: increase the linked reload image performance
        self.linked_indices = {}
        #self.currentMaxI=None
    
        #Optical flow engine
        self.optflow = OptflowFarneback(name=self.list_id)
        
        if self.data_available and init:
            self.load()
    
    @property
    def next(self):
        """Next image"""
        return self.loaded_images["next"]
        
    @property
    def DARK_CORR_OPT(self):
        """Return the current dark correction mode
        
        The following modes are available:

            0   =>  no dark correction possible (is e.g. set if camera is 
                    unspecified)
            1   =>  individual correction with separate dark and offset 
                    (e.g. ECII data)
            2   =>  one dark image which is subtracted (including the offset, 
                    e.g. HD cam data)
        
        For details see documentation of :class:`CameraBaseInfo` 
        """
        return self._dark_corr_opt
        
    
    @DARK_CORR_OPT.setter
    def DARK_CORR_OPT(self, val):
        try:
            val = int(val)
            if not val in [0, 1, 2]:
                raise ValueError
            self._dark_corr_opt = val
        except:
            warn("Failed to update dark correction option")
    
    @property
    def BG_MODEL_MODE(self):
        """Current background image modelling mode"""
        return self.bg_model.mode
        
    @BG_MODEL_MODE.setter
    def BG_MODEL_MODE(self, val):
        self.bg_model.mode = val
        self.load()
            
    @property
    def darkcorr_mode(self):
        """Returns current list darkcorr mode"""
        return self._list_modes["darkcorr"]
        
    @darkcorr_mode.setter
    def darkcorr_mode(self, value):
        """Change current list darkcorr mode
        
        Wrapper for :func:`activate_darkcorr`        
        """
        return self.activate_darkcorr(value)
    
    @property
    def optflow_mode(self):
        """Activate / deactivate optical flow calc on image load"""
        return self._list_modes["optflow"]
    
    @optflow_mode.setter
    def optflow_mode(self, val):
        self.activate_optflow_mode(val)
    
    @property
    def vigncorr_mode(self):
        """Activate / deactivate optical flow calc on image load"""
        return int(self._list_modes["vigncorr"])
    
    @vigncorr_mode.setter
    def vigncorr_mode(self, val):
        self.activate_vigncorr(val)
    
    @property
    def sensitivity_corr_mode(self):
        """Activate / deactivate AA sensitivity correction mode"""
        return self._list_modes["senscorr"]
        
    @sensitivity_corr_mode.setter 
    def sensitivity_corr_mode(self, val):
        """Activate / deactivate AA sensitivity correction mode"""
        if val == self._list_modes["senscorr"]:
            return
        if val:    
            self.aa_corr_mask #raises AttributeError if mask is not available
            if not self.aa_mode:
                raise AttributeError("AA sensitivity correction mode can only "
                    "be activated in list aa_mode, please activate aa_mode "
                    "first...")
   
        self._list_modes["senscorr"] = val
        self.load()
        
    @property
    def tau_mode(self):
        """Returns current list tau mode"""
        return self._list_modes["tau"]
        
    @tau_mode.setter
    def tau_mode(self, value):
        """Change current list tau mode
        
        Wrapper for :func:`activate_tau_mode`        
        """
        self.activate_tau_mode(value)
     
    @property
    def aa_mode(self):
        """Returns current list AA mode"""
        return self._list_modes["aa"]
        
    @aa_mode.setter
    def aa_mode(self, value):
        """Change current list AA mode
        
        Wrapper for :func:`activate_aa_mode`        
        """
        self.activate_aa_mode(value)
        
    @property
    def calib_mode(self):
        """Acitivate / deactivate current list gas calibration mode"""
        return self._list_modes["gascalib"]
        
    @calib_mode.setter
    def calib_mode(self, val):
        """Change current list calibration mode"""
        if val == self._list_modes["gascalib"]:
            return
        if val:
            if not self.tau_mode:
                warn("Calib mode was activated without active tau mode")
            if not self.aa_mode:
                warn("Calib mode was activated without active AA mode")
            if not self.sensitivity_corr_mode:
                warn("AA sensitivity correction mode is deactivated. This "
                    "may yield erroneous results at the image edges")
            self.calib_data(self.current_img())
            
        self._list_modes["gascalib"] = val
        self.load()
    
    @property
    def ext_coeff(self):
        """Current extinction coefficient"""
        if not isinstance(self.ext_coeffs, Series):
            raise AttributeError("Extinction coefficients not available in "
                "image list %s" %self.list_id)
        elif len(self.ext_coeffs) == self.nof:
            #assuming that time stamps correspond to list time stamps
            return self.ext_coeffs[self.cfn]
        else:
            idx = closest_index(self.current_time(), self.ext_coeffs.index)
            return self.ext_coeffs[idx]
            
    @property
    def ext_coeffs(self):
        """Dilution extinction coefficients"""
        return self._ext_coeffs
        
    @ext_coeffs.setter
    def ext_coeffs(self, val):
        if isinstance(val, float):
            val = Series(val, self.acq_times[0])
        if not isinstance(val, Series):
            raise ValueError("Need pandas Series object")
        self._ext_coeffs = val
        
    @property
    def bg_img(self):
        """Return background image based on current vignetting corr setting"""
        img = None
        if self.which_bg == "img":
            try:
                img = self._bg_imgs[self.loaded_images["this"].\
                    edit_log["vigncorr"]]
            except:
                warn("No background image assigned to list %s" %self.list_id)
        else:
            lst = self.bg_list
            try:
                img = lst.current_img()
            except:
                pass
        return img
    
    @bg_img.setter
    def bg_img(self, val):
        self.set_bg_img(val)
        
    @property
    def bg_list(self):
        """Returns background image list (if assigned)"""
        try:
            return self.linked_lists[self._bg_list_id]
        except KeyError:
            warn("No linked list with ID %s found in list %s. " 
                            %(self._bg_list_id, self.list_id))
                
    @bg_list.setter
    def bg_list(self, val):
        if isinstance(val, str):
            if not self.linked_lists.has_key(val):
                raise AttributeError("No linked list with ID %s found in image"
                    " list %s" %(self.list_id, val))
            self._bg_list_id = val
        elif isinstance(val, ImgList):
            lid = "bg_" + self.list_id
            self.link_imglist(val, list_id=lid)
            self._bg_list_id = lid
        else:
            raise ValueError("Invalid input for assignment of background image"
                " list. Please provide either a string of one of the image "
                "lists already linked to this list or provide an ImgList object"
                "containing BG images")
                
    @property
    def which_bg(self):
        """Specifies from where background images are accessed"""
        return self._which_bg
        
    @which_bg.setter
    def which_bg(self, val):
        if val in ["img", "list"]:
            self._which_bg = val
            #warns if no background image is available for this access method
            self.bg_img 
        else:
            raise ValueError("Invalid input: choose from img or list")
    
    @property
    def aa_corr_mask(self):
        """Get / set AA correction mask"""
        if isinstance(self._aa_corr_mask, ndarray):
            warn("AA correction mask in list %s is numpy array and"
            "will be converted into Img object" %self.list_id)
            self._aa_corr_mask = Img(self._aa_corr_mask)
        if not isinstance(self._aa_corr_mask, Img):
            raise AttributeError("AA correction mask is not available...")
        return self._aa_corr_mask
        
    @aa_corr_mask.setter
    def aa_corr_mask(self, val):
        """Setter for AA correction mask"""
        if isinstance(val, ndarray):
            warn("Input for AA correction mask in list %s is numpy array and"
                    "will be converted into Img object" %self.list_id)
            val = Img(val)
        if not isinstance(val, Img):
            raise TypeError("Invalid input for AA correction mask: need Img"
                " object (or numpy array)")
        if not val.pyrlevel == 0:
            warn("AA correction mask is required to be at pyramid level 0 and "
                "will be converted")
            val.to_pyrlevel(0)
        img_temp = Img(self.files[self.cfn],
                       import_method=self.camera.image_import_method)
        if val.shape != img_temp.shape:
            try:
                val = val.to_pyrlevel(img_temp.pyrlevel)
                if val.shape != img_temp.shape:        
                    raise ValueError
            except:
                raise ValueError("Img shape mismatch between AA correction "
                    "mask and list images")
                
        self._aa_corr_mask = val
     
    @property
    def calib_data(self):
        """Get set object to perform calibration"""
        from pyplis.cellcalib import CellCalibEngine as cc
        from pyplis.doascalib import DoasCalibData as dc
        if not any([isinstance(self._calib_data, x) for x in [cc, dc]]):
            warn("No calibration data available in imglist %s" %self.list_id)
        return self._calib_data
    
    @calib_data.setter
    def calib_data(self, val):
        from pyplis.cellcalib import CellCalibEngine as cc
        from pyplis.doascalib import DoasCalibData as dc
        if not any([isinstance(val, x) for x in [cc, dc]]):
            raise TypeError("Could not set calibration data in imglist %s: "
            "need CellCalibData obj or DoasCalibData obj" %self.list_id)
        try:
            val(0.1) #try converting a fake tau value into a gas column
        except ValueError:
            raise ValueError("Cannot set calibration data in image list, "
                "calibration object is not ready")
        self._calib_data = val
        
    @property
    def doas_fov(self):
        """Try access DOAS FOV info (in case cailbration data is available)"""
        try:
            return self.calib_data.fov
        except:
            warn("No DOAS FOV information available")
        
    @property
    def img_mode(self):
        """Checks and returns current img mode (tau, aa, or raw)
        
        :return:
            - "tau", if ``self._list_modes["tau"] == True``
            - "aa", if ``self._list_modes["aa"] == True``
            - "raw", else
        """
        if self._list_modes["tau"] == True:
            return "tau"
        elif self._list_modes["aa"] == True:
            return "aa"
        else:
            return "raw"
            
    """RESETTING AND INIT METHODS"""      
    def init_filelist(self):
        """Adding functionality to filelist init"""
        super(ImgList, self).init_filelist()
        
    def init_bg_model(self, **kwargs):
        """Init clear sky reference areas in background model"""
        self.bg_model.update(**kwargs)
        self.bg_model.set_missing_ref_areas(self.current_img())
        
    """LIST MODE MANAGEMENT METHODS"""        
    def activate_darkcorr(self, val=True):
        """Activate or deactivate dark and offset correction of images
        
        If dark correction turned on, dark image access is attempted, if that
        fails, Exception is raised including information what did not work 
        out.
        
        Parameters
        ----------
        val : bool
            new mode
        """
        if val is self.darkcorr_mode: #do nothing
            return
        if not val and self._load_edit["darkcorr"]:
            raise ImgMetaError("Cannot deactivate dark correction, original"
                "image file was already dark corrected")
        if val:
            if self.this.edit_log["darkcorr"]:
                warn("Cannot activate dark correction in image list %s: "
                     "current image is already corrected for dark current"
                     %self.list_id)
                return
            self.get_dark_image()
            self.update_index_dark_offset_lists()
                    
        self._list_modes["darkcorr"] = val
        self.load()
        
    def activate_vigncorr(self, val=True):
        """Activate / deactivate vignetting correction on image load
        
        Note
        ----
        
        Requires ``self.vign_mask`` to be set or an background image 
        to be available (from which ``self.vign_mask`` is then determined)
        
        Parameters
        ----------
        val : bool
            new mode
        """
        if val is self.vigncorr_mode: #do nothing
            return
        elif val:
            if self.this.edit_log["vigncorr"]:
                warn("Cannot activate vignetting correction in image list %s: "
                     "current image is already corrected for vignetting"
                     %self.list_id)
                return 
            if isinstance(self.vign_mask, Img):
                self.vign_mask = self.vign_mask.img
            if not isinstance(self.vign_mask, ndarray):
                self.det_vign_mask_from_bg_img() 
            sh = Img(self.files[self.cfn],
                     import_method=self.camera.image_import_method).img.shape
            if not self.vign_mask.shape == sh:
                raise ValueError("Shape of vignetting mask %s deviates from "
                            "raw img shape %s" %(list(self.vign_mask.shape),
                            list(sh)))
        self._list_modes["vigncorr"] = val
        self.load()
    
    def activate_tau_mode(self, val=1):
        """Activate tau mode
        
        In tau mode, images will be loaded as tau images (if background image
        data is available). 
        
        Parameters
        ----------
        val : bool
            new mode
            
        """
        if val is self.tau_mode: #do nothing
            return
        if val:
            if self.this.edit_log["is_tau"]:
                warn("Cannot activate tau mode in image list %s: "
                     "current image is already a tau image"
                     %self.list_id)
                return
            vc = self.vigncorr_mode
            self.vigncorr_mode = False
            cim = Img(self.files[self.cfn],
                      import_method=self.camera.image_import_method)
            bg_img = None
            self.bg_model.set_missing_ref_areas(cim)
            if self.bg_model.mode == 0:
                print ("Background correction mode is 0, initiating settings "
                    "for poly surface fit")
                try:
                    mask = self.prepare_bg_fit_mask(dilation=True)
                    self.bg_model.surface_fit_mask = mask
                except:
                    warn("Background access mask could not be retrieved for "
                        "PolySurfaceFit in background model of image list %s"
                        %self.list_id)
                
            else:
                if not self.has_bg_img():
                    raise AttributeError("no background image available in "
                        "list %s, please set a suitable background image "
                        "using method set_bg_img, or change current bg " 
                        "modelling mode to 0 using self.bg_model.mode=0)" 
                        %self.list_id)
                bg_img = self.bg_img
            self.bg_model.get_tau_image(self.this, bg_img)
            self.vigncorr_mode = vc
        self._list_modes["tau"] = val
        self.load()
    
    def activate_aa_mode(self, val=True):
        """Activates AA mode (i.e. images are loaded as AA images)
        
        In order for this to work, the following prerequisites need to be
        fulfilled:
        
            1. This list needs to be an on band list 
            (``self.list_type = "on"``)
            #. At least one offband list must be linked to this list (if more
            offband lists are linked and input param off_id is unspecified, 
            then the first offband list found is used)
            #. The number of images in the off band list must exceed a minimum
            of 50% of the images in this list
            
        """
        if val is self.aa_mode:
            return
        if not self.list_type == "on":
            raise TypeError("This list is not an on band list")
        aa_test = None
        if val:
            if self.this.edit_log["is_aa"]:
                warn("Cannot activate AA mode in image list %s: "
                     "current image is already AA image"
                     %self.list_id)
                return
                
            offlist = self.get_off_list()
            if not isinstance(offlist, ImgList):
                raise Exception("Linked off band list could not be found")
            if not offlist.nof / float(self.nof) > 0.25:
                raise IndexError("Off band list does not have enough images...")
            if self.bg_model.mode != 0:
                if not self.has_bg_img():
                    raise AttributeError("no background image available, "
                        "please set suitable background image using method "
                        "set_bg_img or set background modelling mode = 0")
                if not offlist.has_bg_img():
                    raise AttributeError("no background image available in "
                        "off band list. Please set suitable background image "
                        "using method set_bg_img or set background modelling "
                        "mode = 0")
            #offlist.update_img_prep(**self.img_prep)
            #offlist.init_bg_model(mode = self.bg_model.mode)
            self.tau_mode = 0
            offlist.tau_mode = 0
    
            aa_test = self._aa_test_img(offlist)
        self._list_modes["aa"] = val
        
        self.load()

        return aa_test
    
    def activate_optflow_mode(self, val=True, draw=False):
        """Activate / deactivate optical flow calculation on image load
        
        Parameters
        ----------
        val : bool
            activate / deactivate
        draw : bool
            if True, flow field is plotted into current image
            

        """
        if val is self.optflow_mode:
            return 
#==============================================================================
#         if self.crop:
#             raise ValueError("Optical flow analysis can only be applied to "
#                 "uncropped images, please deactivate crop mode")
#==============================================================================
        if val:
            try:
                self.set_flow_images()
            except IndexError:
                raise IndexError("Optical flow mode cannot be activated in "
                    "image list %s: list is at last index, please change list "
                    "index and retry")
            self.optflow.calc_flow()
#==============================================================================
#             len_im = self.optflow.get_flow_vector_length_img() #is at pyrlevel
#             img = self.current_img()
#             if img.edit_log["is_tau"]:
# #==============================================================================
# #                 cond = logical_and(img.img > -0.03, img.img < 0.03) #tau values around 0
# #                 if cond.sum() == 0:
# #                     raise Exception("Fatal: could not activate optical flow: "
# #                         "retrieval of noise ref area failed, since current "
# #                         "list image is flagged as tau image but does not "
# #                         "contain pixels showing values around zero using cond: "
# #                         "(-0.03 < value < 0.03)")
# #                         
# #                 sub = len_im[cond]
# #==============================================================================
#                 min_len = 1.0
#             else:
#                 if self.bg_model.scale_rect is None:
#                     self.bg_model.guess_missing_settings(img)
#                 roi = map_roi(self.bg_model.scale_rect, self.pyrlevel)
#                 sub = len_im[roi[1]:roi[3], roi[0]:roi[2]]
#                 min_len = ceil(sub.mean() + 3 * sub.std()) + 0.5
#             self.optflow.settings.min_length = min_len
#==============================================================================
            if draw:
                self.optflow.draw_flow()
        self._list_modes["optflow"] = val
    
    """GETTERS"""
    def get_dark_image(self, key="this"):
        """Prepares the current dark image dependent on ``DARK_CORR_OPT``
        
        The code checks current dark correction mode and, if applicable, 
        prepares the dark image. 

            1. ``self.DARK_CORR_OPT == 0`` (no dark correction)
                return False
                
            2. ``self.DARK_CORR_OPT == 1`` (model dark image from a sample dark
                and offset image)
                Try to access current dark and offset image from 
                ``self.dark_lists`` and ``self.offset_lists`` (so these must
                exist). If this fails for some reason, set 
                ``self.DARK_CORR_OPT = 2``, else model dark image using
                :func:`model_dark_image` and return this image
                
            3. ``self.DARK_CORR_OPT == 2`` (subtract dark image if exposure 
                times of current image does not deviate by more than 20% to 
                current dark image)
                Try access current dark image in ``self.dark_lists``, if this 
                fails, try to access current dark image in ``self.darkImg``
                (which can be set manually using :func:`set_dark_image`). If 
                this also fails, set ``self.DARK_CORR_OPT = 0`` and return 
                False. If a dark image could be found and the exposure time
                differs by more than 20%, set ``self.DARK_CORR_OPT = 0`` and 
                raise ValueError. Else, return this dark image.
                
        """
        if self.DARK_CORR_OPT == 0:
            raise ValueError("Dark image could not be accessed in list %s: "
                "DARK_CORR_OPT is zero, please set DARK_CORR_OPT according "
                "to your data type")
                
        img = self.current_img(key)
        read_gain = img.meta["read_gain"]
        self.update_index_dark_offset_lists()
        if self.DARK_CORR_OPT == 1:
            try:
                dark = self.dark_lists[read_gain]["list"].current_img()
                offset = self.offset_lists[read_gain]["list"].current_img()
                dark = model_dark_image(img, dark, offset)
            except:
                try:
                    dark = model_dark_image(img, self.master_dark,
                                            self.master_offset)
                except:
                    raise ValueError("Dark image could not be accessed in "
                            "image list %s (DARK_CORR_OPT=1)")

        if self.DARK_CORR_OPT == 2:
            try:
                dark = self.dark_lists[read_gain]["list"].current_img()
                if not isinstance(dark, Img):
                    raise ValueError
            except:
                dark = self.master_dark
                if not isinstance(dark, Img):
                    raise ValueError("Dark image could not be accessed in "
                            "image list %s (DARK_CORR_OPT=2)")
        try:
            texp_ratio = img.meta["texp"] / dark.meta["texp"]
            if not 0.8 <= texp_ratio <= 1.2:
                warn("Exposure time of current dark image in list %s "
                     "deviates by more than 20% from list image %s "
                     "(current list index: %d)"
                     %(self.list_id, key, self.cfn))
        except:
            pass
    
        return dark
    
    def get_off_list(self, list_id=None):
        """Search off band list in linked lists
        
        :param str list_id: specify the ID of the list. If unspecified (None), 
            the default off band filter key is attempted to be accessed
            (``self.camera.filter_setup.default_key_off``) and if this fails,
            the first off band list found is returned.
            
            
        """
        if list_id is None:
            try:
                list_id = self.camera.filter_setup.default_key_off
                #print "Found default off band key %s" %list_id
            except:
                pass
        for lst in self.linked_lists.values():
            if lst.list_type == "off":
                if list_id is None or list_id == lst.list_id:
                    return lst
        raise AttributeError("No linked offband list was found")
        
    """SETTERS: ATTRIBUTE ASSIGNMENT METHODS"""
    def set_bg_img(self, bg_img):
        """Update the current background image object
        
        Check input background image and, in case a vignetting mask is not 
        available in this list, determine a vignetting mask from the background
        image. Furthermore, if the input image is not blurred it is blurred 
        using current list blurring factor and in case the latter is 0, then 
        it is blurred with a Gaussian filter of width 1.
        
        The image is then stored twice, 1. as is and 2. corrected for 
        vignetting.
        
        Parameters
        ----------
        bg_img : Img
            the background image object used for plume background modelling 
            (modes 1 - 6 in :class:`PlumeBackgroundModel`)        
        """
        if not isinstance(bg_img, Img):
            print ("Could not set background image in ImgList %s: "
                ": wrong input type, need Img object" %self.list_id)
            return False
        vign_mask = self.vign_mask
#==============================================================================
#         bl = bg_img.edit_log["blurring"]
#         if bl == 0 or bl < self.gaussian_blurring:
#             blur = self.gaussian_blurring - bl
#             if blur == 0:
#                 blur += 1
#             bg_img.add_gaussian_blurring(blur)
#==============================================================================
        if vign_mask is None:
            if bg_img.edit_log["vigncorr"]:
                raise AttributeError("Input background image is vignetting "
                    "corrected and cannot be used to calculate vignetting corr"
                    "mask.")
        
            self._bg_imgs[0] = bg_img
            vign_mask = self.det_vign_mask_from_bg_img()
            self._bg_imgs[1] = bg_img.duplicate().correct_vignetting(vign_mask)
        else:
            if not bg_img.edit_log["vigncorr"]:
                bg = bg_img
                bg_vigncorr = bg_img.duplicate().correct_vignetting(vign_mask)
            else:
                bg_vigncorr = bg_img
                bg = bg_img.duplicate().correct_vignetting(vign_mask,
                                                           new_state=0)
            self._bg_imgs = [bg, bg_vigncorr]
            
    def set_bg_corr_mode(self, mode=1):
        """Update the current background correction mode in ``self.bg_model``
        
        Parameters
        ----------
        mode : int
            valid bakground modelling mode
        """
        self.BG_MODEL_MODE = mode
    
    def set_flow_images(self):  
        """Update images for optical flow determination 
        
        The images are updated in :attr:`optflow` 
        (:class:`OptflowFarneback` object) using method :func:`set_images`
        
        Raises
        
        object, i.e. `self.loaded_images["this"]` and `self.loaded_images["next"]`
        """
        if self.cfn == self.nof - 1:
            self.optflow.reset_flow()
            raise IndexError("Optical flow images cannot be set in ImgList %s:"
                " reached last image ..." %self.list_id)
            
        self.optflow.set_images(self.loaded_images["this"],
                                self.loaded_images["next"])
    
    def set_optical_flow(self, optflow):
        """Set the current optical flow object 
        
        Currently only support for type :class:`OptflowFarneback`
        
        Parameters
        ----------
        optflow : OptflowFarneback
            the optical flow engine
        """
        if not isinstance(optflow, OptflowFarneback):
            raise ValueError("Need class OptflowFarneback")
        self.optflow = optflow
    
    def set_darkcorr_mode(self, mode):
        """Update dark correction mode
        
        :param int mode (1): new mode
        """
        if 0 <= mode <= 2:
            self.camera.DARK_CORR_OPT = mode
            return True
        return False
    
    def add_master_dark_image(self, dark, acq_time=datetime(1900, 1, 1),
                              texp=0.0, read_gain=0):
        """Add a (master) dark image data to list
        
        Sets a dark image, which is used for dark correction in case, 
        no dark / offset image lists are linked to this object or the data 
        extraction from these lists does not work for some reason.
        
        :param (Img, ndarray) dark: dark image data 
        :param datetime acq_time: image acquisition time (only updated if input 
            image is numpy array or if acqtime in Img object is default), 
            default: (1900, 1, 1)
        :param float texp: optional input for exposure time in units of
            s (i.e. is used if img input is ndarray or if exposure time is not
            set in the input img)
        
        The image is stored at::
        
            stored at self.master_dark
            
        """
        if not any([isinstance(dark, x) for x in [Img, ndarray]]):
            raise TypeError("Could not set dark image in image list, invalid"
                " input type")
        elif isinstance(dark, Img):
            if dark.meta["texp"] == 0.0: 
                if texp == 0.0:
                    raise ValueError("Could not set dark image in image "
                            "list, missing input for texp")       
                dark.meta["texp"] = texp
                
        elif isinstance(dark, ndarray):
            if texp == None:
                raise ValueError("Could not add dark image in image list, "
                    "missing input for texp")
            dark = Img(dark, texp=texp)

        if (acq_time != datetime(1900,1,1) and 
            dark.meta["start_acq"] == datetime(1900,1,1)):
            dark.meta["start_acq"] = acq_time
        dark.meta["read_gain"] = read_gain
            
        self.master_dark = dark
    
    
    def add_master_offset_image(self, offset, acq_time=datetime(1900, 1, 1),
                                texp=0.0, read_gain=0):
        """Add a (master) offset image to list
        
        Sets a offset image, which is used for dark correction in case, 
        no dark / offset image lists are linked to this object or the data 
        extraction from these lists does not work for some reason.
        
        :param (Img, ndarray) offset: offset image data 
        :param datetime acq_time: image acquisition time (only used if input
            image is numpy array or if acqtime in Img object is default)
        :param float texp: optional input for exposure time in units of
            s (i.e. is used if img input is ndarray or if exposure time is not
            set in the input img)
            
        The image is stored at::
        
            self.master_offset
                    
        """
        if not any([isinstance(offset, x) for x in [Img, ndarray]]):
            raise TypeError("Could not set offset image in image list, invalid"
                " input type")
        elif isinstance(offset, Img):
            if offset.meta["texp"] == 0.0: 
                if texp == 0.0:
                    raise ValueError("Could not set offset image in image "
                            "list, missing input for texp")       
                offset.meta["texp"] = texp
                
        elif isinstance(offset, ndarray):
            if texp == None:
                raise ValueError("Could not add offset image in image list, "
                    "missing input for texp")
            offset = Img(offset, texp=texp)

        if (acq_time != datetime(1900,1,1) 
                and offset.meta["start_acq"] == datetime(1900,1,1)):
            offset.meta["start_acq"] = acq_time
        offset.meta["read_gain"] = read_gain
        self.master_offset = offset
    
    def set_bg_img_from_polyfit(self, mask=None, **kwargs):
        """Sets background image from results of a poly surface fit
        
        Parameters
        ----------
        mask : array
            mask specifying sky background pixels, if None (default) then this
            mask is determined automatically using :func:`prepare_bg_fit_mask`
        **kwargs:
            additional keyword arguments for :class:`PolySurfaceFit
        Returns
        -------
        Img
            fitted background image
        """
        if mask is None:
            mask = self.prepare_bg_fit_mask(dilation=True)
        fit = PolySurfaceFit(self.current_img(), mask, **kwargs)
        bg = fit.model
        try:
            low = self.get_dark_image().mean()
        except:
            low = finfo(float).eps
        print "LOW: %s" %low
        bg [bg <= low] = low
        self.bg_img = Img(bg)
    
    def set_closest_dark_offset(self):
        """Updates the index of the current dark and offset images 
        
        The index is updated in all existing dark and offset lists. 
        """
        try:
            num = self.index
            for read_gain, info in self.dark_lists.iteritems():
                darknum = info["idx"][num]
                if darknum != info["list"].index:
                    print ("Dark image index (read_gain %s) was changed in "
                            "list %s from %s to %s" %(read_gain, self.list_id, 
                                                  info["list"].index, darknum))
                    info["list"].goto_img(darknum)
            
            if self.DARK_CORR_OPT == 1:
                for read_gain, info in self.offset_lists.iteritems():
                    offsnum = info["idx"][num]
                    if offsnum != info["list"].index:
                        print ("Offset image index (read_gain %s) was changed "
                            "in list %s from %s to %s" %(read_gain, 
                                self.list_id, info["list"].index, offsnum))
                        info["list"].goto_img(offsnum)
        except Exception:
            print ("Failed to update index of dark and offset lists")
            return False
        return True
        
    """LINKING OF OTHER IMAGE LIST OBJECTS"""       
    def link_imglist(self, other_list, list_id=None):
        """Link another image list to this list
        
        :param other_list: another image list object
        
        """
        if list_id is None:
            list_id = other_list.list_id
        self.current_img(), other_list.current_img()
        self.linked_lists[list_id] = other_list
        self.linked_indices[list_id] = {}
        idx_array = self.assign_indices_linked_list(other_list)
        self.linked_indices[list_id] = idx_array
        self.update_index_linked_lists()  
        self.load()

    def disconnect_linked_imglist(self, list_id):
        """Disconnect a linked list from this object
        
        :param str list_id: string id of linked list
        """
        if not list_id in self.linked_lists.keys():
            print ("Error: no linked list found with ID " + str(list_id))
            return 0
        del self.linked_lists[list_id]
        del self.linked_indices[list_id]
    
        
    def link_dark_offset_lists(self, list_dict):
        """Assign dark and offset image lists to this object
        
        Set dark and offset image lists: get "closest-in-time" indices of dark 
        list with respect to the capture times of the images in this list. Then
        get "closest-in-time" indices of offset list with respect to dark list.
        The latter is done to ensure, that dark and offset set used for image
        correction are recorded subsequently and not individual from each other
        (i.e. only closest in time to the current image)
        """
        dark_assigned = False
        offset_assigned = False
        try:
            texp = self.current_img().texp
            if texp == 0 or isnan(texp):
                raise ValueError
        except:
            warn("Exposure time could not be accessed in ImgList %s"
                %self.list_id)
                
        warnings = []
        # if input contains multiple lists for one of the two types (e.g. 2
        # type "dark" lists), then try to assign dark list with the smallest 
        # difference in image exposure time. Here two helpers are initiated 
        # for logging the difference in exposure (this method is for instance
        # relevant for the HD cam), requires flag: texp_access = True (see 
        # above)
        dtexp_dark, dtexp_offset = 999999, 999999
        for lst in list_dict.values():
            if isinstance(lst, DarkImgList):
                if lst.list_type == "dark":
                    try:
                        dt = abs(texp - lst.current_img().texp)
                        if isnan(dt):
                            raise ValueError
                        elif dt < dtexp_dark\
                                or not self.dark_lists.has_key(lst.read_gain):
                            self.dark_lists[lst.read_gain] = od()
                            self.dark_lists[lst.read_gain]["list"] = lst
                            dtexp_dark = dt
                            dark_assigned = True
                    except:
                        self.dark_lists[lst.read_gain] = od()
                        self.dark_lists[lst.read_gain]["list"] = lst
                        dark_assigned = True
        
                elif lst.list_type == "offset":
                    try:
                        dt = abs(texp - lst.current_img().texp)
                        if dt < dtexp_offset or not\
                                self.offset_lists.has_key(lst.read_gain):
                            self.offset_lists[lst.read_gain] = od()
                            self.offset_lists[lst.read_gain]["list"] = lst
                            dtexp_offset = dt
                            offset_assigned = True
                    except:
                        self.offset_lists[lst.read_gain] = od()
                        self.offset_lists[lst.read_gain]["list"] = lst
                        offset_assigned = True
                        
                else:
              
                    warnings.append("List %s, type %s could not be linked "
                        %(lst.list_id, lst.list_type))
            else:
                warnings.append("Obj of type %s could not be linked, need "
                                        " DarkImgList " %type(lst))
                                        
        for gain, value in self.dark_lists.iteritems():
            value["idx"] = self.assign_indices_linked_list(value["list"])
        for gain, value in self.offset_lists.iteritems():
            value["idx"] = self.assign_indices_linked_list(value["list"])
        _print_list(warnings) 
        return dark_assigned, offset_assigned
    
    """INDEX AND IMAGE LOAD MANAGEMENT"""
    def load(self):
        """Try load current and next image"""
        self.update_index_linked_lists() #based on current index in this list
        if not super(ImgList, self).load():
            print ("Image load aborted...")
            return False
        if self.nof > 1:
#==============================================================================
#             prev_file = self.files[self.prev_index]
#             self.loaded_images["prev"] = Img(prev_file,
#                             import_method=self.camera.image_import_method,
#                             **self.get_img_meta_from_filename(prev_file))
#             self._apply_edit("prev")
#==============================================================================
            
            next_file = self.files[self.next_index]
            self.loaded_images["next"] = Img(next_file,
                            import_method=self.camera.image_import_method,                            
                            **self.get_img_meta_from_filename(next_file))
            self._apply_edit("next")
        else:
            #self.loaded_images["prev"] = self.loaded_images["this"]
            self.loaded_images["next"] = self.loaded_images["this"]
        
        if self.optflow_mode:  
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                warn("Reached last index in image list, optflow_mode will be "
                    "deactivated")
                self.optflow_mode = 0
        return True
    
    def update_index_linked_lists(self):
        """Update current index in all linked lists based on ``cfn``"""
        for key, lst in self.linked_lists.iteritems():
            lst.change_index(self.linked_indices[key][self.index])
            
    def load_next(self):
        """Load next image in list"""
        if self.nof < 2 or not self._auto_reload:
            print ("Could not load next image, number of files in list: " +
                str(self.nof))
            return False
        self.index = self.next_index
        self.update_prev_next_index()
        
        self.update_index_linked_lists() #loads new images in all linked lists
        
        #self.loaded_images["prev"] = self.loaded_images["this"]
        self.loaded_images["this"] = self.loaded_images["next"]
        
        next_file = self.files[self.next_index]
        self.loaded_images["next"] = Img(next_file,
                            import_method=self.camera.image_import_method,                            
                            **self.get_img_meta_from_filename(next_file))
    
        self._apply_edit("next")
        if self.optflow_mode:  
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                warn("Reached last index in image list, optflow_mode will be "
                    "deactivated")
                self.optflow_mode = 0
        return True
        
    def change_index(self, idx):
        """Change current image based on index of file list
        
        :param idx: index in `self.files` which is supposed to be loaded
        
        Dependend on the input index, the following scenarii are possible:
        If..
        
            1. idx < 0 or idx > `self.nof`
                then: do nothing (return)
            #. idx == `self.index`
                then: do nothing
            #. idx == `self.next_index`
                then: call :func:`next_img`
            #. idx == `self.prev_index`
                then: call :func:`prev_img`
            #. else
                then: call :func:`goto_img`
        """
        if not -1 < idx < self.nof or idx == self.index:
            return
        elif idx == self.next_index:
            self.next_img()
            return
        elif idx == self.prev_index:
            self.prev_img()
            return
        #: goto_img calls :func:`load` which calls prepare_additional_data
        self.goto_img(idx)

        return self.loaded_images["this"]
    
    """PROCESSING AND ANALYSIS METHODS""" 
    def optflow_histo_analysis(self, lines=[], start_idx=0, stop_idx=None, 
                               intensity_thresh=0, **optflow_settings):
        cfn_tmp = self.cfn
        flm = self.optflow_mode
        self.goto_img(start_idx)
        self.optflow.settings.update(**optflow_settings)
        props = []  
        for line in lines:
            if isinstance(line, LineOnImage):
                props.append(LocalPlumeProperties(line.line_id, 
                                                  color=line.color))

        if len(props) == 0:
            lines=[None]
            props.append(LocalPlumeProperties("thresh_%.1f" %intensity_thresh))

        if stop_idx is None:
            stop_idx = self.nof - 1
            
        self.optflow_mode = True
        for k in range(start_idx, stop_idx):
            plume_mask = self.get_thresh_mask(intensity_thresh)
            for i in range(len(props)):
                props[i].get_and_append_from_farneback(self.optflow, 
                                                      line=lines[i],
                                                      pix_mask=plume_mask)

            self.next_img()
        self.goto_img(cfn_tmp)
        self.optflow_mode = flm
        return props
        
    def get_thresh_mask(self, thresh=None, this_and_next=True):
        """Get bool mask based on intensity threshold
        
        Parameters
        ----------
        thresh : :obj:`float`, optional
            intensity threshold
        this_and_next : bool
            if True, uses the current AND next image to determine mask
            
        Returns
        -------
        array
            mask specifying pixels that exceed the threshold
        """
        mask = self.this.duplicate().to_binary(thresh).img
        if this_and_next and not self.cfn == self.nof - 1:
            mask = logical_or(mask, 
                              self.next.duplicate().to_binary(thresh).img)
        return mask

    def det_vign_mask_from_bg_img(self):
        """Determine vignetting mask from current background image
        
        The mask is determined using a blurred (:math:`\sigma = 3`) 
        background image which is normalised to one.
        
        The mask is stored in ``self.vign_mask``
        
        Returns
        -------
        Img
            vignetting mask
        """
        if not self.has_bg_img():
            raise AttributeError("Please set a background image first")
        mask = self._bg_imgs[0].duplicate()
        if mask.edit_log["blurring"] < 3:
            mask.add_gaussian_blurring(3)
        mask.img = mask.img / mask.img.max()
        self.vign_mask = mask
        return mask
    
    def prepare_bg_fit_mask(self, dilation=False, dilate_kernel=None,
                            optflow_blur=1, optflow_median=10, 
                            plot_masks=False, **flow_settings):
        """Prepare mask for background fit based on analysis of current image
        
        The mask is determined based on intensities in the 3 reference 
        recangular areas in the plume background model (if they are not 
        assigned then they are retrieved using :func:`set_missing_ref_areas`
        in :attr:`bg_model`). Furthermore, an optical flow analysis is 
        performed in order to exclude image pixels where movement could be 
        detected.
        
        Note
        ----
        
        This is a beta version
        
        Parameters
        ----------
        dilation : bool
            if True, the mask is dilated
        dilate_kernel : array
            if None, uses 30x30 pix kernel
        optflow_blur : int
            amount of Gaussian blurring applied to images before optical flow
            is determined
        optflow_median : int
            apply median filter of specified size to length image of optical 
            flow vectors (can be useful in order to remove artifacts and only
            mask out movement areas spanning a reasonable pixel neighbourhood)
        plot_masks : bool
            if True, creates subplot showing indidivual masks used to determine
            the background mask (1. is based on intensity thresh, second based
            on detected movement)
        **flow_settings 
            keyword arguments for optical flow settings
            
        Returns
        -------
        array 
            mask specifying detected background pixels
            
        """
        # remember some settings
        fl_mode = self.optflow_mode
        bl = self.gaussian_blurring
        s_temp = self.optflow.settings.duplicate()
        
        img = self.current_img().duplicate()
        mask = ones(img.shape)
        
        self.bg_model.set_missing_ref_areas(img)
    
        mean, low, high = self.bg_model.mean_in_rects(img)
        thresh = mean - mean * 0.1
    
        cond_low = (img.img < thresh).astype(uint8)
        
        s = self.optflow.settings
        s.auto_update = False
        keys = flow_settings.keys()
        if not "i_min" in keys:
            flow_settings["i_min"] = low
        elif not "i_max" in keys:
            flow_settings["i_max"] = img.max()
        
        s.update(**flow_settings)
        self.gaussian_blurring = optflow_blur
        print "I MIN: %s" %self.optflow.settings.i_min
        print "I MAX: %s" %self.optflow.settings.i_max
        self.optflow_mode = True
        len_im = Img(self.optflow.get_flow_vector_length_img())
        if optflow_median > 0:
            len_im = len_im.apply_median_filter(optflow_median)
        cond_movement = (len_im.img > s.min_length).astype(uint8)
        
        if dilate:
            if dilate_kernel is None:
                dilate_kernel = ones((30, 30), dtype=uint8)   
            cond_low = dilate(cond_low, dilate_kernel)
            cond_movement = dilate(cond_movement, dilate_kernel)
        
        mask = mask * (1 - cond_low) * (1 - cond_movement)
        
        if plot_masks:
            fig, ax = subplots(1,2, figsize=(18,8))
            
            ax[0].imshow(cond_low, cmap="gray")
            ax[0].set_title("Below intensity thresh %.1f" %thresh)
            ax[1].imshow(cond_movement, cmap="gray")
            ax[1].set_title("Movement detected")
        self.optflow.settings = s_temp
        self.optflow_mode = fl_mode
        self.gaussian_blurring = bl
        self.bg_model.surface_fit_mask = mask
        return mask.astype(bool)
        
    def prep_data_dilutioncorr(self, tau_thresh=0.05, plume_pix_mask=None, 
                               plume_dists=None, ext_coeff=None):
        """Get parameters relevant for dilution correction
        
        Relevant parameters are:
        
            1. Current plume background
            #. Plume distance estimate (either global or on a pixel basis)
            #. Plume pixel mask (only plume pixels are corrected)
           
        Note
        ----
        This method changes the current image preparation state such that tau
        mode is deactivated and vigncorr mode is activated. 
        
        Parameters
        ----------
        tau_thresh : float
            tau threshold for retrieval of plume pixel mask. Is only used in
            case next :param:`plume_mask` is unspecified or invalid. In this
            case the plume mask is retrieved using :func:`get_thresh_mask`.
        plume_pix_mask : :obj:`array`, :obj:`Img`, optional
            mask specifying plume pixels. If valid, it will be passed through 
            and no threshold mask will retrieved (see :param:`tau_thresh`)
        plume_dists : :obj:`array`, :obj:`Img`, :obj:`float`, optional
            plume distance(s) in m. If input is numpy array or :class:`Img` 
            then, it must have the same shape as the current image
        ext_coeff : :obj:`float`, optional
            atmospheric extinction coefficient. If unspecified, try access 
            via :attr:`ext_coeff` which returns the current extinction 
            coefficient and raise :obj:`AttributeError` in case, no coeffs are
            assigned to this list
            
        Returns
        -------
        tuple
            5-element tuple containing input for dilution correction
            
            - :obj:`Img`, current vignetting corrected image 
            - :obj:`float`, current extinction coefficient 
            - :obj:`Img`, current plume background
            - (:obj:`array`, :obj:`float`), plume distance(s)
            - :obj:`array`, mask specifying plume pixels
        """
        # check input distance and if invalid try retrieve using measurement
        # geometry
        try:
            try:
                plume_pix_mask = plume_pix_mask.img
            except:
                pass
            
            if plume_pix_mask.shape == self.current_img().shape:
                mask_ok = True
            else:
                mask_ok = False
        except:
            mask_ok = False
        
        
        dists = plume_dists
        
        if dists is None:
            try:
                (_, 
                 _, 
                 dists) = self.meas_geometry.get_all_pix_to_pix_dists(
                             pyrlevel=self.pyrlevel) 
                dists = dists.img
            except:
                raise ValueError("Measurement geometry not ready for access "
                    "of plume distances in image list %s. Please provide "
                    "plume distance using input parameter plume_dist_m" 
                    %self.list_id)
        # get current extinction coefficient, raises AttributeError if not 
        # available
        try:
            ext_coeff = float(ext_coeff)
        except:
            ext_coeff = self.ext_coeff 
        self.vigncorr_mode = False
        self.tau_mode = True
        tau0 = self.current_img().duplicate()
        self.vigncorr_mode = True
        #bg = self.bg_model.current_plume_background
        #bg.edit_log["vigncorr"] = True
        if not mask_ok:
            print "Retrieving plume pixel mask in list %s" %self.list_id
            plume_pix_mask = self.get_thresh_mask(tau_thresh)    
        self.tau_mode = False
        bg = self.current_img() * exp(tau0.img)
        return (self.current_img(), ext_coeff, bg, dists, plume_pix_mask)
    
    def correct_dilution(self, tau_thresh=0.05, plume_pix_mask=None,
                         plume_dists=None, ext_coeff=None):
        """Correct current image for signal dilution
        
        Requires measurement geometry (:attr:`meas_geometry`) and extinction
        coefficients (:attr:`ext_coeffs`) to be available in this list.
        Further, the list needs to be prepared such that :attr:`tau_mode` can
        be activated since the correction requires an accurate estimation of
        the current plume background. The latter can be retrieved from
        :attr:`bg_model` (:class:`PlumeBackgroundModel`) after 
        :func:`get_tau_image` was called therein.
        
        Parameters
        ----------
        tau_thresh : :obj:`float`, optional
            tau threshold applied to determine plume pixel mask (retrieved 
            using :attr:`tau_mode`, not :attr:`aa_mode`)
        plume_pix_mask : :obj:`float`, optional
            mask specifying plume pixels. If valid, it will be passed through 
            and no threshold mask will retrieved (see :param:`tau_thresh`)
        plume_dists : :obj:`array`, :obj:`Img`, :obj:`float`
            plume distance(s) in m. If input is numpy array or :class:`Img` 
            then, it must have the same shape as the current image
        ext_coeff : :obj:`float`, optional
            atmospheric extinction coefficient. If unspecified, try access 
            via :attr:`ext_coeff` which returns the current extinction 
            coefficient and raise :obj:`AttributeError` in case, no coeffs are
            assigned to this list
            
        Returns
        -------
        tuple
            3-element tuple containing input for dilution correction
            
            - :obj:`Img`, vignetting and dilution corrected image
            - :obj:`Img`, corresponding plume background
            - :obj:`array`, mask specifying plume pixels

        """
        from .dilutioncorr import correct_img        
        (img, 
         ext_coeff, 
         bg, 
         plume_dists, 
         plume_pix_mask) = self.prep_data_dilutioncorr(tau_thresh,
                                                       plume_pix_mask, 
                                                       plume_dists, 
                                                       ext_coeff)
                                                       
                                                       
        corr = correct_img(img, ext_coeff, bg, plume_dists, plume_pix_mask)
                                  
        bad_pix = corr.img <= 0
        corr.img[bad_pix] = self.current_img().img[bad_pix]
            
        return (corr, bg, plume_pix_mask)
   
    def correct_dilution_all(self, tau_thresh=0.05, ext_on=None, ext_off=None,
                             add_off_list=True, save_dir=None, 
                             save_masks=False, save_bg_imgs=False, 
                             save_tau_prev=False, vmin_tau_prev=None, 
                             vmax_tau_prev=None, **kwargs):
        """Correct all images for signal dilution
        
        Correct and save all images in this list for the signal dilution 
        effect. See :func:`correct_dilution` and :func:`prep_data_dilutioncorr` 
        for details about requirements and additional input options.
        
        Note
        ----
        The vignetting and dilution corrected images are stored with all 
        additional image preparation settings applied (e.g. dark correction, 
        blurring)
        
        Parameters        
        ----------
        tau_thresh : :obj:`float`, optional
            tau threshold applied to determine plume pixel mask (retrieved 
            using :attr:`tau_mode`, not :attr:`aa_mode`)
        ext_on : :obj:`float`, optional
            atmospheric extinction coefficient at on-band wavelength, if None
            (default), try access via :attr:`ext_coeff`
        ext_off : :obj:`float`, optional
            atmospheric extinction coefficient at off-band wavelength. Only 
            relevant if input param ``add_off_list`` is True. If None (default)
            and ``add_off_list=True`` try access via :attr:`ext_coeff` in off
            band list.
        add_off_list : bool
            if True, also the images in a linked off-band image list 
            (using :func:`get_off_list`) are corrected as well. For the 
            correction of the off-band images, the current plume pixel mask 
            of this list is used.
        save_dir : :obj:`str`, optional
            base directory for saving the corrected images. If None (default),
            then a new directory ``dilcorr`` is created at the storage location 
            of the first image in this list
        save_masks : bool
            if True,  a folder *plume_pix_masks* is created within 
            :param:`save_dir` in which all plume pixel masks are stored as
            FITS
        save_bg_imgs : bool 
            if True, a folder *bg_imgs* is created which is used to store 
            modelled plume background images for each image in this list. This 
            folder can be used on re-import of the data in order to save 
            background modelling time using background modelling mode 99.
        save_tau_prev : bool
            if True, png previews of dilution corrected tau images are saved
        vmin_tau_prev : :obj:`float`, optional
            lower tau value for tau image preview plots
        vmax_tau_prev : :obj:`float`, optional
            upper tau value for tau image preview plots
        **kwargs 
            additional keyword args for dilution correction functions
            :func:`correct_dilution` and :func:`prep_data_dilutioncorr`
        """
        ioff()
        if save_dir is None or not exists(save_dir):
            save_dir = abspath(join(dirname(self.files[0]), ".."))
        save_dir = join(save_dir, "dilutioncorr")
        
        if not exists(save_dir):
            mkdir(save_dir)
        if save_masks:
            mask_dir = join(save_dir, "plume_pix_masks")
            if not exists(mask_dir):
                mkdir(mask_dir)
        if save_bg_imgs:
            bg_dir = join(save_dir, "bg_imgs")
            if not exists(bg_dir):
                mkdir(bg_dir)
        if save_tau_prev:
            tau_dir = join(save_dir, "tau_prev")
            if not exists(tau_dir):
                mkdir(tau_dir)
        # initiate settings
        self.goto_img(0)
        try:
            (img, 
             ext_coeff, 
             bg, 
             plume_dists, 
             plume_pix_mask) = self.prep_data_dilutioncorr(tau_thresh,
                                                           ext_coeff=ext_on)
            if add_off_list:
                off = self.get_off_list()
                (img_off, 
                 ext_coeff_off, 
                 bg_off, 
                 plume_dists, 
                 plume_pix_mask) = off.prep_data_dilutioncorr(plume_pix_mask=
                                                              plume_pix_mask,
                                                              plume_dists=
                                                              plume_dists,
                                                              ext_coeff=ext_off)
        except:
            raise Exception("Failed to initiate dilution correction with "
                "error:\n%s" %format_exc())
        saved_off = []
        num = self.nof
        for k in range(num):
            (corr, 
             bg, 
             plume_pix_mask) = self.correct_dilution(tau_thresh=tau_thresh,
                                                     plume_dists=plume_dists,
                                                     ext_coeff=ext_on)
            corr.save_as_fits(save_dir)
            fname = corr.meta["file_name"]
            if save_masks:
                Img(plume_pix_mask, dtype=uint8, 
                    file_name=fname).save_as_fits(mask_dir)
            if save_bg_imgs:
                bg.save_as_fits(bg_dir, fname)
            if save_tau_prev:
                tau = corr.to_tau(bg)
                fig = self.bg_model.plot_tau_result(tau, 
                                                    tau_min=vmin_tau_prev,
                                                    tau_max=vmax_tau_prev)
                name = fname.split(".")[0] + ".png"
                fig.savefig(join(tau_dir, name))
                close("all")
                del fig
            if add_off_list:
                if not off.current_img().meta["file_name"] in saved_off:
                    # use on band plume pixel mask
                    (corr_off, 
                     bg_off, 
                     _) = off.correct_dilution(plume_pix_mask=plume_pix_mask,
                                               plume_dists=plume_dists)
                    saved_off.append(corr_off.save_as_fits(save_dir))
                    if save_bg_imgs:
                        bg_off.save_as_fits(bg_dir, corr_off.meta["file_name"])    
            
            self.next_img()
        ion()
                
    """I/O"""
    def import_ext_coeffs_csv(self, file_path, header_id, **kwargs):
        """Import extinction coefficients from csv 
        
        The text file requires datetime information in the first column and
        a header which can be used to identify the column. The import is 
        performed using :func:`pandas.DataFrame.from_csv` 
        
        Parameters
        ----------
        file_path : str
            the csv data file
        header_id : str
            header string for column containing ext. coeffs
        **kwargs :
            additionald keyword args passed to :func:`pandas.DataFrame.from_csv`
            
        Returns
        -------
        Series
            pandas Series containing extinction coeffs
            
        Todo
        ----
        
        This is a Beta version, insert try / except block after testing
        
        """
        df = DataFrame.from_csv(file_path, **kwargs)
        self.ext_coeffs = df[header_id]
        return self.ext_coeffs
    
    """HELPERS"""
    def has_bg_img(self):
        """Returns boolean whether or not background image is available"""
        if not isinstance(self.bg_img, Img):
            return False
        return True
        
    def update_index_dark_offset_lists(self):
        """Check and update current dark image (if possible / applicable)"""
        if self.DARK_CORR_OPT == 0:
            return
        t_last = self.time_last_dark_check

        ctime = self.current_time()

        if not (t_last - timedelta(minutes = self.update_dark_ival)) < ctime <\
                        (t_last + timedelta(minutes = self.update_dark_ival)):
            if self.set_closest_dark_offset():
                print ("Updated dark / offset in img_list %s at %s"
                        %(self.list_id, ctime))
                self.time_last_dark_check = ctime
                
        
    """Private methods"""
    def _apply_edit(self, key):
        """Applies the current image edit settings to image
        
        :param str key: image id (e.g. this)
        """
        if not self.edit_active:
            warn("Edit not active in img_list %s: no image preparation will "
                "be performed" %self.list_id)
            return
        if key == "this":
            upd_bgmodel = True
        else:
            upd_bgmodel = False
        img = self.loaded_images[key]
        if self.darkcorr_mode:
            dark = self.get_dark_image(key)
            img.subtract_dark_image(dark)
        if self.vigncorr_mode:
            img.correct_vignetting(self.vign_mask, new_state=True)
        if self.tau_mode:
            bg = self.bg_img.to_pyrlevel(img.pyrlevel)
            img = self.bg_model.get_tau_image(plume_img=img, 
                                              bg_img=bg,
                                              update_imgs=upd_bgmodel)
        elif self.aa_mode:
            bg_on = self.bg_img.to_pyrlevel(img.pyrlevel)
            off_list = self.get_off_list()
            off_img = off_list.current_img().to_pyrlevel(img.pyrlevel)
            bg_off = off_list.bg_img.to_pyrlevel(img.pyrlevel)
            
            img = self.bg_model.get_aa_image(plume_on=img, 
                                             plume_off=off_img,
                                             bg_on=bg_on,
                                             bg_off=bg_off,
                                             update_imgs=upd_bgmodel)
            if self.sensitivity_corr_mode:
                
                img = img / self.aa_corr_mask
                img.edit_log["senscorr"] = 1
        # Direct calibration of tau images possible
        if self.calib_mode:
            img.img = self.calib_data(img.img)
            img.edit_log["gascalib"] = True
    
        img.to_pyrlevel(self.img_prep["pyrlevel"])
        if self.img_prep["crop"]:
            img.crop(self.roi_abs)
        if self.img_prep["8bit"]:
            img._to_8bit_int(new_im = False)
        # do this at last, since it can be time consuming and is therfore much
        # faster in case pyrlevel > 0 or crop applied
        img.add_gaussian_blurring(self.img_prep["blurring"])
        img.apply_median_filter(self.img_prep["median"])
        self.loaded_images[key] = img
        
    def _aa_test_img(self, off_list):
        """Try to determine an AA image"""
        on = Img(self.files[self.cfn],
                 import_method=self.camera.image_import_method)
        off = Img(off_list.files[off_list.cfn],
                  import_method=self.camera.image_import_method)
        bg_on = self.bg_img.to_pyrlevel(on.pyrlevel)
        bg_off = off_list.bg_img.to_pyrlevel(off.pyrlevel)
        return self.bg_model.get_aa_image(on, off, bg_on, bg_off)
        
class CellImgList(ImgList):
    """Image list object for cell images
    
    Whenever cell calibration is performed, one calibration cell is put in 
    front of the lense for a certain time and the camera takes one (or ideally)
    a certain amount of images. 
    
    This image list corresponds to such a list of images with one specific
    cell in the camera FOV. It is a :class:`BaseImgList` only extended by 
    the variable ``self.gas_cd`` specifying the amount of gas (column 
    density) in this cell.
    """
    def __init__(self, files=[], list_id=None, list_type=None, camera=None,
                 geometry=None, cell_id="", gas_cd=0.0, gas_cd_err=0.0):
        
        super(CellImgList, self).__init__(files, list_id, list_type, camera,
                                          geometry)
        self.cell_id = cell_id
        self.gas_cd = gas_cd
        self.gas_cd_err = gas_cd_err
        
    def update_cell_info(self, cell_id, gas_cd, gas_cd_err):
        """Update cell_id and gas_cd amount"""
        self.cell_id = cell_id
        self.gas_cd = gas_cd
        self.gas_cd_err = gas_cd_err

from astropy.io import fits
import numpy as np
        
class ImgListMultiFits(ImgList):
    """Image list object which can be used with mulitple fit files (comtessa project)
    
    Additional features:
        
            1. Indexing using double index: Filename and image plane
            2. Function which returns a DataFrame of all available data
            
    Parameters
    ----------
    files : list
        list containing image file paths, defaults to ``[]`` (i.e. empty list)
    list_id : :obj:`str`, optional
        string ID of this list, defaults to None
    list_type : :obj:`str`, optional
        string specifying type of image data in this list (e.g. on, off)
    camera : :obj:`Camera`, optional
        camera specifications, defaults to None
    geometry : :obj:`MeasGeometry`, optional
        measurement geometry
    init : bool
        if True, the first two images in list ``files`` are loaded
    
    """
    
    def __init__(self, files=[], meta=None, list_id=None, list_type=None, camera=None,
                 geometry=None, init=True):
        ''' let's assume I already detected all relevant fits files and give them as a parameter
        Load ALL images inside the files (only multiple of minute intervals possible)
        Alternatively, the the imagelist can be loaded by giving a meta DataFrame
        --> speed up the initialisation; in this case self.fitsfiles can remain empty
        
        Camera can be defined by loading the first image (fov, pixel, etc...)
        '''
        # uses the init method from ImgList but does not load the files!
        super(ImgListMultiFits, self).__init__(files, list_id, list_type, camera, 
                                      geometry, init=False)
        
        # redefinition of several paramters and new parameter

        # n files with m_n images
        # datafiles (every file only once)
        # only needed for fast referencing
        # make a function/property out of it!
        self.fitsfiles = files
        
        if isinstance(meta, DataFrame):
            try:
                self.metaData = meta
            except:
                self.metaData = self.get_img_meta_all()
        else:
            self.metaData = self.get_img_meta_all()

        # filename subindex (file is repeated m_n times)
        self.files = self.metaData['file'].values
        # image plane subindex
        self.img_plane = self.metaData['hdu_nr'].values
                                      
        if self.data_available and init:
            self.load()


################################################################################
    def get_img_meta_from_filename(self, file_path):
        """Loads and prepares img meta input dict for Img object
        
        Note
        ----
        Convenience method only rewritten in order to not break the code.
        Loads meta data of first image plane in fits file_path
        
        Parameters
        ----------
        file_path : str 
            file path of image
        
        Returns
        -------
        dict
            dictionary containing retrieved values for ``start_acq`` and 
            ``texp``
        """
        hdulist = fits.open(file_path)
        time = datetime.strptime(hdulist[0].header['ENDTIME'], '%Y.%m.%dZ%H:%M:%S.%f') 
        texp = int(hdulist[0].header['EXP'])
        return {"start_acq" : time, "texp": texp}
    
    
        
    def get_img_meta_all_filenames(self):   
        """ returns the same data as expected
        from ImgList.get_img_meta_all_filenames()
        
        Note
        ----
        Convenience method only rewritten in order to not break the code
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - list, list containing all retrieved acq. time stamps
            - list, containing all retrieved exposure times

        """
        meta = self.metaData
        times = meta.start_acq.values
        texps = meta.exposure.values         
        return times, texps
    
    
    def get_img_meta_all(self):
        import pandas as pd
        """ Load all available meta data from fits files
        
        Note:
            For some reason memmap=True works in fits.open() only when
            do_not_scale_image_data=False. With memmap=True the (image) data
            is not directly loaded in memory and thus saves processing time.
            Due to do_not_scale_image_data=False, the images cannot be used
            for evaluation. In this function only the headers should be readout,
            thus it can be used
            
        return:
            dataFrame containing all metadata"""
            
        imgFile = []
        imgFilehdu = []
        imgFileTime = []
        imgFileExp = []
        imgFileTemp = []
        
        # Exatract information of every image in every fits file in fitsFile
        for filePath in self.fitsfiles:
            #open the file, returning a list containg Header-Data Units (HDU)
            hdulist = fits.open(filePath,
                            memmap=True, #data is not read all at once)
                            do_not_scale_image_data=False) # Should be True, see Note
            imgPerFile = np.size(hdulist)
            hdulist.close()
            for hdu in range(imgPerFile):    
                # Info needed for loading the image            
                imgFile.append(filePath)
                imgFilehdu.append(hdu)              
                # Info from header
                imageHeader = hdulist[hdu].header
                ### TODO Load as dictionary
                imgFileTime.append(imageHeader['ENDTIME'])
                imgFileExp.append(int(imageHeader['EXP']))
                imgFileTemp.append(int(imageHeader['TCAM']))
           
        imgFileDatetime = [ datetime.strptime(time, '%Y.%m.%dZ%H:%M:%S.%f' ) for time in imgFileTime]
            
        meta = DataFrame(data={'img_id':arange(0,len(imgFile),1),
                               'file':imgFile,
                               'hdu_nr':imgFilehdu,
                               'exposure':imgFileExp,
                               'temperature':imgFileTemp,
                               'start_acq':imgFileDatetime},
                                index=imgFileDatetime)
        meta.index = pd.to_datetime(meta.index)
        return meta
        
###############################################################################
    
    def load_img(self, index):
        """ Load an image
        Don't apply any image processing!
         Check out if this could be put inro camera.import method
         """
        img_file = self.files[index]
        img_hdu = self.img_plane[index]            
        hdulist = fits.open(img_file)#,
                            #memmap=True,
                            #do_not_scale_image_data=False) #memmap keyword: data is not read all at once
        image = hdulist[img_hdu].data
        hdulist.close() #close the file, the header is stil available
        # load meta data
        imageHeader = hdulist[img_hdu].header
        imageMeta = {"start_acq"    : datetime.strptime(imageHeader['ENDTIME'],
                                                        '%Y.%m.%dZ%H:%M:%S.%f'),
                    "texp"         : int(imageHeader['EXP']),
                    "temperature"  : int(imageHeader['TCAM'])}

        # replace binary time stamp
        image[0,0:14] = image[1,0:14]            
        #Define pyplis image
        return Img(image, **imageMeta)
        

    """INDEX AND IMAGE LOAD MANAGEMENT"""
    def load(self):
        """Try load current and next image"""
        self.update_index_linked_lists() #based on current index in this list
        
        ### redefine load funtion
        #if not super(ImgList, self).load():
        #    print ("Image load aborted...")
        #    return False
        
        ### Parent funtions
        
        """Load current image
        
        Try to load the current file ``self.files[self.cfn]`` and if remove the 
        file from the list if the import fails
        Raises
        ------
        
        Returns
        -------
        bool
            if True, image was loaded, if False not        
        """
        if not self._auto_reload:
            print ("Automatic image reload deactivated in image list %s"\
                                                                %self.list_id)
            return False        
        try:
    
            img = self.load_img(self.index)
            self.loaded_images["this"] = img
            self._load_edit.update(img.edit_log)
            
            if img.vign_mask is not None:
                self.vign_mask = img.vign_mask
            
            self.update_prev_next_index()
            self._apply_edit("this")
                    
        except IOError:
            warn("Invalid file encountered at list index %s, file will"
                " be removed from list" %self.index)
            self.pop()
            if self.nof == 0:
                raise IndexError("No filepaths left in image list...")
            self.load()
            
        except IndexError:
            try:
                self.init_filelist()
                self.load()
            except:
                raise IndexError("Could not load image in list %s: file list "
                    " is empty" %(self.list_id))
        
        if self.nof > 1:                
            self.loaded_images["next"] = self.load_img(self.next_index)
            self._apply_edit("next")
        else:
            #self.loaded_images["prev"] = self.loaded_images["this"]
            self.loaded_images["next"] = self.loaded_images["this"]
            
        if self.optflow_mode:  
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                warn("Reached last index in image list, optflow_mode will be "
                        "deactivated")
                self.optflow_mode = 0
        return True

            
    def load_next(self):
        """Load next image in list"""
        if self.nof < 2 or not self._auto_reload:
            print ("Could not load next image, number of files in list: " +
                str(self.nof))
            return False
        self.index = self.next_index
        self.update_prev_next_index()
        
        self.update_index_linked_lists() #loads new images in all linked lists
        
        #self.loaded_images["prev"] = self.loaded_images["this"]
        self.loaded_images["this"] = self.loaded_images["next"]
        
        self.loaded_images["next"] = self.load_img(self.next_index)
        self._apply_edit("next")
        if self.optflow_mode:  
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                warn("Reached last index in image list, optflow_mode will be "
                    "deactivated")
                self.optflow_mode = 0
        return True

    def activate_tau_mode(self, val=1):
            """Activate tau mode
            
            In tau mode, images will be loaded as tau images (if background image
            data is available). 
            
            Parameters
            ----------
            val : bool
                new mode
                
            """
            if val is self.tau_mode: #do nothing
                return
            if val:
                if self.this.edit_log["is_tau"]:
                    warn("Cannot activate tau mode in image list %s: "
                         "current image is already a tau image"
                         %self.list_id)
                    return
                vc = self.vigncorr_mode
                self.vigncorr_mode = False
                cim = self.load_img(self.index) # this was changed
                bg_img = None
                self.bg_model.set_missing_ref_areas(cim)
                try:
                    dark = self.get_dark_image("this")
                    cim.subtract_dark_image(dark)
                except:
                    warn("Dark images not available")
                if self.bg_model.mode == 0:
                    
                    print("Customed changed by Solvejg."
                          "Surface map is calcualted on intensity thresholds"
                          "directly in plumebackground model.")
#                    print ("Background correction mode is 0, initiating settings "
#                        "for poly surface fit")
#                    try:
#                        mask = self.prepare_bg_fit_mask(dilation=True)
#                        self.bg_model.surface_fit_mask = mask
#                    except:
#                        warn("Background access mask could not be retrieved for "
#                            "PolySurfaceFit in background model of image list %s"
#                            %self.list_id)
                    
                else:
                    if not self.has_bg_img():
                        raise AttributeError("no background image available in "
                            "list %s, please set a suitable background image "
                            "using method set_bg_img, or change current bg " 
                            "modelling mode to 0 using self.bg_model.mode=0)" 
                            %self.list_id)
                    bg_img = self.bg_img
                self.bg_model.get_tau_image(cim, bg_img)
                self.vigncorr_mode = vc
            self._list_modes["tau"] = val
            self.load()

        
    def change_index(self, idx):
        """Change current image based on index of file list
        
        :param idx: index in `self.files` which is supposed to be loaded
        
        Dependend on the input index, the following scenarii are possible:
        If..
        
            1. idx < 0 or idx > `self.nof`
                then: do nothing (return)
            #. idx == `self.index`
                then: do nothing
            #. idx == `self.next_index`
                then: call :func:`next_img`
            #. idx == `self.prev_index`
                then: call :func:`prev_img`
            #. else
                then: call :func:`goto_img`
        """
        if not -1 < idx < self.nof or idx == self.index:
            return
        elif idx == self.next_index:
            self.next_img()
            return
        elif idx == self.prev_index:
            self.prev_img()
            return
        #: goto_img calls :func:`load` which calls prepare_additional_data
        self.goto_img(idx)

        return self.loaded_images["this"]
    
    def activate_vigncorr(self, val=True):
        """Activate / deactivate vignetting correction on image load
        
        Note
        ----
        
        Requires ``self.vign_mask`` to be set or an background image 
        to be available (from which ``self.vign_mask`` is then determined)
        
        Parameters
        ----------
        val : bool
            new mode
        """
        if val is self.vigncorr_mode: #do nothing
            return
        elif val:
            if self.this.edit_log["vigncorr"]:
                warn("Cannot activate vignetting correction in image list %s: "
                     "current image is already corrected for vignetting"
                     %self.list_id)
                return 
            if isinstance(self.vign_mask, Img):
                self.vign_mask = self.vign_mask.img
            if not isinstance(self.vign_mask, ndarray):
                self.det_vign_mask_from_bg_img()
            sh = (self.load_img(self.index)).img.shape
            #sh = Img(self.files[self.cfn],
            #        import_method=self.camera.image_import_method).img.shape
            if not self.vign_mask.shape == sh:
                raise ValueError("Shape of vignetting mask %s deviates from "
                            "raw img shape %s" %(list(self.vign_mask.shape),
                            list(sh)))
        self._list_modes["vigncorr"] = val
        self.load()
