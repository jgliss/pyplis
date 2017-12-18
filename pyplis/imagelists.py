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
Image list objects of pyplis library   
"""
from numpy import asarray, zeros, argmin, arange, ndarray, float32,\
    isnan, logical_or, uint8, finfo, exp, ones
from datetime import timedelta, datetime, date
#from bunch import Bunch
from pandas import Series, DataFrame, to_datetime, concat
from matplotlib.pyplot import figure, draw, subplots, ion, ioff, close
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
from warnings import warn
from os.path import exists, abspath, dirname, join, basename
from os import mkdir
from collections import OrderedDict as od

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
from .helpers import check_roi, map_roi, _print_list, closest_index,exponent,\
    isnum, get_pyr_factor_rel

# For custom defined ImgListMultiFits which needs to access the images for meta info
from .custom_image_import import load_comtessa, _read_binary_timestamp

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
        
        #these variables can be accessed using corresponding @property 
        #attributes
        self._integration_step_lengths = None
        self._plume_dists = None
        
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
        
        self._vign_mask = None #a vignetting correction mask can be stored here
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
    def plume_dists(self):
        """Distance to plume
        
        Can be an image were each pixel value corresponds to the plume distance
        at each pixel position (e.g. computed using the MeasGeometry) or can
        also be a single value, which may be appropriate under certain 
        measurement setups (e.g. distant plume perpendicular to CFOV of camera)
        
        Note
        ----
        This method checks if a value is accessible in :attr:`_plume_dists` and
        if not tries to compute plume distances by calling 
        :func:`compute_all_integration_step_lengths` of the 
        :class:`MeasGeometry` object assigned to this ImgList. If this fails, 
        then an AttributeError is raised
        
        Returns
        -------
        float or Img or ndarray
            Plume distances in m. If plume distances are accessible per image 
            pixel. Note that the corresponding data is converted to pyramid 
            level 0 (required for dilution correction).
        """
        v = self._plume_dists
        if isnum(v):
            return v
        elif isinstance(v, Img):
            return v.to_pyrlevel(0)
        self._get_and_set_geometry_info()
        return self._plume_dists

    @plume_dists.setter
    def plume_dists(self, value):
        if not (isnum(value) or isinstance(value, Img)):
            raise TypeError("Need Img or numerical data type (e.g. float, int)")
        if isinstance(value, Img):
            value = value#.to_pyrlevel(self.pyrlevel)
# =============================================================================
#             if not value.shape == self.this.shape:
#                 raise ValueError("Cannot set plume distance image: shape "
#                                  "mismatch between input and images in list")
# =============================================================================
        self._plume_dists = value
        
    @property
    def vign_mask(self):
        """Current vignetting correction mask"""
        if not any([isinstance(self._vign_mask, x) for x in (Img, ndarray)]):
            raise AttributeError("Vignetting mask is not available in list")
        return self._vign_mask
                
    @vign_mask.setter
    def vign_mask(self, value):
        if not any([isinstance(value, x) for x in (Img, ndarray)]):
            raise AttributeError("Invalid input for vignetting mask, need "
                                 "Img object or numpy ndarray")
        try:
            value=Img(value)
        except:
            pass
        pyrlevel_rel = get_pyr_factor_rel(self.this.img, value.img)
        if pyrlevel_rel != 0:
            if pyrlevel_rel < 0: 
                value.pyr_down(pyrlevel_rel)
            else:
                value.pyr_up(pyrlevel_rel)
        self._vign_mask = value
        
    @property
    def integration_step_length(self):
        """The integration step length for emission-rate analyses
        
        The intgration step length corresponds to the physical distance in 
        m between two pixels within the plume and is central for computing
        emission-rate. It may be an image were each pixel value corresponds to 
        the integreation step length at each pixel position (e.g. computed 
        using the MeasGeometry) or it can also be a single value, which may be 
        appropriate under certain measurement setups (e.g. distant plume 
        perpendicular to CFOV of camera).
        
        Note
        ----
        This method checks if a value is accessible in
        :attr:`_integration_step_lengths` and if not tries to compute them by 
        calling :func:`compute_all_integration_step_lengths` of the 
        :class:`MeasGeometry` object assigned to this ImgList. If this fails, 
        an AttributeError is raised
        
        Returns
        -------
        float or Img or ndarray
            Integration step lengths in m. If plume distances are accessible 
            per image pixel, then the corresponding data IS converted to the 
            current pyramid level
        """
        v = self._integration_step_lengths
        if isnum(v):
            return v
        elif isinstance(v, Img):
            return v.to_pyrlevel(self.pyrlevel)
        self._get_and_set_geometry_info()
        return self._integration_step_lengths

    @integration_step_length.setter
    def integration_step_length(self, value):
        if not (isnum(value) or isinstance(value, Img)):
            raise TypeError("Need Img or numerical data type (e.g. float, int)")
        if isinstance(value, Img):
            value = value.to_pyrlevel(self.pyrlevel)
            if not value.shape == self.this.shape:
                raise ValueError("Cannot set plume distance image: shape "
                                 "mismatch between input and images in list")
        self._integration_step_lengths = value
        
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
            plume_dist_img = self.meas_geometry.\
                compute_all_integration_step_lengths()[2]  
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
            print("Changing image list pyrlevel from %d to %d"\
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
                print("Building img-stack from list %s, progress: (%s | %s)" 
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
            self.goto_next()  
        stack.start_acq = asarray(stack.start_acq)
        stack.texps = asarray(stack.texps)
        stack.roi_abs = self._roi_abs
        
        print("Img stack calculation finished, rolling back to intial list"
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
                self.goto_next()
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
            
            self.goto_next()
        
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
            
            self.goto_next()
        
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
            #raise IndexError("Last image reached in ImgList")
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

    def load_img(self, index):
        """ Loads a single img; wrappes the custom_image_import to a single 
        method needing only the index
        
        Note
        ----
        Can be redefined in child classes without need of redifing methods like
        load(), load_next() etc
        
        Parameters
        ----------
        index : int
            index of image which should be loaded
        Returns
        -------
        pyplis.Img
            loaded image including meta data
        """
        img_file = self.files[self.index]
        image = Img(img_file, import_method=self.camera.image_import_method,
                     **self.get_img_meta_from_filename(img_file))
        return image
        
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
    
    def goto_next(self):
        """Go to next image 
        
        Calls :func:`load_next` 
        """
        self.load_next()
        return self.loaded_images["this"]

    def next_img(self):
        """Old name of method goto_next"""
        warn("This method was renamed (but still works). Please use method "
             "goto_next in the future")
        return self.goto_next()

    def goto_prev(self):
        """Go to previous image
        
        Calls :func:`load_prev`
        """
        self.load_prev()
        return self.loaded_images["this"]

    def prev_img(self):
        """Old name of method goto_next"""
        warn("This method was renamed (but still works). Please use method "
             "goto_prev in the future")
        return self.goto_prev()
    
    def append(self, file_path):
        """Append image file to list
        
        :param str file_path: valid file path
        """
        if not exists(file_path):
            raise IOError("Image file path does not exist %s" %file_path)
        
        self.files.append(file_path)
        
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
    
    def _this_raw_fromfile(self):
        """Reloads and returns current image
        
        This method is used for test purposes and does not change the list
        state. See for instance :func:`activate_dilution_corr` in 
        :class:`ImgList`
        
        Returns
        -------
        Img
            the current image loaded and unmodified from file    
        """
        return Img(self.files[self.cfn],
                   import_method=self.camera.image_import_method)
        
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
            img_arr = self.goto_next().img
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
    Private methods
    """
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
            
    def _get_and_set_geometry_info(self):
        """Compute and write plume and pix-to-pix distances from MeasGeometry"""
        try:
            (int_steps, 
            _, 
            dists)=\
            self.meas_geometry.compute_all_integration_step_lengths(
                     pyrlevel=self.pyrlevel) 
            self._plume_dists = dists.to_pyrlevel(0)
            self._integration_step_lengths = int_steps
            print ("Computed and updated list attributes plume_dist and "
                   "integration_step_length in ImgList from MeasGeometry")
        except:
            raise ValueError("Measurement geometry not ready for access "
                "of plume distances and integration steps in image list %s." 
                %self.list_id)
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
 
class AutoDilcorrSettings(object):
    """This class stores settings for automatic dilution correction in ImgLists

    Attributes
    ----------
    tau_thresh : float
        OD threshold for computation of plume pixel mask
    erode_mask_size : int
        size of erosion kernel applied to plume pixel mask
    dilate_mask_size : int
        size of dilation kernel applied to plume pixel mask after 
        erosion was applied
    bg_model : PlumeBackgroundModel
        plume background model used to compute tau images (i.e. 
        correction mode 99, is e.g. used in :func:`_apply_edit` of 
        :class:`ImgList`)
    
    Parameters
    ----------
    tau_thresh : float
        OD threshold for computation of plume pixel mask
    erode_mask_size : int
        size of erosion kernel applied to plume pixel mask
    dilate_mask_size : int
        size of dilation kernel applied to plume pixel mask after 
        erosion was applied
    """
    def __init__(self, tau_thresh=0.05, erode_mask_size=3, 
                 dilate_mask_size=4):
        self.tau_thresh = tau_thresh
        self.erode_mask_size = erode_mask_size
        self.dilate_mask_size = dilate_mask_size
        self.bg_model = PlumeBackgroundModel(mode=99)
        
    def __str__(self):
        """String representation"""
        for k, v in self.__dict__.iteritems():
            print "%s: %s" %(k, v)
                 
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
    **dilcorr_settings
        additional keyword args corresponding to settings for automatic 
        dilution correction passed to __init__ of :class:´AutoDilcorrSettings`
    
    """
    def __init__(self, files=[], list_id=None, list_type=None, camera=None,
                 geometry=None, init=True, **dilcorr_settings):

        super(ImgList, self).__init__(files, list_id, list_type, camera, 
                                      geometry, init=False)
                                      
        self.loaded_images.update({"next": None})
    
        #: List modes (currently only tau) are flags for different list states
        #: and need to be activated / deactivated using the corresponding
        #: method (e.g. :func:`activate_tau_mode`) to be changed, dont change
        #: them directly via this private dictionary
        self._list_modes.update({"darkcorr"  :  False, #dark correction
                                 "optflow"   :  False, #compute optical flow
                                 "vigncorr"  :  False, #load vignetting corrected images
                                 "dilcorr"   :  False, #load as dilution corrected images
                                 "tau"       :  False, #load as OD images
                                 "aa"        :  False, #load as AA images
                                 "senscorr"  :  False, #correct for cross-detector sensitivity variations
                                 "gascalib"  :  False})#load as calibrated SO2 images
                                 
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
        
        # tau threshold for calculation of plume pixel mask fro dilution 
        # correction
        self.dilcorr_settings = AutoDilcorrSettings(**dilcorr_settings)
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
    
# =============================================================================
#     @property
#     def next(self):
#         """Next image"""
#         return self.loaded_images["next"]
# =============================================================================
        
    
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
        """Activate / deactivate vignetting correction on image load"""
        return int(self._list_modes["vigncorr"])
    
    @vigncorr_mode.setter
    def vigncorr_mode(self, val):
        self.activate_vigncorr(val)
    
    @property
    def dilcorr_mode(self):
        """Activate / deactivate dilution correction on image load"""
        return int(self._list_modes["dilcorr"])
    
    @dilcorr_mode.setter
    def dilcorr_mode(self, val):
        self.activate_dilcorr_mode(val)
        
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
    def calib_mode(self, value):
        """Change current list calibration mode"""
        self.activate_calib_mode(value)

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
            val = Series(val, [self.acq_times[0]])
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
    def dark_img(self):
        """Current dark image"""
        return self.get_dark_image()
        
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
    def activate_darkcorr(self, value=True):
        """Activate or deactivate dark and offset correction of images
        
        If dark correction turned on, dark image access is attempted, if that
        fails, Exception is raised including information what did not work 
        out.
        
        Parameters
        ----------
        val : bool
            new mode
        """
        if value is self.darkcorr_mode: #do nothing
            return
        if not value and self._load_edit["darkcorr"]:
            raise ImgMetaError("Cannot deactivate dark correction, original"
                "image file was already dark corrected")
        if value:
            if self.this.edit_log["darkcorr"]:
                warn("Cannot activate dark correction in image list %s: "
                     "current image is already corrected for dark current"
                     %self.list_id)
                return
            self.get_dark_image()
            self.update_index_dark_offset_lists()
                    
        self._list_modes["darkcorr"] = value
        self.load()
        
    def activate_vigncorr(self, value=True):
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
        if value is self.vigncorr_mode: #do nothing
            return
        elif value:
            if self.this.edit_log["vigncorr"]:
                warn("Cannot activate vignetting correction in image list %s: "
                     "current image is already corrected for vignetting"
                     %self.list_id)
                return 
            try:
                self.vign_mask
            except:
                self.det_vign_mask_from_bg_img() 
            sh = (self.load_img(self.index)).img.shape
            if not self.vign_mask.shape == sh:
                raise ValueError("Shape of vignetting mask %s deviates from "
                            "raw img shape %s" %(list(self.vign_mask.shape),
                            list(sh)))
        self._list_modes["vigncorr"] = value
        self.load()
    
    def activate_tau_mode(self, value=True):
        """Activate tau mode
        
        In tau mode, images will be loaded as tau images (if background image
        data is available). 
        
        Parameters
        ----------
        val : bool
            new mode
            
        """
        if value is self.tau_mode: #do nothing
            return
        if value:
            if self.this.edit_log["is_tau"]:
                warn("Cannot activate tau mode in image list %s: "
                     "current image is already a tau image"
                     %self.list_id)
                return
            vc = self.vigncorr_mode
            self.vigncorr_mode = False
            cim = self.load_img(self.cfn)
            try:
                dark = self.get_dark_image("this")
                cim.subtract_dark_image(dark)
            except:
                warn("Dark images not available")
            bg_img = None
            self.bg_model.set_missing_ref_areas(cim)
            if self.bg_model.mode == 0:
                print ("Background correction mode is 0, initiating "
                       "settings for poly surface fit")
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
    
    def activate_aa_mode(self, value=True):
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
         
        Parameters
        ----------
        val : bool
            Activate / deactivate AA mode
            
        """
        if value is self.aa_mode:
            return
        if not self.list_type == "on":
            raise TypeError("AA mode could not be actu")
        aa_test = None
        if value:
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
            self._list_modes["tau"] = False
            offlist._list_modes["tau"] = False
            aa_test = self._aa_test_img(offlist)
        self._list_modes["aa"] = value
        
        self.load()

        return aa_test
    
    def activate_calib_mode(self, value=True):
        """Activate calibration mode"""
        if value == self._list_modes["gascalib"]:
            return
        if value:    
            if not self.aa_mode:
                self._list_modes["aa"] = True
                warn("List is not in AA mode")
                
            if not self.sensitivity_corr_mode:
                warn("AA sensitivity correction mode is deactivated. This "
                    "may yield erroneous results at the image edges")
            self.calib_data(self.current_img())
            
        self._list_modes["gascalib"] = value
        self.load()
    
    def activate_dilcorr_mode(self, value=True):
        """Activate dilution correction mode
        
        Please see :func:`correct_dilution` for details.
        
        Parameters
        ----------
        value : bool
            New mode: True or False 
        """
        if value == self._list_modes["dilcorr"]:
            return
        if value:
            img = self._this_raw_fromfile()
            _,_,mask = self.correct_dilution(img)
            # now make sure that in case and off-band list is assigned, it can
            # also be used to perform a dilution correction (i.e. bg_model 
            # ready)
            try:
                off_list = self.get_off_list()
                off_img = off_list._this_raw_fromfile().to_pyrlevel(off_list.pyrlevel)
                mask = mask.to_pyrlevel(off_list.pyrlevel)
                try:
                    off_list.correct_dilution(off_img, plume_pix_mask=mask)
                except:
                    off_list.bg_model.update(**self.bg_model.settings_dict())
            except AttributeError as e:
                print repr(e)
            
        self._list_modes["dilcorr"] = value
        self.load()
        
    def activate_optflow_mode(self, value=True, draw=False):
        """Activate / deactivate optical flow calculation on image load
        
        Parameters
        ----------
        val : bool
            activate / deactivate
        draw : bool
            if True, flow field is plotted into current image
            

        """
        if value is self.optflow_mode:
            return 
        if value:
            try:
                self.set_flow_images()
            except IndexError:
                raise IndexError("Optical flow mode cannot be activated in "
                    "image list %s: list is at last index, please change list "
                    "index and retry")
            self.optflow.calc_flow()
            if draw:
                self.optflow.draw_flow()
        self._list_modes["optflow"] = value
    
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
        
        Parameters
        ----------
        list_id : :obj:`str`, optional
            ID of the list. If unspecified (None), the default off band filter 
            key is attempted to be accessed 
            (``self.camera.filter_setup.default_key_off``) and if this fails,
            the first off band list found is returned.
          
        Raises
        ------
        AttributeError
            if not offband list can be assigned
        
        Returns
        -------
        ImgList 
            the corresponding off-band list
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
        available in this list, determine a vignetting mask from the 
        background image. Furthermore, if the input image is not blurred it 
        is blurred using current list blurring factor and in case the 
        latter is 0, then it is blurred with a Gaussian filter of width 1.
        
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
        try:
            vign_mask = self.vign_mask
        except:
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
        ------
        IndexError
            object, i.e. `self.loaded_images["this"]` and 
            `self.loaded_images["next"]`
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
      
    def link_dark_offset_lists(self, *lists):
        """Assign dark and offset image lists to this object
        
        Assign dark and offset image lists: get "closest-in-time" indices of dark 
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
        for lst in lists:
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
            self.goto_next()
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

            self.goto_next()
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
                              self.loaded_images["next"].duplicate().to_binary(thresh).img)
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
        self.vign_mask = Img(mask)
        return self.vign_mask
    
    def calc_sky_background_mask(self, lower_thresh=None,
                                apply_movement_search=True,
                                **settings_movement_search):
        """Retrieve and set background mask for 2D poly surface fit
        
        Wrapper for method :func:`find_sky_background` 
        
        Calculates mask specifying sky radiance pixels for background 
        modelling mode 0 
        
        Parameters
        ----------
        lower_thresh : :obj:`float`, optional
            lower intensity threshold. If provided, this value is used, 
            else, the minimum value is derived from the minimum intensity 
            in the plume image within the current 3 sky reference 
            rectangles 
        **settings_movement_search
            additional keyword arguments passed to :func:`find_movement`. 
            Note that these may include settings for the optical flow 
            calculation which are further passed to the 
            initiation of the :class:`FarnebackSettings` class 
        
        Returns
        -------
        array
            2D-numpy boolean numpy array specifying sky background pixels
        """
        mask = self.bg_model.\
            calc_sky_background_mask(self.this, 
                                     self.loaded_images["next"],
                                     lower_thresh,
                                     apply_movement_search,
                                     **settings_movement_search)
        self.surface_fit_mask = mask
        return mask
    
    def prepare_bg_fit_mask(self, **kwargs):
        """Calculate mask specifying sky-reference pixels in current image
        
        Note
        ----
        
        1. The method was redefined and renamed, please see (and use) 
            :func:`calc_sky_background_mask` instead
        2. This is a beta version
        
        """
        warn("Old name (wrapper) for method calc_sky_background_mask")
        
        return self.calc_sky_background_mask(**kwargs)
    
    def prep_data_dilutioncorr_old(self, tau_thresh=0.05, plume_pix_mask=None, 
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
            
            if plume_pix_mask.shape == self.this.shape:
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
                 dists)=\
                 self.meas_geometry.compute_all_integration_step_lengths(
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
            #print "Retrieving plume pixel mask in list %s" %self.list_id
            plume_pix_mask = self.get_thresh_mask(tau_thresh)    
        self.tau_mode = False
        bg = self.current_img() * exp(tau0.img)
        return (self.current_img(), ext_coeff, bg, dists, plume_pix_mask)
    
    def correct_dilution(self, img, tau_thresh=0.10, ext_coeff=None,  
                         plume_pix_mask=None, plume_dists=None, 
                         vigncorr_mask=None, erode_mask_size=0, 
                         dilate_mask_size=0):
        """Correct a plume image for the signal dilution effect
        
        The provided plume image needs to be in intensity space, meaning the
        pixel values need to be intensities and not optical densities or 
        calibrated gas-CDs. The correction is based on Campion et al., 2015 
        and requires knowledge of the atmospheric scattering extinction 
        coefficients (``ext_coeff``) in the viewing direction of the camera.
        These can be provided using the corresponding input parameter 
        ``ext_coeff`` or can be assigned to the list beforehand (up to you). 
        See example script no. 11 to check out how you can retrieve the
        extinction coefficients using dark terrain features in the plume image.
        The correction furthermore requires knowledge of the plume distance
        (in the best case on the pixel-level) and it must be possible to 
        compute optical density images, hence the :attr:`bg_model`
        (instance of :class:`PlumeBackgroundModel`) needs to be ready for 
        tau image computation. In addition, a vignetting correction mask must
        be available.
        
        Parameters
        ----------
        img : Img
            the plume image object
        tau_thresh : float
            OD (tau) threshold to compute plume pixel mask (irrelevant if
            next :param:`plume_pix_mask` is provided)
        ext_coeff : :obj:`float`, optional
            atmospheric extinction coefficient. If unspecified, try access 
            via :attr:`ext_coeff` which returns the current extinction 
            coefficient and raises :obj:`AttributeError` in case, no coeffs are
            assigned to this list
        vigncorr_mask : :obj:`ndarray` or :obj:`Img`, optional
            mask used for vignetting correction
        plume_pix_mask : :obj:`Img`, optional
            binary mask specifying plume pixels in the image, is retrieved
            automatically if input is None
        erode_mask_size : int
            if not zero, the morphological operation erosion is applied 
            to the plume pixel mask (e.g. to remove noise outliers) using
            an appropriate quadratic kernel corresponding to the input size
        dilate_mask_size : int
            if not zero, the morphological operation dilation is applied 
            to the plume pixel mask (e.g. to slightly extend the borders of 
            the detected plume) using an appropriate quadratic kernel 
            corresponding to the input size
 
        Returns
        -------
        tuple
            3-element tuple containing
            
            - :obj:`Img`, dilution corrected image (vignetting corrected)
            - :obj:`Img`, corresponding vignetting corrected plume background
            - :obj:`array`, mask specifying plume pixels
        """
        if img.is_tau or img.is_aa or img.is_calibrated:
            raise ValueError("Img must not be an OD, AA or calibrated CD img")
        try:
            self.vign_mask = vigncorr_mask
        except:
            pass
        vign_mask = self.vign_mask #raises Exception if not available
        try:
            try:
                plume_pix_mask = plume_pix_mask.img
            except:
                pass
            if plume_pix_mask.shape == self.this.shape:
                mask_ok = True
            else:
                mask_ok = False
        except:
            mask_ok = False
        if plume_dists is None:
            plume_dists = self.plume_dists
        # get current extinction coefficient, raises AttributeError if not 
        # available
        try:
            ext_coeff = float(ext_coeff)
        except:
            ext_coeff = self.ext_coeff 
        if img.is_vignetting_corrected:
            idx=1
        else:
            idx=0
        tau0 = self.bg_model.get_tau_image(img, self._bg_imgs[idx])
        if not idx:
            img.correct_vignetting(vign_mask, new_state=True)
        #bg = self.bg_model.current_plume_background
        #bg.edit_log["vigncorr"] = True
        if not mask_ok:
            #print "Retrieving plume pixel mask in list %s" %self.list_id
            plume_pix_mask = tau0.to_binary(threshold=tau_thresh,
                                            new_img=True)
            if erode_mask_size > 0:
                plume_pix_mask.erode(ones((erode_mask_size,
                                           erode_mask_size),dtype=uint8))
            if dilate_mask_size > 0:
                plume_pix_mask.dilate(ones((dilate_mask_size,
                                            dilate_mask_size),dtype=uint8))
        bg = img * exp(tau0.img)
        from .dilutioncorr import correct_img
        corr = correct_img(img, ext_coeff, bg, plume_dists, plume_pix_mask)
                                  
        bad_pix = corr.img <= 0
        corr.img[bad_pix] = img.img[bad_pix]
            
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
        if self.calib_mode or self.aa_mode or self.tau_mode:
            raise AttributeError("List must not be in tau, AA or calib mode")
        self.darkcorr_mode=True
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
    
        self.goto_img(0)
        saved_off = []
        num = self.nof
        if add_off_list:
            off = self.get_off_list()
            off.bg_model.update(**self.bg_model.settings_dict())
        for k in range(num):
            (corr, 
             bg, 
             plume_pix_mask) = self.correct_dilution(self.this,
                                                     tau_thresh=tau_thresh,
                                                     ext_coeff=ext_on,   
                                                     **kwargs)
            corr.save_as_fits(save_dir)
            fname = corr.meta["file_name"]
            if save_masks:
                Img(plume_pix_mask.img, dtype=uint8, 
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
                     _) = off.correct_dilution(off.this, ext_coeff=ext_off,
                                               plume_pix_mask=plume_pix_mask,
                                               **kwargs)
                    saved_off.append(corr_off.save_as_fits(save_dir))
                    if save_bg_imgs:
                        bg_off.save_as_fits(bg_dir, corr_off.meta["file_name"])    
            self.goto_next()
        ion()
            
    """I/O"""
    def import_ext_coeffs_csv(self, file_path, header_id=None, **kwargs):
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
        try:
            df = DataFrame.from_csv(file_path, **kwargs)
            s=df[header_id]
        except:
            s = Series.from_csv(file_path, **kwargs)
        self.ext_coeffs = s#
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
            update_bgmodel = True
        else:
            update_bgmodel = False
        img = self.loaded_images[key]
        bg = None
        if self.darkcorr_mode:
            dark = self.get_dark_image(key)
            img.subtract_dark_image(dark)
        bg_model = self.bg_model
        if self.dilcorr_mode:
            s = self.dilcorr_settings
            # update bg_model in case dilution correction is active, the model
            # stored in the settings class is set at mode 99, i.e. no modelling
            # is performed
            bg_model = s.bg_model
            (img, bg, mask)=self.correct_dilution(img,
                                               s.tau_thresh,
                                               erode_mask_size=s.erode_mask_size, 
                                               dilate_mask_size=s.dilate_mask_size)
        elif self.vigncorr_mode: #elif because if dilcorr is active the image is already vign corrected
            img.correct_vignetting(self.vign_mask, new_state=True)
        if self.tau_mode:
            if bg is None: #dilution_corr is not active
                bg = self.bg_img.to_pyrlevel(img.pyrlevel)
            img = bg_model.get_tau_image(plume_img=img, 
                                         bg_img=bg,
                                         update_imgs=upd_bgmodel)
        elif self.aa_mode:
            off_list = self.get_off_list()
            if off_list.dilcorr_mode:
                raise AttributeError("Linked off-band list has dilution "
                                     "correction mode activated. Please "
                                     "deactivate.")
            #off_list.dilcorr_mode = self.dilcorr_mode
            if bg is None:
                bg = self.bg_img.to_pyrlevel(img.pyrlevel)
            img_off = off_list.this
            # make sure, the dilution correction mode is activated in the off
            # list if it is activated here
            if self.dilcorr_mode:
                mask = mask.to_pyrlevel(off_list.pyrlevel)
                (img_off, 
                 bg_off, 
                 _)=off_list.correct_dilution(img_off,
                                              plume_pix_mask=mask)
            else:
                bg_off = off_list.bg_img
            img_off.to_pyrlevel(img.pyrlevel)
            bg_off.to_pyrlevel(img.pyrlevel)
            
            img = bg_model.get_aa_image(plume_on=img, 
                                        plume_off=img_off,
                                        bg_on=bg,
                                        bg_off=bg_off,
                                        update_imgs=upd_bgmodel)
            if self.sensitivity_corr_mode:
                img = img / self.aa_corr_mask
                img.edit_log["senscorr"] = 1

        if self.calib_mode:
            img.img = self.calib_data(img.img)
            img.edit_log["gascalib"] = True
          
        img.to_pyrlevel(self.img_prep["pyrlevel"])
        if self.img_prep["crop"]:
            img.crop(self.roi_abs)
        if self.img_prep["8bit"]:
            img._to_8bit_int(new_im=False)
        # do this at last, since it can be time consuming and is therefore much
        # faster in case pyrlevel > 0 or crop applied
        img.add_gaussian_blurring(self.img_prep["blurring"])
        img.apply_median_filter(self.img_prep["median"])
        self.loaded_images[key] = img
    
    def _aa_test_img(self, off_list):
        """Try to compute an AA test-image"""
        on = Img(self.files[self.cfn],
                 import_method=self.camera.image_import_method)
        off = Img(off_list.files[off_list.cfn],
                  import_method=self.camera.image_import_method)
        if self.which_bg == "img":
            # the stored images may be vignetting corrected, then also a
            # vignetting corrected BG image is required. The attribute
            # _bg_imgs is a list that contains two images: one that is not 
            # corrected for vignetting (index 0), and one that is corrected
            # for vignetting (index 1). Thus, the right bg image can simply
            # be accessed passing the img state variable "is_vigncorr"
            bg_on = self._bg_imgs[on.is_vigncorr].to_pyrlevel(on.pyrlevel)
            bg_off = off_list._bg_imgs[off.is_vigncorr].to_pyrlevel(off.pyrlevel)
        else:
            bg_on = self.bg_list.this
            bg_off = off_list.bg_list.this
        return self.bg_model.get_aa_image(on, off, bg_on, bg_off)
    
    """
    SORTED OUT METHODS
    """
    def correct_dilution_old(self, img, tau_thresh=0.05, plume_pix_mask=None,
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
            coefficient and raises :obj:`AttributeError` in case, no coeffs are
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
    
    def correct_dilution_all_old(self, tau_thresh=0.05, ext_on=None, 
                                 ext_off=None,
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
            
            self.goto_next()
        ion()
        
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
        
        # that should be done in a different way!
        self.camera.image_import_method = load_comtessa
        
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
        # image subindex inside fits file
        self.hdu_nr = self.metaData['hdu_nr'].values
                                      
        if self.data_available and init:
            self.load()


################################################################################
    """ META DATA HANDLING """
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
        
        warn('This method does not make a lot of sense for the ImgListMultiFits!'
             ' Returns the meta data of the first image in file_path.'
             ' metaData attribute to access meta information.')
        
        hdulist = fits.open(file_path)
        # Load the image
        image = hdulist[0].data
        time = _read_binary_timestamp(image) 
        texp = float(hdulist[0].header['EXP']) / 1000. # in s
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
    
    def get_img_meta_one_fitsfile(self, file_path):
        """ Load all meta data from all images of one fits file """
        
        # temporary lists of parameters
        imgFileStart = []
        imgFileStop = []
        imgFileMin = []
        imgFileMax = []
        imgFileMean = []
        imgFileExp = []
        imgFileTemp = []
        
        #open the file, returning a list containg Header-Data Units (HDU)
        hdulist = fits.open(file_path)
        imgPerFile = np.size(hdulist)
#            hdulist.close()
        for hdu in range(imgPerFile):
            # Info from image
            image = hdulist[hdu].data    
            imgFileStop.append(_read_binary_timestamp(image))
            image[0,0:14] = image[1,0:14] #replace binary timestamp
            imgFileMin.append(image.min())
            imgFileMax.append(image.max())
            imgFileMean.append(image.mean())
            # Info from header
            imageHeader = hdulist[hdu].header
            imgFileStart.append(imgFileStop[-1] - timedelta(microseconds=int(imageHeader['EXP'])*1000))
            imgFileExp.append(float(imageHeader['EXP']) / 1000.) # in s
            imgFileTemp.append(float(imageHeader['TCAM']))
        
        # Combine the temporary lists to a dataFrame and return it
        meta = DataFrame(data={'file'       : [file_path]*imgPerFile,
                               'hdu_nr'     : range(imgPerFile),
                               'start_acq'  : imgFileStart,
                               'stop_acq'   : imgFileStop,
                               'exposure'   : imgFileExp,
                               'temperature': imgFileTemp,
                               'min'        : imgFileMin,
                               'max'        : imgFileMax,
                               'mean'       : imgFileMean},
                            index=imgFileStart)
        meta.index = to_datetime(meta.index)
        return meta
        
    
    def get_img_meta_all(self):
        """ Load all available meta data from fits files            
        Returns
        -------
        dataFrame
            containing all metadata
        """
        
        if self.fitsfiles == []:
            print("ImgListMultiFits was intialised without providing the "
                  "fitsfile (e.g. only by meta file). self.get_img_meta_all "
                  "will return the existing metaData.")
            return self.metaData
        
        # Exatract information of every image in every fits file in fitsFile
        meta_single = [self.get_img_meta_one_fitsfile(file_path) for file_path in self.fitsfiles]
        meta = concat(meta_single)
        meta['img_id'] = arange(0,len(meta),1)
        meta.index = to_datetime(meta.index)
        return meta
        
###############################################################################
    """INDEX AND IMAGE LOAD MANAGEMENT"""
    def load_img(self, index):
        """ Loads a single image
        
        Parameters
        ----------
        index : int
            index of image which should be loaded
        Returns
        -------
        pyplis.Img
            loaded image including meta data
         """
        img_file = self.files[index]
        img_hdu = self.hdu_nr[index]
        try:
            image =  Img(input=img_file,
                         import_method=self.camera.image_import_method,                            
                         **{'img_idx':img_hdu})
        except:
            print('Couldnt load image with self.camera.image_import_method.'
                  'Used load_comtessa instead.')
            image =  Img(input=img_file,
                         import_method=load_comtessa,                            
                         **{'img_idx':img_hdu})
        return image

    def activate_tau_mode(self, value=1):
            """Activate tau mode
            
            In tau mode, images will be loaded as tau images (if background image
            data is available). 
            
            Parameters
            ----------
            val : bool
                new mode
                
            """
            if value is self.tau_mode: #do nothing if already fullfilled
                return
            if value:
                if self.this.edit_log["is_tau"]:
                    warn("Cannot activate tau mode in image list %s: "
                         "current image is already a tau image"
                         %self.list_id)
                    return
                
                ### Why can't I set a load() for the new images?
                
                vc_original = self.vigncorr_mode
                # Reload image without vignetting correction
                self.vigncorr_mode = False
                cim = self.load_img(self.index) # this was changed
                # what is this needed for?
                self.bg_model.set_missing_ref_areas(cim) 
                # Dark correction should go before logically?
                try:
                    dark = self.get_dark_image("this")
                    cim.subtract_dark_image(dark)
                except:
                    warn("Dark images not available")
                    
                if self.bg_model.mode == 0:
                    # Create a empty backgound image
                    bg_img = None                    
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
                self.vigncorr_mode = vc_original
            self._list_modes["tau"] = value
            self.load()