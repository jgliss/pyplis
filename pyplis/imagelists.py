# -*- coding: utf-8 -*-
"""
Image list objects of pyplis library

    1. :class:`BaseImgList` the base class for image list objects (includes 
        functionality for file management and iteration and basic editing)
    #. :class:`DarkImgList`: Inherits from :class:`BaseImgList` and was only 
        extended by attribute "read_gain". Introducing a separate list object
        for dark and offset images was also done to have a clearer structure 
        (e.g. when using identifiers such as :func:`isinstance` or 
        :func:`type`)
    #. :class:`ImgList`: the central list object in pyplis, e.g. for
        representing on and off band lists of plume image data. Inherits from
        :class:`BaseImgList` and was extended by functionality for dark image
        correction, optical flow calculations and background image modelling 
        (tau image determination). Furthermore, it is possible to link other 
        image lists which are then automatically updated based on acquisition
        time (e.g. link a off and on band lists can make life easier).
    #. :class:`CellImgList`: This list can be used to store images of one 
        calibration cell (covering the whole camera FOV). It is extended by 
        the attributes cell_id, gas_cd and gas_cd_err. It inherits
        from :class:`ImgList` (to enable dark and offset correction) and is 
        mainly used in the :class:`pyplis.Calibration.CellCalib` object.

"""
from numpy import asarray, zeros, argmin, arange, ndarray, float32, ceil,\
    isnan, ones, nan
from ntpath import basename
from datetime import timedelta, datetime
#from bunch import Bunch
from matplotlib.pyplot import figure, draw
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter
from warnings import warn
from os.path import exists
from collections import OrderedDict as od

from traceback import format_exc

from .image import Img
from .inout import load_img_dummy
from .exceptions import ImgMetaError
from .setupclasses import Camera
from .geometry import MeasGeometry
from .processing import ImgStack, PixelMeanTimeSeries, LineOnImage,\
                                                            model_dark_image
from .plumebackground import PlumeBackgroundModel
from .plumespeed import OpticalFlowFarneback
from .helpers import check_roi, map_roi, _print_list

class BaseImgList(object):
    """Basic image list object
    Basic image list object providing indexing and image loading functionality
    
    In this, only the current image is loaded at a time while :class:`ImgList` 
    loads previous, current and next image whenever the index is changed
    
    This object and all objects inheriting from this are fundamentally based 
    on a list of image file paths, which are dynamically loaded during usage.
    
    """
    def __init__(self, files=[], list_id=None, list_type=None,
                 camera=None, init=True, **img_prep_settings):
        """Init image list
        
        :param list files: list with file names
        :param str list_id: id of list (e.g. "my bad pics")
        :param str list_type: type of images in list (e.g. "on")
        :param Camera camera: camera setup
        :param bool init: if True, list will be initiated and files loaded 
            (given that image files are provided on input)
        """
        #this list will be filled with filepaths
        self.files = []
        #id of this list
        self.list_id = list_id
        self.list_type = list_type
        
        self.filter = None #can be used to store filter information
        
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
            
        if self.data_available and init:
            self.load()
         
    def update_img_prep(self, **settings):
        """Update image preparation settings and reload"""
        for key, val in settings.iteritems():
            if self.img_prep.has_key(key) and\
                        isinstance(val, type(self.img_prep[key])):
                self.img_prep[key] = val
        self.load()
        
    @property
    def auto_reload(self):
        """Activate / deactivate automatic reload of images"""
        return self._auto_reload
        
    @auto_reload.setter
    def auto_reload(self, val):
        """Set mode"""
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
        """Activate / deactivate crop mode"""
        return self.img_prep["pyrlevel"]
    
    @pyrlevel.setter
    def pyrlevel(self, value):
        """Update pyrlevel and reload images"""
        self.img_prep["pyrlevel"] = int(value)
        self.load()
    
    @property
    def gaussian_blurring(self):
        """Get / set current blurring level"""
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
        """Return current ROI (in relative coordinates)
        
        The ROI is returned with respect to the current pyrlevel
        """
        return map_roi(self._roi_abs, self.pyrlevel)
    
    @roi.setter
    def roi(self):
        """Raises Attribute Error with some information"""
        raise AttributeError("Please use roi_abs to set the current ROI in "
            "absolute image coordinates. :func:`roi` is used to access the "
            "current ROI for the actual pyramide level.")
    @property 
    def roi_abs(self):
        """Returns current roi (in consideration of current pyrlevel)"""
        #return map_roi(self._roi_abs, self.img_prep["pyrlevel"])
        return self._roi_abs
        
    @roi_abs.setter
    def roi_abs(self, val):
        """Updates current ROI"""
        if check_roi(val):
            self._roi_abs = val
            self.load()
            
    @property
    def cfn(self):
        """Returns current file number"""
        return self.index
        
    @property
    def nof(self):
        """Returns the number of files in this list"""
        return len(self.files)
        
    @property
    def last_index(self):
        """Returns index of last image"""
        return len(self.files) - 1
    
    def has_files(self):
        """Returns boolean whether or not images are available in list"""
        return bool(self.nof)
        
    @property
    def data_available(self):
        """Wrapper for :func:`has_files`"""
        return self.has_files()
    
    @property
    def img_mode(self):
        """Checks and returs current img mode (tau, aa, raw)
        
        This function is overwritten in :class:`ImgList` where more states
        are allowed. It is, for instance used in :func:`make_stack`.
        
        :return:
            - "raw" (:class:`BaseImgList` does not support tau or aa image 
                determination)
        """
        return "raw"
        
    def clear(self):
        """Empty file list ``self.files``"""
        self.files = []
    
    def separate_by_substr_filename(self, sub_str, sub_str_pos, delim="_"):
        """Separate this list by filename specifications
        
        :param str sub_str: string identification used to identify the image 
            type which is supposed to be kept within this list object
        :param int sub_str_pos: position of sub string after filename was split 
            (using input param delim)
        :param str delim ("_"): filename delimiter
        :returns: tuple 
            - :class:`ImgList`, list contains images matching the requirement
            - :class:`ImgList`, list containing all other images
            
        The function checks all current filenames, and keeps those, which have
        idStr at position idPos after splitting the file name using the input
        delimiter. All other files are added to a new image list which is 
        returned
        """
        match = []
        rest = []
        for p in self.files:
            spl = basename(p).split(".")[0].split(delim)
            if spl[sub_str_pos] == sub_str:
                match.append(p)
            else:
                rest.append(p)
        
        lst_match = ImgList(match, list_id = "match", camera = self.camera)
        lst_rest = ImgList(rest, list_id = "rest", camera = self.camera)
        return (lst_match, lst_rest)
        
    def add_files(self, file_list):
        """Import a list with filepaths
        
        :param list file_list: list with file paths
        
        If input is okay, then this list is initiated using :func:`init_filelist`
        """
        if isinstance(file_list, str):
            print ("Warning in add_files: import list is a string and"
                    " will be converted into a list")
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
        
        :param int num (0): index of image which will be loaded (`self.index`)
        """
        print 
        print "Init ImgList %s" %self.list_id
        print "-------------------------"
        self.index = num
        
        #self.selectedFilesIndex = zeros(self.nof)            
        for key, val in self.loaded_images.iteritems():
            self.loaded_images[key] = None
            
        #self.load_images()  
        print "Number of files: " + str(self.nof)
        print "-----------------------------------------"
        
    def set_dummy(self):
        """Loads and sets the dummy image"""
        dummy = Img(load_img_dummy())
        for key in self.loaded_images:
            self.loaded_images[key] = dummy
            
    def set_camera(self, camera = None, cam_id = None):
        """Set the current camera
        
        Two options:
            
            1. set :class:`pyplis.setup.Camera`directly
            #. provide one of the default camera IDs (e.g. "ecII", "hdcam")
        
        :param Camera camera: camera setup, or, alternatively
        :param str cam_id: valid camera ID (None)
        """
        if not isinstance(camera, Camera):
            camera = Camera(cam_id)
        self.camera = camera
    
    def reset_img_prep(self):
        """Init image pre edit settings"""
        self.img_prep = dict.fromkeys(self.img_prep, 0)
        self._roi_abs = [0, 0, 9999, 9999]
        if self.nof > 0:
            self.load()
    
    def get_img_meta_from_filename(self, file_path):
        """Loads and prepares img meta input dict for Img object
        
        :param str img_file: file path of image
        :returns: dict, keys: "start_acq" and "texp"
        """
        info = self.camera.get_img_meta_from_filename(file_path)
        return {"start_acq" : info[0], "texp": info[3]}
    
    @property
    def acq_times(self):
        """Returns list of acquisition time stamps
        
        The time stamps are extracted from the file names
        """
        ts = self.get_img_meta_all_filenames()[0]
        if ts is False:
            raise ValueError("Image acquisition times could not be extracted"
                " from file names")
        return ts
        
    def get_img_meta_all_filenames(self):   
        """Try to load acquisition and exposure times from filenames
        
        :return: 
            - list, containing img acquisition time stamps for all images in 
                this list (if accessible, else False)
            - list, containing all img exposure times 
                (if accessible, else False)
            
        .. note:: 
        
            Only works if relevant information is specified in ``self.camera``
        
        """
        times, texps = asarray([None] * self.nof), asarray([None] * self.nof) 
        
        for k in range(self.nof):
            try:
                info = self.camera.get_img_meta_from_filename(self.files[k])
                times[k] = info[0]
                texps[k] = info[3]
            except:
                pass
    
        return times, texps

    def assign_indices_linked_list(self, lst):
        """Create a look up table for fast indexing between image lists
        
        :param BaseImgList lst: image list supposed to be linked
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
        
        :returns bool: False if not the same, True else
        """
        sd = self.img_prep
        for key, val in settings_dict.iteritems():
            if sd.has_key(key):
                if not sd[key] == val:
                    return False
        return True
    
    def make_stack(self, stack_id=None, pyrlevel=None, roi_abs=None,
                   dtype=float32):
        """Stack all images in this list 
        
        The stacking is performed using the current image preparation
        settings (blurring, dark correction etc). Only stack ROI and pyrlevel
        can be set explicitely.
        """
        self.activate_edit()
        
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
            stack_id = self.img_mode
        if stack_id in ["raw", "tau"]:
            stack_id += "_%s" %self.list_id
        #create a new settings object for stack preparation
        self.goto_img(0)
        self.auto_reload = True
        h, w = self.current_img().shape
        stack = ImgStack(h, w, self.nof, dtype, stack_id, camera=self.camera,
                         img_prep=self.current_img().edit_log)
        nof = self.nof
        lid = self.list_id
        for k in range(nof):
            print "Building img-stack from list %s (%s | %s)" %(lid, k, nof-1)
            img = self.loaded_images["this"]
            #print im.meta["start_acq"]
            stack.append_img(img.img, img.meta["start_acq"],\
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
        return stack
    
    def get_mean_img(self, start_index=0, stop_index=None):
        """Determines average image from list images
        
        Parameters
        ----------
        start_index : int
            index of first considered image
        stop_index : int
            index of last considered image (if None, the last image in this 
            list is used)
            
        Returns
        -------
        Img 
            average image 
        """
        cfn = self.cfn
        self.goto_img(start_index)
        if stop_index is None or stop_index > self.nof:
            print "Setting stop_index to last list index"
            stop_index = self.nof
        img = Img(zeros(self.current_img().shape))
        img.edit_log = self.current_img().edit_log
        img.meta["start_acq"] = self.current_time()
        added = 0
        texps = []
        for k in range(start_index, stop_index):
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
            :class:`PixelMeanTimeSeries` objects for each ROI
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
        lid=self.list_id
        for k in range(num):
            print ("Calc pixel mean t-series in list %s (%d | %d)" 
                                            %(lid,(k+1),num))
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
        input param roi, image preparation can be turned on or off as well as
        tau mode (for which an background image must be available)
        
        :param list roi: rectangular region of interest ``[x0, y0, x1, y1]``, 
            default is [0, 0, 9999, 9999] (i.e. whole image)
        :param bool apply_img_prep: if True, img preparation is performed
            as set in ``self.img_prep`` dictionary  (True)        
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
        for k in range(num):
            print ("Calc pixel mean t-series in list %s (%d | %d)" 
                                            %(lid,(k+1),num))
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
        """Returns the current image preparation settings
        
        These are applied by default when images are loaded and if
        ``self.edit_active == True`` (which is the default)
        """
        return self.current_img().edit_log
        
    def edit_info(self):
        """Print the current image preparation settings"""
        d = self.current_img().edit_log
        print("\nImgList %s, image edit info\n----------------------------" 
            %self.list_id)
        for key, val in d.iteritems():
            print "%s: %s" %(key, val)
        
    def _make_header(self):
        """Make header string for current image (using image meta information)            
        """
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
        """Merges the current list edit state with the image state on load"""
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
        """Load current image"""
        if not self._auto_reload:
            print ("Automatic image reload deactivated in image list %s"\
                                                                %self.list_id)
            return False
        img_file = self.files[self.index]
        try:
            img = Img(img_file,
                      import_method=self.camera.image_import_method,
                      **self.get_img_meta_from_filename(img_file))
            self.loaded_images["this"] = img
            self._load_edit.update(img.edit_log)
            
            if img.vign_mask is not None:
                self.vign_mask = img.vign_mask
            
            self.update_prev_next_index()
            self.apply_current_edit("this")
            
        except IOError:
            print ("Invalid file encountered at list index %s, file will"
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
    def add_gaussian_blurring(self, sigma = 1):
        """Increase amount of gaussian blurring on image load
        
        :param int sigma (1): Add width gaussian blurring kernel
        """
        self.img_prep["blurring"] += sigma
        self.load()
        
    def apply_current_edit(self, key):
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
    
    """List modes"""    
    def activate_edit(self, val = True):
        """Activate / deactivate image edit mode
        
        :param bool val: new mode
        
        If inactive, images will be loaded raw without any editing or 
        further calculations (e.g. determination of optical flow, or updates of
        linked image lists). Images will be reloaded.
        """
        if val == self.edit_active:
            return
        self.edit_active = val
        self.load()
        
    def cam_id(self):
        """Get the current camera ID (if camera is available)"""
        return self.camera.cam_id

            
    def current_time(self):
        """Get the acquisition time of the current image from image meta data
        
        :returns datetime:
        """
        return self.current_img().meta["start_acq"]
        
    def current_time_str(self, format='%H:%M:%S'):
        """Returns a string of the current acq time"""
        return self.loaded_images["this"].meta["start_acq"].strftime(format)
        
    def current_img(self, key="this"):
        """Get the current image object
        
        :param str key ("this"): "prev", "this" or "next"
        :returns Img:
        """
        img = self.loaded_images[key]
        if not isinstance(img, Img):
            try:
                self.load()
                img = self.loaded_images[key]
            except IndexError:
                raise IndexError("Image list %s does not contain images" 
                            %self.list_id)
            except:
                raise
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
    def plot_mean_value(self, roi = [0, 0, 9999, 9999], apply_img_prep = True,\
                                    yerr = False, ax = None):
        """Plot mean value of image time series
        
        :param list roi: rectangular ROI in which mean is determined (default
            ``[0, 0, 9999, 9999]``, i.e. whole image)
        :param bool yerr: include errorbars (std)
        :param ax: matplotlib axes object
        """
        if ax is None:
            fig = figure()#figsize=(16, 6))
            ax = fig.add_subplot(1, 1, 1)
        mean = self.get_mean_value()
        ax = mean.plot(yerr = yerr, ax = ax)
        return ax
    
    def plot_tseries_vert_profile(self, pos_x, start_y = 0, stop_y = None,\
                                                step_size = 0.1, blur = 4):
        """Plot the temporal evolution of a line profile
        
        :param int pos_x: number of pixel column
        :param int start_y: Start row of profile (y coordinate, default: 10)
        :param int stop_y: Stop row of profile (is set to rownum - 10pix if
            input is None)
        :param float step_size: stretch between different line profiles of
            the evolution (0.1)
        :param int blur: blurring of individual profiles (4)
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
        return fig, ax1, ax2

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
    """Image list object with expanded functionality compared to 
    :class:`BaseImgList`. 
    
    Additional features:
    
        1. Linking of other image lists 
            The index (i.e. the currently loaded image) of all linked lists is 
            automatically updated whenever the index in this list is changed 
            based on acquisition time such that the corresponding images in 
            other lists are the ones closest in time to this image
    
    """
    def __init__(self, files=[], list_id=None, list_type=None, camera=None,
                 init=True):
        """
            
        Extended version of :class:`BaseImgList` object, additional
        features:
        
            1. Optical flow determination
            #. Linking of lists (e.g. on and offband lists)
            #. Dark and offset image correction (*write a bit more here*)
            #. Plume background modelling and tau image determination
            #. Include
            
        """
        super(ImgList, self).__init__(files, list_id, list_type,\
                                                camera, init = False)
        self.loaded_images.update({"prev": None, 
                                   "next": None})
    
        self._meas_geometry = None
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
        
        self._bg_imgs = [None, None] #sets bg images, 
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
        self.optflow = OpticalFlowFarneback(name=self.list_id)
        
        if self.data_available and init:
            self.load()
        
    def init_filelist(self):
        """Adding functionality to filelist init"""
        super(ImgList, self).init_filelist()
#==============================================================================
#         if not self.data_available:
#             self.set_dummy()
#==============================================================================
    
    def set_bg_img(self, bg_img):
        """Update the current background image object
        
        :param Img bg_img: the background image object used for plume 
            background modelling (modes 1 - 6 in :class:`PlumeBackgroundModel`)        
        """
        if not isinstance(bg_img, Img):
            print ("Could not set background image in ImgList %s: "
                ": wrong input type, need Img object" %self.list_id)
            return False
        if self.vign_mask is None:
            if bg_img.edit_log["vigncorr"]:
                raise AttributeError("Input background image is vignetting "
                    "corrected and cannot be used to calculate vignetting corr"
                    "mask.")
        
            self._bg_imgs[0] = bg_img
            mask = self.det_vign_mask_from_bg_img()
            self._bg_imgs[1] = bg_img.duplicate().correct_vignetting(mask.img)
        else:
            if not bg_img.edit_log["vigncorr"]:
                bg = bg_img
                bg_vigncorr = bg_img.duplicate().correct_vignetting(mask.img)
            else:
                bg_vigncorr = bg_img
                bg = bg_img.duplicate().correct_vignetting(mask.img,
                                                           new_state=0)
            self._bg_imgs = [bg, bg_vigncorr]
    
    @property
    def bg_img(self):
        """Return background image based on current vignetting corr setting"""
        return self._bg_imgs[self.loaded_images["this"].edit_log["vigncorr"]]
    
    @bg_img.setter
    def bg_img(self, val):
        self.set_bg_img(val)
        
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
            raise ValueError("Img shape mismatch between AA correction mask "
                "and list images")
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
            
    def set_bg_corr_mode(self, mode=1):
        """Update the current background correction mode in ``self.bg_model``
        
        :param int mode (1): one of the default background correction modes of
            the background model. Valid input is 0 - 6, call::
            
                self.bg_model.print_mode_info()
                
            for more information.
            
        """
        if not (0 <= mode <= 6):
            raise ValueError("Invalid background correction mode")
        elif (1 <= mode <= 6) and not isinstance(self.bg_img, Img):
            raise TypeError("Could not set mode %s, this mode requires an "
                "additional sky background image which is not set in ImgList: "
                "please set a background image first, using ``set_bg_img``")
        self.bg_model.CORR_MODE = mode
        
    def init_bg_model(self, **kwargs):
        """Init clear sky reference areas in background model"""
        self.bg_model.update(**kwargs)
        self.bg_model.guess_missing_settings(self.current_img())
        
    def set_optical_flow(self, optflow):
        """Set the current optical flow object (type 
        :class:`pyplis.Processing.OpticalFlowFarneback`)
        """
        self.optflow = optflow
    
    @property
    def meas_geometry(self):
        """Get / set current measurement geometry"""
        return self._meas_geometry
    
    @meas_geometry.setter
    def meas_geometry(self, val):
        """Set :class:`pyplis.Utils.MeasGeometry` object"""
        if not isinstance(val, MeasGeometry):
            raise TypeError("Could not set meas_geometry, need MeasGeometry "
                "object")
        self._meas_geometry = val
                
    def link_imglist(self, other_list):
        """Link another image list to this list
        
        :param other_list: another image list object
        
        """
        self.current_img(), other_list.current_img()
        list_id = other_list.list_id            
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
                        elif dt < dtexp_dark:
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
                        if dt < dtexp_offset:
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
            dark = Img(dark, texp = texp)

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
        
    @property
    def dark_info(self):
        """Print information about current dark image access"""
        s = "Dark image info list %s\n------------------------------+n"
        s += "DARK_CORR_OPT=%d\n" 
        raise NotImplementedError
        
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
        
    def activate_darkcorr(self, val=True):
        """Activate or deactivate dark and offset correction of images
        
        :param bool val: Active / Inactive
        
        If dark correction turned on, dark image access is attempted, if that
        fails, Exception is raised including information what did not work 
        out.
        """
        if not val and self._load_edit["darkcorr"]:
            raise ImgMetaError("Cannot deactivate dark correction, original"
                "image file was already dark corrected")
        if val:
            self.get_dark_image()
            self.update_index_dark_offset_lists()
                    
        self._list_modes["darkcorr"] = val
        self.load()
    
    def _update_prep_bg_image(self):
        """Update image preparation status of BG image"""
        cim = self.current_img()
        bg = self.bg_img
        log = cim.edit_log
        if bg.edit_log["blurring"] < log["blurring"]:
            print ("Update blurring of BG image in img list %s, new val: %d" 
                %(self.list_id, log["blurring"]))
            bg.add_gaussian_blurring(sigma_final = log["blurring"])
        elif bg.edit_log["blurring"] > log["blurring"]:
            warn("Blurring factor of current background image in list %s "
                "exceeds current list blurring" %self.list_id)
        if bg.edit_log["median"] < log["median"]:
            print ("Update median filtering of BG image in img list %s, "
                "new val: %d" %(self.list_id, log["median"]))
            bg.apply_median_filter(log["median"])
        elif bg.edit_log["median"] > log["median"]:
            warn("Median filter kernel of current background image in list %s "
                "exceeds that of list" %self.list_id)    
        if not log["vigncorr"] == bg.edit_log["vigncorr"]:
            raise ValueError("Check vignette")
        
    def activate_tau_mode(self, val = 1):
        """Activate tau mode
        
        In tau mode, images will be loaded as tau images (if background image
        data is available). 
        """
        if val is self.tau_mode: #do nothing
            return
        if val:
            if not self.has_bg_img():
                raise AttributeError("no background image available in list %s"
                ", please set suitable background image using method "
                "set_bg_img" %self.list_id)
            #self._update_prep_bg_image()
            self.bg_model.guess_missing_settings(self.loaded_images["this"])
            self.bg_model.get_tau_image(self.loaded_images["this"],\
                                                            self.bg_img)
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
            offlist = self.get_off_list()
            if not isinstance(offlist, ImgList):
                raise Exception("Linked off band list could not be found")
            if not offlist.nof / float(self.nof) > 0.5:
                raise IndexError("Off band list does not have enough images...")
            if not self.has_bg_img():
                raise AttributeError("no background image available, please set "
                    "suitable background image using method set_bg_img")
            if not offlist.has_bg_img():
                raise AttributeError("no background image available in off "
                    "band list")
            #offlist.update_img_prep(**self.img_prep)
            #offlist.init_bg_model(CORR_MODE = self.bg_model.CORR_MODE)
            self.tau_mode = 0
            offlist.tau_mode = 0
    
            aa_test = self._aa_test_img(offlist)
        self._list_modes["aa"] = val

        self.load()

        return aa_test
     
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
    
    def activate_optflow_mode(self, val=True):
        """Activate / deactivate optical flow calculation on image load"""
        if val is self.optflow_mode:
            return 
        if self.crop:
            raise ValueError("Optical flow analysis can only be applied to "
                "uncropped images, please deactivate crop mode")
        if val:
            try:
                self.set_flow_images()
            except IndexError:
                raise IndexError("Optical flow mode cannot be activated in "
                    "image list %s: list is at last index, please change list "
                    "index and retry")
            self.optflow.calc_flow()
            self.optflow.draw_flow()
            if self.bg_model.scale_rect is None:
                if self.current_img().edit_log["is_tau"]:
                    raise Exception("Fatal: scale rectangle is not set in "
                        "background model of list %s, but current image is "
                        "flagged tau image" %self.list_id)
                self.bg_model.guess_missing_settings(self.current_img())
            roi = map_roi(self.bg_model.scale_rect, self.pyrlevel)
            len_im = self.optflow.get_flow_vector_length_img() #is at pyrlevel
            sub = len_im[roi[1]:roi[3], roi[0]:roi[2]]
            min_len = ceil(sub.mean() + 3 * sub.std()) + 1
            self.optflow.settings.min_length = min_len
                
        self._list_modes["optflow"] = val
    
    def det_vign_mask_from_bg_img(self):
        """Determine vignetting mask from current background image
        
        The mask is determined using a blurred (:math:`\sigma = 3`) 
        background image which is normalised to one.
        
        The mask is stored in ``self.vign_mask``
        
        :return: 
            - Img, vignetting mask
        """
        if not self.has_bg_img():
            raise AttributeError("Please set a background image first")
        mask = self._bg_imgs[0].duplicate()
        if mask.edit_log["blurring"] < 3:
            mask.add_gaussian_blurring(3)
        mask.img = mask.img / mask.img.max()
        self.vign_mask = mask
        return mask
        
    def activate_vigncorr(self, val=True):
        """Activate / deactivate vignetting correction on image load
        
        :param bool val: new mode 
        
        .. note::
        
            Requires ``self.vign_mask`` to be set or an background image 
            to be available (from which ``self.vign_mask`` is then determined)
            
        """
        if isinstance(self.vign_mask, Img):
            self.vign_mask = self.vign_mask.img
        if not isinstance(self.vign_mask, ndarray):
            self.det_vign_mask_from_bg_img() 
        shape_orig = Img(self.files[self.cfn]).img.shape
        if not self.vign_mask.shape == shape_orig:
            raise ValueError("Shape of vignetting mask %s deviates from raw "
                "img shape %s" %(list(self.vign_mask.shape),
                                 list(shape_orig)))
        self._list_modes["vigncorr"] = val
        self.load()

    """
    Image loading functions 
    """
    def load(self):
        """Try load current image and previous and next image"""
        self.update_index_linked_lists() #based on current index in this list
        if not super(ImgList, self).load():
            print ("Image load aborted...")
            return False
        if self.nof > 1:
            prev_file = self.files[self.prev_index]
            self.loaded_images["prev"] = Img(prev_file,
                            import_method=self.camera.image_import_method,
                            **self.get_img_meta_from_filename(prev_file))
            self.apply_current_edit("prev")
            
            next_file = self.files[self.next_index]
            self.loaded_images["next"] = Img(next_file,
                            import_method=self.camera.image_import_method,                            
                            **self.get_img_meta_from_filename(next_file))
            self.apply_current_edit("next")
        else:
            self.loaded_images["prev"] = self.loaded_images["this"]
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
        
        self.loaded_images["prev"] = self.loaded_images["this"]
        self.loaded_images["this"] = self.loaded_images["next"]
        
        next_file = self.files[self.next_index]
        self.loaded_images["next"] = Img(next_file,
                            import_method=self.camera.image_import_method,                            
                            **self.get_img_meta_from_filename(next_file))
    

        self.apply_current_edit("next")
        if self.optflow_mode:  
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                warn("Reached last index in image list, optflow_mode will be "
                    "deactivated")
                self.optflow_mode = 0
        return True
        #self.prepare_additional_data()
        
    def load_prev(self):   
        """Load previous image in list"""
        if self.nof < 2 or not self._auto_reload:
            print ("Could not load previous image, number of files in list: " +
                str(self.nof))
            return
        self.index = self.prev_index
        self.update_prev_next_index()
        
        self.update_index_linked_lists() #loads new images in all linked lists
    
        self.loaded_images["next"] = self.loaded_images["this"]
        self.loaded_images["this"] = self.loaded_images["prev"]
        
        prev_file = self.files[self.prev_index]
        self.loaded_images["prev"] = Img(prev_file,
                        import_method=self.camera.image_import_method,
                        **self.get_img_meta_from_filename(prev_file))
        
        self.apply_current_edit("prev")
        if self.optflow_mode:  
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                warn("Reached last index in image list, optflow_mode will be "
                    "deactivated")
                self.optflow_mode = 0
 
    def apply_current_edit(self, key):
        """Applies the current image edit settings to image
        
        :param str key: image id (e.g. this)
        """
        if not self.edit_active:
            warn("Edit not active in img_list %s: no image preparation will "
                "be performed" %self.list_id)
            return
        img = self.loaded_images[key]
        if self.darkcorr_mode:
            dark = self.get_dark_image(key)
            img.subtract_dark_image(dark)
        img.correct_vignetting(self.vign_mask, new_state=self.vigncorr_mode)
        if self.tau_mode:
            img = self.bg_model.get_tau_image(img, self.bg_img)
        elif self.aa_mode:
            off = self.get_off_list()
            img = self.bg_model.get_aa_image(img, off.current_img(),
                                             self.bg_img, off.bg_img)
            if self.sensitivity_corr_mode:
                img = img / self.aa_corr_mask
                img.edit_log["senscorr"] = 1
            if self.calib_mode:
                img.img = self.calib_data(img.img)
                img.edit_log["gascalib"] = True
       
        img.to_pyrlevel(self.img_prep["pyrlevel"])
        if self.img_prep["crop"]:
            img.crop(self.roi_abs)
        img.add_gaussian_blurring(self.img_prep["blurring"])
        img.apply_median_filter(self.img_prep["median"])
        if self.img_prep["8bit"]:
            img._to_8bit_int(new_im = False)
        self.loaded_images[key] = img
            
      
    def set_flow_images(self):  
        """Update images for optical flow determination 
        
        The images are updated in :attr:`optflow` 
        (:class:`OpticalFlowFarneback` object) using method :func:`set_images`
        
        Raises
        
        object, i.e. `self.loaded_images["this"]` and `self.loaded_images["next"]`
        """
        if self.cfn == self.nof - 1:
            self.optflow.reset_flow()
            raise IndexError("Optical flow images cannot be set in ImgList %s:"
                " reached last image ..." %self.list_id)
            
        self.optflow.set_images(self.loaded_images["this"],
                                self.loaded_images["next"])

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
        
    """Helpers"""
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
            if not self.aa_mode:
                self.aa_mode = True
                print "Activated AA mode in image list"
            if not self.sensitivity_corr_mode:
                warn("AA sensitivity correction mode is deactivated. This "
                    "may yield erroneous results at the image edges")
            self.calib_data(self.current_img())
            
        self._list_modes["gascalib"] = val
        self.load()
    
    def has_bg_img(self):
        """Returns boolean whether or not background image is available"""
        if not isinstance(self.bg_img, Img):
            return False
        return True
        
    """Private methods"""        
    def _aa_test_img(self, off_list):
        """Try to determine an AA image"""
        on = Img(self.files[self.cfn],
                 import_method=self.camera.image_import_method)
        off = Img(off_list.files[off_list.cfn],
                  import_method=self.camera.image_import_method)
        return self.bg_model.get_aa_image(on, off, self.bg_img,
                                          off_list.bg_img)
        
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
                 cell_id="", gas_cd=0.0, gas_cd_err=0.0):
        
        super(CellImgList, self).__init__(files, list_id, list_type, camera)
        self.cell_id = cell_id
        self.gas_cd = gas_cd
        self.gas_cd_err = gas_cd_err
        
    def update_cell_info(self, cell_id, gas_cd, gas_cd_err):
        """Update cell_id and gas_cd amount"""
        self.cell_id = cell_id
        self.gas_cd = gas_cd
        self.gas_cd_err = gas_cd_err
                   