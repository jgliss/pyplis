# -*- coding: utf-8 -*-
from matplotlib.pyplot import subplots
from warnings import warn
from numpy import float, log, arange, polyfit, poly1d, linspace, isnan,\
    diff, mean, argmin, ceil, round
from matplotlib.pyplot import Figure
from datetime import timedelta
from os.path import exists
from collections import OrderedDict as od

from .dataset import Dataset
from .setupclasses import MeasSetup
from .processing import ImgStack, PixelMeanTimeSeries#, ImgListStack, ImagePreparation
from .imagelists import ImgList, CellImgList
from .exceptions import CellSearchError, ImgMetaError
from .image import Img
from .helpers import subimg_shape, map_coordinates_sub_img, exponent
from .doascalib import DoasFOV
from .optimisation import PolySurfaceFit
      
class CellSearchInfo(object):
    """Class for collecting cell search results in a given time window"""
    def __init__(self, filter_id, id, y_max):
        """Class initialisation
        
        :param str filter_id: string ID of filter 
        :param str id: additional identifier (e.g. "bg", "cell1")
        :param float y_max: a reference intensity
        
        """
        self.filter_id = filter_id
        self.id = id
        #self.abbr = None
        self.y_max = y_max
        self.mean_vals = []
        self.mean_vals_err = []
        self.file_paths = []
        self.start_acq = []
        self.texps = []
        
        self.img_list = None
        
    @property
    def start(self):
        """Returns first time stamp of ``start_acq``"""
        return self.start_acq[0]
        
    @property
    def stop(self):
        """Returns last time stamp of ``start_acq``"""
        return self.start_acq[-1]
    @property
    def tot_num(self):
        """Returns number of datapoints in ``self.mean_vals``"""
        return len(self.mean_vals)
    
    def from_img_list(self, img_list):
        """Fill values using all images from a specific image list
        
        :param ImgList img_list: load all values from an image list by 
            determining the mean value time series (in the full image)        
        """
        if not isinstance(img_list, ImgList):
            raise TypeError("Wrong input type")
        dat = img_list.get_mean_value()        
        self.start = dat.index[0]
        self.stop = dat.index[-1]
        self.file_paths = img_list.files
        self.mean_vals = dat.values
        self.mean_vals_err = dat.std
        self.start_acq = dat.index
        self.y_max = max(dat.values)
    
    @property
    def mean_err(self):
        """Returns average std of mean value time series"""
        return mean(self.mean_vals_err)
    
    @property
    def mid_point_val(self):
        """Returns the mean value in the middle of the time series"""
        num = len(self.mean_vals)
        if num < 1:
            raise Exception ("No data available in CellSearchInfo")
        elif num == 1:
            return self.mean_vals[0]
        elif num == 2:
            return mean(self.mean_vals)
        else:
            mid_index = int(num/2)
            return self.mean_vals[mid_index]
            
    def point_ok(self, idx):
        """Checks data point at given index
    
        :param int idx: index of datapoint
        """
        try:
            val = self.mean_vals[idx]
            if abs(self.mid_point_val - val) < self.mean_err:
                return True
            return False
        except IndexError as e:
            print repr(e)
        except:
            raise
    
    def create_image_list(self, camera):
        """Create image list containing all valid cell images 
        
        :param Camera camera: the camera used
        :returns CellImgList: image list containing all (valid) cell images
        
        The validity of one specific image file is checked by analysing 
        its mean intensity values with respect to the intensity value of 
        the midpoint of this dataset (must be smaller than ``self.mean_err``)
        and by ensuring the intensity decrease of this point (offs) exceeds
        ``self.mean_err``
        
        """
        lst = CellImgList(list_id = self.filter_id, cell_id = self.id,\
                                                            camera = camera)
        for idx in range(len(self.mean_vals)):
            if self.point_ok(idx):
                lst.files.append(self.file_paths[idx])
        lst.init_filelist()
        self.img_list = lst
        if lst.nof < 1:
            raise CellSearchError("No suitable %s images found on creation of "
                "image list for cell %s" %(self.filter_id, self.id))
        print ("Succesfully created image list %s for cell with ID %s from cell "
            " search results" %(self.filter_id, self.id))
            
    @property
    def offs(self):
        return self.y_max - self.mean_vals

class CellAutoSearchResults(object):
    """Helper class collecting results from automatic cell detection algorithm 
    
    This object is included in :class:`CellCalibEngine` object and will be filled
    with :class:`CellSearchInfo` in case, the cell autodetection is used
    """
    def __init__(self):
        self.cell_info = {}
        self.bg_info = {}
        self.restInfo = {}
    
    def add_cell_search_result(self, filter_id, cell_info, bg_info, restInfo):
        """Adds a collection of :class:`CellSearchInfo` objects
        
        :param filter_id: ID of filter
        
        """
        self.cell_info[filter_id] = {}
        for cell_id, res in cell_info.iteritems():
            if isinstance(res, CellSearchInfo):
                self.cell_info[filter_id][cell_id] = res
        if isinstance(bg_info, CellSearchInfo):
            self.bg_info[filter_id] = bg_info
        if isinstance(restInfo, CellSearchInfo):
            self.restInfo[filter_id] = restInfo

class CellCalibData(object):
    """Object representing cell calibration data
    
    The object mainly consists of a stack containing tau images (e.g. on band
    off band or AA images)
    """
    _calib_id=""
    def __init__(self, tau_stack=None, calib_id="", so2_cds=None):
        
        self.tau_stack = tau_stack
        self.calib_id = calib_id        
                
        try:
            if len(so2_cds) == self.tau_stack.shape[0]:
                self.tau_stack.add_data = so2_cds
        except:
            pass
    
    @property
    def calib_id(self):
        """Get / set calibration ID"""
        return self._calib_id
    
    @calib_id.setter
    def calib_id(self, val):
        if not isinstance(val, str):
            raise TypeError("Invalid input for calib_id, need str")
        self._calib_id = val
        try:
            self.tau_stack.stack_id = val
        except:
            pass
        
    def add_tau_image(self, tau_img, cell_cd):
        """Add one cell image to the data
        
        :param Img tau_img: the actual tau image (must not be cropped, is 
            converted to pyramid level of stack, must be dark corrected, flag
            "is_tau" must be set).
        :param float cell_cd: corresponding gas coloumn density in cell
        
        .. note::
        
            Alpha version: not tested
            
        """
        raise NotImplementedError
                
    @property
    def cell_so2_cds(self):
        """return vector containing cell SO2 CDs"""
        return self.tau_stack.add_data
     
    def poly(self, pos_x_abs=None, pos_y_abs=None, radius_abs=1, mask=None, 
             polyorder=1):
        """Retrieve calibration polynomial within a certain pixel neighbourhood
        
        :param str filter_id: image type ID (e.g. "on", "off")
        :param int pos_x_abs: detector x position (col) in absolute detector 
                                                                    coords
        :param int pos_y_abs: detector y position (row) in absolute detector 
                                                                    coords
        :param float radius_abs: radius of pixel disk on detector (centered
            around pos_x_abs, pos_y_abs, default: 1)
        :param ndarray mask: boolean mask for image pixel access, 
            default is None, if the mask is specified and valid (i.e. same
            shape than images in stack) then the other three input parameters
            are ignored (None)
        :param int polyorder: order of polynomial for fit (1)
        :returns: tuple containing 
            - poly1d, fitted polynomial
            - ndarray, array with tau values 
            - ndarray, array with corresponding gas CDs
        
        """
        stack = self.tau_stack
        try:
            x_rel, y_rel = map_coordinates_sub_img(pos_x_abs, pos_y_abs,
                                                   stack.roi_abs,
                                                   stack.img_prep["pyrlevel"])
        except:
            print "Using image center coordinates"
            h, w = stack.shape[1:]
            x_rel, y_rel = int(w / 2.0), int(h / 2.0)
        try:
            rad_rel = int(ceil(float(radius_abs) /\
                            2**stack.img_prep["pyrlevel"]))
        except:
            print "Using radius of 3"
            rad_rel = 3
        tau_arr = stack.get_time_series(x_rel, y_rel, rad_rel, mask)[0].values
        so2_arr = stack.add_data
        return poly1d(polyfit(tau_arr, so2_arr, polyorder)), tau_arr, so2_arr
    
    def get_sensitivity_corr_mask(self, doas_fov=None, cell_cd=1e16,
                                  surface_fit_pyrlevel=2):
        """Get sensitivity correction mask 
        
        Prepares a sensitivity correction mask to corrector for filter 
        transmission shifts. These shifts result in increaing optical densities
        towards the image edges for a given gas column density.
        
        The mask is determined for original image resolution, i.e. pyramid 
        level 0.
        
        The mask is determined from a specified cell optical density image 
        (aa, tau_on, tau_off) which is normalised either
        with respect to the pixel position of a DOAS field of view within the
        images, or, alternatively with respect to the image center coordinate.
        
        Plume AA (or tau_on, tau_off) images can then be corrected for these
        sensitivity variations by division with the mask. If DOAS calibration 
        is used, the calibration polynomial can then be used for all image 
        pixels. If only cell calibration is used, the mask is normalised with
        respect to the image center, the corresponding cell calibration 
        polynomial should then be retrieved in the center coordinate which
        is the default polynomial when using :func:`get_calibration_polynomial` 
        or func:`__call__`) if not explicitely specified. You may then 
        calibrate a given aa image (``aa_img``) as follows with using a 
        :class:`CellCalibData` object (denoted with ``cellcalib``)::
        
            mask, cell_cd = cellcalib.get_sensitivity_corr_mask()
            aa_corr = aa_img.duplicate()
            aa_corr.img = aa_img.img / mask
            #this is retrieved in the image center if not other specified
            so2img = cellcalib(aa_corr)
            so2img.show()
            
        :param DoasFov doas_fov: DOAS field of view class, if unspecified, the
            correction mask is determined with respect to the image center
        :param str filter_id: mask is determined from the corresponding calib
            data (e.g. "on", "off", "aa")
        :param float cell_cd: use the cell which is closest to this column
        :param int surface_fit_pyrlevel: additional downscaling factor for 
            2D polynomial surface fit
        :return: 
            - ndarray, correction mask
            - float, column density of corresponding cell image
        
        .. note::
        
            This function was only tested for AA images and not for on / off
            cell tau images
            
        """
        stack = self.tau_stack
        #convert stack to pyramid level 0
        stack = stack.to_pyrlevel(0)
        try:
            if stack.img_prep["crop"]:
                raise ValueError("Stack is cropped: sensitivity mask can only"
                    "be determined for uncropped images")
        except:
            pass
        idx = argmin(abs(stack.add_data - cell_cd))
        cell_img, cd = stack.stack[idx], stack.add_data[idx]
        if isinstance(doas_fov, DoasFOV):
            fov_x, fov_y = doas_fov.pixel_position_center(abs_coords=True)
            fov_extend = doas_fov.pixel_extend(abs_coords=True)
            
        else:
            print "Using image center coordinates and radius 3"
            h, w = stack.shape[1:]
            fov_x, fov_y = int(w / 2.0), int(h / 2.0)
            fov_extend = 3
        fov_mask = stack.make_circular_access_mask(fov_x, fov_y, fov_extend)
        try:
            cell_img = PolySurfaceFit(cell_img, 
                                      pyrlevel=surface_fit_pyrlevel).model
        except:
            warn("2D polyfit failed while determine sensitivity correction "
                "mask, using original cell tau image for mask determination")
        mean = (cell_img * fov_mask).sum() / fov_mask.sum()
        mask = cell_img / mean
        return mask, cd
        
    def plot(self, pos_x_abs, pos_y_abs, radius_abs=1, mask=None,
             ax=None):
        """Plot all available calibration curves in a certain pixel region
        
        :param int pos_x_abs (None): x position of center pixel on detector
        :param int pos_y_abs (None): y position of center pixel on detector
        :param float radius_abs (1): radius of pixel disk on detector (centered
            around pos_x_abs, pos_y_abs)
        :param ndarray mask (None): boolean mask for image pixel access, 
            default is None, if the mask is specified and valid (i.e. same
            shape than images in stack) then the other three input parameters
            are ignored
        :param ax (None): matplotlib axes (if None, a new figure with axes
            will be created)
            
        """
        add_to = True
        if ax is None:
            fig, ax = subplots(1, 1)
            add_to = False
        poly, tau, so2 = self.poly(pos_x_abs, pos_y_abs, radius_abs, mask)
        
        taus = linspace(0, tau.max() * 1.05, 100)
        ax.plot(tau, so2, " ^", label = "Cell data %s" %self.calib_id)
        ax.plot(taus, poly(taus),"--", label = "Fit: %s" %poly)
        
        if not add_to:
            ax.set_ylabel(r"S$_{SO2}$ [cm$^{-2}$]", fontsize=18)
            ax.set_xlabel(r"$\tau$", fontsize = 18)    
            ax.grid()
        ax.legend(loc = "best", fancybox = True, framealpha = 0.5,\
                                                        fontsize = 14)
        return ax
        
    def __sub__(self, other):
        """Subtraction (e.g. for AA calib determination)"""
        if not isinstance(other, CellCalibData):
            raise TypeError("Need CellCalibData object for subtraction")
        st = self.tau_stack - other.tau_stack
        return CellCalibData(tau_stack=st)

    def __call__(self, value, **kwargs):
        """Define call function to apply calibration
        
        :param float value: tau or AA value
        :return: corresponding column density
        """
        try:
            poly = self.get_calibration_polynomial(**kwargs)[0]
        except:
            raise ValueError("Calibration data not available")
        if isinstance(value, Img):
            calib_im = value.duplicate()
            calib_im.img = poly(calib_im.img)
            calib_im.edit_log["gascalib"] = True
            return calib_im
        elif isinstance(value, ImgStack):
            try:
                value = value.duplicate()
            except MemoryError:
                warn("Stack cannot be duplicated, applying calibration to "
                "input stack")
            value.stack=poly(value.stack)
            value.img_prep["gascalib"] = True
            return value
        return poly(value) - poly.coeffs[-1]
        
class CellCalibEngine(Dataset):
    """Class for performing automatic cell calibration 
    
    This class is designed to define datasets related to time windows, where 
    cell calibration was performed, i.e. the camera pointing into a gas (and
    cloud) free area of the sky with a number of calibration cells are put
    in front of the lense consecutively (ideally, the cells should cover the
    whole FOV of the camera in order to be able to retrieve calibration 
    polynomials for each image pixel individually).
    Individual time windows for each cell are extracted by analysing the time
    series of pixel mean intensities for all images that fall into the start
    / stop interval. Cells can be identified by dips of decreased intensities
    in the time series. The individual cells are then assigned automatically
    based on the depth of each dip (in the on band) and the column densities
    of the cells used (the latter need to be provided).
    
    Is initialised as :class:`piscope.Datasets.Dataset` object, i.e. normal
    setup is like plume data using a :class:`MeasSetup` object (make sure 
    that ``cell_info_dict`` is set in the setup class).
    """
    def __init__(self, setup = None, init = 1):
        print 
        print "INIT CALIB DATASET OBJECT"
        print
        super(CellCalibEngine, self).__init__(setup, init)
                        
        self.type = "cell_calib"
        
        self.cell_search_performed = 0
        self._cell_info_auto_search = od()
        
        if isinstance(self.setup, MeasSetup):
            self.set_cell_info_dict_autosearch(self.setup.cell_info_dict)
        
        self.cell_lists = od()
        self.bg_lists = od()
        
        self.search_results = CellAutoSearchResults()
        
        self.pix_mean_tseries = od()
        self.bg_tseries = od()
        
        self.calib_data = od()
        print 
        print "FILELISTS IN CALIB DATASET OBJECT INITIALISED"
        print
    
    def add_cell_images(self, img_paths, cell_gas_cd, cell_id, filter_id):
        """Add list corresponding to cell measurement
        
        :param list img_paths: list containing image file paths
        :param float cell_gas_cd: column amount of gas in cell
        :param str cell_id: string identification of cell
        :param str filter_id: filter ID for images (e.g. "on", "off")
            
        """
        try:
            #input is not a list but a valid path to (hopefully) an image 
            if exists(img_paths): 
                img_paths = [img_paths,]
        except:
            pass
        
        paths = [p for p in img_paths if exists(p)]
        if not len(paths) > 0:
            raise TypeError("No valid filepaths could be identified")
        
        lst = CellImgList(files = paths, list_id = filter_id, camera =\
            self.camera, cell_id = cell_id, gas_cd = cell_gas_cd)
        self.add_cell_img_list(lst)
            
    def add_cell_img_list(self, lst):
        """Add a :class:`CellImgList` object in ``self.cell_lists``
        
        :param CellImgList lst: if, valid input, the list is added to dictionary
            ``self.cell_lists`` with it's filter ID as first key and its cell_id as 
            second, e.g. ``self.cell_lists["on"]["a53"]``
        """
        if not isinstance(lst, CellImgList):
            raise TypeError("Error adding cell image list, need CellImgList "
                "object, got %s" %type(lst))
        elif not lst.nof > 0:
            raise IOError("No files available in cell ImgList %s, %s" 
            %(lst.id, lst.cell_id))
        elif any([lst.gas_cd == x for x in [0, None]]) or isnan(lst.gas_cd):
            raise ValueError("Error adding cell image list, invalid value encountered for"
                "attribute gas_cd: %s" %lst.gas_cd)
        if not self.cell_lists.has_key(lst.list_id):
            self.cell_lists[lst.list_id] = {}
        self.cell_lists[lst.list_id][lst.cell_id] = lst
    
    def add_bg_img_list(self, lst):
        """Add a :class:`ImgList` object in ``self.bg_lists``
        
        :param ImgList lst: if, valid input, the list is added to dictionary
            ``self.bg_lists`` with it's ID as key
        """
        if not isinstance(lst, ImgList):
            raise TypeError("Error adding bg image list, need ImgList object, "
                "got %s" %type(lst))
        elif not lst.nof > 0:
            raise IOError("No files available in bg ImgList %s, %s" 
            %(lst.list_id, lst.cell_id))
        self.bg_lists[lst.list_id] = lst
        
    def det_bg_mean_pix_timeseries(self, filter_id):
        """Determine (or get) pixel mean values of background image list
        
        :param str filter_id: list ID
        
        Gets the average pixel intenisty (conisdering the whole image) for 
        all images in background image list and loads it as 
        :class:`PixelMeanTimeSeries` object, which is then stored in 
        ``self.bg_tseries`` and can be used to interpolate background 
        intensities for cell image time stamps (this might be important for
        large SZA measurements where the background radiance changes 
        fastly).
        """
#==============================================================================
#         if filter_id in self.search_results.bg_info.keys():
#             info = self.search_results.bg_info[filter_id]
#             ts = PixelMeanTimeSeries(info.mean_vals, info.start_acq,\
#                 info.mean_vals_err, info.texps)
#         else:
#==============================================================================
        ts = self.bg_lists[filter_id].get_mean_value()
        ts.fit_polynomial()
        ts.img_prep.update(self.bg_lists[filter_id].current_img().edit_log)
        self.bg_tseries[filter_id] = ts
        return ts
            
    def find_cells(self, filter_id = "on", threshold = 0.10,\
                                        accept_last_in_dip = False):
        """Autodetection of cell images and bg images using mean value series
        
        :param str filter_id: filter ID (mean value series is determined from 
            corresponding :class:`ImgList` object)
        :param float threshold: threshold in percent by which intensity 
            decreases are identified
        :param bool accept_last_in_dip (False): if true, also the last image in 
            one of the Cell intensity dips is considered a valid cell image
            (by default, the first and the last images of one dip are not 
            considered)
            
        This algorithm tries to separate individual cell images and background 
        images by analysing the 1st derivative of the mean pixel intensity of 
        each image in the time span specified in this object (``self.start``,
        ``self.stop``). 
        
        The separation of the individual cell images is performed by 
        identifying dips in the mean intensity evolution and assignment of 
        all image files belonging to each dip. 
    
        """
        print 
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "++++++++SEARCHING CELL TIME WINDOWS ", filter_id, " ++++++++++++++"
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print
        l = self.get_list(filter_id)
        ts = l.get_mean_value()
        ts.name = filter_id
        x, y, yerr, texps = ts.index, ts.values, ts.std, ts.texps
        #this will be overwritten in the loop to find the BG image with the 
        #lowest standard deviation, which is then set as current bg image
        #yerrCurrentBG = 9999 
        ydiff = diff(y) #1st derivative (finite differences)
        y_max = max(y)
        bg_info = CellSearchInfo(filter_id, "bg", y_max)
        rest = CellSearchInfo(filter_id, "rest", y_max)
        cell_info = od() #will be filled with CellSearchInfo objects in the loop        
        cell_count = 0 #counter for number of cells detected
        on_cell = 0 #flag which is set when cell entry_cond is fulfilled
        for k in range(len(y) - 2):
            #Look for dip in intensity => candidate for calib cell time stamp
        
            #Define entry and leave acceptance condition for detection of time window
            entry_cond = ((1 - abs(y[k + 1] / y_max)) > threshold and abs(\
                ydiff[k]) / y_max < threshold and abs(ydiff[k - 1]) /\
                y_max > threshold)
                
            leave_cond = (1 - abs(y[k] / y_max)) > threshold and abs(ydiff[k]) /\
                y_max > threshold and abs(ydiff[k - 1]) / y_max < threshold 
                
            bg_cond = (1 - abs(y[k] / y_max)) < threshold and abs(ydiff[k]) /\
                y_max < threshold and abs(ydiff[k - 1]) / y_max < threshold
                    
            if not accept_last_in_dip:
                #print "Adapting entry condition for cell time window"
                entry_cond = entry_cond and abs(ydiff[k + 1]) / y_max < threshold
                leave_cond = (1 - abs(y[k] / y_max)) > threshold and\
                    abs(ydiff[k]) / y_max<threshold and abs(ydiff[k - 1]) /\
                    y_max < threshold and abs(ydiff[k + 1]) / y_max > threshold
            
            if entry_cond:
                if on_cell:
                    ts.plot()
                    raise Exception("Fatal: found cell dip within cell dip"
                        "plotted time series...")
                    
                print "Found cell at %s, %s" %(k,x[k])
                on_cell = 1
                cell_count += 1
                cell_id = "Cell%s" %cell_count
                result = CellSearchInfo(filter_id, cell_id, y_max)
                
            #Look for increase in intensity => candidate for removal of calib cell
            elif leave_cond and on_cell:
            #and onFilter:
                print "Reached end of cell DIP at %s, %s" %(k,x[k])
                #result.stop = x[k]
                on_cell = 0
                result.mean_vals.append(y[k])
                result.mean_vals_err.append(yerr[k])
                result.file_paths.append(l.files[k])
                result.start_acq.append(x[k])
                result.texps.append(texps[k])
                cell_info[result.id] = result
                #onFilter=0
            elif bg_cond:
                print "Found BG candidate at %s, %s" %(k,x[k])
                bg_info.mean_vals.append(y[k])
                bg_info.mean_vals_err.append(yerr[k])
                bg_info.file_paths.append(l.files[k])
                bg_info.start_acq.append(x[k])
                bg_info.texps.append(texps[k])
            else: 
                if on_cell:
                    result.mean_vals.append(y[k])
                    result.mean_vals_err.append(yerr[k])
                    result.file_paths.append(l.files[k])
                    result.start_acq.append(x[k])
                    result.texps.append(texps[k])
                else:
                    rest.mean_vals.append(y[k])
                    rest.mean_vals_err.append(yerr[k])
                    rest.file_paths.append(l.files[k])
                    rest.start_acq.append(x[k])
                    rest.texps.append(texps[k])
            k += 1
        
        if not len(self._cell_info_auto_search.keys()) == len(cell_info.keys()):
            raise CellSearchError("Number of detected cells (%s) is different "
                "from number of cells specified in cellSpecInfo (%s) " 
                %(len(self._cell_info_auto_search.keys()), len(cell_info.keys())))
        
        #Create new image lists from search results for background images
        #and one list for each cell that was detected
        bg_info.create_image_list(self.camera)
        bg_info.img_list.update_cell_info("bg", 0.0, 0.0)
        self.assign_dark_offset_lists(into_list = bg_info.img_list)
        for cell_id, info in cell_info.iteritems():
            info.create_image_list(self.camera)
            self.assign_dark_offset_lists(into_list = info.img_list)
            
        self.search_results.add_cell_search_result(filter_id, cell_info,\
                                                                bg_info, rest)
        self.pix_mean_tseries["%s_auto_search" %filter_id] = ts
        #link background image list to earliest cell list
        cell_info.items()[0][1].img_list.link_imglist(bg_info.img_list)
        
        #bg_info.img_list.change_index(argmin(bg_info.mean_vals_err))
        
        
    def _assign_calib_specs(self, filter_id = None):
        """Assign the SO2 amounts to search results for all filter lists
        
        :param str filter_id: ID of filter used (e.g. "on") for assignment. Uses 
            default onband filter if input is unspecified (None) or if imagelist 
            for this filter key does not exist
          
        This function assigns gas amounts to the results of a cell search
        stored in ``self.search_results``, which is a 
        :class:`CellAutoSearchResults` object which contains image lists for
        all filters for which search was performed (e.g. "on", "off") 
        separated by background images (``self.search_results.bg_info``) and
        individual cells (``self.search_results.cell_info``).
        This function access the latter and assigns the gas amounts by measure
        of the magnitude of the corrseponding intensity decrease. 
        
        .. note::
        
            1. In order for this to work, the automatic cell search must have 
                been performed
            2. This function does not change class attributes which are
                actually used for calibration. These are stored in 
                ``self.cell_lists`` and ``self.bg_lists`` and have to be 
                assigned specifically
            
        """
        # check input list ID and set default if invalid
        if not self.lists_access_info.has_key(filter_id):
            self.filters.check_default_filters()
            filter_id = self.filters.default_key_on
        #the info about columns in the cells
        cell_info = self._cell_info_auto_search
        #init temporary dicts (will be filled below)
        offs_dict = {}
        cell_cd_dict = {}
        #the results of the cell search
        res = self.search_results.cell_info
        for val in res[filter_id].values():
            offs_dict[val.id] = val.offs.mean()
        
        #read the gas column amounts
        for key, val in cell_info.iteritems():
            cell_cd_dict[key] = val[0]
        #sort the dicionaries by column amount or intensity decrease
        s0 = sorted(offs_dict, key = offs_dict.get)
        s1 = sorted(cell_cd_dict, key = cell_cd_dict.get)
        print "Cell search keys sorted by depth of Dip: %s" %s0
        print "Cell amounts sorted by depth of Dip: %s" %s1
        filter_ids = res.keys()
        for k in range(len(s0)):
            cell_id = s1[k]
            gas_cd, gas_cd_err = cell_info[s1[k]][0], cell_info[s1[k]][1]
            print ("Search key: %s\nDel I: %s\nCell abbr: %s\nGasCol %s +/- %s"
                        %(s0[k], offs_dict[s0[k]], cell_id, gas_cd, gas_cd_err))
            #now add gas column to corresponding list in search result object
            for filter_id in filter_ids:
                res[filter_id][s0[k]].img_list.update_cell_info(cell_id, gas_cd,\
                                                                gas_cd_err)
    
    def add_search_results(self):
        """Add results from automatic cell detection to calibration
        
        This method analyses ``self.search_results`` for valid cell image lists
        (i.e. lists that contain images and have the gas column assigned)
                
        """
        for filter_id, info in self.search_results.bg_info.iteritems():
            self.add_bg_img_list(info.img_list)
        dff = self.filters.default_key_on
        #link all background image lists to default background image list such
        for filter_id, lst in self.bg_lists.iteritems():
            if filter_id != dff:
                self.bg_lists[dff].link_imglist(self.bg_lists[filter_id])
        for filter_id, cellDict in self.search_results.cell_info.iteritems():
            for cell_id, cell in cellDict.iteritems():
                lst = cell.img_list
                if lst.has_files() and lst.gas_cd > 0:
                    self.add_cell_img_list(lst)
            
            
    def find_and_assign_cells_all_filter_lists(self, threshold = 0.10):
        """High level function for automatic cell and background image search"""
        for filter_id in self.filters.filters.keys():
            try:
                self.find_cells(filter_id, threshold, False)
            except:
                self.find_cells(filter_id, threshold, True)
        
        self._assign_calib_specs()
        self.add_search_results()
        self.check_all_lists()
        self.cell_search_performed = 1
            
    def bg_img_available(self, filter_id):
        """Checks if a background image is available
        
        :param str filter_id: filter ID of image list
        """
        try:
            if isinstance(self.bg_lists[filter_id], Img):
                return True
            raise Exception
        except:
            self.check_image_lists(filter_id)
            if isinstance(self.bg_lists[filter_id], Img):
                return True
            return False
    
    def check_image_list(self, l):
        """Check if image list contains files and has images ready (loaded)
        
        :param ImgList l: image list object
        """
        if not l.nof > 0:
            raise IndexError("Error, image list %s does not contain images" 
                                                                        %l.list_id)
        if not isinstance(l.current_img(), Img):
            if not l.load():
                raise Exception("Unexpected error...")
        #raises Exception is gas column is not a number
        float(l.gas_cd)
        
    def check_all_lists(self):
        """Check if image lists for a given filter contain images and if images 
        are loaded if not, load images
        """
        filter_ids = self.cell_lists.keys()
        cell_ids = self.cell_lists[filter_ids[0]].keys()
        #get number of cells for first filter ID
        first_cell_num = len(self.cell_lists[filter_ids[0]])
        
        for filter_id in filter_ids:
            if not len(self.cell_lists[filter_id]) == first_cell_num:
                raise Exception("Mismatch in number of cells in "
                    "self.cell_lists between filter list %s and %s" 
                    %(filter_ids[0], filter_id))
            for cell_id in cell_ids:
                self.check_image_list(self.cell_lists[filter_id][cell_id])
            if not self.bg_lists.has_key(filter_id):
                raise KeyError("Error: BG image data (list) for filter ID %s "
                    "is not available" %filter_id)  
            else:
                self.check_image_list(self.bg_lists[filter_id])

        return True
        
    def check_cell_info_dict_autosearch(self, cell_info_dict):
        """Checks if dictionary including cell gas column info is right format
        
        :param dict cell_info_dict: keys: cell ids (e.g. "a57"), 
            values: list of gas column density and uncertainty in cm-2:
            ``[value, error]``
        """
        for key, val in cell_info_dict.iteritems():
            if not isinstance(key, str) and not isinstance(key, unicode):
                raise KeyError("Invalid key: %s" %key)
            if not isinstance(val, list):
                raise ValueError("Invalid cell column specification, need"
                    "list containing [value, uncertainty] of gas column with"
                    "id %s, got %s" %(key, val))
            else:
                if len(val) != 2:
                    raise ValueError("Invalid cell column specification, need"
                    "list containing [value, uncertainty] of gas column with"
                    "id %s, got %s" %(key, val))
                for k in range(len(val)):
                    if not isinstance(val[k], (int, long, float, complex)):
                        raise ValueError("Invalid data type for cell gas"
                            " column specification %s" %val)
    
    def set_cell_info_dict_autosearch(self, cell_info_dict):
        """Set attribute ``self._cell_info_auto_search`` (dictionary)
        
        :param dict cell_info_dict: dictionary containing cell information        
        """
        self.check_cell_info_dict_autosearch(cell_info_dict)
        self._cell_info_auto_search = cell_info_dict
    
    @property
    def cell_lists_ready(self):
        """Checks if all current cell lists contain images and gas CD info"""
        return self.check_all_lists()
    
    def prepare_calib_data(self, on_id, off_id, darkcorr=True, blurring=1,
                           pyrlevel=0):
        """High level function to prepare all calibration stacks (on, off, aa)
        
        :param str on_id: ID of onband filter used to determine calib curve
        :param str off_id: ID of offband filter
        :param bool darkcorr: perform dark correction before determining tau
            images
        :param int blurring: width of gaussian filter applied to tau images
        :param int pyrlevel: downscale factor
        :returns bool: success
        """
        ids = [on_id, off_id]
        self.check_all_lists()
        for filter_id in ids:
            if not self.prepare_tau_calib(filter_id, darkcorr, blurring,
                                          pyrlevel):
                print ("Failed to prepare cell calibration, check tau stack "
                    "determination for filter ID: %s" %filter_id)
                return 0
        if not self.prepare_aa_calib(on_id, off_id):
            print ("Failed to prepare cell AA calibration, check tau stack "
                    "determination for filter ID: %s" %filter_id)
            return 0
        return 1
    
    def prepare_tau_calib(self, filter_id, darkcorr=True, blurring=1,
                          pyrlevel=0):
        """Prepare a stack of tau images for input filter images
        
        Prepares a stack of tau images with each layer corresponding to one
        calibration cell image. The tau images are all prepared using the 
        same background image (self.bg_info[filter_id].img_list.current_img())
        
        :param str filter_id: ID of image lists used for tau calc 
            (e.g. "on", "off")
        :param bool darkcorr: bool specifying whether dark correction is
            supposed to be applied to data (True)
        :param int blurring: specify amount of gaussian blurring (1)
        :param int pyrlevel: Specify size reduction factor using gaussian 
            pyramide
        """

        bg_list = self.bg_lists[filter_id]
        bg_list.update_img_prep(blurring = blurring)
        bg_list.darkcorr_mode = darkcorr
        bg_mean_tseries = self.det_bg_mean_pix_timeseries(filter_id)
        bg_img = bg_list.current_img()
        bg_mean = bg_img.img.mean()
         
        h, w = subimg_shape(bg_list.current_img().img.shape,\
                                                pyrlevel = pyrlevel)
        num = len(self.cell_lists[filter_id])
        tau_stack = ImgStack(h, w, num, stack_id = filter_id)
        
        for cell_id, lst in self.cell_lists[filter_id].iteritems():
            lst.update_img_prep(blurring = blurring)
            lst.darkcorr_mode = True
            cell_img = lst.current_img()
            try:
                bg_mean_now = bg_mean_tseries.get_poly_vals(cell_img.meta[\
                                                                "start_acq"])
                offset = bg_mean - bg_mean_now
            except:
                warn("Warning in tau image stack calculation for filter "
                " %s: Time series data for background list (background poly) "
                " is not available. Calculating tau image for cell image  %s, "
                " %s based on unchanged background image recorded at %s"
                %(filter_id, cell_id, cell_img.meta["start_acq"],\
                                                bg_img.meta["start_acq"]))
                    
                offset = 0.0
            
            bg_img = bg_img - offset
            tau_img = cell_img.duplicate()
            if bg_img.edit_log["darkcorr"] != cell_img.edit_log["darkcorr"]:
                raise ImgMetaError("Fatal: cannot determine tau stack, bg "
                    "image and cell image have different darkcorr modes")
            tau_img.img = log(bg_img.img / cell_img.img)
            
            tau_img.to_pyrlevel(pyrlevel)
            tau_stack.append_img(tau_img.img, start_acq =\
                    cell_img.meta["start_acq"], texp = cell_img.meta["texp"],\
                                    add_data = lst.gas_cd)
        tau_stack.img_prep.update(tau_img.edit_log)
        self.calib_data[filter_id] = CellCalibData(tau_stack=tau_stack,
                                                   calib_id=filter_id)
        return tau_stack
    
    def prepare_aa_calib(self, on_id="on", off_id="off", calib_id="aa"):
        """Prepare stack containing AA images
        
        :param str on_id ("on"): ID of on band filter
        :param str off_id ("off"): ID of offband filter
        :param str calib_id ("aa"): ID of AA image stack
        
        The imagery data is retrieved from ``self.calib_data`` so, before calling
        this function, make sure, the corresponding on and offband stacks were
        created using :func:`prepare_tau_calib`
        
        The new AA stack is added to ``self.calib_data`` dictionary
        
        """
        try:
            aa_calib = self.calib_data[on_id] - self.calib_data[off_id]
        except:
            warn("Tau on and / or tau off calib data is not available for AA "
                "stack calculation, trying to determine these ... ")
            self.prepare_tau_calib(on_id, darkcorr=True)
            self.prepare_tau_calib(off_id, darkcorr=True)
            aa_calib = self.calib_data[on_id] - self.calib_data[off_id]
        aa_calib.calib_id  = "aa"
        self.calib_data[calib_id] = aa_calib
        return aa_calib
    
    def get_sensitivity_corr_mask(self, calib_id="aa", **kwargs):
        """Get sensitivity correction mask 
        
        For a detailed description see corresponding method in 
        :class:`CellCalibData`.
            
        :param str calib_id: mask is determined from the corresponding calib
            data (e.g. "on", "off", "aa")
        :param **kwargs: keyword args (see corresponding method in 
            :class:`CellCalibData`)
            
    
        """
        if not calib_id in self.calib_data.keys():
            raise ValueError("%s calibration data is not available" %calib_id)
        return self.calib_data[calib_id].get_senitivity_corr_mask(**kwargs)
          
    def get_calibration_polynomial(self, calib_id="aa", **kwargs):
        """Retrieve calibration polynomial within a certain pixel neighbourhood
        
        :param str calib_id: image type ID (e.g. "on", "off")
        :param **kwargs: additional keyword arguments passed to :func:`poly`
            of :class:`CellCalibData` object corresponding to ``calib_id``.
        :returns: tuple containing 
            - poly1d, fitted polynomial
            - ndarray, array with tau values 
            - ndarray, array with corresponding gas CDs
        
        """
        return self.calib_data[calib_id].poly(**kwargs)
    
    """
    Redefinitions
    """
    def get_list(self, list_id, cell_id=None):
        """Expanding funcionality of this method from :class:`Dataset` object
        
        :param str list_id: filter ID of list (e.g. on, off). If parameter 
            ``cell_id`` is None, then this function returns the initial
            Dataset list (containing all images, not the ones separated by 
            cells / background).
        :param cell_id: if input is specified (type str) and valid (available
            cell img list), then the corresponding list is returned which only
            contains images from this cell. The string "bg" might be used to 
            access the background image list of the filter specified with 
            parameter ``list_id``
        :return: - ImgList, the actual list object
        """
        if cell_id is not None and isinstance(cell_id, str):
            if cell_id in self.cell_lists[list_id].keys():
                return self.cell_lists[list_id][cell_id]
            elif cell_id == "bg":
                return self.bg_lists[list_id]
        return super(CellCalibEngine, self).get_list(list_id)
        
    """
    Plotting etc
    """           
    def plot_cell_search_result(self, filter_id = "on", for_app = 0,\
                                        include_tit = True, ax = None):
        """High level plotting function for results from automatic cell search
        
        :param str filter_id ("on"): image type ID (e.g. "on", "off"), i.e. 
        usually ID of filter used
            
        """
        # get stored time series (was automatically saved in :func:`find_cells`)
        ts_all = self.pix_mean_tseries[("%s_auto_search" %filter_id)]
        # get cell search results
        res = self.search_results
        if not res.cell_info.has_key(filter_id) or\
                                len(res.cell_info[filter_id]) < 1:
            print ("Error plotting cell search results: no results found...")
            return 0
        if for_app:
            fig = Figure()#figsize = (16, 6))
            ax = fig.add_subplot(111)
        else:
            if ax is None:
                fig, ax = subplots(1,1)
        
        ts_all.plot(include_tit = include_tit, ax = ax)
        info = res.cell_info[filter_id]
        ts = ts_all.index
        dt = timedelta(0, (ts[-1] - ts[0]).total_seconds() /\
                                                (len(ts_all) * 10))
        for cell in info.values():                
            lbl = "Cell %s: %.2e cm-2" %(cell.img_list.cell_id,\
                                            cell.img_list.gas_cd)
            p = ax.plot(cell.start_acq, cell.mean_vals,' o', ms = 10,\
                label = lbl, markeredgecolor = "None",\
                markeredgewidth = 1, alpha = 0.6)
            c = p[0].get_color()
            ax.fill_betweenx(arange(0, ts_all.max()*1.05, 1), cell.start - dt,\
                                    cell.stop + dt, facecolor = c, alpha = 0.1)
    
        if filter_id in res.bg_info.keys():
            bg_info = res.bg_info[filter_id]
            ax.plot(bg_info.start_acq, bg_info.mean_vals,' o', ms = 10,\
                markerfacecolor = "None", markeredgecolor = 'c',\
                mew = 2,label = 'BG image candidates')
            ts = PixelMeanTimeSeries(bg_info.mean_vals, bg_info.start_acq)
            ts.fit_polynomial(2)
            bg_poly_vals = ts.get_poly_vals(bg_info.start_acq,
                                            ext_border_secs=30)
                                            
            ax.plot(bg_info.start_acq, bg_poly_vals,'-', c = 'lime', lw = 2,
                                                label = 'Fitted BG polynomial')
            
            cfn = bg_info.img_list.cfn
            ax.plot(bg_info.start_acq[cfn], bg_info.mean_vals[cfn],\
            ' +r', ms = 14, mew = 4, label = 'Current BG image')
            
        ax.legend(loc = 4, fancybox = True, framealpha = 0.5, fontsize=10)
        ax.set_ylabel("Avg. pixel intensity", fontsize = 16)
        return ax
    
    def plot_calib_curve(self, calib_id, **kwargs):
        """Plot calibration curve 
        
        :param str filter_id: image type ID (e.g. "aa")
        :param **kwargs: plotting keyword arguments passed to :func:`plot` of
            corresponding :class:`CellCalibData` object
            
        """
        return self.calib_data[calib_id].plot(**kwargs)
        
    
    def plot_all_calib_curves(self, ax=None, **kwargs):
        """Plot all available calibration curves in a certain pixel region
        
        :param **kwargs: plotting keyword arguments passed to 
            :func:`get_calibration_polynomial` of corresponding 
            :class:`CellCalibData` objects
            
        """
        if ax is None:
            fig, ax = subplots(1,1)
        tau_max = -10
        y_min = 1e20
        for calib_id, calib in self.calib_data.iteritems():
            poly, tau, so2 = calib.poly(**kwargs)
            
            taus = linspace(0, tau.max() * 1.2, 100)
            pl = ax.plot(tau, so2, " ^", label = "Data %s" %calib_id)
            ax.plot(taus, poly(taus),"-", color=pl[0].get_color(),
                    label = "Poly %s" %poly_str(poly))
            tm = tau.max()
            if tm > tau_max:
                tau_max = tm
            if poly(0) < y_min:
                y_min = poly(0)
                
        ax.set_ylabel(r"S$_{SO2}$ [cm$^{-2}$]", fontsize=18)
        ax.set_xlabel(r"$\tau$", fontsize = 18)
        ax.set_ylim([y_min - so2.min() * 0.1, so2.max()*1.05])
        ax.set_xlim([0, tau_max * 1.05])
        ax.grid()
        ax.legend(loc = "best", fancybox = True,\
                            framealpha = 0.5, fontsize = 10)
        return ax
    
    def __call__(self, value, calib_id="aa", **kwargs):
        """Apply calibration to input value (i.e. convert into gas CD)
        
        :param float value: tau or AA value
        :param str calib_id: ID of calibration data supposed to be used
        :param **kwargs: keyword arguments to extract calibration information
            (e.g. pos_x_abs, pos_y_abs, radius_abs)
        :return: corresponding column density
        """
        return self.calib_data[calib_id](value, **kwargs)
        
def poly_str(poly):
    """Return custom string representation of polynomial"""
    exp = exponent(poly.coeffs[0])
    p = poly1d(round(poly / 10**(exp - 2))/10**2)
    return "%s E%+d" %(p, exp)

        

