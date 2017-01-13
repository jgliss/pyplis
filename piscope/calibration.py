# -*- coding: utf-8 -*-
from matplotlib.pyplot import subplots, subplot,draw,tight_layout
from numpy import float, log, arange, polyfit, poly1d, linspace, isnan,\
                                            isfinite, diff, mean, argmin
from traceback import format_exc
from pandas import Series

from copy import deepcopy
from matplotlib.patches import Circle
from matplotlib.pyplot import Figure
from datetime import timedelta
from os.path import exists

from piscope import PYDOASAVAILABLE
if PYDOASAVAILABLE:
    from pydoas.analysis import DoasResults

from .dataset import Dataset
from .setupclasses import AutoCellCalibSetup
from .processing import ImgStack, PixelMeanTimeSeries#, ImgListStack, ImagePreparation
from .imagelists import ImgList, CellImgList
from .exceptions import CellSearchError
from .image import Img
from .helpers import subimg_shape, map_coordinates_sub_img
id
class CalibrationCollection(object):
    """High level class combined DOAS and cell calibration stuff
    
    .. todo:: 
    
        Write some more when finished
        
    """
    def __init__(self, cell_calib = None, doas_calib = None):
        """Class initialisation"""
        self._calib_objects = {"cell"   :   None,
                               "doas"   :   None}
        
        try:
            self.cell = cell_calib
        except:
            pass
        try:
            self.doas = doas_calib
        except:
            pass
    
    @property
    def cell(self):
        """Getter for cell calibration object"""
        return self._calib_objects["cell"]
    
    @cell.setter
    def cell(self, value):
        """Setter for cell calibration object"""        
        if not isinstance(value, CellCalibEngine):
            raise TypeError("Need CellCalibEngine object")
        self._calib_objects["cell"] = value
    
    @property
    def doas(self):
        """Getter for cell calibration object"""
        return self._calib_objects["doas"]
    
    @doas.setter
    def doas(self, value):
        """Setter for cell calibration object"""        
        if not isinstance(value, DoasCalibEngine):
            raise TypeError("Need DoasCalib object")
        self._calib_objects["doas"] = value
    
    def get_cell_calib_poly(self, filter_id, pos_x, pos_y, radius = 1):
        """Retrieve cell calibration polynomial for a certain filter
        
        Get the cell calibration for one of the camera filters (or aa) within
        a specific image region. Calls :func:`get_calibration_polynomial`
        
        :param str filter_id: filter ID (e.g. "on")
        :param int pos_x: pixel X coordinate 
        :param int pos_y: pixel Y coordinate         
        :param int radius: get tau values for calibration for using the mean 
            tau of all pixels within radial mask around centre pixel 
            (default = 1, i.e. for a single pixel)
        
        :returns: list, containing
            - polynomial
            - ndarray, tau data array
            - ndarray, so2 column data array]
        
        """
        return self.cell.get_calibration_polynomial(filter_id, pos_x, pos_y,\
                                                                        radius)
    
    def plot_calibration_curves(self, pos_x, pos_y, radius = 1, on_id = "on",\
                                                 off_id = "off", aa_id = "aa"):
                                    
        """High level function to plot calibration curves for all filters
        
        .. todo::
            
            Include params
            
        """
        fig, axes = subplots(3, 1)
        ids = [on_id, off_id, aa_id]
        for k in range(3):
            self.plot_calibration_curve_overlay(ids[k], pos_x, pos_y, radius,\
                                                                ax = axes[k])
        
    def plot_calibration_curve_overlay(self, filter_id, pos_x_cell, pos_y_cell,\
            radius_cell = 1, ax = None):
        """Plot all available calibration polynomials
        
        :param str filter_id: filter ID (e.g. "on")
        :param int pos_x_cell: x position of pixel for cell calibration
        :param int pos_y_cell: y position of pixel for cell calibration
        :param int radius_cell: radius of cell calibration retrieval 
            (default = 1) 
        :param list tau_range: plotted min / max range for tau values
        
        """
        if ax is None:
            ax = subplot(1,1,1)
        #set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))
        polys = {}
        print ("Plotting calibration curve of filter " + str(filter_id) + " @ (x|y) "
            + "(" + str(pos_x_cell) + "|" + str(pos_y_cell) + "), radius: " + \
                                                        str(radius_cell) + "\n")
        calib = self.doas
        dev_id = calib.dev_id
        
        if calib.calib_available(filter_id):
            c = calib.calibration[filter_id]
            sx = c["tau_series"]
            sy = c["doas_cd_series"]
            sx, sy = calib.merge_series_interval(sx, sy)
            p = ax.plot(sx, sy," x",label = dev_id + " data")
            xp = linspace(sx.min(), sx.max(), 10)
            poly = c["poly"]
            ax.plot(xp, poly(xp), "--", color = p[0].get_color(), label =\
                                                            dev_id + " poly")
            polys["doas"] = poly
        
        # this block needs an exception handle
        poly_c, tau_c, cd_c = self.cell.get_calibration_polynomial(\
                            filter_id, pos_x_cell, pos_y_cell, radius_cell)
        
        p = ax.plot(tau_c, cd_c," o",label = "Cell data")
        xp = linspace(tau_c.min(), tau_c.max(), 10)
        ax.plot(xp, poly_c(xp), "--", color = p[0].get_color(),\
                                                label = "Cell poly")
        polys["cell"] = poly_c

#==============================================================================
#         ax.set_xlim(tau_range)
#         ax.set_ylim(so2Range)
#==============================================================================
        ax.legend(loc = "best", fancybox = True, framealpha = 0.5).draggable()
        
        ax.set_title("Calibration curves: " + str(filter_id))
        ax.set_xlabel("tau")
        ax.set_ylabel("Gas CDs [cm-2]")
        tight_layout()
        ax.grid()
        
        return ax, polys
    
    def __str__(self):
        """String representation"""
        return "Coming soon..."
#==============================================================================
#     def show_fovs_in_image(self):
#         try:
#             id0=self.spectral.keys()[0]
#             img=self.spectral[id0].rawInput.img_lists.on.loadedImages.this.img
#             fig,ax=subplots(1,1)
#             ax.imshow(img,cmap="gray")
#             self.insert_fovs_axes(ax)
#             return fig, ax
#         except:
#             raise()
#             
#     def insert_fovs_axes(self,ax):
#         colors=["c","r","y","lime"]
#         k=0
#         for id,calib in self.spectral.iteritems():
#             x,y=calib.fov.pos_abs
#             r=calib.fov.radius
#             if r ==1:
#                 r=2
#             c = Circle((x, y), r, color=colors[k],linewidth=1, fill=False)
#             ax.add_patch(c)
#             k+=1
#             if k==4:
#                 k=0
# 
#==============================================================================
        
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
        """Fill values using all images from a specific image list"""
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
        return mean(self.mean_vals_err)
    
    @property
    def mid_point_val(self):
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
        
class CellCalibEngine(Dataset):
    """Class representing cell calibration functionality
    
    This class id designed to define datasets related to time windows, where 
    calibration cells were put in front of lense. It provides functionality to
    define the cells used (i.e. SO2-SCDs in each cell) and to identify sub time
    windows corresponding to the individual cells. Inherits from 
    :class:`piscope.Datasets.Dataset`
    """
    def __init__(self, setup = None, init = 1):
        print 
        print "INIT CALIB DATASET OBJECT"
        print
        super(CellCalibEngine, self).__init__(setup, init)
                        
        self.type = "cell_calib"
        
        self.cell_search_performed = 0
        self._cell_info_auto_search = {}
        
        if isinstance(self.setup, AutoCellCalibSetup):
            self.set_cell_info_dict_autosearch(self.setup.cell_info_dict)
        
        self.cell_lists = {}
        self.bg_lists = {}
        
        self.search_results = CellAutoSearchResults()
        
        self.pix_mean_tseries = {}
        self.bg_tseries = {}
        
        self.tau_stacks = {}
        
        self.warnings = []
 
        print 
        print "FILELISTS IN CALIB DATASET OBJECT INITIALISED"
        print
    
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
            raise TypeError("Error adding bg image list, need ImgList object, got "
                "%s" %type(lst))
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
        if filter_id in self.search_results.bg_info.keys():
            info = self.search_results.bg_info[filter_id]
            ts = PixelMeanTimeSeries(info.mean_vals, info.start_acq,\
                info.mean_vals_err, info.texps)
        else:
            ts = self.bg_lists[filter_id].get_mean_value()
        ts.fit_polynomial()
        self.bg_tseries[filter_id] = ts
            
    def add_cell_images(self, img_paths, cell_gas_cd, cell_id, filter_id):
        """Add list corresponding to cell measurement
        
        :param list img_paths: list containing imagery file paths
        :param float cell_gas_cd: column amount of gas in cell
        :param str cell_id: string identification of cell
        :param str filter_id: filter ID for images (e.g. "on", "off")
        
        .. note:: 
        
            No input check performed
            
        """
        try:
            if exists(img_paths): #input is not a list but a valid path
                img_paths = [img_paths,]
        except:
            pass
        
        paths = [p for p in img_paths if exists(p)]
        if not len(paths) > 0:
            raise TypeError("No valid filepaths could be identified")
        
        lst = CellImgList(files = paths, list_id = filter_id, camera =\
            self.camera, cell_id = cell_id, gas_cd = cell_gas_cd)
        
        
    def find_cells(self, filter_id = "on", threshold = 0.10, accept_last_in_dip = False):
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
        x, y, yErr, texps = ts.index, ts.values, ts.std, ts.texps
        #this will be overwritten in the loop to find the BG image with the 
        #lowest standard deviation, which is then set as current bg image
        #yErrCurrentBG = 9999 
        ydiff = diff(y) #1st derivative (finite differences)
        y_max = max(y)
        bg_info = CellSearchInfo(filter_id, "bg", y_max)
        rest = CellSearchInfo(filter_id, "rest", y_max)
        cell_info = {} #will be filled with CellSearchInfo objects in the loop        
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
                    raise Exception("Fatal error, found cell dip within cell dip")
                    
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
                result.mean_vals_err.append(yErr[k])
                result.file_paths.append(l.files[k])
                result.start_acq.append(x[k])
                result.texps.append(texps[k])
                cell_info[result.id] = result
                #onFilter=0
            elif bg_cond:
                print "Found BG candidate at %s, %s" %(k,x[k])
                bg_info.mean_vals.append(y[k])
                bg_info.mean_vals_err.append(yErr[k])
                bg_info.file_paths.append(l.files[k])
                bg_info.start_acq.append(x[k])
                bg_info.texps.append(texps[k])
            else: 
                if on_cell:
                    result.mean_vals.append(y[k])
                    result.mean_vals_err.append(yErr[k])
                    result.file_paths.append(l.files[k])
                    result.start_acq.append(x[k])
                    result.texps.append(texps[k])
                else:
                    rest.mean_vals.append(y[k])
                    rest.mean_vals_err.append(yErr[k])
                    rest.file_paths.append(l.files[k])
                    rest.start_acq.append(x[k])
                    rest.texps.append(texps[k])
            k += 1
        
        if not len(self._cell_info_auto_search.keys()) == len(cell_info.keys()):
            raise CellSearchError("Number of detected cells (%s) is different "
                "from number of cells specified in cellSpecInfo (%s) " 
                %(len(self._cell_info_auto_search.keys()), len(cell_info.keys())))
        
        bg_info.create_image_list(self.camera)
        bg_info.img_list.update_cell_info("bg", 0.0, 0.0)
        self.assign_dark_offset_lists(into_list = bg_info.img_list)
        for cell_id, info in cell_info.iteritems():
            info.create_image_list(self.camera)
            self.assign_dark_offset_lists(into_list = info.img_list)
            
        self.search_results.add_cell_search_result(filter_id, cell_info,\
                                                                bg_info, rest)
        self.store_pixel_mean_time_series(ts, ("%s_autoSearch" %filter_id))
        bg_info.img_list.change_index(argmin(bg_info.mean_vals_err))
        
        
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
#==============================================================================
#                 if not self.cell_lists.has_key(filter_id):
#                     self.cell_lists[filter_id] = {}
#==============================================================================
                
                #self.add_cell_img_list(res[filter_id][s0[k]].img_list)
        
    
    def add_search_results(self):
        """Add results from automatic cell detection to calibration
        
        This method analyses ``self.search_results`` for valid cell image lists
        (i.e. lists that contain images and have the gas column assigned)
                
        """
        for filter_id, info in self.search_results.bg_info.iteritems():
            self.add_bg_img_list(info.img_list)
            self.det_bg_mean_pix_timeseries(filter_id)
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
                try:
                    self.find_cells(filter_id, threshold, True)
                except:
                    raise
        self._assign_calib_specs()
        self.add_search_results()
        self.check_all_lists()
        self.cell_search_performed = 1
        
#==============================================================================
#     def copy_dark_offset_info(self, into_list, fromListKey):
#         """Copy dark and offset infor from one list into another
#         
#         :param str fromListKey: key of base list (``self.get_list(key)``)
#         :param BaseImgList into_list: destination list
#         
#         """
#         into_list.darkLists = self.get_list(fromListKey).darkLists      
#         into_list.offsetLists = self.get_list(fromListKey).offsetLists        
#         print ("Succesfully copied dark and offset info from %s to %s " 
#                                                 %(fromListKey, into_list.id))
#==============================================================================
#==============================================================================
#     def ready_2_go(self,filter_id):
#         if filter_id in self.stacks.tau.keys():
#             return 1
#         return 0
#==============================================================================
            
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
        cellNum0 = len(self.cell_lists[filter_ids[0]])
        
        for filter_id in filter_ids:
            if not len(self.cell_lists[filter_id]) == cellNum0:
                raise Exception("Mismatch in number of cells in self.cell_lists "
                    " between filter list %s and %s" %(filter_ids[0], filter_id))
            for cell_id in cell_ids:
                print "filter_id: " + str(filter_id) + ", CellId: " + str(cell_id)
                self.check_image_list(self.cell_lists[filter_id][cell_id])
            if not self.bg_lists.has_key(filter_id):
                raise KeyError("Error: BG image data (list) for filter ID %s "
                    "is not available" %filter_id)  
            else:
                self.check_image_list(self.bg_lists[filter_id])

        return True
        
    def check_cell_info_dict_autosearch(self, cell_info_dict):
        """Checks if dictionary including cell gas column info is in right format
        
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
    
    def create_background_dataset(self):
        """Creates a :class:`piscope.dataset.Dataset` object from bg image data
        """
        raise NotImplementedError
    def prepare_calibration_stacks(self, on_id, off_id, darkCorr = True,
            blurring = 1):
        """High level function to prepare all stacks etc to retrieve the actual
        calibration
        
        :param str on_id: ID of onband filter used to determine calib curve
        :param str off_id: ID of offband filter
        :returns bool: success
        """
        ids = [on_id, off_id]
        self.check_all_lists()
        for filter_id in ids:
            if not self.prepare_tau_stack(filter_id):
                print ("Failed to prepare cell calibration, check tau stack "
                    "determination for filter ID: " + str(filter_id))
                return 0
        if not self.prepare_aa_stack(on_id, off_id):
            print ("Failed to prepare cell AA calibration, check tau stack "
                    "determination for filter ID: " + str(filter_id))
            return 0
        return 1
       
    def prepare_tau_stack(self, filter_id, darkcorr = True, blurring = 1,\
                                                            pyrlevel = 0):
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
            pyramide (2)
        
        """

        bg_list = self.bg_lists[filter_id]
        bg_list.update_img_prep(blurring = blurring)
        bg_list.activate_dark_corr()
        #prep = bg_list.img_prep.update({"pyrlevel" : pyrlevel})
        
        bg_img = bg_list.current_img()
        bg_mean = bg_img.img.mean()
        bg_mean_tseries = self.bg_tseries[filter_id]
        h, w = subimg_shape(bg_list.current_img().img.shape,\
                                                pyrlevel = pyrlevel)
        num = len(self.cell_lists[filter_id])
        tau_stack = ImgStack(h, w, num, stack_id = filter_id)
        tau_stack.img_prep.update(bg_list.img_prep, darkcorr = darkcorr,\
                                                      pyrlevel = pyrlevel)
        
        for cell_id, lst in self.cell_lists[filter_id].iteritems():
            lst.update_img_prep(blurring = blurring)
            lst.activate_dark_corr()
            cell_img = lst.current_img()
            try:
                bg_mean_now = bg_mean_tseries.get_poly_vals(cell_img.meta["start_acq"])
                offset = bg_mean - bg_mean_now
            except:
                print ("Warning while calculating tau image stack for filter "
                " %s: Time series data for background list (background poly) "
                " is not available. Calculating tau image for cell image  %s, "
                " %s based on unchanged background image recorded at %s"
                %(filter_id, cell_id, cell_img.meta["start_acq"],\
                                                bg_img.meta["start_acq"]))
                    
                offset = 0.0
            bg_data = bg_img.img - offset
            tau_img = Img(log(bg_data / cell_img.img))
            tau_img.pyr_down(pyrlevel)
            tau_stack.append_img(tau_img.img, start_acq =\
                    cell_img.meta["start_acq"], texp = cell_img.meta["texp"],\
                                    add_data = lst.gas_cd)
        
        self.tau_stacks[filter_id] = tau_stack
    
    def prepare_aa_stack(self, on_id = "on", off_id = "off", stack_name = "aa"):
        """Prepare stack containing AA images
        
        :param str on_id ("on"): ID of on band filter
        :param str off_id ("off"): ID of offband filter
        :param str stack_name ("aa"): ID of AA image stack
        
        The imagery data is retrieved from ``self.tau_stacks`` so, before calling
        this function, make sure, the corresponding on and offband stacks were
        created using :func:`prepare_tau_stack`
        
        The new AA stack is added to ``self.tau_stacks`` dictionary
        
        """
        aa = self.tau_stacks[on_id] - self.tau_stacks[off_id]
        self.tau_stacks[stack_name] = aa
        return aa
                
    def get_calibration_polynomial(self, filter_id, pos_x_abs, pos_y_abs,\
                                radius_abs = 1, mask = None, polyorder = 1):
        """Retrieve calibration polynomial within a certain pixel neighbourhood
        
        :param str filter_id: image type ID (e.g. "on", "off")
        :param int pos_x_abs: detector x position (col) in absolute detector 
                                                                    coords
        :param int pos_y_abs: detector y position (row) in absolute detector 
                                                                    coords
        :param float radius_abs: radius of pixel disk on detector (centered
            around pos_x, pos_y, default: 1)
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
        stack = self.tau_stacks[filter_id]
        lvl = stack.img_prep["pyrlevel"]
        print lvl
        roi = [0, 0, 9999, 9999]
        if stack.img_prep.has_key("roi"):
            roi = stack.img_prep["roi"]
        x_rel, y_rel = map_coordinates_sub_img(pos_x_abs, pos_y_abs, roi,\
                                                   stack.img_prep["pyrlevel"])
        rad_rel = radius_abs / 2**stack.img_prep["pyrlevel"]
        print "ABS: %s, %s, %s" %(pos_x_abs, pos_y_abs, radius_abs)
        print "REL: %s, %s, %s" %(x_rel, y_rel, rad_rel)
        tau_arr = stack.get_time_series(x_rel, y_rel,\
                                        rad_rel, mask).values
        so2_arr = stack.add_data
        return poly1d(polyfit(tau_arr, so2_arr, polyorder)), tau_arr, so2_arr
    
    """
    Plotting etc
    """           
    def plot_cell_search_result(self, filter_id = "on", for_app = 0,\
                                        include_tit = True, ax = None):
        """High level plotting function for results from automatic cell search
        
        :param str filter_id ("on"): image type ID (e.g. "on", "off"), i.e. usually
            ID of filter used
            
        """
        # get stored time series (was automatically saved in :func:`find_cells`)
        ts_all = self.pix_mean_tseries[("%s_autoSearch" %filter_id)]
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
        dt = timedelta(0, (ts[-1] - ts[0]).total_seconds() / (len(ts_all) * 10))
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
                mew = 2,label = 'BG image canditates')
                
            bg_poly_vals = self.bg_tseries[filter_id].get_poly_vals(\
                                    bg_info.start_acq, ext_border_secs = 30)
            ax.plot(bg_info.start_acq, bg_poly_vals,'-', c = 'lime', lw = 2,
                                                label = 'Fitted BG polynomial')
            
            cfn = bg_info.img_list.cfn
            ax.plot(bg_info.start_acq[cfn], bg_info.mean_vals[cfn],\
            ' +r', ms = 14, mew = 4, label = 'Current BG image')
            
        ax.legend(loc = 4, fancybox = True, framealpha = 0.5).draggable()
        ax.set_ylabel("Avg. pixel intensity", fontsize = 16)
        return ax.figure, ax
    
    def plot_all_calib_curves(self, pos_x = None, pos_y = None, radius = 1,\
                                                    mask = None, ax = None):
        """Plot all available calibration curves in a certain image pixel region
        
        :param str filter_id: image type ID (e.g. "on", "off")
        :param int pos_x (None): x position of center pixel on detector
        :param int pos_y (None): y position of center pixel on detector
        :param float radius (1): radius of pixel disk on detector (centered
            around pos_x, pos_y)
        :param ndarray mask (None): boolean mask for image pixel access, 
            default is None, if the mask is specified and valid (i.e. same
            shape than images in stack) then the other three input parameters
            are ignored
        :param ax (None): matplotlib axes (if None, a new figure with axes
            will be created)
            
        """
        if ax is None:
            fig, ax = subplots(1,1)
        tau_max = -10
        y_min = 1e20
        for filter_id, stack in self.tau_stacks.iteritems():
            poly, tau, so2 = self.get_calibration_polynomial(filter_id, pos_x,\
                                                        pos_y, radius, mask)
            
            taus = linspace(0, tau.max() * 1.2, 100)
            pl = ax.plot(tau, so2, " ^", label = "Data %s" %filter_id)
            ax.plot(taus, poly(taus),"-", color = pl[0].get_color(), label =\
                                                                "Poly %s" %filter_id)
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
        ax.legend(loc = "best", fancybox = True, framealpha = 0.5, fontsize = 14)
        return ax
        
#==============================================================================
#     def plot_calib_curve(self, filter_id, pos_x = None, pos_y = None, radius = 1,\
#                                                     mask = None, ax = None):
#         """Plot the calibration curve in a certain image pixel region
#         
#         :param str filter_id: image type ID (e.g. "on", "off")
#         :param int pos_x (None): x position of center pixel on detector
#         :param int pos_y (None): y position of center pixel on detector
#         :param float radius (1): radius of pixel disk on detector (centered
#             around pos_x, pos_y)
#         :param ndarray mask (None): boolean mask for image pixel access, 
#             default is None, if the mask is specified and valid (i.e. same
#             shape than images in stack) then the other three input parameters
#             are ignored
#         :param ax (None): matplotlib axes (if None, a new figure with axes
#             will be created)
#             
#         """
#         if ax is None:
#             fig, ax = subplots(1,1)
#         #self.tau_stacks[filter_id].get_time_series(pos_x, pos_y, radius, mask)
#         poly, tau_arr, so2_arr = self.fit_calib_poly(pos_x,pos_y,filter_id,radius)
#         ax=subplot(1,1,1)
#         ax.plot(tau_arr,so2_arr," ob",label="Cell data " + str(filter_id))
#         xp=linspace(0,0.5,10)
#         ax.plot(xp,poly(xp), "--r", label="Cell poly " + str(filter_id))            
#         ax.set_title("Cell calib " + filter_id + \
#         " at pos (x,y,r) : (" + str(pos_x) + "," + str(pos_y) + "," + str(radius) + ")")
#         ax.set_xlabel("Tau")
#         ax.set_ylabel("SO2-SCD [cm-2]")
#         ax.grid()
#         ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
#         return ax
#==============================================================================
            
#==============================================================================
#     def plot_tau_image(self, cellAcro, filter_id = "on", scaleMinMax = 1):
#         """Plot tau image of one specific cell
#         
#         :param str cellAcro: cell ID of this image
#         """
#         if filter_id, stack in self.tau_stacks.iteritems():
#             tauMin=stack.min()
#             tau_max=stack.max()
#             fig,axes=subplots(1,3, figsize=(18,5))
#             for k in range(self.numberOfCells):
#                 tit=("Cell " + str(self.stacks.suppl.abbr[k]) + ", SO2-SCD: "
#                     + str(self.stacks.suppl.so2[k]) + " cm-2")
#                 if scaleMinMax:
#                     disp=axes[k].imshow(stack[:,:,k], vmin=tauMin, vmax=tau_max)
#                 else:
#                     disp=axes[k].imshow(stack[:,:,k])
#                 axes[k].set_title(tit)
#                 fig.colorbar(disp,ax=axes[k])
#             return fig,axes
#==============================================================================
            
    def store_pixel_mean_time_series(self, pix_mean_ts, save_id):
        """Option to store :class:`PixelMeanTimeSeries` in this class
        
        :param PixelMeanTimeSeries pix_mean_ts: the object to be saved
        :param str save_id: string ID used for storing

        The object is saved in ``self.pix_mean_tseries`` dictionary
        
        .. note::
        
            existing objects with same ID will be overwritten without 
            warning
            
        """
        if not isinstance(pix_mean_ts, PixelMeanTimeSeries):
            raise TypeError("Invalid input type %s: need PixelMeanTimeSeries "
                " object" %type(pix_mean_ts))
        self.pix_mean_tseries[save_id] = pix_mean_ts

        
class DoasCalibData(object):
    """Object to store DOAS calibration data"""
    def __init__(self, poly = None, filter_id = None, tau_series = None,\
                            doas_cd_series = None, doas_fov = None):
        """Class initialisation
        
        :param poly1d poly: calibration polynomial object 
        :param str filter_id: ID of filter (e.g. "on", "off")
        :param Series tau_series: pandas time series containing image tau 
            data for a considered time period from images recorded with filter
            specified with input param "filter_id". The tau data is supposed
            to be convolved with the FOV parametrisation of the DOAS instrument
            within the image detector and merged in time with the DOAS data 
            (i.e. same number of data points as input param ``doas_cd_series``)
        :param Series doas_cd_series: DOAS column density time series data. The
            data needs to be merged in time with input tau data
        :param DoasFOV doas_fov: DOAS FOV information
        """
        
        self.filter_id = filter_id
        self.tau_series = tau_series
        self.doas_cd_series = doas_cd_series
        self.doas_fov = doas_fov
        self.poly = poly
        
class DoasCalibEngine(object):
    """Class for determination of camera calibration curve using DOAS data"""
    DOAS_START_STOP_AVAILABLE = False    
    def __init__(self, doas_data, *img_lists):
                                        
        """Class initiation
        
        :param pandas.Series doas_data: DOAS column density time series
             (e.g. SO2 CDs)
        :param *img_lists: image list object
        
        """                
        self.doas_data = doas_data
        self.img_lists = {}
        self.calibration = {} # will be filled with DoasCalibData objects
        
        if PYDOASAVAILABLE and isinstance(doas_data, DoasResults): 
            if doas_data.has_start_stop_timestamps():
                DOAS_START_STOP_AVAILABLE = True
        
    def add_img_lists(self, *img_lists):
        """Add image lists to this calibration object"""
        for lst in img_lists:
            if isinstance(lst, ImgList) and lst.data_available:
                self.img_lists[lst.list_id] = lst
                
    def get_time_fov_info_file(self):
        """Get the datetime corresponding to current FOV file"""
        raise NotImplementedError
      
    def perform_fov_search(self, list_id = "on", from_tau_imgs = True,\
            data_merge_type = "averaging", method = "pearson"):
        """High level function to search the FOV 
        
        The main analysis steps are:
            
            1. Get image list for specified filter
        
        Perform search of FOV of spectrometer within FOV of camera (High level)
        
        Calls::
        
            self.searchFovTool.perform_fov_search()
        
        """
        raise NotImplementedError
        
    def load_fov_search_results(self):
        """
        Load the information about the FOV of spectrometer from another
        :class:`SpectralCalibration`
        """
        if not self.searchFovTool.searchInfo.results_available():
            msg=("Could not load results from FOV search, not all necessary "
            "results are available")
            print msg
            self.searchFovTool.searchInfo.print_result_overview()
            return
        
        self.fov = deepcopy(self.searchFovTool.searchInfo.fov)
             
    def do_calibration(self, filter_id):
        """
        Check if FOV inforamtion is available and if so, check if the fov search
        object provides the right stack (same img_list, tau-stack, time averaged
        to spectral resolution) to do the analysis.
        """
        raise NotImplementedError("Temporarily not available")
        if None in self.fov.values():
            print ("No FOV information available, please do FOV search first or"
            " load the FOV information from another SpectralCalibration object"
            " using self.load_fov_info(**kwargs)")
            return
        info = self.searchFovTool
        if info is not None and info.searchInfo.supplInfo.listId == filter_id and\
                        info.searchInfo.supplInfo.mergeType == "binning" and\
                                    info.searchInfo.supplInfo.tauMode:
            """Save some time if possible, i.e. if the settings for the FOV 
            search are such, that they correspond to the settings for 
            calibration
            """
            
            print "Using stack from FOV search tool for calibration"
            sTau=info.searchInfo.results.radius.tau_series
            if self.specCorrFactor != info.doas_dataCorrFac:
                raise ValueError("WARNING: spectral retrieval correction factor in FOV "
                    "search tool is different from the one used for spectral "
                    "calibration")
            #sSpec=info.searchInfo.results.radius.doas_data
        else:
            specRes=self.rawInput.doas_data
            l=self.rawInput.img_lists[filter_id]
            stackObj=ImgListStack(l,l.list_id)
            stackObj.activate_tau_mode()
            self.stackTemp=stackObj
            #initiate the settings for the stack determiantion (i.e. ROI and
            #steps down in gaussian pyramide)
            stackObj.img_prep["pyrlevel"]=0
            xAbs,yAbs=self.fov.pos_abs
            r=self.fov.radius
            #cut out a ROI which fits the radial FOV (+10 pixels)
            pm=int(r+10)
            stackObj.img_prep["roi"]=[xAbs-pm,xAbs+pm,yAbs-pm,yAbs+pm]
            if not stackObj.make_stack():
                print "Failed to determine image stack for spectral calibration.."
                return 0
            stackObj.make_stack_time_average(specRes.specimes.start,specRes.specTimes.stop)
#==============================================================================
#             bgCube=stackObj.stack_single_image(l.bgModel.bgModel[filter_id],"bgCube")
#             stackObj.stack=log(bgCube/stackObj.stack)
#==============================================================================
            mask=stackObj.make_radial_mask(xAbs,yAbs,r)
            stack=stackObj.stack
            means=[]
            for i in range(stack.shape[2]):
                subIm=stack[:,:,i]
                means.append(subIm[mask].mean())
            sTau=Series(means,stackObj.start_acq)
#==============================================================================
#             #now make sure that there are no nans in the series objects used 
#             #for polyfit (because then the fit does not work)
#             cond=invert(sTau.isnull().values)
#             sTau=sTau[cond]    
#==============================================================================
        sSpec=self.rawInput.doas_data.data*self.doas_cd_offset#[cond]
        self.calibration[filter_id]["tau_series"]=sTau
        self.calibration[filter_id]["doas_cd_series"]=sSpec
        poly=self.fit_calib_polynomial(filter_id)
        return poly
        
    def determine_aa_calibration(self,on_id, off_id, id="aa"):
        """
        Get the AA calibration from the two single filter calibration bunches.
        """
        if self.has_calib_data(on_id) and self.has_calib_data(off_id):
            c=self._new_calibration_bunch(id)
            if c is None:
                print("Abort determination of aa calibration in : " + 
                    self.__str__())
                return
            
            c.tau_series=self.calibration[on_id].tau_series-\
                                        self.calibration[off_id].tau_series
            c.doas_cd_series=self.calibration[on_id].doas_cd_series
            self.fit_calib_polynomial(id)
        else:
            print ("Error in " + self.__str__() + ": could not determine AA "
                "calibration, information missing")
            
        
    def merge_series_interval(self,s1,s2):
        """
        :param s1,s2: pandas series with same frequency
        """
        if len(s1)>len(s2):
            s1=s1[s2.index]
        elif len(s1)<len(s2):
            s2=s2[s1.index]
        return s1,s2
        
    def fit_calib_polynomial(self,filter_id,order=1,addSO2Offset=0):
        if self.has_calib_data(filter_id):
            sx=self.calibration[filter_id]["tau_series"]
            sy=self.calibration[filter_id]["doas_cd_series"]
            #the series should have the same frequency in order for this to 
            #work. The following cuts out the right range (arrays must have same)
            #length
            if len(sx) != len(sy):
                sx,sy=self.merge_series_interval(sx,sy)
            m=isfinite(sx.values) & isfinite(sy.values)
            x=sx[m]
            y=sy[m]
            coeffs=polyfit(x,y,order)
            poly=poly1d(coeffs)
            
            self.calibration[filter_id]["coeffs"]=coeffs
            self.calibration[filter_id]["poly"]=poly
            
            return poly
    
    def has_calib_data(self, filter_id):
        try:
            c=self.calibration[filter_id]
            if c.doas_cd_series is None or c.tau_series is None:
                return 0
            return 1
        except:
            print(format_exc())
            return 0
        
    def calib_available(self, filter_id):
        try:
            for val in self.calibration[filter_id].values():
                if val is None:
                    "Nooooooooooooooooooooooo"
                    return 0
            return 1
        except:
            print(format_exc())
            return 0
        
    
    def plot_calibration_curve(self, filter_id, ax=None,xMin=None,xMax=None):
        if self.has_calib_data(filter_id):
            if ax is None:
                fig,ax=subplots(1,1)
            sx=self.calibration[filter_id]["tau_series"]
            sy=self.calibration[filter_id]["doas_cd_series"]
            sx,sy=self.merge_series_interval(sx,sy)
            ax.plot(sx,sy," x",label=str(self.dev_id) + " data " + str(filter_id))
            ax.set_xlabel("Img Tau Series FOV Spectrometer")
            ax.set_ylabel("SO2-SCD cm-2")
            i=0
            f=sx.max()*1.05
            if xMin:
                i=xMin
            if xMax:
                f=xMax
            if self.calib_available(filter_id):
                xR=linspace(i,f,100)
                poly=self.calibration[filter_id]["poly"]
                yp=poly(xR)
                ax.plot(xR,yp,"--",label=str(self.dev_id) + " poly")
            #ax.legend(loc=2)
            ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
            ax.grid()
        
    def plot_calibration_curves(self,ax=None):
        if ax is None:
            fig,ax=subplots(1,1)
        for key in self.calibration:
            self.plot_calibration_curve(key, ax=ax)
        ax.set_title("Calibration curves " + str(self.dev_id))
        return ax 
    
    def plot_spectral_results(self, ax=None):
        """Plot the raw spectral results and if a correction factor is active
        do a second plot with the corrected SCDs
        """
        if ax is None:
            fig,ax=subplots(1,1)
        self.rawInput.doas_data.plot(style="--b", ax=ax, label="Raw")
        if self.doas_cd_offset != 1.0:
            sCorr=self.rawInput.doas_data.data*self.doas_cd_offset
            sCorr.plot(style="-r", ax=ax, label="Corrected")    
            
    def plot_timeseries_overlay(self, filter_id, ax = None):
        if not self.has_calib_data(filter_id):
            print "Could not plot timeseries overlay..."
            return 0
            
        if ax is None:
            fig,ax=subplots(1,1)
        s1=self.calibration[filter_id].tau_series
        s2=self.calibration[filter_id].doas_cd_series
        p1=ax.plot(s1.index.values,s1.values,"--xb", label="tau")
        ax.set_ylabel("Tau")
        ax2=ax.twinx()
        lStr="SO2 "
        if self.specCorrFactor != 1.0:
            s3=self.rawInput.doas_data.data          
            p3=ax2.plot(s3.index.values, s3.values,":xr", label=lStr + "(uncorrected)")
            lStr=lStr + "(Corr fac: " + str(self.specCorrFactor) + ")"
            
        p2=ax2.plot(s2.index.values, s2.values,"--xr", label=lStr)
        ax2.set_ylabel("SO2-SCD [cm-2]")
        ax.set_title("Time series overlay " + str(filter_id))
        vs=s1.values[isfinite(s1.values)]
        if all(vs>=0):
            ax.set_ylim([0,vs.max()])
        so2Max=max([s2.values.max(), s3.values.max()])
        ax2.set_ylim([0,so2Max])
        # added these three lines
        ps =p1+p2+p3
        labs = [l.get_label() for l in ps]
        ax.legend(ps, labs, loc="best",fancybox=True, framealpha=0.5)
        draw()
        return ax, ax2


