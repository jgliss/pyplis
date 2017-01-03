# -*- coding: utf-8 -*-
"""
piSCOPE setup classes

This module contains several setup classes related to measurement data and 
analysis, these are:

    1. :class:`Source`: emission source specifications
    #. :class:`FilterSetup`: Collection of interference filters used
    #. :class:`Camera`: Camera specifications
    #. :class:`MeasSetup`: Basic measurement setup 
    #. :class:`AutoCellCalibSetup`: Setup for cell calibration data

.. thumbnail::  ../data/illustrations/flowchart_setup.png
   
   Flowchart of basic measurement setup classes
   
"""
from dill import dump
from os import path, mkdir, getcwd
from datetime import datetime, timedelta
from collections import OrderedDict as od
from os.path import exists
from numpy import isnan, nan
from abc import ABCMeta
from copy import deepcopy
from PyQt4.QtGui import QApplication
from sys import argv

from piscope import PYDOASAVAILABLE
from piscope import _LIBDIR
if PYDOASAVAILABLE:
    from pydoas.dataimport import ResultImportSetup

from .forms import LineCollection, RectCollection  
from .inout import get_source_info#, get_all_valid_cam_ids
from .utils import Filter, CameraBaseInfo
from .geometry import MeasGeometry

        
class Source(object):
    """Object for source information"""
    def __init__(self, name = None, info_dict = {}):
        """Initiation of object
        
        :param str name (None): string ID of source
        :param dict info_dict (None): dictrinary contatining source info (only
            loaded if all necessary parameters are available and in the right
            format)
        
        .. note:: 
        
            if input name is valid (info can be found in database) then any 
            additional input using ``info_dict`` is ignored
            
        .. todo::
        
            Allow to specify gases emitted by the source 
            (e.g. "so2", "no2", "ch4") and also (if available) to assign 
            average emission amounts of these gases in order to access xs 
            of these species and make a prediction of relative optical 
            densities for a given camera in combination with a camera setup, 
            (i.e. filter wavelength specs). One example could be a power plant
            which emits so2 and no2 to correct for interferences in the optical
            densities in a certain wavelength range.
        
        """
        self.name = name
        self.lon = nan
        self.lat = nan
        self.altitude = nan
        
        self.gases = "Coming soon...."
        
        self.suppl_info = od([("status"        ,   ""),
                             ("country"       ,   ""),
                             ("region"        ,   ""),
                             ("type"          ,   ""),
                             ("last_eruption" ,   "")])
                           
        if isinstance(name, str):
            info = self.get_info(name)
            if bool(info):
                info_dict = info
#==============================================================================
#         if not bool(info_dict):
#             info_dict = self.get_info("Etna")
#==============================================================================
                
        self.load_source_info(info_dict)
        
    @property
    def source_id(self):
        return self.name
        
    @property
    def info_available(self):
        """Checks if main information is available"""
        return all([x is not None for x in [self.lon, self.lat, self.altitude]])
    
    @property
    def geo_data(self):
        return od([("lon"          ,   self.lon),
                   ("lat"          ,   self.lat),
                   ("altitude"     ,   self.altitude)])
                   
    @property
    def _type_dict(self):
        """Dict of all attributes and corresponding string conversion funcs"""
        return od([("name"          ,   str),
                   ("lat"           ,   float),
                   ("lon"           ,   float),
                   ("altitude"      ,   float),
                   ("status"        ,   str),
                   ("country"       ,   str),
                   ("region"        ,   str),
                   ("type"          ,   str),
                   ("last_eruption" ,   str)])
    
    def to_dict(self):
        """Returns dictionary of all parameters"""
        d = self.geo_data
        d["name"] = self.name
        d.update(self.suppl_info)
        return d
        
    def load_source_info(self, info_dict):
        """Try access default information of source"""
        types = self._type_dict
        if not isinstance(info_dict, dict):
            raise TypeError("need dictionary like object for source info update")
        err = []
        for key, val in info_dict.iteritems():
            if types.has_key(key):
                func = types[key]
            else:
                func = str
            try:
                self[key] = func(val)
            except:
                err.append(key)
        if bool(err) > 0:
            print "Failed to load the following source parameters\n%s" %err
        
        return self.info_available
    
    def get_info(self, name):
        """Load info dict form database
        
        :param str cam_id: string ID of camera 
        """
        res = get_source_info(name)
        num = len(res)
        if num == 0:
            return {}
        elif num == 1:
            return res.values()[0]
        else:
            print "\nMultiple occurences found for %s" %name
            ok=0
            print res.keys()
            while not ok:
                try:
                    inp=input("\nEnter, key:\n")
                    return res[inp]
                except:
                    print res.keys()
                    print "Retry..."
        return res
    """
    Helpers
    """
    def _all_params(self):
        """Return list of all relevant source attributes"""
        return self._type_dict.keys()    
        
    def __str__(self):
        """String representation of source"""
        s=("\npiSCOPE Source\n-------------------------\n")
        for key, val in self._type_dict.iteritems():
            s=s + "%s: %s\n" %(key, self(key))
        return s
        
    def __setitem__(self, key, value):
        """Update class item"""
        if self.__dict__.has_key(key):
            self.__dict__[key]=value
        else: 
            self.suppl_info[key] = value
            
    def __getitem__(self, name):
        """Load value of class item"""
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        if self.suppl_info.has_key(name):
            return self.suppl_info[name]
            
    def __call__(self, key):
        """Make object callable (access item)"""
        return self.__getitem__(key)

#==============================================================================
# class DarkOffsetSetup(object):
#     """Collection of :class:`DarkOffsetInfo` objects for a camera"""
#     def __init__(self, infoList = []):
#         """Initialisation"""
#         self.init_objects(infoList)
#     
#     def init_objects(self, infoList):
#         self.objects = od()
#         try:
#             for obj in infoList:
#                 if isinstance(obj, DarkOffsetInfo):
#                     self.objects[obj.acronym] = obj
#         except:
#             pass
#         
#     def get_acros_low_gain(self):
#         """Get acronyms of all :class:`DarkOffsetInfo` objects which correspond
#         to low gain measurements"""
#         acros = []
#         for acro, obj in self.objects:
#             if obj.gain == 'LOW':
#                 acros.append(acro)
#         return acros
# 
#     def get_acros_high_gain(self):
#         """Get acronyms of all :class:`DarkOffsetInfo` objects which correspond
#         to high gain measurements"""
#         acros = []
#         for acro, obj in self.objects:
#             if obj.gain == 'HIGH':
#                 acros.append(acro)
#         return acros
#==============================================================================
        
class FilterSetup(object):
    """A collection of :mod:`Filter` objects 
    
    This collection specifies a filter setup for a camera, normally it consists
    of one on and one offband filter, but it can also include more filters 
    """
    def __init__(self, filter_list = [], default_key_on = None,\
                                                default_key_off = None):
        """Class initialisation
        
        :param list filters: list of :class:`Filter` objects specifying filters. 
        :param str default_key_on: Key of central filter object (e.g. "on")
        :param str default_key_off: Key of default offband filter object (e.g. 
            "off")
        
        """
        self.init_filters(filter_list)
        
        self.default_key_on = None
        self.default_key_off = None
 
        self.set_default_filter_keys(default_key_on, default_key_off)
            
    def init_filters(self, filters):
        """Initiate the filters (old settings will be deleted)
        
        :param listlike filters: list with :class:`Filter` objects
        
        The filters will be written in the ordered dictionary ``self.filters``
        in the list order, keys are the filter ids
        """
        self.filters = od()
        try:
            for f in filters:
                if isinstance(f, Filter):
                    self.filters[f.id] = f
        except:
            pass
        if not bool(self.filters):
            self.filters["on"] = Filter("on")
            
    def update_filters_from_dict(self, filterDict):
        """Add filter objects from a dictionary
        
        :param dict filterDict: dictionary, containing filter information
        """
        for f in filterDict.values():
            if isinstance(f, Filter):
                if self.filters.has_key(f.id):
                    print "Filter %s was overwritten" %f.id
                self.filters[f.id] = f
    
    def set_default_filter_keys(self, default_key_on = None,\
                                                    default_key_off = None):
        """Set default filter IDs for on and offband
        
        :param str default_key_on: Key of central filter object (e.g. "on")
        :param str default_key_off: Key of default offband filter object 
            (e.g. "off")
            
        If input parameters are unspecified, the first entries from the current setup are
        used.
        """
        ids_on, ids_off = self.get_ids_on_off()
        if not ids_on:
            raise ValueError("No onband filter specified in FilterSetup")
        if default_key_on is None or default_key_on not in ids_on:
            print "No onband default key specified, use 1st entry in FilterDict"
            self.default_key_on = ids_on[0]
        else:
            self.default_key_on = default_key_on
        if ids_off:
            if default_key_off is None or default_key_off not in ids_off:
                print "No offband default key specified, use 1st entry in FilterDict"
                self.default_key_off = ids_off[0]
            else:
                self.default_key_off = default_key_off
        else:
            print ("Offband default key could not be specified, no offband "
                "filter available in FilterSetup")
    
    def check_default_filters(self):
        """Checks if default filter keys are set"""
        if self.has_on:
            ids_on = self.ids_on
            if self.default_key_on is None or not self.default_key_on in ids_on:
                print "Updating default onband in FilterSetup %s" %ids_on[0]
                self.default_key_on = ids_on[0]
        if self.has_off:
            ids_off = self.ids_off
            if self.default_key_off is None or not self.default_key_off in ids_off:
                print "Updating default offband in FilterSetup %s" %ids_off[0]
                self.default_key_off = ids_off[0]
                
    @property
    def ids_off(self):
        """Get list with all offband filter ids"""
        return self.get_ids_on_off()[1]
        
    @property
    def ids_on(self):
        """Get list with all onband filter ids"""
        return self.get_ids_on_off()[0]
        
    @property
    def has_on(self):
        """Check if collection contains an onband filter"""
        if bool(self.ids_on):
            return True
        return False
    
    @property
    def has_off(self):
        """Check if collection contains an onband filter"""
        if bool(self.ids_off):
            return True
        return False    
        
    def get_ids_on_off(self):
        """Get all filters sorted by their type (On or Off)
        
        :returns:
            - list, contains all on band IDs
            - list, contains all off band IDs
        """
        ids_on, ids_off = [], []
        for key in self.filters:
            if self.filters[key].type == "on":
                ids_on.append(key)
            elif self.filters[key].type == "off":
                ids_off.append(key)
        return ids_on, ids_off
        
    """
    Helpers
    """      
    @property
    def number_of_filters(self):
        """Returns the current number of filters in this collection"""
        return len(self.filters)
    
    """
    Helpers, convenience stuff...
    """
    def print_setup(self):
        """Prints the current setup"""
        s=("piSCOPE FilterSetup\n------------------------------\n"
            "All filters:\n\n")
        for flt in self.filters.values():
            s += ("%s" %flt)
        s += "Default Filter: %s\n\n" %self.default_key_on
        return s
    
    def __call__(self, filter_id):
        """Returns the filter corresponding to the input ID
                
        :param str filter_id: string ID of filter
        """
        return self.filters[filter_id]
        
    def __str__(self):
        """String representation"""
        s = ""
        for f in self.filters.values():
            s += "%s (%s): %s nm\n" %(f.type, f.acronym, f.center_wavelength) 
        s += "Default Filter: %s\n" %self.default_key_on
        return s
            
class Camera(CameraBaseInfo):
    """Base class to specify a camera setup
    
    Class representing a UV camera system including detector specifications, 
    optics, file naming convention and :class:`FilterSetup`
    """
    def __init__(self, cam_id = None, geom_data = {}, filter_list = [],\
                    default_filter_key_on = None, default_filter_key_off =\
                                                None, ser_no = 9999, **kwargs):
        """Initiation of object
        
        :param str cam_id (""): camera ID (e.g "ecII")
        :param int ser_no (9999): camera serial number
        :param dict filterInfoDict (None)
        """
        super(Camera, self).__init__(cam_id, **kwargs)
        #specify the filters used in the camera and the main filter (e.g. On)        
        self.ser_no = ser_no #identifier of camera
        self.geom_data = od([("lon"          ,   None),
                             ("lat"          ,   None),
                             ("altitude"     ,   None),
                             ("azim"         ,   None),
                             ("azim_err"     ,   None),
                             ("elev"         ,   None),
                             ("elev_err"     ,   None)])
                               
        self.filter_setup = None
  
        self.prepare_filter_setup(filter_list, default_filter_key_on,\
                                                    default_filter_key_off)
        self.update_geom_data(geom_data)
    
        
    def update_geom_data(self, geom_info_dict):
        """Update geometry info in self.geom_data
        
        :param dict geom_info_dict: dictionary containing valid geometry info
            (see dict ``self.geom_data`` for valid input keys)
        """
        if not isinstance(geom_info_dict, dict):
            return 
        for key, val in geom_info_dict.iteritems():
            if key in self.geom_data.keys():
                self.geom_data[key] = val 
            
    def prepare_filter_setup(self, filter_list = None, default_key_on = None,\
                                                    default_key_off = None):
        """Create :class:`FilterSetup` object (collection of bandpass filters)
        
        :param list filter_list: list containing :class:`Filter` objects
        :param default_filter_key_on: string specifiying the string ID of the 
            main onband filter of the camera (usually "on"). If unspecified 
            (None), then the ID of the first available on bandfilter in the 
            filter input list will be used.
        :param default_filter_key_off: string specifiying the string ID of the 
            main offband filter of the camera (usually "on"). If unspecified 
            (None), then the ID of the first available off band filter in the 
            filter input list will be used.
        """
        if not isinstance(filter_list, list) or not bool(filter_list):
            filter_list = self.default_filters
            default_key_on = self.main_filter_id

        self.filter_setup = FilterSetup(filter_list, default_key_on,\
                                                        default_key_off)
                     
    """
    Helpers, Convenience stuff
    """
    def to_dict(self):
        """Convert this object into a dictionary"""
        d = super(Camera, self).to_dict()
        d["ser_no"] = self.ser_no
        for key, val in self.geom_data.iteritems():
            d[key] = val
        return d
        
    def change_camera(self, cam_id = None, make_new = False, **kwargs):
        """Change current camera type
        
        :param str cam_id: ID of new camera
        :param bool make_new: if True, a new instance will be created and 
            returned
        :param **kwargs: additional keyword args (see :func:`__init__`)
        """
        if not "geom_data" in kwargs:
            kwargs["geom_data"] = self.geom_data
        if make_new:
            return Camera(cam_id, **kwargs)
        
        self.__init__(cam_id, **kwargs)
        return self

        
    """Magic methods
    """
    def __str__(self):
        """String representation of this setup"""
        s = ("%s, serno. %s\n-------------------------\n"
                %(self.cam_id, self.ser_no))
        s += self._short_str()
        s += "\nFilters\n----------------------\n"
        s += str(self.filter_setup)
       
        s += "\nGeometry info\n----------------------\n"
        for key, val in self.geom_data.iteritems():
            try:
                s += "%s: %.3f\n" %(key, val)
            except:
                s += "%s: %s\n" %(key, val)
        
        return s
        
    def __setitem__(self, key, value):
        """Set item method"""
        if self.__dict__.has_key(key):
            self.__dict__[key] = value
        elif self.geom_data.has_key(key):
            self.geom_data[key] = value
        
    def __getitem__(self, name):
        """Get class item"""
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        for k, v in self.__dict__.iteritems():
            try:
                if v.has_key(name):
                    return v[name]
            except:
                pass
            
class FormSetup(object):
    """A setup class for all forms (lines, rectangles etc.) supposed to be used
    for the evaluation"""
    def __init__(self, line_dict = {}, rect_dict = {}):
        self.id = "forms"
        self.lines = LineCollection(line_dict)
        self.rects = RectCollection(rect_dict)
        
    def __str__(self):
        """String representation"""
        s = "piSCOPE FormSetup\n-----------------------------------\n\n"
        s += "Lines: %s\n" %self.lines
        s += "Rects: %s\n" %self.rects
        return s
        

class BaseSetup(object):
    """abstract base class for basic measurement setup, 
    
    Specifies image base path and start / stop time stamps of measurement
    as well as the following boolean access flags:
    
        1. :attr:`USE_ALL_FILES`
        #. :attr:`SEPARATE_FILTERS`
        #. :attr:`USE_ALL_FILE_TYPES`
        #. :attr:`INCLUDE_SUB_DIRS`
        
    """
    __metaclass__ = ABCMeta

    def __init__(self, base_path, start, stop, **opts):
        """Class initialisation
        
        :param str base_path: Path were e.g. imagery data lies
        :param datetime start: start time of Dataset
        :param datetime stop: stop time of Dataset
        :param **opts: setup options for file handling (currently only 
            INCLUDE_SUB_DIRS option)
            
        """
        self.id = "base"
        self.base_path = base_path
        self.save_path = base_path
        self.start = start
        self.stop = stop
        self.options = od([("USE_ALL_FILES"      ,   False),
                           ("SEPARATE_FILTERS"   ,   True),
                           ("USE_ALL_FILE_TYPES" ,   False),
                           ("INCLUDE_SUB_DIRS"   ,   False)])
                           
        self.check_timestamps()
        
        for k, v in opts.iteritems():
            if self.options.has_key(k):
                self.options[k] = v
        
        #self.check_paths()
        
    def check_timestamps(self):
        """Check if timestamps are valid and set to current time if not"""
        if not isinstance(self.start, datetime):
            self.options["USE_ALL_FILES"] = True
            self.start = datetime.now()
        if not isinstance(self.stop, datetime):
            self.stop = self.start + timedelta(1) #add one day to start time
        if self.start > self.stop:
            self.start, self.stop = self.stop, self.start
            
    @property
    def USE_ALL_FILES(self):
        """File import option (boolean)
        
        If True, all files in image base folder are used (i.e. start / stop 
        time stamps are disregarded)
        """
        return self.options["USE_ALL_FILES"]
    
    @USE_ALL_FILES.setter
    def USE_ALL_FILES(self, value):
        """Setter for this option"""
        if not value in [0, 1]:
            raise ValueError("need boolean")
        self.options["USE_ALL_FILES"] = value
        
    @property
    def SEPARATE_FILTERS(self):
        """File import option (boolean)
        
        If true, files are separated by filter type (e.g. "on", "off")
        """
        return self.options["SEPARATE_FILTERS"]
    
    @SEPARATE_FILTERS.setter
    def SEPARATE_FILTERS(self, value):
        """Setter for this option"""
        if not value in [0, 1]:
            raise ValueError("need boolean")
        self.options["SEPARATE_FILTERS"] = value
        
    @property
    def USE_ALL_FILE_TYPES(self):
        """File import option (boolean)
        
        If True, all files found are imported, disregarding the file type 
        (i.e. if image file type is not specified, we strongly recommend NOT to 
        use this option)
        """
        return self.options["USE_ALL_FILE_TYPES"]
    
    @USE_ALL_FILE_TYPES.setter
    def USE_ALL_FILE_TYPES(self, value):
        """Setter for this option"""
        if not value in [0, 1]:
            raise ValueError("need boolean")
        self.options["USE_ALL_FILE_TYPES"] = value
        
    @property
    def INCLUDE_SUB_DIRS(self):
        """File import option (boolean)
        
        If True, sub directories are included into image search
        """
        return self.options["INCLUDE_SUB_DIRS"]
    
    @INCLUDE_SUB_DIRS.setter
    def INCLUDE_SUB_DIRS(self, value):
        """Setter for this option"""
        if not value in [0, 1]:
            raise ValueError("need boolean")
        self.options["INCLUDE_SUB_DIRS"] = value
        
    def check_paths(self):
        """Check if current paths (base_path, save_path) are ok
        
        Sets library dir as base and / or save_path if current values are no
        valid location on machine.
        
        """
        if not isinstance(self.base_path, str) or not exists(self.base_path):
            self.base_path = _LIBDIR
        if not isinstance(self.save_path, str) or not exists(self.save_path):
            self.save_path = self.base_path
        
    def base_info_check(self):
        """Checks if all necessary information if available 
        
        Checks if path and times are valid
        
        :returns: tuple, containing
            - bool, True or False
            - str, information
            
        """
        ok = 1
        s=("Base info check\n-----------------------------\n")
        if not self.base_path or not exists(self.base_path):
            ok = 0
            s += "BasePath does not exist\n"
        if not self.USE_ALL_FILES:
            if not isinstance(self.start, datetime) or\
                                not isinstance(self.stop, datetime):
                s += "Start / Stop info wrong datatype (need datetime)\n"
                ok = 0
            elif not self.start < self.stop:
                s += "Start time exceeds stop time"
                ok = 0
        return (ok, s)
    
    def _check_if_number(self, val):
        """Check if input is integer or float and not nan"""
        if isinstance(val, (int, float)) and not isnan(val):
            return 1
        return 0
    
    def set_save_path(self,p):
        """set the base path for results to be stored"""
        if not path.exists(p):
            print ("Could not set save base path in\n\n" + self._save_name
                + ":\nPath does not exist")
            return
        self.save_path = p
        
    @property
    def _save_name(self):
        """Name according to saving convention"""
        d = self.start.strftime('%Y%m%d')
        i, f = self.start.strftime('%H%M'),self.stop.strftime('%H%M')
        return "piscope_setup_%s_%s_%s_%s" %(self.id, d, i, f)
    
    def _dict_miss_info_str(self, key, val):
        """string notification for invalid value"""
        return "Missing / wrong information: %s, %s\n" %(key, val)
        
    def __str__(self):
        """String representation of this class"""
        s=("\nSetup\n---------\n\n"
            "ID: %s\n"
            "Base path: %s\n" 
            "Save path: %s\n"
            "Start: %s\n"
            "Stop: %s\n"
            "Options:\n"
            %(self.id, self.base_path, self.save_path, self.start, self.stop))

        for key, val in self.options.iteritems():
            s = s + "%s: %s\n" %(key, val)
            
        return s   
    
class MeasSetup(BaseSetup):
    """Setup class for measurement 
    
    Class specifying a full measurement setup (e.g. plume image data). Inherits 
    from and is initiated as :class:`BaseSetup`, i.e. includes image base path, 
    start / stop time stamps, and furthermore source specs (:class:`Source`), 
    camera specs (:class:`Camera`) and meteorology information (wind 
    direction and velocity, stored as Python directory). 
    :class:`MeasSetup` objects can for instance be used as input for 
    :class:`piscope.Datasets.PlumeData` objects or 
    :class:`piscope.Datasets.BackgroundData` objects.
    """
    def __init__(self, base_path = None, start = None, stop = None,\
            camera = None, source = None, wind_info = None, rects = {},\
                                                        lines = {}, **opts):
        """
        :param str base_path: Path were e.g. imagery data lies
        :param datetime start: start time of Dataset
        :param datetime stop: stop time of Dataset
        :param Camera camera: general information about the camera used
        :param Source source: emission source object 
        :param **opts: setup options for file handling (currently only 
            INCLUDE_SUB_DIRS option)
            
        """
        super(MeasSetup, self).__init__(base_path, start, stop, **opts)
        self.id = "meas"
        
        if not isinstance(camera, Camera):
            camera = Camera()
        if not isinstance(source, Source):
            source = Source()
        
        self._cam_source_dict = {"camera"   :   camera,
                                 "source"   :   source}
        
        self.forms = FormSetup(lines, rects)

        self.wind_info = od([("dir"     ,   None),
                             ("dir_err" ,   None),
                             ("vel"     ,   None),    
                             ("vel_err" ,   None)])
                          
        self.meas_geometry = MeasGeometry()
        
        if isinstance(wind_info, dict):
            self.update_wind_info(wind_info)
            
        self.update_meas_geometry()
    
    @property
    def source(self):
        """Getter of property source"""
        return self._cam_source_dict["source"]
    
    @source.setter
    def source(self, value):
        """Setter of private attribute source
        
        :param Source value: a source object
        """
        if not isinstance(value, Source):
            raise TypeError("Invalid input type, need Source object")
        self._cam_source_dict["source"] = value
    
    @property
    def camera(self):
        """Getter of property source"""
        return self._cam_source_dict["camera"]
    
    @camera.setter
    def camera(self, value):
        """Setter of private attribute source
        
        :param Source value: a source object
        """
        if not isinstance(value, Camera):
            raise TypeError("Invalid input type, need Camera object")
        self._cam_source_dict["camera"] = value
        
    def set_source(self, source):
        """Set the current source object"""
        self.source = source
    
    def set_camera(self, camera):
        """Set the current camera setup"""
        self.camera = camera
    
    def update_wind_info(self, info_dict):
        """Update current wind info dict using valid entries from input dict"""
        for key, val in info_dict.iteritems():
            if self.wind_info.has_key(key):
                self.wind_info[key] = val
    
    def base_info_check(self):
        """Checks if all necessary information if available in order to create
        a DataSet object and determine measurement geometry for all sources
        """
        ok = 1
        s = ("Base info check\n-----------------------------\n")
        if not self.base_path or not exists(self.base_path):
            ok = 0
            s += "Image base path does not exist\n"
        if not self.USE_ALL_FILES:
            if not isinstance(self.start, datetime) or\
                                not isinstance(self.stop, datetime):
                s +=  "Start / Stop info wrong datatype (need datetime)\n"
                ok = 0
        ok, info = self.check_geometry_info()
        s += info
            
        return ok, s
            
    def check_geometry_info(self):
        """Checks if all necessary information is available for the determination 
        of measurement geometry, these are:
        
            1. Lon, Lat of
                i. source
                #. camera
            
            #. Meteorology info
                i. Wind direction
                #. Wind velocity (rough estimate)
                
            #. Viewing direction of camera
                i. Azimuth (N)
                #. Elvation(from horizon)

            #. Alitude of camera and source

            #. Camera optics
                i. Pixel size
                #. Number of pixels detector
                #. focal length
        """
        if not isinstance(self.source, Source):
            return 0
        source = self.source
        ok = 1 
        s=("\n------------------------------\nChecking basic geometry info"
            "\n------------------------------\n")
        for key, val in self.camera.geom_data.iteritems():
            if not self._check_if_number(val):
                ok = 0
                s += "Missing info in Camera setup\n"
                s += self._dict_miss_info_str(key, val)
        for key in self.camera.optics_keys:
            val = self.camera[key]
            if not self._check_if_number(val):
                ok = 0
                s += self._dict_miss_info_str(key, val)
        for key, val in source.geo_data.iteritems():
            if not self._check_if_number(val):
                ok = 0
                s += "Missing info in Source: %s\n" %source.name
                s += self._dict_miss_info_str(key, val)
        for key, val in self.wind_info.iteritems():
            if not self._check_if_number(val):
                ok = 0
                s += "Missing Meteorology info\n"
                s += self._dict_miss_info_str(key, val)
        if ok:
            s += "All necessary information available\n"
        print s
        return ok, s
    
    def update_meas_geometry(self):
        """Update the meas geometry based on current settings"""
        self.meas_geometry.__init__(self.source.to_dict(),\
                        self.camera.to_dict(), self.wind_info)
        
    """I/O stuff and helpers
    """
    @property
    def _save_name(self):
        """Returns the save name using piscope naming convention"""
        name = super(BaseSetup, self)._save_name
        try:
            name += "_%s" %self.source.name
        except:
            name += "_noSource"
        try:
            name += "_%s_%s" %(self.camera.cam_id, self.camera.ser_no)
        except:
            name += "_NoCamID_NoCamSerNo"
        return name
        
    def save(self, p = None):
        """save this object at a given location"""
        if p is None:
            p = self.save_path
        if not path.exists(p):
            self.save_path = p = getcwd()
        name = self._save_name + ".stp"
        f_path = path.join(p, name)
        dump(self, open(f_path, "wb"))
        return f_path
    
    def edit_in_gui(self):
        """Edit the current dataSet object"""
        from gui.SetupWidgets import MeasSetupEdit
        QApplication(argv)
        dial = MeasSetupEdit(deepcopy(self))
        dial.exec_()
        return dial
#==============================================================================
#         if dial.changesAccepted:
#             #self.dataSet.update_base_info(self.dataSet.setup)
#             self.dataSet.set_setup(stp)
#             self.analysis.setup.set_plume_data_setup(stp)
#             self.dataSet.init_image_lists()            
#             self.init_viewers()
#             self.update_actions()
#==============================================================================
    
    def short_str(self):
        """A short info string"""
        s = super(BaseSetup, self).__str__() + "\n"
        return s + "Camera: %s\nSource: %s" %(self.camera.cam_id,\
                                                        self.source.name)
    
    """Magic methods
    """
    def __setitem__(self, key, value):
        """Update class item"""
        if self.__dict__.has_key(key):
            self.__dict__[key] = value

    def __getitem__(self, key):
        """Load value of class item"""
        if self.__dict__.has_key(key):
            return self.__dict__[key]
            
    def __str__(self):
        """Detailed information string"""
        s = super(BaseSetup, self).__str__() + "\n\n"
        s = s + "Meteorology info\n-----------------------\n"
        for key, val in self.wind_info.iteritems():
            s = s + "%s: %s\n" %(key, val)
        s = s +"\n" + str(self.camera) +"\n"
        s = s + str(self.source)
        return s

    
class AutoCellCalibSetup(MeasSetup):
    """Setup class for cell calibration (i.e. input for DataSetCalib objects)
    
    A measurement setup for cell calibration data. Inherits from and is
    initiated as :class:`MeasSetup` (i.e. includes image path, start / stop 
    time stamps and camera specs) and was extended by parameter cell_info_dict, 
    a dictionary containing information about the gas columns of the 
    calibration cells used. 
    :class:`AutoCellCalibSetup` objects are supposed to be used as input for 
    :class:`piscope.Calibration.CellCalib` objects.
    """
    def __init__(self, cell_info_dict = {}, *args, **kwargs):
        """Class initialisation
        
        Initates :class:`BaseSetup` (see specs there) and extends the 
        initialisation by setting 
        
        :param dict cell_info_dict: dictionary containing cell information
            where keys are cell string abbreveations and values are lists with
            gas columns (first entry) and gas column uncertainties (second 
            entry) in units of cm-2
        
        An exemplary cell info dictionary could look like::
        
            cell_info_dict = {"cell1" : [3.2e18, 1.0e17],
                              "cell2" : [9.2e17, 1.1e17],
                              "cell3" : [3.1e17, 5.8e16]}
                              
        """
        super(AutoCellCalibSetup, self).__init__(*args,**kwargs)
        self.id = "cellcalib"
        self.cell_info_dict = cell_info_dict
    
    def short_str(self):
        """Short string representation"""
        s = "Cell specifications:\n"
        for key, val in self.cell_info_dict.iteritems():
            s += "%s: %s +/- %s\n" %(key, val[0], val[1]) 
        return super(AutoCellCalibSetup, self).short_str() + s
        
    def __str__(self):
        s="\nCell specifications\n---------------------------------\n"
        for key, val in self.cell_info_dict.iteritems():
            s=s+ str(key) + ": " + str(val[0]) + " +/- " + str(val[1]) + "\n" 
        return super(AutoCellCalibSetup, self).__str__() + s
        
#==============================================================================
# class EvalSettings(object):
#     """High level class to specify settings for emission rate analysis
#     
#     This class includes all relevant settings to perform emission rate analysis
#     of plume imagery data. These are mainly:
#     
#         1. Image pre edit and preparation settings (e.g. dark correction...)
#         #. Calibration settings (e.g. cell or spectral calibration, or both)
#         #. Wind speed retrieval settings (e.g. optical flow)
#         #. The plume cross sections used to retrieve emission rates
#     """
#     def __init__(self, settings_dict = {}):
#         
#         self.imgPrep = {"darkcorr"        :   1,
#                         "blurring"        :   0,
#                         "pyrlevel"        :   0,
#                         "roi"             :   [0, 0, 9999, 9999]}
#         
#         self._CALIBTYPES = ["spectral", "cell", "hybrid"]
#         
#         self.doas_calib_dev_id = None
#         
#         self.pcs_ids = []
#         
#         #self.optFlowSettings=OpticalFlowFarnebackSettings()
#         self.update_settings(settings_dict)
#     
#     
#     def update_settings(self, settingsDict):
#         """Update evaluation settings"""
#         if isinstance(settingsDict, dict):
#             for key, val in settingsDict.iteritems():
#                 self.__setitem__(key,val)
#     
#     def __setitem__(self, key, value):
#         if self.__dict__.has_key(key):
#             self.__dict__[key]=value
#             return
#         for v in self.__dict__.values():
#             if isinstance(v,dict) and v.has_key():
#                 v[key]=value
#             else:
#                 try:
#                     v[key]=value
#                 except:
#                     pass
#                 
#     def __call__(self, key):
#         if self.__dict__.has_key(key):
#             return self.__dict__[key]
#         for val in self.__dict__.values():
#             if isinstance(val, dict) and val.has_key(key):
#                 return val[key]
#         raise KeyError("Unkown input parameter")
#==============================================================================
            
class EmissionRateAnalysisSetup(object):
    """High level setup class for emission rate analysis
    
    Basically a "putting it all together" class to determine emission rates
    from a set of plume images. :class:`EmissionRateAnalysisSetup` objects can 
    be used as input for :class:`piscope.Evaluation.EmissionRateAnalysis` 
    objects
    """
    def __init__(self, plume_data_setup = None, auto_cell_calib_setup = None,\
            bg_img_access_setup = None, doas_result_import_setup = None):
        """Class initialisation
        
        :param MeasSetup plume_data_setup: the setup specifying plume data
        :param AutoCellCalibSetup auto_cell_calib_setup: setup specifying 
            automatic cell calibration (optional)
        :param MeasSetup bg_img_access_setup: setup specifying time information
            about background imagery data (optional)
            
        .. todo::
        
            This needs some review, need more flexibility with bg images, 
            calibration coeffs, etc. Best case would be a minimum input 
            solution, at least the possibility to provide a path to a single
            background image
        
        """
        #The base setup objects
        self.id = "analysis"
        
        #self.evalSettings = EvalSettings()
        
        self.save_path = None
        self.plume_data_setup = None #:class:`BaseSetup`
        self.auto_cell_calib_setup = None #:class:`CellCalibSetup`
        self.bg_img_access_setup = None
        
        self.doas_result_setups = {} #:class:`SpectralResultsSetup`
        
        #self.bgModelSetup = BackgroundAnalysisSetup()
        
        self.set_plume_data_setup(plume_data_setup)
        self.set_cellcalib_setup(auto_cell_calib_setup)
    
    @property
    def saveBase(self):
        """Returns the save path of ``self.plume_data_setup``"""
        return self.plume_data_setup.save_path
    
    def _check_path(self, p):
        """Check if input is valid path"""
        if not (isinstance(p, str) and exists(p)):
            return False
        return True
        
    def create_folder_structure(self):
        """Create the folder structure for saving / reloading"""
        if not self._check_path(self.saveBase):
            raise IOError("Invalid path for saveBase variable...")
        name = self.__str__() + "/"
        self.save_path = self.saveBase + name
        if not path.exists(self.save_path):
            mkdir(self.save_path)
    
#==============================================================================
#     def set_eval_settings(self, evalSettings):
#         """Change the evaluation settings object"""
#         if isinstance(evalSettings, EvalSettings):
#             self.evalSettings = evalSettings
#         raise TypeError("Invalid input while attempt to set :class:`EvalSettings`")
#==============================================================================
        
    def set_plume_data_setup(self, plume_data_setup = None):
        """Set the current :class:`BaseSetup` object"""
        if not isinstance(plume_data_setup, MeasSetup):
            print ("Creating new BaseSetup in EvalSetup...\n")
            plume_data_setup = MeasSetup()
        self.plume_data_setup = plume_data_setup
    
    def set_cellcalib_setup(self, auto_cell_calib_setup = None):
        """Set the current :class:`CellCalibSetup` object"""
        if not isinstance(auto_cell_calib_setup, AutoCellCalibSetup):
            print ("Creating new CellCalibSetup in EvalSetup...\n")
            auto_cell_calib_setup = AutoCellCalibSetup()
        self.auto_cell_calib_setup = auto_cell_calib_setup
    
            
    def add_doas_results_setup(self, doas_result_setup):
        """Add one current :class:`SpectralResultsSetup` object"""
        if not isinstance(doas_result_setup, ResultImportSetup):
            raise TypeError("Could not add ResultImportSetup to EvalSetup: "
                "wrong input type %s" %type(doas_result_setup))
        dev_id = doas_result_setup.dev_id
        self.doas_result_setups[dev_id] = doas_result_setup
        
    def __str__(self):
        """String representation of this class"""
        stp = self.plume_data_setup
        d = stp.start.strftime('%Y%m%d')
        i, f = stp.start.strftime('%H%M'), stp.stop.strftime('%H%M')
        name = "stp_%s_%s_%s_%s" %(self.id, d, i, f)
        try:
            name += "_%s" %stp.source.name
        except:
            name += "_noSource"
        try:
            name += "_%s_%s" %(stp.camera.cam_id, stp.camera.ser_no)
        except:
            name += "_noCamID_noCamSerNo"
        return name
    
    def save(self):
        """save this object at self.save_path"""
        if self.save_path is None or not path.exists(self.save_path):
            print ("Could not save " + self.__str__() + ": save path does not exists")
            return
        print ("Saving " + self.__str__() + "at " + str(self.save_path))
        name = self.__str__() + ".stp"
        print ("FileName: " + name)
        file_path=self.save_path + name
        dump(self, open(file_path, "wb"))
        return file_path 
        
"""SORTED OUT STUFF"""
#==============================================================================
# 
# class DoasCalibSetup(object):
#     """Setup for storing DOAS calibration information"""
#     def __init__(self, dev_id, speciesId, fitId):
#         self.dev_id = dev_id
#         self.speciesId = speciesId
#         self.fitId = fitId
#==============================================================================
#==============================================================================
# class BackgroundAnalysisSetup(object):
#     """A class to collect settings for background image treatment"""
#     def __init__(self):
#         self.id="bgModel"
#         self.settings=Bunch({"scale"    :   1,
#                              "ygradient":   0,
#                              "polyMode" :   0,
#                              "surfaceFit":  0})
#                              
#         self.bgPolyFiles=[] #list of background polynomial files
#         self.bgPolyTimeIntervals=Bunch({"scale"     :   None,
#                                         "ygradient" :   None})
#     
#     def add_bg_poly_file(self, f):
#         if exists(f) and f.split(".")[-1] == "bgp" and not f in self.bgPolyFiles:
#             self.bgPolyFiles.append(f)
#             
#     def __str__(self):
#         s=("\npiSCOPE background model setup\n-------------------------\n" +
#             "Settings:\n")
#         for key, val in self.settings.iteritems():
#             s=s + key + ": " + str(val) + "\n"
#         s = s +"\nBackground polynomial files:\n"
#         for f in self.bgPolyFiles:
#             s = s+ f + "\n"
#         return s
#==============================================================================       
        
        