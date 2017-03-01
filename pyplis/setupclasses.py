# -*- coding: utf-8 -*-
"""
This module contains several setup classes related to measurement data and 
analysis, the most important ones are:

    1. :class:`Source`: emission source specifications
    #. :class:`FilterSetup`: collection of interference filters used
    #. :class:`Camera`: camera specifications
    #. :class:`MeasSetup`: full measurement setup  
"""
from datetime import datetime
from collections import OrderedDict as od
from copy import deepcopy
from os.path import exists
from numpy import nan, rad2deg, arctan
from abc import ABCMeta
from warnings import warn

from .forms import LineCollection, RectCollection  
from .helpers import isnum, to_datetime
from .exceptions import MetaAccessError
from .inout import get_source_info
from .utils import Filter, CameraBaseInfo
from .geometry import MeasGeometry

class Source(object):
    """Object containing information about emission source"""
    def __init__(self, name="", info_dict={}):
        """Class initialisation
        
        :param str name: string ID of source (default is "")
        :param dict info_dict: dictionary contatining source information (is 
            only loaded if all necessary parameters are available and in the 
            right format)
        
        .. note:: 
        
            if input param ``name`` is a valid default ID (e.g. "Etna") then 
            the source information is extracted from the database and the 
            input param ``info_dict`` is ignored
            
        """
        self.name = name
        self.lon = nan
        self.lat = nan
        self.altitude = nan
        
        self.suppl_info = od([("status"        ,   ""),
                              ("country"       ,   ""),
                              ("region"        ,   ""),
                              ("type"          ,   ""),
                              ("last_eruption" ,   "")])
                           
        if isinstance(name, str):
            info = self.get_info(name, try_online=False)
            if bool(info):
                info_dict = info
                
        self.load_source_info(info_dict)
        
    @property
    def source_id(self):
        """Returns ``self.name``"""
        return self.name
        
    @property
    def info_available(self):
        """Checks if main information is available"""
        return all([x is not None for x in [self.lon, self.lat, self.altitude]])
    
    @property
    def geo_data(self):
        """Return dictionary containing lon, lat and altitude"""
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
        """Try access default information of source
        
        :param dict info_dict: dictonary containing source information (valid
            keys are keys of dictionary ``self._type_dict``, e.g. ``lon``, 
            ``lat``, ``altitude``)        
        """
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
    
    def get_info(self, name, **kwargs):
        """Load info dict from database (includes online search)
        
        :param str name: source ID
        """
        res = get_source_info(name, **kwargs)
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
        s=("\npyplis Source\n-------------------------\n")
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
        
class FilterSetup(object):
    """A collection of :class:`pyplis.utils.Filter` objects
    
    This collection specifies a filter setup for a camera. A typical setup 
    would be one on and one off band filter. 
    """
    def __init__(self, filter_list=[], default_key_on=None,
                 default_key_off=None):
        """Class initialisation
        
        :param list filters: list of :class:`pyplis.utils.Filter` objects
            specifying camera filter setup
        :param str default_key_on: string ID of default on band filter (only
            relevant if collection contains more than one on band filter)
        :param str default_key_off: string ID of default off band filter (only
            relevant if collection contains more than one off band filter)
        
        """
        self.init_filters(filter_list)
        
        self.default_key_on = None
        self.default_key_off = None
 
        self.set_default_filter_keys(default_key_on, default_key_off)
    
    @property
    def on_band(self):
        """Returns default on band filter"""
        return self.filters[self.default_key_on]
    
    @property
    def off_band(self):
        """Returns default on band filter"""
        try:
            return self.filters[self.default_key_off]    
        except:
            raise TypeError("Collection does not contain off band filter")
            
    def init_filters(self, filters):
        """Initiate the filter collection (old settings will be deleted)
        
        The filters will be written into the dictionary ``self.filters``
        in the list order, keys are the filter ids
        
        :param list filters: list of :class:`pyplis.utils.Filter` objects
            specifying camera filter setup
        
        
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
            
    def update_filters_from_dict(self, filter_dict):
        """Add filter objects from a dictionary
        
        :param dict filter_dict: dictionary, containing filter information
        """
        for f in filter_dict.values():
            if isinstance(f, Filter):
                if self.filters.has_key(f.id):
                    print "Filter %s was overwritten" %f.id
                self.filters[f.id] = f
    
    def set_default_filter_keys(self, default_key_on=None,
                                default_key_off=None):
        """Set default filter IDs for on and offband
        
        If input parameters are unspecified, the first entries from the current 
        setup are used.
        
        :param str default_key_on: string ID of default on band filter (only
            relevant if collection contains more than one on band filter)
        :param str default_key_off: string ID of default off band filter (only
            relevant if collection contains more than one off band filter)
            
        
        """
        ids_on, ids_off = self.get_ids_on_off()
        if not ids_on:
            raise ValueError("No onband filter specified in FilterSetup")
        if default_key_on is None or default_key_on not in ids_on:
            self.default_key_on = ids_on[0]
        else:
            self.default_key_on = default_key_on
        if ids_off:
            if default_key_off is None or default_key_off not in ids_off:
                self.default_key_off = ids_off[0]
            else:
                self.default_key_off = default_key_off
    
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
        s=("pyplis FilterSetup\n------------------------------\n"
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
            s += ("%s, type: %s (%s): %s nm\n" 
                    %(f.id, f.type, f.acronym, f.center_wavelength))
        s += "Default Filter: %s\n" %self.default_key_on
        return s
            
class Camera(CameraBaseInfo):
    """Base class to specify a camera setup
    
    Class representing a UV camera system including detector specifications, 
    optics, file naming convention and :class:`FilterSetup`
    """
    def __init__(self, cam_id=None, filter_list=[], default_filter_on=None,
                 default_filter_off=None, ser_no=9999, **geom_info):
        """Initiation of object
        
        :param str cam_id: camera ID (e.g "ecII"), if this ID corresponds to 
            one of the default cameras, the information is automatically 
            loaded from supplementary file *cam_info.txt* 
        :param list filter_list: list containing :class:`pyplis.utils.Filter`
            objects specifying the camera filter setup. If unspecified (empty
            list) and input param ``cam_id`` is a valid default ID, then the 
            default filter setup of the camera will be loaded.
        :param str default_filter_on: string ID of default on band filter (only
            relevant if collection contains more than one on band filter)
        :param str default_filter_off: string ID of default off band filter (only
            relevant if collection contains more than one off band filter)
        :param int ser_no (9999): optional, camera serial number
        :param **geom_info: additional keyword args specifying geometrical 
            information, e.g. lon, lat, altitude, elev, azim
            
        Example creating a new camera (using ECII default info with custom
        filter setup)::
        
            import pyplis
    
            #the custom filter setup
            filters= [pyplis.utils.Filter(type="on", acronym="F01"),
                      pyplis.utils.Filter(type="off", acronym="F02")]
            
            cam = pyplis.setupclasses.Camera(cam_id=ecII", filter_list=filters,
                                              lon=15.11, lat=37.73, elev=18.0,
                                              elev_err=3, azim=270.0,
                                              azim_err=10.0, focal_lengh=25e-3)
            print cam
            
        """
        super(Camera, self).__init__(cam_id)
    
        #specify the filters used in the camera and the main filter (e.g. On)        
        self.ser_no = ser_no #identifier of camera
        self.geom_data = od([("lon"         ,   None),
                             ("lat"         ,   None),
                             ("altitude"    ,   None),
                             ("azim"        ,   None),
                             ("azim_err"    ,   None),
                             ("elev"        ,   None),
                             ("elev_err"    ,   None),
                             ("alt_offset"  ,   0.0)])
        
        for k, v in geom_info.iteritems():
            self[k] = v
                       
        self.filter_setup = None
  
        self.prepare_filter_setup(filter_list, default_filter_on,
                                  default_filter_off)
    
    
    @property
    def lon(self):
        """Returns longitude"""
        return self.geom_data["lon"]

    @lon.setter
    def lon(self, val):
        """Set longitude"""
        if not -180 <= val <= 180:
            raise ValueError("Invalid input for longitude, must be between"
                "-180 and 180")
        self.geom_data["lon"] = val
    
    @property
    def lat(self):
        """Returns latitude"""
        return self.geom_data["lat"]

    @lat.setter
    def lat(self, val):
        """Set longitude"""
        if not -90 <= val <= 90:
            raise ValueError("Invalid input for longitude, must be between"
                "-90 and 90")
        self.geom_data["lat"] = val
        
    def update_settings(self, **settings):
        """Update geometry info in self.geom_data
        
        :param dict geom_info_dict: dictionary containing valid geometry info
            (see dict ``self.geom_data`` for valid input keys)
        """
        for key, val in settings.iteritems():
            self[key] = val
           
    def prepare_filter_setup(self, filter_list=None, default_key_on=None,
                             default_key_off=None):
        """Create :class:`FilterSetup` object (collection of bandpass filters)
        
        :param list filter_list: list containing :class:`pyplis.utils.Filter`
            objects
        :param default_filter_on: string specifiying the string ID of the 
            main onband filter of the camera (usually "on"). If unspecified 
            (None), then the ID of the first available on bandfilter in the 
            filter input list will be used.
        :param default_filter_off: string specifiying the string ID of the 
            main offband filter of the camera (usually "on"). If unspecified 
            (None), then the ID of the first available off band filter in the 
            filter input list will be used.
        """
        if not isinstance(filter_list, list) or not bool(filter_list):
            filter_list = self.default_filters
            default_key_on = self.main_filter_id
        
        filter_list = deepcopy(filter_list)
        self.filter_setup = FilterSetup(filter_list, default_key_on,
                                        default_key_off)
        #overwrite default filter information
        self.default_filters = []
        for f in filter_list:
            self.default_filters.append(f)
                     
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

    """Simple processing stuff
    """
    def dx_to_decimal_degree(self, pix_num_x):
        """Convert horizontal distance (in pixel units) into angular range
        
        :param int pix_num_x: number of pixels for which angular range is 
            determined        
        """
        try:
            len_phys = self.pix_width * pix_num_x
            return rad2deg(arctan(len_phys / self.focal_length))
        except:
            raise MetaAccessError("Please check availability of focal length, "
                "and pixel pitch (pix_width)")
    
    def dy_to_decimal_degree(self, pix_num_y):
        """Convert vertical distance (in pixel units) into angular range
        
        :param int pix_num_y: number of pixels for which angular range is 
            determined        
        """
        try:
            len_phys = self.pix_height * pix_num_y
            return rad2deg(arctan(len_phys / self.focal_length))
        except:
            raise MetaAccessError("Please check availability of focal length, "
                "and pixel pitch (pix_height)")
        
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
        s = "pyplis FormSetup\n-----------------------------------\n\n"
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
    _start = None
    _stop = None
    def __init__(self, base_dir, start, stop, **opts):
        """Class initialisation
        
        :param str base_dir: Path were e.g. imagery data lies
        :param datetime start: start time of Dataset (can also be 
            datetime.time)
        :param datetime stop: stop time of Dataset (can also be datetime.time)
        :param **opts: setup options for file handling (currently only 
            INCLUDE_SUB_DIRS option)
            
        """
        self.base_dir = base_dir
        self.save_dir = base_dir
        
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
    
    @property
    def start(self):
        """Getter / setter method for start time"""
        return self._start
        
    @start.setter
    def start(self, val):
        try:
            self._start = to_datetime(val)
        except:
            warn("Start time stamp was not set in Setup class")
    
    @property
    def stop(self):
        """Getter / setter method for start time"""
        return self._stop
        
    @stop.setter
    def stop(self, val):
        try:
            self._stop = to_datetime(val)
        except:
            warn("Stop time stamp was not set in Setup class")
            
    def check_timestamps(self):
        """Check if timestamps are valid and set to current time if not"""
        if not isinstance(self.start, datetime):
            self.options["USE_ALL_FILES"] = True
            self.start = datetime(1900, 1, 1)
        if not isinstance(self.stop, datetime):
            self.stop = datetime(1900, 1, 1)
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
        self.options["USE_ALL_FILES"] = bool(value)
        
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
                
    def base_info_check(self):
        """Checks if all necessary information if available 
        
        Checks if path and times are valid
        
        :returns: tuple, containing
            - bool, True or False
            - str, information
            
        """
        ok = 1
        s=("Base info check\n-----------------------------\n")
        if not self.base_dir or not exists(self.base_dir):
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
        return isnum(val)
    
#==============================================================================
#     def set_save_dir(self,p):
#         """set the base path for results to be stored"""
#         if not path.exists(p):
#             print ("Could not set save base path in\n\n" + self._save_name
#                 + ":\nPath does not exist")
#             return
#         self.save_dir = p
#         
#     @property
#     def _save_name(self):
#         """Name according to saving convention"""
#         d = self.start.strftime('%Y%m%d')
#         i, f = self.start.strftime('%H%M'),self.stop.strftime('%H%M')
#         return "pyplis_setup_%s_%s_%s_%s" %(self.id, d, i, f)
#     
#==============================================================================
    def _dict_miss_info_str(self, key, val):
        """string notification for invalid value"""
        return "Missing / wrong information: %s, %s\n" %(key, val)
        
    def __str__(self):
        """String representation of this class"""
        s=("\nSetup\n---------\n\n"
            "Base path: %s\n" 
            "Save path: %s\n"
            "Start: %s\n"
            "Stop: %s\n"
            "Options:\n"
            %(self.base_dir, self.save_dir, self.start, self.stop))

        for key, val in self.options.iteritems():
            s = s + "%s: %s\n" %(key, val)
            
        return s   
    
class MeasSetup(BaseSetup):
    """**Setup class for plume image data** 
    
    In this class, everything related to a full measurement setup is 
    defined, i.e. includes image base directory, start / stop time stamps, 
    :class:`Source`, :class:`Camera` and meteorology information (wind 
    direction and velocity, stored as Python dictionary). 
    :class:`MeasSetup` objects are the default input for 
    :class:`pyplis.dataset.Dataset` objects (i.e. also 
    :class:)
    """
    def __init__(self, base_dir=None, start=None, stop=None, camera=None,
                 source=None, wind_info=None, cell_info_dict={}, rects={},
                 lines={}, **opts):
        """
        :param str base_dir: Path were e.g. imagery data lies
        :param datetime start: start time of Dataset
        :param datetime stop: stop time of Dataset
        :param Camera camera: general information about the camera used
        :param Source source: emission source object 
        :param **opts: setup options for file handling (currently only 
            INCLUDE_SUB_DIRS option)
            
        """
        super(MeasSetup, self).__init__(base_dir, start, stop, **opts)
        self.id = "meas"
        
        if not isinstance(camera, Camera):
            camera = Camera()
        if not isinstance(source, Source):
            source = Source()
        
        self._cam_source_dict = {"camera"   :   camera,
                                 "source"   :   source}
        
        self.cell_info_dict = cell_info_dict
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
        if not self.base_dir or not exists(self.base_dir):
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
        s += "Meteorology info\n-----------------------\n"
        for key, val in self.wind_info.iteritems():
            s += "%s: %s\n" %(key, val)
        s += "\n" + str(self.camera) +"\n"
        s += str(self.source)
        if self.cell_info_dict.keys():
            s += "\nCell specifications:\n"
            for key, val in self.cell_info_dict.iteritems():
                s += "%s: %s +/- %s\n" %(key, val[0], val[1]) 
        return s
#==============================================================================
#     @property
#     def _save_name(self):
#         """Returns the save name using pyplis naming convention"""
#         name = super(BaseSetup, self)._save_name
#         try:
#             name += "_%s" %self.source.name
#         except:
#             name += "_noSource"
#         try:
#             name += "_%s_%s" %(self.camera.cam_id, self.camera.ser_no)
#         except:
#             name += "_NoCamID_NoCamSerNo"
#         return name
#         
#     def save(self, p = None):
#         """save this object at a given location"""
#         if p is None:
#             p = self.save_dir
#         if not path.exists(p):
#             self.save_dir = p = getcwd()
#         name = self._save_name + ".stp"
#         f_dir = path.join(p, name)
#         dump(self, open(f_dir, "wb"))
#         return f_dir
#     
#==============================================================================
#==============================================================================
#     def edit_in_gui(self):
#         """Edit the current dataSet object"""
#         from pyplis.gui_features.setup_widgets import MeasSetupEdit
#         app=QApplication(argv)
#         dial = MeasSetupEdit(deepcopy(self))
#         dial.exec_()
#         return dial
#==============================================================================
#==============================================================================
#         if dial.changesAccepted:
#             #self.dataSet.update_base_info(self.dataSet.setup)
#             self.dataSet.set_setup(stp)
#             self.analysis.setup.set_plume_data_setup(stp)
#             self.dataSet.init_image_lists()            
#             self.init_viewers()
#             self.update_actions()
#==============================================================================
    
