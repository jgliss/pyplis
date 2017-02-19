# -*- coding: utf-8 -*-
"""Low level utilities"""
from datetime import datetime as dt
from numpy import nan
from collections import OrderedDict as od
from pandas import Series
from os.path import basename, exists

if not __name__ == "__main__":
    from .inout import get_camera_info, save_new_default_camera
else: 
    from pyplis.inout import get_camera_info, save_new_default_camera
        
class CameraBaseInfo(object):
    """Low level base class for camera specific information 
    
    Mainly detector definitions (pixel geometries, size, etc, detector size),
    image file convention issues and how to handle dark image correction
    
    .. todo::
    
        1. Include binning
        
    """
    def __init__(self, cam_id=None, **kwargs):
        """Init object
        
        :param str cam_id: string ID of camera (e.g. "ecII")
        :param dict info_dict: dictionary containing camera info (only loaded
            if all parameters are available in the right format)
            
        .. note:: 
        
            if input cam_id is valid (i.e. can be found in database) then any 
            additional input using ``info_dict`` is ignored.
            
        """
        self.cam_id = None
        self.delim = None #""
        self.time_info_pos = None # nan
        self.time_info_str = None #""
        self.filter_id_pos = None #nan
        self.file_type = None #""
        self.default_filters = []
        self.main_filter_id = None#"on"
        self.texp_pos = None
        self.texp_unit = "ms"
        
        self.meas_type_pos = None#nan
        #:the next flag (self.DARK_CORR_OPT) is set for image lists created using this fileconvention
        #:and is supposed to define the way dark image correction is performed.
        #:for definition of the modes, see in :class:`BaseImgList` documentation)
        self.DARK_CORR_OPT = 0
        
        self.dark_info = []
        
        self.focal_length = None #in m
        self.pix_height = None # in m
        self.pix_width = None # in m
        self.pixnum_x = None#nan
        self.pixnum_y = None# nan
        #the following attribute keys are relevant for measurement geometry calcs
        self.optics_keys = ["pix_width", "pix_height", "focal_length"] 
        
        self._fname_access_flags = {"filter_id"    :   False,
                                    "texp"         :   False,
                                    "meas_type"    :   False,
                                    "start_acq"    :   False}
         
        try:
            self.load_default(cam_id)
        except:
            pass
        type_conv = self._type_dict
        for k,v in kwargs.iteritems():
            if type_conv.has_key(k):
                self[k] = type_conv[k](v)
                
        if self.meas_type_pos is None:
            self.meas_type_pos = self.filter_id_pos
    
    def update_file_access_flags(self):
        """Check which info can (potentially) be extracted from filename
        
        This function is based on the assumption that all settings in 
        cam_info.txt are set without mistakes and. It sets the 
        access flags accordingly on this theoretical base that everything works
        and the filenames are set right.
        
        See also :func:`get_img_meta_from_filename` which actually 
        checks if the access works for an given input file.
        
        """
        #for this camera and set the flags.
        flags = self._fname_access_flags        
        if isinstance(self.filter_id_pos, int):
            flags["filter_id"] = True
        if isinstance(self.meas_type_pos, int):
            flags["meas_type"] = True
        if isinstance(self.texp_pos, int):
            flags["texp"] = True
        if isinstance(self.time_info_pos, int) and\
                            isinstance(self.time_info_str, str):
            flags["start_acq"] = True
    
    def get_img_meta_from_filename(self, file_path):
        """Extract as much as possible from filename and updated flags
        
        Checks if all declared import information works for a given filetype 
        and update all flags for which it does not work 

        :param str file_path: file path used for info import check
        
        """
        acq_time, filter_id, meas_type, texp = None, None, None, None
        if not exists(file_path):
            raise IOError("Input file location does not exist %s" %file_path)
        warnings = []
        flags = self._fname_access_flags
        if self.delim is None:
            for key in flags:
                flags[key] = False
            warnings.append("Filename delimiter is not set")
               
        spl = basename(file_path).split(".")[0].split(self.delim)
        try:
            acq_time = dt.strptime(spl[self.time_info_pos], self.time_info_str)
            flags["start_acq"] = True
        except:
            warnings.append("Failed to extract start_acq from filename")
            flags["start_acq"] = False
        try:
            filter_id = spl[self.filter_id_pos]
            flags["filter_id"] = True
        except:
            warnings.append("Failed to extract filter_id from filename")
            flags["filter_id"] = False
        try:
            meas_type = spl[self.meas_type_pos]
            flags["meas_type"] = True
        except:
            warnings.append("Failed to extract meas_type from filename")
            flags["meas_type"] = False
        try:
            texp = float(spl[self.texp_pos])
            if self.texp_unit == "ms":
                texp = texp / 1000.0 #convert to s
            flags["texp"] = True
        except:
            warnings.append("Failed to extract texp from filename")
            flags["texp"] = False
        return acq_time, filter_id, meas_type, texp, warnings
    
        
    """Decorators / dynamic class attributes"""
    @property
    def default_filter_acronyms(self):
        """Get acronyms of all default filters"""
        acros = []
        for f in self.default_filters:
            acros.append(f.acronym)
        return acros    
        
    @property
    def _type_dict(self):
        """Dict of all attributes and corresponding string conversion funcs"""
        return od([("cam_id"          ,     str),
                   ("delim"           ,     str),
                   ("time_info_pos"   ,     int),
                   ("time_info_str"   ,     str),
                   ("filter_id_pos"   ,     int),
                   ("texp_pos"        ,     int),
                   ("texp_unit"       ,     str),
                   ("file_type"       ,     str),
                   ("main_filter_id"  ,     str),
                   ("meas_type_pos"   ,     int),
                   ("DARK_CORR_OPT"   ,     int),
                   ("focal_length"    ,     float),
                   ("pix_height"      ,     float),
                   ("pix_width"       ,     float),
                   ("pixnum_x"        ,     int),
                   ("pixnum_y"        ,     int),
                   ("default_filters" ,     list),                   
                   ("dark_info"       ,     list)])
    
    @property
    def _info_dict(self):
        """Returns dict containing information strings for all attributes"""
        return od([("cam_id"          ,   "ID of camera within code"),
                   ("delim"           ,   "Filename delimiter for info access"),
                   ("time_info_pos"   ,   ("Position (int) of acquisition time"
                                           " info in filename after splitting "
                                           "using delim")),
                   ("time_info_str"   ,   ("String formatting information for "
                                           "datetimes in filename")),
                   ("filter_id_pos"   ,   ("Position (int) of filter acronym "
                                           "string in filename after splitting"
                                           " using delim")),
                   ("texp_pos"        ,   ("Position of acquisition time info "
                                           "filename after splitting using "
                                           "delim")),
                   ("texp_unit"       ,   ("Unit of exposure time in filename"
                                           "choose between s or ms")),                       
                   ("file_type"       ,   "Filetype information (e.g. tiff)"),
                   ("main_filter_id"  ,   "String ID of main filter (e.g. on)"),
                   ("meas_type_pos"   ,   ("Position of meastype specification "
                                           "in filename after splitting using "
                                           "delim. Only applies to certain "
                                           "cameras(e.g. HD cam)")),
                   ("DARK_CORR_OPT"  ,   "Camera dark correction mode"),
                   ("focal_length"    ,   "Camera focal length in m"),
                   ("pix_height"      ,   "Detector pixel height in m"),
                   ("pix_width"       ,   "Detector pixel width in m"),
                   ("pixnum_x"        ,   "Detector number of pixels in x dir"),
                   ("pixnum_y"        ,   "Detector number of pixels in y dir"),
                   ("default_filters" ,   ("A Python list containing pyplis"
                                           "Filter objects")),                   
                   ("dark_info"       ,   ("A Python list containing pyplis"
                                           "DarkOffsetInfo objects"))])
        
             
    def load_info_dict(self, info_dict):
        """Loads all valid data from input dict
        
        :param dict info_dict: dictionary specifying camera information
        
        """
        types = self._type_dict
        filters = []
        dark_info = []
        missed = []
        err = []
        for key, func in types.iteritems():
            if info_dict.has_key(key):
                try:
                    val = func(info_dict[key])
                    if key == "default_filters":
                        for f in val:
                            filters.append(Filter(id=f[0], type=f[1],
                                                  acronym=f[2],
                                                  meas_type_acro=f[3],
                                                  center_wavelength=f[4]))
                    elif key == "dark_info":
                        for f in val:
                            dark_info.append(DarkOffsetInfo(id=f[0], type=f[1],
                                                            acronym=f[2],
                                                            meas_type_acro=f[3],
                                                            read_gain=int(f[4])))
                    else:
                        self[key] = val
                except:
                    err.append(key)
            else:
                missed.append(key)
        self.default_filters = filters
        self.dark_info = dark_info
        
        self.update_file_access_flags()
            
        return missed, err
    
    
    def load_default(self, cam_id):
        """Load information from one of the implemented default cameras
        
        :param str cam_id: id of camera (e.g. "ecII")
        """
        self.load_info_dict(get_camera_info(cam_id))

    """
    Helpers, supplemental stuff...
    """ 
    def get_acronym_dark_offset_corr(self, read_gain = 0):
        """Get file name acronyms for dark and offset image identification
        
        :param str read_gain (0): detector read gain
        """
        offs = None
        dark = None
        for info in self.dark_img_info:
            if info.type == "dark" and info.read_gain == read_gain:
                dark = info.acronym
            elif info.type == "offset" and info.read_gain == read_gain:
                offs = info.acronym
        return offs, dark
        
    """
    Helpers
    """
    def to_dict(self):
        """Writes specs into dictionary which is returned"""
        d = od()
        for k in self._type_dict.keys():
            if k in ["default_filters", "dark_info"]:
                d[k] = []
                for f in self[k]:
                    print f
                    print k
                    d[k].append(f.to_list())
            else:
                d[k] = self[k]
        return d
    
    def save_as_default(self, *add_cam_ids):
        """Saves this camera to default data base"""
        cam_ids = [self.cam_id]
        cam_ids.extend(add_cam_ids)
        print cam_ids
        d = od([("cam_ids"    ,   cam_ids)])
        d.update(self.to_dict())
        print d.keys()
        save_new_default_camera(d)
        
    def _all_params(self):
        """Return list of all relevant source attributes"""
        return self._type_dict.keys()
    
    def _short_str(self):
        """Short string repr"""
        s = ""
        for key in self._type_dict:
            #print key in ["defaultFilterSetup", "dark_img_info"]
            val = self(key)
            if key in ["default_filters", "dark_info"]:
                pass
            else:
                s = s + "%s: %s\n" %(key,val)
        s = s + "\nDark & offset info\n------------------------\n"
        for i in self.dark_info:
            s = s + ("ID: %s, Type: %s, Acro: %s, MeasTypeAcro: %s," 
                     "Read gain: %s\n" 
                     %(i.id, i.type, i.acronym, i.meas_type_acro, i.read_gain))
        return s
    """
    Magic methods
    """
    def __str__(self):
        """String representation"""
        s=("\npyplis CameraBaseInfo\n-------------------------\n\n")
        for key in self._type_dict:
            #print key in ["defaultFilterSetup", "dark_img_info"]
            val = self(key)
            if key in ["default_filters", "dark_info"]:
                for info in val:
                    s += "%s\n" %info
            else:
                s += "%s: %s\n" %(key,val)
        return s
        
    def __setitem__(self, key, value):
        """Set class item
        
        :param str key: valid class attribute
        :param value: new value
        """
        if self.__dict__.has_key(key):
            self.__dict__[key] = value
            
    def __getitem__(self, key):
        """Get current item
        
        :param str key: valid class attribute        
        """
        if self.__dict__.has_key(key):
            return self.__dict__[key]
            
    def __call__(self, key):
        """Make object callable (access item)"""
        return self.__getitem__(key)
        
class Filter(object):
    """Object representing an interference filter
    
    A low level helper class to store information of interference filters.    
    """          
    def __init__(self, id = None, type = "on", acronym = "default",\
                                meas_type_acro = None, center_wavelength = nan):      
        """Initiation of object
        
        :param str id ("on"): string identification of this object for 
            working environment
        :param str type ("on"): Type of object (choose from "on" and "off")
        :param str acronym (""): acronym for identification in filename
        :param str meas_type_acro (""): acronym for meastype identification in 
            filename
        :param str center_wavelength (nan): center transmission wabvelength of filter
        """
        if not type in ["on","off"]:
            raise ValueError("Invalid type specification for filter: %s, "
                "please use on or off as type")
        if id is None:
            id = type
        if meas_type_acro is None:
            meas_type_acro = acronym
    
        self.id = id
        self.type = type
        
        #filter acronym (e.g. F01, i.e. as used in filename)
        self.acronym = acronym      
        self.meas_type_acro = meas_type_acro
        
        #filter central wavelength
        self.center_wavelength = center_wavelength
        self.trans_curve = None
        #filter peak transmission

        if self.id is None:
            self.id = self.type

    def to_list(self):
        """Return filter info as list"""
        return [self.id, self.type, self.acronym, self.meas_type_acro,
                self.center_wavelength]
                
    def set_trans_curve(self, data, wavelengths = None):
        """Assign transmission curve to this filter
        
        :param ndarray data: transmission data
        :param ndarray wavelengths: corresponding wavelength array
        :returns: :class:`pandas.Series` object
        
        .. note::
        
            Also accepts :class:`pandas.Series` as input using input param
            data and leaving wavelength empty, in this case, the Series index
            is assumed to be the wavelenght data
            
        """
        if isinstance(data, Series):
            self.trans_curve = data
        else:
            try:
                self.trans_urve = Series(data, wavelengths)
            except:
                print ("Failed to set transmission curve in Filter %s" %self.id)
        
    def __str__(self):
        """String representation"""
        s = ("\nFilter\n---------------------------------\n"
              "ID: %s\n"
              "Type: %s\n"
              "Acronym: %s\n" 
              "Meastype acronym: %s\n" 
              "Central wavelength [nm]: %s\n" 
              %(self.id, self.type, self.acronym, self.meas_type_acro,\
                                                  self.center_wavelength))
        return s
        
    def print_specs(self):
        """print __str__"""
        print self.__str__()

class DarkOffsetInfo(object):
    """Base class for storage of dark offset information 
    
    Similar to :class:`Filter`. This object can be used to store relevant 
    information of different types of dark and offset images. The attribute 
    "read_gain" is set 0 by default. For some camera types (e.g. Hamamastsu 
    c8484 16c as used in the ECII SO2 camera), the signal can be enhancened 
    with an electronic read_gain (measured in dB) on read. This can be helpful in 
    low light conditions. However, it significantly increases the noise in the
    images and therefore also the dark image signal. 
    """
    def __init__(self, id="dark", type="dark", acronym="", meas_type_acro=None,
                 read_gain=0):
        """Initiation of object
        
        :param str id: string identification of this object for 
            working environment  (default: "dark")
        :param str type: Type of object (e.g. dark or offset, default: "dark")
        :param str acronym: acronym for identification in filename
        :param str meas_type_acro: acronym for meastype identification in 
            filename
        :param str read_gain: string specifying read_gain mode of this object
            (use 0 or 1, default is 0)
        """
        if not type in ["dark","offset"]:
            raise ValueError("Invalid type specification for DarkOffsetInfo: %s, "
                "please use dark  or offset as type")
        self.id = id
        self.type = type
        self.acronym = acronym
        if meas_type_acro is None:
            meas_type_acro = acronym
        self.meas_type_acro = meas_type_acro
        self.read_gain = read_gain
    
    def to_list(self):
        """Return parameters as list"""
        return [self.id, self.type, self.acronym, self.meas_type_acro,
                self.read_gain]
                
    def __str__(self):
        """String representation"""
        s = ("\nDarkOffsetInfo\n---------------------------------\n"
              "ID: %s\n"
              "Type: %s\n"
              "Acronym: %s\n" 
              "Meas type acronym: %s\n" 
              "Read gain: %s\n" %(self.id, self.type, self.acronym,\
                                        self.meas_type_acro, self.read_gain))
        return s

