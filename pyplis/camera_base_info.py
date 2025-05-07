import re
from pathlib import Path
from typing import Any, Optional
from numpy import nan
from warnings import warn
from os.path import basename
from datetime import datetime as dt
from collections import OrderedDict as od
from pyplis.utils import DarkOffsetInfo, Filter
from pyplis import custom_image_import
from pyplis.inout import save_new_default_camera, get_camera_info

class CameraBaseInfo(object):
    """Low level base class for camera specific information.

    Mainly detector definitions (pixel geometries, size, etc, detector size),
    image file convention issues and how to handle dark image correction

    """

    def __init__(
            self, 
            cam_id=None, 
            cam_info_file: Optional[Path]=None, 
            try_load_from_registry=True,
            **kwargs):
        """Init object.
    
        :param str cam_id: string ID of camera (e.g. "ecII")
        :param Path cam_info_file: path to cam_info.txt file to be used to load default.

        .. note::

            if input cam_id is valid (i.e. can be found in database) then any
            additional input using ``info_dict`` is ignored.

        """
        self.cam_id = None
        self.delim = "." 
        self.time_info_pos = None 
        self.time_info_str = ""  
        self._time_info_subnum = 1
        self.filename_regexp = None

        # Specify filter ID
        self.filter_id_pos = None  
        self._fid_subnum_max = 1

        self.file_type = None  
        self.default_filters = []
        self.main_filter_id = None  
        self.texp_pos = None
        self.texp_unit = "ms"

        self.io_opts = {}

        self.image_import_method = None

        self.meas_type_pos = None 
        # maximum length of meastype substrings after splitting using delim
        self._mtype_subnum_max = 1

        # the next flag (self.darkcorr_opt) is set for image lists created
        # using this fileconvention
        # and is supposed to define the way dark image correction is performed.
        # for definition of the modes, see in :class:`BaseImgList`
        # documentation)
        self.darkcorr_opt = 0

        self.dark_info = []

        self.reg_shift_off = [0.0, 0.0]

        self.focal_length = None  # in m
        self.pix_height = None  # in m
        self.pix_width = None  # in m
        self.pixnum_x = None  # nan
        self.pixnum_y = None  # nan
        # the following attribute keys are relevant for measurement
        # geometry calcs
        self.optics_keys = ["pix_width", "pix_height", "focal_length"]

        self._fname_access_flags = {"filter_id": False,
                                    "texp": False,
                                    "meas_type": False,
                                    "start_acq": False}
        # Helper to avoid unnecessary warnings
        self._fname_access_checked = False
        if cam_id and try_load_from_registry:
            try:
                self.load_default(cam_id=cam_id, cam_info_file=cam_info_file)
            except Exception as e:
                warn(f"Failed to load camera information for cam_id {cam_id}:\n{e}")
        type_conv = self._type_dict
        for k, v in kwargs.items():
            if k in type_conv:
                self[k] = type_conv[k](v)

        if self.meas_type_pos is None:
            self.meas_type_pos = self.filter_id_pos
        self._init_access_substring_info()

    @property
    def acronym_pos(self):
        """Wrap filter_id_pos."""
        return self.filter_id_pos

    @acronym_pos.setter
    def acronym_pos(self, val):
        self.filter_id_pos = val

    @property
    def meas_type_acro_pos(self):
        """Get / set ``meas_type_pos``."""
        return self.meas_type_pos

    @meas_type_acro_pos.setter
    def meas_type_acro_pos(self, val):
        """Get / set for ``meas_type_pos``."""
        self.meas_type_pos = val

    def _init_access_substring_info(self):
        """Check number of sub strings for specific access values after split.
        """
        if not self.delim:
            warn("Cannot init filename access info in Camera. Delimiter is "
                 "unspecified.")
            return False
        self._time_info_subnum = len(self.time_info_str.split(self.delim))
        for f in self.default_filters:
            len_acro = len(f.acronym.split(self.delim))
            len_mtype = len(f.meas_type_acro.split(self.delim))
            if len_acro > self._fid_subnum_max:
                self._fid_subnum_max = len_acro
            if len_mtype > self._mtype_subnum_max:
                self._mtype_subnum_max = len_mtype
        for f in self.dark_info:
            len_acro = len(f.acronym.split(self.delim))
            len_mtype = len(f.meas_type_acro.split(self.delim))
            if len_acro > self._fid_subnum_max:
                self._fid_subnum_max = len_acro
            if len_mtype > self._mtype_subnum_max:
                self._mtype_subnum_max = len_mtype
        return True

    def update_file_access_flags(self):
        """Check which info can (potentially) be extracted from filename.

        This function assumes that all settings in cam_info.txt are correct
        and sets the access flags accordingly.

        See also :func:`get_img_meta_from_filename` which actually
        checks if the access works for an given input file.
        """
        flags = self._fname_access_flags
        if isinstance(self.filter_id_pos, int):
            flags["filter_id"] = True
        if isinstance(self.meas_type_pos, int):
            flags["meas_type"] = True
        if isinstance(self.texp_pos, int):
            flags["texp"] = True
        if (isinstance(self.time_info_pos, int) and
                isinstance(self.time_info_str, str)):
            flags["start_acq"] = True

    def parse_meta_from_filename_regexp(self, filename, values, conf):
        """Extract information from filename using a camera defined or custom regexp.

        Parse a date from a string using a grouped regular expression.

        :param str file_path: file path used for info import check
        """
        regexp = re.compile(conf["filename_regexp"])
        values.update(re.match(regexp, filename).groupdict())
        return values
        
    def parse_meta_from_filename(self, filename, values, config):
        """Extract metadata from filename using a delimiter.
        
        Args:
            filename: filename to be parsed for metadata.
            config: dictionary containing filename parsing information.
        
        Returns:
            dict: dictionary containing parsed metadata. In specific, the
                following keys are used:
                - acq_time: acquisition time (datetime object) 
                - filter_id: filter ID (string)
                - meas_type: measurement type (string)
                - texp: exposure time (float)
        """
        spl = filename.rsplit(".", 1)[0].split(config["delim"])
        try:
            start = config["time_info_pos"]
            end = config["time_info_pos"] + config["time_info_subnum"]
            values["date"] = config["delim"].join(spl[start:end])
        except BaseException:
            pass

        try:
            start = config["filter_id_pos"]
            end = config["filter_id_pos"] + config["fid_subnum_max"]
            values["filter_id"] = config["delim"].join(spl[start:end])
        except BaseException:
            pass

        try:
            start = config["meas_type_pos"]
            end = config["meas_type_pos"] + config["mtype_subnum_max"]
            values["meas_type"] = config["delim"].join(spl[start:end])
        except BaseException:
            pass

        try:
            values["texp"] = float(spl[config["texp_pos"]])
        except BaseException:
            pass

        return values

    @property
    def filename_info_cfg(self) -> dict:
        return {
            "delim": self.delim,
            "time_info_pos": self.time_info_pos,
            "time_info_subnum": self._time_info_subnum,
            "time_info_str": self.time_info_str,
            "filter_id_pos": self.filter_id_pos,
            "fid_subnum_max": self._fid_subnum_max,
            "meas_type_pos": self.meas_type_pos,
            "mtype_subnum_max": self._mtype_subnum_max,
            "texp_pos": self.texp_pos,
            "texp_unit": self.texp_unit,
            "filename_regexp": self.filename_regexp
        }
    
    def get_img_meta_from_filename(self, file_path):
        """Extract as much as possible from filename and update access flags.

        Checks if all declared import information works for a given filetype
        and update all flags for which it does not.

        Args:
            file_path: image file path used to extract metadata from 
                filename.
        
        Returns:
            tuple: (start_acq, filter_id, meas_type, texp, warnings)
        """
        filename = basename(file_path)
        config = self.filename_info_cfg
        
        # init metadata of interest  
        extracted_metadata = {
            "start_acq": None, #datetime object
            "filter_id": None, #str
            "meas_type": None, #str
            "texp": None, #float
            "date": None #str
        }
        if self.filename_regexp:
            extracted_metadata = self.parse_meta_from_filename_regexp(filename, extracted_metadata, config)
        elif self.delim:
            extracted_metadata = self.parse_meta_from_filename(filename, extracted_metadata, config)
        else:
            return (None, None, None, None, ["Neither filename delimiter or regexp is set"])

        if extracted_metadata.get("date"):
            extracted_metadata["start_acq"] = dt.strptime(extracted_metadata["date"], config["time_info_str"])

        if extracted_metadata.get("texp") and config["texp_unit"] == "ms":
            extracted_metadata["texp"] = extracted_metadata["texp"] / 1000.0  # convert to s

        warnings = []
        for key, val in self._fname_access_flags.items():
            new_val_bool = bool(extracted_metadata[key]) 
            if new_val_bool != val:
                self._fname_access_flags[key] = new_val_bool
                if self._fname_access_checked:
                    warnings.append(f"Filename access flag {key} changed from {val} to {new_val_bool}")
        # if the filename access check was not done before, set the flag to True
        self._fname_access_checked = True
        return (
            extracted_metadata["start_acq"], 
            extracted_metadata["filter_id"], 
            extracted_metadata["meas_type"],
            extracted_metadata["texp"], 
            warnings)

    @property
    def default_filter_acronyms(self):
        """Get acronyms of all default filters."""
        acros = []
        for f in self.default_filters:
            acros.append(f.acronym)
        return acros

    @property
    def _type_dict(self):
        """Get dict of all attributes and corresponding string conversion funcs.
        """
        return od([("cam_id", str),
                   ("delim", str),
                   ("time_info_pos", int),
                   ("time_info_str", str),
                   ("filter_id_pos", int),
                   ("texp_pos", int),
                   ("texp_unit", str),
                   ("file_type", str),
                   ("main_filter_id", str),
                   ("meas_type_pos", int),
                   ("darkcorr_opt", int),
                   ("focal_length", float),
                   ("pix_height", float),
                   ("pix_width", float),
                   ("pixnum_x", int),
                   ("pixnum_y", int),
                   ("default_filters", list),
                   ("dark_info", list),
                   ("reg_shift_off", list),
                   ("io_opts", dict)])

    @property
    def _info_dict(self):
        """Return dict containing information strings for all attributes."""
        return od([("cam_id", "ID of camera within code"),
                   ("delim", "Filename delimiter for info access"),
                   ("time_info_pos", ("Position (int) of acquisition time"
                                      " info in filename after splitting "
                                      "using delim")),
                   ("time_info_str", ("String formatting information for "
                                      "datetimes in filename")),
                   ("filter_id_pos", ("Position (int) of filter acronym "
                                      "string in filename after splitting"
                                      " using delim")),
                   ("texp_pos", ("Position of acquisition time info "
                                 "filename after splitting using delim")),
                   ("texp_unit", ("Unit of exposure time in filename"
                                  "choose between s or ms")),
                   ("file_type", "Filetype information (e.g. tiff)"),
                   ("main_filter_id", "String ID of main filter (e.g. on)"),
                   ("meas_type_pos", ("Position of meastype specification "
                                      "in filename after splitting using "
                                      "delim. Only applies to certain "
                                      "cameras(e.g. HD cam)")),
                   ("darkcorr_opt", "Camera dark correction mode"),
                   ("focal_length", "Camera focal length in m"),
                   ("pix_height", "Detector pixel height in m"),
                   ("pix_width", "Detector pixel width in m"),
                   ("pixnum_x", "Detector number of pixels in x dir"),
                   ("pixnum_y", "Detector number of pixels in y dir"),
                   ("default_filters", ("A Python list containing pyplis"
                                        "Filter objects")),
                   ("dark_info", ("A Python list containing pyplis"
                                  "DarkOffsetInfo objects"))])

    def load_info_dict(self, info_dict):
        """Load all valid data from input dict.

        :param dict info_dict: dictionary specifying camera information

        """
        types = self._type_dict
        filters = []
        dark_info = []
        missed = []
        err = []
        for key, func in types.items():
            if key in info_dict:
                try:
                    val = func(info_dict[key])
                    if key == "default_filters":
                        for f in val:
                            try:
                                wl = f[4]
                            except BaseException:
                                wl = nan
                            try:
                                f = Filter(id=f[0], type=f[1],
                                           acronym=f[2],
                                           meas_type_acro=f[3],
                                           center_wavelength=wl)
                                filters.append(f)
                            except BaseException:
                                warn("Failed to convert %s into Filter"
                                     " class in Camera" % f)

                    elif key == "dark_info":
                        for f in val:
                            try:
                                rg = int(f[4])
                            except BaseException:
                                rg = 0
                            try:
                                i = DarkOffsetInfo(id=f[0], type=f[1],
                                                   acronym=f[2],
                                                   meas_type_acro=f[3],
                                                   read_gain=rg)
                                dark_info.append(i)
                            except BaseException:
                                warn("Failed to convert %s into DarkOffsetInfo"
                                     " class in Camera" % f)
                    else:
                        self[key] = val
                except BaseException:
                    err.append(key)
            else:
                missed.append(key)

        try:
            self.image_import_method = getattr(custom_image_import,
                                               info_dict["image_import_method"]
                                               )

        except BaseException:
            pass

        self.default_filters = filters
        self.dark_info = dark_info
        self.io_opts = info_dict["io_opts"]
        self.update_file_access_flags()

        return missed, err

    def load_default(self, cam_id, cam_info_file: Optional[Path] = None):
        """Load information from one of the implemented default cameras.

        :param str cam_id: id of camera (e.g. "ecII")
        :param Path cam_info_file: file from which camera info is to be loaded
        """
        info = get_camera_info(cam_id, cam_info_file)
        self.load_info_dict(info)

    """
    Helpers, supplemental stuff...
    """

    @property
    def dark_acros(self):
        """Return list containing filename access acronyms for dark images."""
        acros = []
        for item in self.dark_info:
            if item.acronym not in acros and item.type == "dark":
                acros.append(item.acronym)
        return acros

    @property
    def dark_meas_type_acros(self):
        """Return list containing meas_type_acros of dark images."""
        acros = []
        for item in self.dark_info:
            if item.meas_type_acro not in acros and item.type == "dark":
                acros.append(item.meas_type_acro)
        return acros

    @property
    def offset_acros(self):
        """Return list containing filename access acronyms for dark images."""
        acros = []
        for item in self.dark_info:
            if item.acronym not in acros and item.type == "offset":
                acros.append(item.acronym)
        return acros

    @property
    def offset_meas_type_acros(self):
        """Return list containing meas_type_acros of dark images."""
        acros = []
        for item in self.dark_info:
            if item.meas_type_acro not in acros and item.type == "offset":
                acros.append(item.meas_type_acro)
        return acros

    def get_acronym_dark_offset_corr(self, read_gain=0):
        """Get file name acronyms for dark and offset image identification.

        Parameters
        -----------
        read_gain : 
            detector read gain. Default is 0.
        
        Returns 
        --------
        offs : 
            offset image acronym (None, if not found).
        dark :
            dark image acronym (None, if not found).
        """
        offs = None
        dark = None
        for info in self.dark_info:
            if info.type == "dark" and info.read_gain == read_gain:
                dark = info.acronym
            elif info.type == "offset" and info.read_gain == read_gain:
                offs = info.acronym
        return offs, dark

    def to_dict(self) -> dict:
        """Convert to dictionary
        
        Returns
        -------
        dictionary containing all camera information.
        """
        d = od()
        for k in self._type_dict.keys():
            if k in ["default_filters", "dark_info"]:
                d[k] = []
                for f in self[k]:
                    d[k].append(f.to_list())
            else:
                d[k] = self[k]
        try:
            ipm = self.image_import_method.__name__
        except BaseException:
            ipm = ""
        d["image_import_method"] = ipm
        return d

    def save_as_default(self, cam_info_file: Optional[Path], *add_cam_ids) -> None:
        """Save this camera in default camera registry
        
        Parameters
        ----------
        cam_info_file : 
            Path to the camera info file to be used for saving the default camera.
        *add_cam_ids :
            Additional camera IDs to be added to the registry.
        """
        cam_ids = [self.cam_id]
        cam_ids.extend(add_cam_ids)
        info_dict = od([("cam_ids", cam_ids)])
        info_dict.update(self.to_dict())
        save_new_default_camera(info_dict=info_dict, cam_info_file=cam_info_file)

    def _all_params(self) -> list:
        """Return list of all relevant camera parameters
        
        Returns
        -------
        list of all camera parameters.
        """
        return list(self._type_dict.keys())

    def _short_str(self):
        """Short string repr."""
        s = ""
        for key in self._type_dict:
            # print key in ["defaultFilterSetup", "dark_img_info"]
            val = self(key)
            if key in ["default_filters", "dark_info"]:
                pass
            else:
                s += f"{key}: {val}\n"

        s += f"image_import_method: {self.image_import_method}\n"
        s += "\nDark & offset info\n------------------------\n"
        for i in self.dark_info:
            s += (f"ID: {i.id}, type: {i.type}, acronym: {i.acronym}, "
                  f"meas_type_acro: {i.meas_type_acro}, read_gain: {i.read_gain}\n")
        return s

    def __str__(self):
        s = ("\nCameraBaseInfo\n-------------------------\n\n")
        for key in self._type_dict:
            val = self(key)
            if key in ["default_filters", "dark_info"]:
                for info in val:
                    s += "%s\n" % info
            else:
                s += "%s: %s\n" % (key, val)
        return s

    def __setitem__(self, key: str, value: Any):
        """Set class item.

        :param str key: valid class attribute
        :param value: new value
        """
        if key in self.__dict__:
            self.__dict__[key] = value

    def __getitem__(self, key):
        """Get current item.

        :param str key: valid class attribute
        """
        if key in self.__dict__:
            return self.__dict__[key]

    def __call__(self, key):
        """Make object callable (access item)."""
        return self.__getitem__(key)
