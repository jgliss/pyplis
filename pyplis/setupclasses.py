# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)
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
"""Setup classes to specify relevant parameters for the emission-rate analysis.

The most important ones are:

    1. :class:`Source`: emission source specifications
    #. :class:`Camera`: camera specifications
    #. :class:`MeasSetup`: full measurement setup
"""
from datetime import datetime
from collections import OrderedDict as od
from copy import deepcopy
from os.path import exists
from pathlib import Path
from typing import Optional
from numpy import nan, rad2deg, arctan, ndarray

from pyplis import logger
from pyplis.forms import LineCollection, RectCollection
from pyplis.helpers import isnum, to_datetime
from pyplis.exceptions import MetaAccessError, DeprecationError
from pyplis.inout import get_source_info, save_default_source
from pyplis.utils import Filter
from pyplis.camera_base_info import CameraBaseInfo
from pyplis.geometry import MeasGeometry


class Source(object):
    """Object containing information about emission source.

    Attributes
    ----------
    name : str
        string ID of source
    lon : float
        longitude of source
    lat : float
        latitude of source
    altitude : float
        altitude of source
    suppl_info : dict
        dictionary containing supplementary information (e.g. source type,
        region, country)

    Parameters
    ----------
    name : str
        string ID of source (default is "")
    info_dict : dict
        dictionary contatining source information (is only loaded if
        all necessary parameters are available and in the right format)

    Note
    ----

    If input param ``name`` is a valid default ID (e.g. "Etna") then
    the source information is extracted from the database and the
    parameter ``info_dict`` is ignored.

    """

    def __init__(self, name=None, info_dict=None, **kwargs):
        if info_dict is None:
            info_dict = {}
        if name is None:
            name = ""
        self.name = name
        self.lon = nan
        self.lat = nan
        self.altitude = nan

        self.suppl_info = od([("status", ""),
                              ("country", ""),
                              ("region", ""),
                              ("type", ""),
                              ("last_eruption", "")])

        if isinstance(name, str):
            info = self.get_info(name, try_online=False)
            if bool(info):
                info_dict.update(info)

        self._import_from_dict(info_dict)
        for k, v in kwargs.items():
            self[k] = v

    @property
    def source_id(self):
        """Get ID of source.

        Returns
        -------
        str
            ``self.name``

        """
        return self.name

    @property
    def info_available(self):
        """Check if main information is available."""
        return all([x is not None for x in [self.lon, self.lat,
                                            self.altitude]])

    @property
    def geo_data(self):
        """Return dictionary containing lon, lat and altitude."""
        return od([("lon", self.lon),
                   ("lat", self.lat),
                   ("altitude", self.altitude)])

    @property
    def _type_dict(self):
        """Get dict of all attributes and corresponding string conversion funcs.
        """
        return od([("name", str),
                   ("lat", float),
                   ("lon", float),
                   ("altitude", float),
                   ("status", str),
                   ("country", str),
                   ("region", str),
                   ("type", str),
                   ("last_eruption", str)])

    def to_dict(self):
        """Return dictionary of all parameters.

        Returns
        -------
        dict
            dictionary representation of class

        """
        d = self.geo_data
        d["name"] = self.name
        d.update(self.suppl_info)
        return d

    def load_source_info(self, name=None, try_online=True):
        """Try to load source info from external database.

        Try to find source info in pyplis database file my_sources.txt and
        if it cannot be found there, try online, if applicable.

        Parameters
        ----------
        name : str
            if provided, a volcano with the corresponding name is searched. If
            not provided, the current name is used
        try_online : bool
            if True, online search is attempted in case information cannot be
            found in my_sources.txt

        """
        info = self.get_info(name, try_online)
        self._import_from_dict(info)

    def _import_from_dict(self, info_dict):
        """Try access default information of source.

        Parameters
        ----------
        info_dict : dict
            dictonary containing source information (valid keys are all
            keys ``self._type_dict``, e.g. ``lon``, ``lat``, ``altitude``)

        Returns
        -------
        bool
            success

        """
        types = self._type_dict
        if not isinstance(info_dict, dict):
            raise TypeError(
                "need dictionary like object for source info update")
        err = []
        for key, val in info_dict.items():
            if key in types:
                func = types[key]
            else:
                func = str
            try:
                self[key] = func(val)
            except BaseException:
                err.append(key)
        if bool(err) > 0:
            logger.info("Failed to load the following source parameters\n%s" % err)

        return self.info_available

    def save_to_database(self):
        """Save the current information as a new source.

        The information is stored in the *my_sources.txt* file that can be
        found in the pyplis installation folder *my_pyplis*
        """
        save_default_source(self.to_dict())

    def get_info(self, name=None, try_online=True):
        """Load source info from database.

        Looks if desired source (specified by argument `name`) can be found in
        the *my_sources.txt* file and if not, tries to find information about
        the source online (if :param:`try_online` is True)

        Parameters
        ----------
        name :  str
            source ID
        try_online : bool
            if True, also search online database

        Returns
        -------
        dict
            Dictionary containing source information

        """
        if name is None:
            name = self.name
        res = get_source_info(name, try_online)
        num = len(res)
        if num == 0:
            return {}
        elif num == 1:
            return list(res.values())[0]
        else:
            logger.info("\nMultiple occurences found for %s" % name)
            ok = 0
            logger.info(list(res.keys()))
            while not ok:
                try:
                    inp = input("\nEnter, key:\n")
                    return res[inp]
                except:
                    logger.info(list(res.keys()))
                    logger.info("Retry...")

    def _all_params(self):
        """Return list of all relevant source attributes."""
        return list(self._type_dict.keys())

    def __str__(self):
        s = ("\nSource\n-------------------------\n")
        for key in self._type_dict.keys():
            s = s + "%s: %s\n" % (key, self(key))
        return s

    def __repr__(self):
        return f"Source({self.name})"
    
    def __setitem__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.suppl_info[key] = value

    def __getitem__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self.suppl_info:
            return self.suppl_info[name]

    def __call__(self, key):
        return self.__getitem__(key)


class FilterSetup:
    """A collection of :class:`pyplis.utils.Filter` objects.

    This collection specifies a filter setup for a camera. A typical setup
    would be one on and one off band filter. An instance of this class is
    created automatically as an attribute of :class:`Camera` objects.

    Parameters
    ----------
    filter_list : list
        list of :class:`pyplis.utils.Filter` objects specifying
        camera filter setup
    default_key_on : str
        string ID of default on band filter (only relevant if collection
        contains more than one on band filter)
    default_key_off : str
        string ID of default off band filter (only relevant if collection
        contains more than one off band filter)

    """

    def __init__(self, filter_list=None, default_key_on=None,
                 default_key_off=None, **filters):
        if filter_list is None:
            filter_list = []

        self._filters = {}
        self.init_filters(filter_list, **filters)

        self._default_key_on = default_key_on
        self._default_key_off = default_key_off

    @property
    def filters(self):
        """Get dict containing filters (only getter, for backwards compat)."""
        return self._filters

    @property
    def on_band(self):
        """Return default on band filter."""
        return self._filters[self.default_key_on]

    @property
    def off_band(self):
        """Return default on band filter."""
        try:
            return self._filters[self.default_key_off]
        except BaseException:
            raise TypeError("Collection does not contain off band filter")

    @property
    def ids_off(self):
        """List with all offband filter ids."""
        return self.get_ids_on_off()[1]

    @property
    def ids_on(self):
        """List with all onband filter ids."""
        return self.get_ids_on_off()[0]

    @property
    def default_key_on(self):
        """Return default onband key."""
        if self._default_key_on is not None:
            return self._default_key_on
        ids_on = self.ids_on
        if len(ids_on) == 0:
            raise KeyError('No onband filter set in FilterSetup')
        return ids_on[0]

    @default_key_on.setter
    def default_key_on(self, val):
        if val not in self.ids_on:
            raise ValueError('Cannot set default key {} for onband filter. No '
                             'such filter available in FilterSetup. Choose '
                             'from; {}'.format(val, self.ids_on))
        self._default_key_on = val

    @property
    def default_key_off(self):
        """Return default offband key."""
        if self._default_key_off is not None:
            return self._default_key_off
        ids_off = self.ids_off
        if len(ids_off) == 0:
            raise KeyError('No offband filter set in FilterSetup')
        return ids_off[0]

    @default_key_off.setter
    def default_key_off(self, val):
        if val not in self.ids_off:
            raise ValueError('Cannot set default key {} for offband filter. No'
                             ' such filter available in FilterSetup. Choose'
                             ' from; {}'.format(val, self.ids_off))
        self._default_key_off = val

    @property
    def has_on(self):
        """Check if collection contains an onband filter."""
        if bool(self.ids_on):
            return True
        return False

    @property
    def has_off(self):
        """Check if collection contains an onband filter."""
        if bool(self.ids_off):
            return True
        return False

    @property
    def number_of_filters(self):
        """Return the current number of filters in this collection."""
        return len(self._filters)

    def init_filters(self, filter_list=None, **filters):
        """Initialize the filter collection (old settings will be deleted).

        The filters will be written into the dictionary ``self._filters``
        in the list order, keys are the filter ids

        Parameters
        ----------
        filters : list
            list of :class:`pyplis.utils.Filter` objects
            specifying camera filter setup
        **filters
            pairs of filter IDs and instances of :class:`Filter` that may be
            used instead of (or in addition to) input `filter_list`

        """
        if isinstance(filter_list, (list, ndarray)):
            for f in filter_list:
                self[f.id] = f
        for fid, f in filters.items():
            self[fid] = f

        if not bool(self._filters):
            self._filters["on"] = Filter("on")

    def __setitem__(self, key, val):
        if not isinstance(val, Filter):
            raise ValueError('Invalid input: need instance of Filter class, '
                             'got {}'.format(val))
        if key in self._filters:
            logger.warning('Filter with ID {} already exists in FilterSetup '
                 'and will be overwritten'.format(key))
        self._filters[key] = val

    def __getitem__(self, key):
        if key not in self._filters:
            raise KeyError('No such filter assigned to FilterSetup: {}. '
                           'Please choose from {}'.format(key, self._filters))
        return self._filters[key]

    def update_filters_from_dict(self, filter_dict):
        """Add filter objects from a dictionary.

        Parameters
        ----------
        filter_dict : dict
            dictionary, containing filter information

        """
        for f in filter_dict.values():
            if isinstance(f, Filter):
                if f.id in self._filters:
                    logger.info("Filter %s was overwritten" % f.id)
                self._filters[f.id] = f

    def set_default_filter_keys(self, default_key_on=None,
                                default_key_off=None):
        """Deprecated."""
        raise DeprecationError('Deprecated since v1.4.4: please use setter '
                               'methods directly for '
                               'attrs. default_key_on and default_key_off')

    def check_default_filters(self):
        """Check if default filter keys are set."""
        raise DeprecationError('Deprecated since v1.4.4: please use setter '
                               'methods directly for '
                               'attrs. default_key_on and default_key_off')

    def get_ids_on_off(self):
        """Get all filters sorted by their type (On or Off).

        Returns
        -------
        tuple
            2-element tuple containing

            - list, contains all on band IDs
            - list, contains all off band IDs

        """
        ids_on, ids_off = [], []
        for key in self._filters:
            if self._filters[key].type == "on":
                ids_on.append(key)
            elif self._filters[key].type == "off":
                ids_off.append(key)
        return (ids_on, ids_off)

    def print_setup(self):
        """Print the current setup.

        Returns
        -------
        str
            print string representation

        """
        s = ("pyplis FilterSetup\n------------------------------\n"
             "All filters:\n\n")
        for flt in self._filters.values():
            s += ("%s" % flt)
        s += "Default Filter: %s\n\n" % self.default_key_on
        logger.info(s)
        return s

    def __len__(self):
        return len(self._filters)

    def __call__(self, filter_id):
        """Return the filter corresponding to the input ID.

        :param str filter_id: string ID of filter
        """
        return self._filters[filter_id]

    def __str__(self):
        s = ""
        for f in self._filters.values():
            s += ("%s, type: %s (%s): %s nm\n"
                  % (f.id, f.type, f.acronym, f.center_wavelength))
        s += "Default Filter: %s\n" % self.default_key_on
        return s


class Camera(CameraBaseInfo):
    """Base class to specify a camera setup.

    Class representing a UV camera system including detector specifications,
    optics, file naming convention and the bandpass filters that are
    equipped with the camera (managed via an instance of the
    :class:`FilterSetup` class).

    Parameters
    ----------
    cam_id : str
        camera ID (e.g "ecII"), if this ID corresponds to one of the
        default cameras, the information is automatically loaded from
        supplementary file *cam_info.txt*
    filter_list : list
        list containing :class:`pyplis.utils.Filter` objects specifying
        the camera filter setup. If unspecified (empty list) and input
        param ``cam_id`` is a valid default ID, then the default filter
        setup of the camera will be loaded.
    default_filter_on : str
        string ID of default on band filter (only relevant if collection
        contains more than one on band filter)
    default_filter_off : str
        string ID of default off band filter (only relevant if collection
        contains more than one off band filter)
    ser_no : int
        optional, camera serial number
    cam_info_file : Path
        camera info file from which the default information for provided 
        `cam_id` shall be imported.
    try_load_from_registry : bool 
        if True, and if `cam_id` is provided, the camera information is 
        loaded from the pyplis camera registry. Default is True.
    **geom_info :
        additional keyword args specifying geometrical information, e.g.
        lon, lat, altitude, elev, azim

    Examples
    --------
    Example creating a new camera (using ECII default info with custom
    filter setup)::

        import pyplis

        #the custom filter setup
        filters= [pyplis.utils.Filter(type="on", acronym="F01"),
                  pyplis.utils.Filter(type="off", acronym="F02")]

        cam = pyplis.setupclasses.Camera(cam_id="ecII", filter_list=filters,
                                          lon=15.11, lat=37.73, elev=18.0,
                                          elev_err=3, azim=270.0,
                                          azim_err=10.0, focal_lengh=25e-3)
        print cam

    """

    def __init__(
            self, 
            cam_id: Optional[str] = None, 
            filter_list: Optional[list] = None, 
            default_filter_on=None,
            default_filter_off=None, 
            ser_no=9999, 
            cam_info_file: Optional[Path]=None,
            try_load_from_registry=True,
            **geom_info):

        if filter_list is None:
            filter_list = []

        super(Camera, self).__init__(cam_id, cam_info_file, try_load_from_registry=try_load_from_registry)

        # specify the filters used in the camera and the main filter (e.g. On)
        self.ser_no = ser_no  # identifier of camera
        self.geom_data = od([("lon", None),
                             ("lat", None),
                             ("altitude", None),
                             ("azim", None),
                             ("azim_err", None),
                             ("elev", None),
                             ("elev_err", None),
                             ("alt_offset", 0.0)])

        for k, v in geom_info.items():
            self[k] = v

        self.filter_setup = None

        self.prepare_filter_setup(filter_list, default_filter_on,
                                  default_filter_off)

    @property
    def lon(self):
        """Camera longitude."""
        return self.geom_data["lon"]

    @lon.setter
    def lon(self, val):
        if not -180 <= val <= 180:
            raise ValueError("Invalid input for longitude, must be between"
                             "-180 and 180")
        self.geom_data["lon"] = val

    @property
    def lat(self):
        """Camera latitude."""
        return self.geom_data["lat"]

    @lat.setter
    def lat(self, val):
        if not -90 <= val <= 90:
            raise ValueError("Invalid input for longitude, must be between"
                             "-90 and 90")
        self.geom_data["lat"] = val

    @property
    def altitude(self):
        """Camera altitude in m.

        Note
        ----
        This is typically the local topography altitude, which can for
        instance be accessed automatically based on camera position (lat, lon)
        using :func:`get_altitude_srtm`. Potential offsets (i.e. elevated
        positioning due to tripod or measurement from a house roof) can be
        specified using :attr:`alt_offset`.
        """
        return self.geom_data["altitude"]

    @altitude.setter
    def altitude(self, val):
        self.geom_data["altitude"] = val

    @property
    def elev(self):
        """Return viewing elevation angle (center pixel) in degrees.

        0 refers to horizon, 90 to zenith
        """
        return self.geom_data["elev"]

    @elev.setter
    def elev(self, val):
        self.geom_data["elev"] = val

    @property
    def elev_err(self):
        """Uncertainty in viewing elevation angle in degrees."""
        return self.geom_data["elev_err"]

    @elev_err.setter
    def elev_err(self, val):
        self.geom_data["elev_err"] = val

    @property
    def azim(self):
        """Return viewing azimuth angle in deg relative to north (center pixel).
        """
        return self.geom_data["azim"]

    @azim.setter
    def azim(self, val):
        self.geom_data["azim"] = val

    @property
    def azim_err(self):
        """Uncertainty in viewing azimuth angle in degrees."""
        return self.geom_data["azim_err"]

    @azim_err.setter
    def azim_err(self, val):
        self.geom_data["azim_err"] = val

    @property
    def alt_offset(self):
        """Height of camera position above topography in m.

        This offset can be added in case the camera is positioned above the
        ground and is only required if :param:`altitude` corresponds to the
        topographic elevation
        """
        return self.geom_data["alt_offset"]

    @alt_offset.setter
    def alt_offset(self, val):
        self.geom_data["alt_offset"] = val

    def load_default(self, cam_id: str, cam_info_file: Optional[Path] = None):
        """Redefinition of method from base class :class:`CameraBaseInfo`."""
        super(Camera, self).load_default(cam_id, cam_info_file=cam_info_file)
        self.prepare_filter_setup()

    def update(self, **settings):
        """Update camera parameters.

        Parameters
        ----------
        settings : dict
            dictionary containing camera parametrs (valid keys are
            all keys of ``self.__dict__`` and from dictionary
            ``self.geom_data``)

        """
        for key, val in settings.items():
            self[key] = val

    def get_altitude_srtm(self):
        """Try load camera altitude based on lon, lat and SRTM topo data.

        Note
        ----
        Requires :mod:`geonum` package to be installed and :attr:`lon` and
        :attr:`lat` to be set.

        """
        try:
            from geonum import GeoPoint
            lon, lat = float(self.lon), float(self.lat)
            self.altitude = GeoPoint(lat, lon).altitude
        except Exception as e:
            logger.warning(f"Failed to automatically access local topography "
                           f"altitude at camera position using SRTM data: {e}")

    def prepare_filter_setup(
            self, 
            filter_list: Optional[list] = None, 
            default_key_on: Optional[str] = None,
            default_key_off: Optional[str] = None):
        """Create :class:`FilterSetup` object.

        This method defines the camera filter setup based on an input list of
        :class:`Filter` instances.

        Parameters
        ----------
        filter_list :
            list containing :class:`pyplis.utils.Filter` objects
        default_key_on :
            string specifiying the string ID of the main onband filter of
            the camera (usually "on"). If unspecified (None), then the
            ID of the first available on bandfilter in the filter input
            list will be used.
        default_key_off : 
            string specifiying the string ID of the main offband filter
            of the camera (usually "on"). If unspecified (None), then the
            ID of the first available off band filter in the
            filter input list will be used.

        """
        if filter_list is None or len(filter_list) == 0:
            filter_list = self.default_filters
            default_key_on = self.main_filter_id

        filter_list = deepcopy(filter_list)
        self.filter_setup = FilterSetup(filter_list, default_key_on,
                                        default_key_off)
        # overwrite default filter information
        self.default_filters = []
        for f in filter_list:
            self.default_filters.append(f)

    def to_dict(self):
        """Convert this object into a dictionary."""
        d = super(Camera, self).to_dict()
        d["ser_no"] = self.ser_no
        for key, val in self.geom_data.items():
            d[key] = val
        return d

    def change_camera(self, cam_id=None, make_new=False, **kwargs):
        """Change current camera type.

        Parameters
        ----------
        cam_id : str
            ID of new camera
        make_new : bool
            if True, a new instance will be created and returned
        **kwargs
            additional keyword args (see :func:`__init__`)

        Returns
        -------
        Camera
            either this object (if :param:`make_new` is False) or else, new
            instance

        """
        if "geom_data" not in kwargs:
            kwargs["geom_data"] = self.geom_data
        if make_new:
            return Camera(cam_id, **kwargs)

        self.__init__(cam_id, **kwargs)
        return self

    def dx_to_decimal_degree(self, pix_num_x):
        """Convert horizontal distance (in pixel units) into angular range.

        Parameters
        ----------
        pix_num_x : int
            number of pixels for which angular range is determined

        Returns
        -------
        float
            dx in units of decimal degrees

        """
        try:
            len_phys = self.pix_width * pix_num_x
            return rad2deg(arctan(len_phys / self.focal_length))
        except BaseException:
            raise MetaAccessError("Please check availability of focal "
                                  "length, and pixel pitch (pix_width)")

    def dy_to_decimal_degree(self, pix_num_y):
        """Convert vertical distance (in pixel units) into angular range.

        Parameters
        ----------
        pix_num_y : int
            number of pixels for which angular range is determined

        Returns
        -------
        float
            dy in units of decimal degrees

        """
        try:
            len_phys = self.pix_height * pix_num_y
            return rad2deg(arctan(len_phys / self.focal_length))
        except BaseException:
            raise MetaAccessError("Please check availability of focal "
                                  "length, and pixel pitch (pix_height)")

    def __str__(self):
        s = ("%s, serno. %s\n-------------------------\n"
             % (self.cam_id, self.ser_no))
        s += self._short_str()
        s += "\nFilters\n----------------------\n"
        s += str(self.filter_setup)

        s += "\nGeometry info\n----------------------\n"
        for key, val in self.geom_data.items():
            try:
                s += "%s: %.3f\n" % (key, val)
            except BaseException:
                s += "%s: %s\n" % (key, val)

        return s
    
    def __repr__(self):
        return f"Camera({self.cam_id})"

    def __setitem__(self, key, value):
        """Set item method."""
        if key in self.__dict__:
            self.__dict__[key] = value
        elif key in self.geom_data:
            self.geom_data[key] = value

    def __getitem__(self, name):
        """Get class item."""
        if name in self.__dict__:
            return self.__dict__[name]
        for k, v in self.__dict__.items():
            try:
                if name in v:
                    return v[name]
            except BaseException:
                pass


class FormSetup(object):
    """Setup class for all forms (lines, rectangles etc.) used for evaluation.
    """

    def __init__(self, line_dict=None, rect_dict=None):
        if line_dict is None:
            line_dict = {}
        if rect_dict is None:
            rect_dict = {}
        self.id = "forms"
        self.lines = LineCollection(line_dict)
        self.rects = RectCollection(rect_dict)

    def __str__(self):
        s = "pyplis FormSetup\n-----------------------------------\n\n"
        s += "Lines: %s\n" % self.lines
        s += "Rects: %s\n" % self.rects
        return s


class BaseSetup(object):
    """Abstract base class for basic measurement setup.

    Specifies image base path and start / stop time stamps of measurement
    as well as the following boolean access flags:

        1. :attr:`USE_ALL_FILES`
        #. :attr:`SEPARATE_FILTERS`
        #. :attr:`USE_ALL_FILE_TYPES`
        #. :attr:`INCLUDE_SUB_DIRS`
        #. :attr:`ON_OFF_SAME_FILE`
        #. :attr:`LINK_OFF_TO_ON`
        #. :attr:`REG_SHIFT_OFF`

    Parameters
    ----------
    base_dir : str
        Path were e.g. imagery data lies
    start : datetime
        start time of Dataset (can also be datetime.time)
    stop : datetime
        stop time of Dataset (can also be datetime.time)
    **opts
        setup options for file import (see specs above)

    """

    def __init__(
            self, 
            base_dir: str, 
            start: Optional[datetime] = None, 
            stop: Optional[datetime] = None, 
            **opts):
        
        if start is None:
            use_all_files = True
            start = datetime(1900, 1, 1)
        else:
            use_all_files = False
        if stop is None:
            stop = datetime(1900, 1, 1)

        self.base_dir = base_dir
        self.save_dir = base_dir

        self._start = None
        self._stop = None

        self.options = od([("USE_ALL_FILES", use_all_files),
                           ("SEPARATE_FILTERS", True),
                           ("USE_ALL_FILE_TYPES", False),
                           ("INCLUDE_SUB_DIRS", False),
                           ("ON_OFF_SAME_FILE", False),
                           ("LINK_OFF_TO_ON", True),
                           ("REG_SHIFT_OFF", False)])
        self.start = start
        self.stop = stop

        for k, v in opts.items():
            self.options[k] = v
            
    @property
    def start(self):
        """Start time of setup."""
        return self._start

    @start.setter
    def start(self, val):
        self._start = to_datetime(val)
        self.USE_ALL_FILES = False
        
    @property
    def stop(self):
        """Stop time of setup."""
        return self._stop

    @stop.setter
    def stop(self, val):
        self._stop = to_datetime(val)
        self.USE_ALL_FILES = False
    
    @property
    def USE_ALL_FILES(self):
        """File import option (boolean).

        If True, all files in image base folder are used (i.e. start / stop
        time stamps are disregarded)
        """
        return self.options["USE_ALL_FILES"]

    @USE_ALL_FILES.setter
    def USE_ALL_FILES(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["USE_ALL_FILES"] = bool(value)

    @property
    def SEPARATE_FILTERS(self):
        """File import option (boolean).

        If true, files are separated by filter type (e.g. "on", "off")
        """
        return self.options["SEPARATE_FILTERS"]

    @SEPARATE_FILTERS.setter
    def SEPARATE_FILTERS(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["SEPARATE_FILTERS"] = value

    @property
    def USE_ALL_FILE_TYPES(self):
        """File import option (boolean).

        If True, all files found are imported, disregarding the file type
        (i.e. if image file type is not specified. It is strongly recommended
        NOT to use this option)
        """
        return self.options["USE_ALL_FILE_TYPES"]

    @USE_ALL_FILE_TYPES.setter
    def USE_ALL_FILE_TYPES(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["USE_ALL_FILE_TYPES"] = value

    @property
    def INCLUDE_SUB_DIRS(self):
        """File import option (boolean).

        If True, sub directories are included into image search
        """
        return self.options["INCLUDE_SUB_DIRS"]

    @INCLUDE_SUB_DIRS.setter
    def INCLUDE_SUB_DIRS(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["INCLUDE_SUB_DIRS"] = value

    @property
    def ON_OFF_SAME_FILE(self):
        """File import option (boolean).

        If True, it is assumed, that each image file contains both on and
        offband images. In this case, both the off and the onband image lists
        are filled with the same file paths. Which image to load in each list
        is then handled within the :class:`ImgList`itself on :func:`load`
        using the attribute :attr:`list_id` which is passed using the key
        ``filter_id`` to the respective customised image import method that
        has to be defined in the :mod:`custom_image_import` file of the pyplis
        installation and linked to your Camera settings in the ``cam_info.txt``
        file which can be found in the data directory of the installation.

        An example for such a file convention is the SO2 camera from CVO (USGS)
        See e.g. :func:`load_usgs_multifits` in :mod:`custom_image_import`.
        """
        return self.options["ON_OFF_SAME_FILE"]

    @ON_OFF_SAME_FILE.setter
    def ON_OFF_SAME_FILE(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["ON_OFF_SAME_FILE"] = value

    @property
    def LINK_OFF_TO_ON(self):
        """File import option (boolean).

        If True, the offband ImgList is automatically linked to the onband
        list on initiation of a :class:`Dataset` object.
        """
        return self.options["LINK_OFF_TO_ON"]

    @LINK_OFF_TO_ON.setter
    def LINK_OFF_TO_ON(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["LINK_OFF_TO_ON"] = value

    @property
    def REG_SHIFT_OFF(self):
        """File import option (boolean).

        If True, the images in an offband image list that is linked to an
        onband image list (cf. :attr:`LINK_OFF_TO_ON`) are shifted using the
        registration offset specified in the  ``reg_shift_off`` attribute
        of the :class:`Camera` instance.
        """
        return self.options["REG_SHIFT_OFF"]

    @REG_SHIFT_OFF.setter
    def REG_SHIFT_OFF(self, value):
        if int(value) not in [0, 1]:
            raise ValueError("need boolean")
        self.options["REG_SHIFT_OFF"] = value

    def base_info_check(self):
        """Check if all necessary information if available.

        Checks if path and times are valid

        Returns
        -------
            2-element tuple, containing

            - bool, True or False
            - str, information

        """
        ok = 1
        s = ("Base info check\n-----------------------------\n")
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
        """Check if input is integer or float and not nan.

        Parameters
        ----------
        val
            object to be tested

        Returns
        -------
        bool

        """
        return isnum(val)

    def _dict_miss_info_str(self, key, val):
        """Return a string notification for invalid value."""
        return "Missing / wrong information: %s, %s\n" % (key, val)

    def __str__(self):
        s = ("\nSetup\n---------\n\n"
             "Base path: %s\n"
             "Save path: %s\n"
             "Start: %s\n"
             "Stop: %s\n"
             "Options:\n"
             % (self.base_dir, self.save_dir, self.start, self.stop))

        for key, val in self.options.items():
            s = s + "%s: %s\n" % (key, val)

        return s


class MeasSetup(BaseSetup):
    """Setup class for plume image data.

    In this class, everything related to a full measurement setup is
    defined. This includes the image base directory, start / stop time
    stamps (if applicable), specifications of the emission source (i.e.
    :class:`Source` object), camera specifications (i.e. :class:`Camera`
    object) as well as meteorology information (i.e. wind direction and
    velocity). The latter is not represented as an own class in Pyplis but
    is stored as a Python dictionary. :class:`MeasSetup` objects are the
    default input for :class:`pyplis.dataset.Dataset` objects (i.e. also
    :class:`pyplis.cellcalib.CellCalibEngine`).

    Parameters
    ----------
    base_dir : str
        Path were e.g. imagery data lies
    start : datetime
        start time of Dataset (may as well be datetime.time)
    stop : datetime
        stop time of Dataset (may as well be datetime.time)
    camera : Camera
        general information about the camera used
    source : Source
        information about emission source (e.g. lon, lat, altitude)
    **opts :
        setup options for file handling (currently only INCLUDE_SUB_DIRS
        option)

    """

    def __init__(self, base_dir=None, start=None, stop=None, camera=None,
                 source=None, wind_info=None, cell_info_dict=None, rects=None,
                 lines=None, auto_topo_access=True, **opts):

        super().__init__(base_dir, start, stop, **opts)

        if cell_info_dict is None:
            cell_info_dict = {}
        if rects is None:
            rects = {}
        if lines is None:
            lines = {}
        if not isinstance(camera, Camera):
            camera = Camera()
        if not isinstance(source, Source):
            source = Source()
        self.auto_topo_access = auto_topo_access

        self.camera = camera
        self.source = source

        self.cell_info_dict = cell_info_dict
        self.forms = FormSetup(lines, rects)

        self.wind_info = od([("dir", None),
                             ("dir_err", None),
                             ("vel", None),
                             ("vel_err", None)])

        if isinstance(wind_info, dict):
            self.update_wind_info(wind_info)

        self.meas_geometry = MeasGeometry(
            self.source.to_dict(),
            self.camera.to_dict(),
            self.wind_info,
            auto_topo_access=self.auto_topo_access
        )
        # If specified in custom camera, update the file I/O options
        # defined in :class:`BaseSetup`
        self.options.update(self.camera.io_opts)

    def update_wind_info(self, info_dict):
        """Update wind info dict using valid entries from input dict.

        Parameters
        ----------
        info_dict : dict
            dictionary containing wind information

        """
        for key, val in info_dict.items():
            if key in self.wind_info:
                self.wind_info[key] = val

    def base_info_check(self):
        """Check if all req. info is available."""
        ok = 1
        s = ("Base info check\n-----------------------------\n")
        if not self.base_dir or not exists(self.base_dir):
            ok = 0
            s += "Image base path does not exist\n"
        if not self.USE_ALL_FILES:
            if not isinstance(self.start, datetime) or\
                    not isinstance(self.stop, datetime):
                s += "Start / Stop info wrong datatype (need datetime)\n"
                ok = 0
        ok, info = self.check_geometry_info()
        s += info

        return ok, s

    def check_geometry_info(self):
        """Check if all req. info for measurement geometry is available.

        Relevant parameters are:

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
        s = ("\n------------------------------\nChecking basic geometry info"
             "\n------------------------------\n")
        for key, val in self.camera.geom_data.items():
            if not self._check_if_number(val):
                ok = 0
                s += "Missing info in Camera setup\n"
                s += self._dict_miss_info_str(key, val)
        for key in self.camera.optics_keys:
            val = self.camera[key]
            if not self._check_if_number(val):
                ok = 0
                s += self._dict_miss_info_str(key, val)
        for key, val in source.geo_data.items():
            if not self._check_if_number(val):
                ok = 0
                s += "Missing info in Source: %s\n" % source.name
                s += self._dict_miss_info_str(key, val)
        for key, val in self.wind_info.items():
            if not self._check_if_number(val):
                ok = 0
                s += "Missing Meteorology info\n"
                s += self._dict_miss_info_str(key, val)
        if ok:
            s += "All necessary information available\n"

        logger.info(s)

        return ok, s

    def update_meas_geometry(self):
        """Update the meas geometry based on current settings."""
        logger.info("Updating MeasGeometry in MeasSetup class")
        self.meas_geometry.__init__(self.source.to_dict(),
                                    self.camera.to_dict(), self.wind_info,
                                    auto_topo_access=self.auto_topo_access)

    def short_str(self):
        """Return a short info string."""
        return f"{super().__str__()}\nCamera: {repr(self.camera)}\nSource: {repr(self.source)}"

    def __setitem__(self, key, value):
        """Update class item."""
        if key in self.__dict__:
            self.__dict__[key] = value

    def __getitem__(self, key):
        """Load value of class item."""
        if key in self.__dict__:
            return self.__dict__[key]

    def __str__(self):
        """Detailed information string."""
        s = super(BaseSetup, self).__str__() + "\n\n"
        s += "Meteorology info\n-----------------------\n"
        for key, val in self.wind_info.items():
            s += "%s: %s\n" % (key, val)
        s += "\n" + str(self.camera) + "\n"
        s += str(self.source)
        if self.cell_info_dict.keys():
            s += "\nCell specifications:\n"
            for key, val in self.cell_info_dict.items():
                s += "%s: %s +/- %s\n" % (key, val[0], val[1])
        return s