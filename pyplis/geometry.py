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
"""Module containing functionality for all relevant geometrical calculations.
"""
from numpy import (nan, arctan, deg2rad, linalg, ndarray, sqrt, abs, array,
                   tan, rad2deg, linspace, isnan, asarray,
                   arange, argmin, newaxis, round, ndarray)
from typing import Tuple
from collections import OrderedDict as od

from matplotlib.pyplot import figure
from copy import deepcopy

from pyplis import logger, print_log
from pyplis.utils import LineOnImage
from pyplis.image import Img
from pyplis.helpers import check_roi, isnum
from pyplis.glob import DEFAULT_ROI

from geonum import GeoSetup, GeoPoint, GeoVector3D, TopoData
from geonum.exceptions import TopoAccessError
from geonum.helpers import haversine_formula

class MeasGeometry(object):
    """Class for calculations and management of the measurement geometry.

    All calculations are based on provided information about camera (stored in
    dictionary :attr:`_cam`, check e.g. ``self._cam.keys()`` for valid keys),
    source (stored in dictionary :attr:`_source`, check e.g.
    ``self._source.keys()`` for valid keys) and meteorological wind direction
    (stored in dictionary :attr:`_wind`). The keys of these dictionaries (i.e.
    identifiers for the variables) are the same as the corresponding attributes
    in the respective classes :class:`pyplis.Camera` and
    :class:`pyplis.Source`.

    If you want to change these parameters, it is recommended to use the
    correpdonding update methods :func:`update_cam_specs`,
    :func:`update_source_specs` and :func:`update_wind_specs` or use the
    provided getter / setter methods for each parameter (e.g.,
    :attr:`cam_elev` for key ``elev`` of :attr:`_cam` dictionary,
    :attr:`cam_azim` for key ``azim`` of :attr:`_cam` dictionary,
    :attr:`cam_lon` for key ``lon`` of :attr:`_cam` dictionary,
    :attr:`cam_lat` for key ``lat`` of :attr:`_cam` dictionary,
    :attr:`source_lon` for key ``lon`` of :attr:`_source` dictionary,
    :attr:`source_lat` for key ``lat`` of :attr:`_source` dictionary,
    :attr:`wind_dir` for key ``dir`` of :attr:`_dir` dictionary.

    Note that in the dictionary based update methods :func:`update_cam_specs`,
    :func:`update_source_specs` and :func:`update_wind_specs`, the dict keys
    are supposed to be inserted, e.g.::

        geom = MeasGeometry()
        # either update using valid keywords as **kwargs ...
        geom.update_cam_specs(lon=10, lat=20, elev=30, elev_err=0.5)

        # ... or update using a dictionary containing camera info, e.g.
        # retrieved from an existing camera ...
        cam = pyplis.Camera(altitude=1234, azim=270, azim_err=10)
        cam_dict = cam.to_dict()

        geom.update_cam_specs(cam_dict)

        # ... or directly using the getter / setter attributes
        print geom.cam_altitude #1234 (value of geom._cam["altitude"])

        geom.cam_altitude=111
        print geom.cam_altitude #111 (new value of geom._cam["altitude"])

        # analogous with source and wind

        # This ...
        geom.update_wind_specs(dir=180, dir_err=22)

        # ... is the same as this:
        geom.wind_dir=180
        geom.wind_dir_err=22

        # load Etna default source info
        source = pyplis.Source("etna")
        geom.update_source_specs(**source.to_dict())


    The latter
    by default also update the most important attribute of this class
    :attr:`geo_setup` which is an instance of the :class:`geonum.GeoSetup`
    class and which is central for all geometrical calculations (e.g. camera
    to plume distance).

    Attributes
    ----------
    geo_setup : GeoSetup
        class containing information about the current measurement setup.
        Most of the relevant geometrical calculations are performed within
        this object
    _source : dict
        dictionary containing information about emission source (valid
        keys: ``name, lon, lat, altitude``)
    _wind : dict
        dictionary containing information about meteorology at source
        position (valid keys: ``dir, dir_err, velo, velo_err``)
    _cam : dict
        dictionary containing information about the camera (valid keys:
        ``cam_id, serno, lon, lat, altitude, elev, elev_err, azim,
        azim_err, focal_length, pix_width, pix_height, pixnum_x, pixnum_y
        alt_offset``

    Parameters
    ----------
    source_info : dict
        dictionary containing source parameters (see :attr:`source` for
        valid keys)
    cam_info : dict
        dictionary containing camera parameters (see :attr:`cam` for
        valid keys)
    wind_info : dict
        dictionary conatining meteorology information (see :attr:`wind`
        for valid keys)

    """

    def __init__(self, source_info=None, cam_info=None, wind_info=None,
                 auto_topo_access=True):

        self._source = od([("name", ""),
                           ("lon", nan),
                           ("lat", nan),
                           ("altitude", nan)])

        # Note 22/02/2018 Removed "velo" and "velo_err" from _wind dict
        # since it is not used
        self._wind = od([("dir", nan),
                         ("dir_err", nan)])

        self._cam = od([("cam_id", ""),
                        ("serno", 9999),
                        ("lon", nan),
                        ("lat", nan),
                        ("altitude", nan),
                        ("elev", nan),
                        ("elev_err", nan),
                        ("azim", nan),
                        ("azim_err", nan),
                        ("focal_length", nan),  # in m
                        ("pix_width", nan),  # in m
                        ("pix_height", nan),  # in m
                        ('pixnum_x', nan),
                        ('pixnum_y', nan),
                        ('alt_offset', 0.0)])  # altitude above
        # topo in m
        self.auto_topo_access = auto_topo_access
        self.geo_setup = GeoSetup(id=self.cam_id)
        if source_info:
            self.update_source_specs(source_info, update_geosetup=False)
        if cam_info:
            self.update_cam_specs(cam_info, update_geosetup=False)
        if wind_info:
            self.update_wind_specs(wind_info, update_geosetup=False)
        if any([bool(x) is True for x in [source_info, cam_info, wind_info]]):
            self.update_geosetup()

    @property
    def _type_dict(self):
        """Return dictionary containing required data types for attributes."""
        return od([("dir", float),
                   ("dir_err", float),
                   ("velo", float),
                   ("velo_err", float),
                   ("name", str),
                   ("cam_id", str),
                   ("serno", int),
                   ("lon", float),
                   ("lat", float),
                   ("altitude", float),
                   ("elev", float),
                   ("elev_err", float),
                   ("azim", float),
                   ("azim_err", float),
                   ("focal_length", float),  # in m
                   ("pix_width", float),  # in m
                   ("pix_height", float),  # in m
                   ('pixnum_x', float),
                   ('pixnum_y', float),
                   ('alt_offset', float)])

    @property
    def cam_id(self):
        """ID of current camera (string)."""
        return self._cam["cam_id"]

    @cam_id.setter
    def cam_id(self, value):
        self._cam["cam_id"] = self._type_dict["cam_id"](value)

    @property
    def cam_serno(self):
        """Return serial number of camera."""
        return self._cam["serno"]

    @cam_serno.setter
    def cam_serno(self, value):
        self._cam["serno"] = self._type_dict["serno"](value)

    @property
    def cam_lon(self):
        """Longitude position of camera."""
        return self._cam["lon"]

    @cam_lon.setter
    def cam_lon(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["lon"] = self._type_dict["lon"](val)
        self.update_geosetup()

    @property
    def cam_lat(self):
        """Latitude position of camera."""
        return self._cam["lat"]

    @cam_lat.setter
    def cam_lat(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["lat"] = self._type_dict["lat"](val)
        self.update_geosetup()

    @property
    def cam_altitude(self):
        """Altitude of camera position."""
        return self._cam["elev"]

    @cam_altitude.setter
    def cam_altitude(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["altitude"] = self._type_dict["altitude"](val)
        self.update_geosetup()

    @property
    def cam_elev(self):
        """Elevation angle of camera viewing direction (CFOV)."""
        return self._cam["elev"]

    @cam_elev.setter
    def cam_elev(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["elev"] = self._type_dict["elev"](val)
        self.update_geosetup()

    @property
    def cam_elev_err(self):
        """Elevation angle error of camera viewing direction (CFOV)."""
        return self._cam["elev_err"]

    @cam_elev_err.setter
    def cam_elev_err(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["elev_err"] = self._type_dict["elev_err"](val)
        self.update_geosetup()

    @property
    def cam_azim(self):
        """Azimuth of camera viewing direction (CFOV)."""
        return self._cam["azim"]

    @cam_azim.setter
    def cam_azim(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["azim"] = self._type_dict["azim"](val)
        self.update_geosetup()

    @property
    def cam_azim_err(self):
        """Azimuth error of camera viewing direction (CFOV)."""
        val = self._cam["azim_err"]
        return val

    @cam_azim_err.setter
    def cam_azim_err(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["azim_err"] = self._type_dict["azim_err"](val)
        self.update_geosetup()

    @property
    def cam_focal_length(self):
        """Focal length of camera."""
        val = self._cam["focal_length"]
        return val

    @cam_focal_length.setter
    def cam_focal_length(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["focal_length"] = self._type_dict["focal_length"](val)

    @property
    def cam_pix_width(self):
        """Pixel width of camera detector (horizonzal pix-to-pix distance)."""
        return self._cam["pix_width"]

    @cam_pix_width.setter
    def cam_pix_width(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["pix_width"] = self._type_dict["pix_width"](val)

    @property
    def cam_pix_height(self):
        """Pixel height of camera detector (vertical pix-to-pix distance)."""
        return self._cam["pix_height"]

    @cam_pix_height.setter
    def cam_pix_height(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["pix_height"] = self._type_dict["pix_height"](val)

    @property
    def cam_pixnum_x(self):
        """Return Number of camera detector pixels in x-direction (horizontal).
        """
        return self._cam["pixnum_x"]

    @cam_pixnum_x.setter
    def cam_pixnum_x(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["pixnum_x"] = self._type_dict["pixnum_x"](val)

    @property
    def cam_pixnum_y(self):
        """Return Number of camera detector pixels in y-direction (vertical).
        """
        return self._cam["pixnum_y"]

    @cam_pixnum_y.setter
    def cam_pixnum_y(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["pixnum_y"] = self._type_dict["pixnum_y"](val)

    @property
    def cam_altitude_offs(self):
        """Camera elevation above topography.

        Note
        ----
        This can be used as offset above the ground, if the camera altitude
        (:attr:`cam_altitude`) is retrieved based on local topography level
        (e.g. using automatic SRTM access based on camera lat and lon).
        """
        return self._cam["alt_offset"]

    @cam_altitude_offs.setter
    def cam_altitude_offs(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._cam["alt_offset"] = self._type_dict["alt_offset"](val)

    @property
    def source_lon(self):
        """Longitude position of source."""
        return self._source["lon"]

    @source_lon.setter
    def source_lon(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._source["lon"] = self._type_dict["lon"](val)
        self.update_geosetup()

    @property
    def source_lat(self):
        """Latitude position of source."""
        return self._source["lat"]

    @source_lat.setter
    def source_lat(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._source["lat"] = self._type_dict["lat"](val)
        self.update_geosetup()

    @property
    def source_altitude(self):
        """Altitude of source position."""
        return self._source["altitude"]

    @source_altitude.setter
    def source_altitude(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._source["altitude"] = self._type_dict["altitude"](val)
        self.update_geosetup()

    @property
    def wind_dir(self):
        """Azimuth of wind direction."""
        return self._wind["dir"]

    @wind_dir.setter
    def wind_dir(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._wind["dir"] = self._type_dict["dir"](val)
        self.update_geosetup()

    @property
    def wind_dir_err(self):
        """Azimuth error of wind direction."""
        return self._wind["dir_err"]

    @wind_dir_err.setter
    def wind_dir_err(self, val):
        if not isnum(val):
            raise ValueError("Invalid value: %s (need numeric)" % val)
        self._wind["dir_err"] = self._type_dict["dir_err"](val)
        self.update_geosetup()

    def _update_specs_helper(self, info_dict: dict, attr_name: str, update_geosetup: bool) -> None:
        """
        Helper function to update camera, source or wind information.

        Args:
            info_dict: Dictionary containing information to update.
            attr_name: Name of the attribute to update. Choose from
                _cam, _source, or _wind.
            update_geosetup: If True, the method :func:`update_geosetup` is called at the end of this method.
        """
        types = self._type_dict
        for key in self.__dict__[attr_name]:
            if key in info_dict:
                raw_val = info_dict[key]
                if raw_val is None:
                    continue
                try:
                    val = types[key](raw_val)
                except Exception as e:
                    raise ValueError(f"Failed to cast value '{raw_val}' (type {type(raw_val)}) to type {types[key]} for key {key} in MeasGeometry.{attr_name}")
                self.__dict__[attr_name][key] = val
        if update_geosetup:
            self.update_geosetup()

    def update_cam_specs(
            self, 
            info_dict: dict, 
            update_geosetup=True):
        """Update camera settings.

        Update dictionary containing geometrical camera information
        (:attr:`cam`) by providing a dictionary containing valid key / value
        pairs for camera parameters.

        Parameters
        ----------
        info_dict : dict
            dictionary containing camera information (see :attr:`cam` for
            valid keys)
        update_geosetup : bool
            If True, the method :func:`update_geosetup` is called at the end
            of this method
        """
        self._update_specs_helper(info_dict, "_cam", update_geosetup)
        
    def update_source_specs(
            self, 
            info_dict: dict, 
            update_geosetup: bool = True
        ) -> None:
        """Update source settings.

        Update source info dictionary (:attr:`source`) either by providing a
        dictionary containing valid key / value pairs (:param:`info_dict` or by
        providing valid key / value pairs directly using :param:`kwargs`)

        Parameters
        ----------
        info_dict : dict
            dictionary containing source information (see :attr:`source`
            for valid keys)
        update_geosetup : bool
            If True, the method :func:`update_geosetup` is called at the end
            of this method
        """
        self._update_specs_helper(info_dict, "_source", update_geosetup)

    def update_wind_specs(self, info_dict, update_geosetup=True):
        """Update meteorological settings.

        Update wind info dictionary (:attr:`wind`) either by providing a
        dictionary containing valid key / value pairs (:param:`info_dict` or by
        providing valid key / value pairs directly using :param:`kwargs`)

        Parameters
        ----------
        info_dict : dict
            dictionary containing meterology information (see :attr:`wind`
            for valid keys)
        update_geosetup : bool
            If True, the method :func:`update_geosetup` is called at the end
            of this method
        """
        self._update_specs_helper(info_dict, "_wind", update_geosetup)

    def _check_all_info_avail_for_geosetup(self):
        """Check if relevant information for :attr:`geo_setup` is ready."""
        check = ["lon", "lat"]
        cam_ok, source_ok = True, True
        for key in check:
            if key in self._cam and not isnum(self._cam[key]):
                cam_ok = False
            if key in self._source and not isnum(self._source[key]):
                source_ok = False
        if not isnum(self._wind["dir"]) and cam_ok and isnum(self.cam_azim):
            logger.info("setting orientation angle of wind direction relative to camera cfov")
            self._wind["dir"] = (self._cam["azim"] + 90) % 360
            self._wind["dir_err"] = 45.0

        return cam_ok, source_ok

    def _update_geopoint(self, point_name: str, point_coords: dict) -> GeoPoint:
        """
        Update a geographic point in the geo setup.
        
        Parameters
        ----------
        point_name : str
            The name of the geographic point to update.
        point_coords : dict
            A dictionary containing the coordinates of the point. 
            Expected keys are "lat" for latitude, "lon" for longitude, 
            and "altitude" for altitude.
        
        Returns
        -------
        GeoPoint which was added the the :attr:`geo_setup`.
        """
        
        altitude = point_coords["altitude"]
        valid_altitude = altitude is not None and not isnan(altitude)
        retrieve_altitude = True if self.auto_topo_access and not valid_altitude else False
        pt = GeoPoint(point_coords["lat"], point_coords["lon"],altitude, 
                       name=point_name, auto_topo_access=retrieve_altitude)
        self.geo_setup.add_geo_point(pt, overwrite_existing=True)
        return pt
    
    def update_geosetup(self) -> bool:
        """Update the current GeoSetup object.

        Note
        ----
        The borders of the range are determined considering cam pos, source
        pos and the position of the cross section of viewing direction with
        plume

        Returns
        -------
        Whether or not all required information for analysing the geometry is available.

        """
        mag = 20.0  # init magnitude in km for lon / lat range of GeoSetup
        cam_ok, source_ok = self._check_all_info_avail_for_geosetup()
        all_ok = True
        if cam_ok:
            cam = self._update_geopoint(point_name="cam", point_coords=self._cam)
            logger.info("Updated camera in GeoSetup of MeasGeometry")
        if source_ok:
            source = self._update_geopoint(point_name="source", point_coords=self._source)
            logger.info("Updated source in GeoSetup of MeasGeometry")
        if cam_ok and source_ok:
            try:
                source2cam = cam - source  # Vector pointing from source to cam
                mag = source2cam.norm  # length of this vector
                source2cam.name = "source2cam"
                self.geo_setup.add_geo_vector(source2cam, overwrite_existing=True)
                logger.info("Updated source2cam GeoVector in GeoSetup of MeasGeometry")
            except Exception:
                print_log.warning("Failed to compute GeoVector between camera and source")
                all_ok = False
            try:
                # vector representing the camera center pix viewing direction
                # (CFOV), anchor at camera position
                cam_view_vec = GeoVector3D(azimuth=self._cam["azim"],
                                           elevation=self._cam["elev"],
                                           dist_hor=mag, anchor=cam,
                                           name="cfov")
                logger.info("Updated camera CFOV vector in GeoSetup of MeasGeometry")
                self.geo_setup.add_geo_vector(cam_view_vec, overwrite_existing=True)
            except BaseException:
                print_log.warning("Failed to compute camera CFOV GeoVector in GeoSetup of MeasGeometry")
                all_ok = False
            try:
                # vector representing the emission plume
                # (anchor at source coordinates)
                plume_vec = GeoVector3D(azimuth=self.plume_dir,
                                        dist_hor=mag, anchor=source,
                                        name="plume_vec")
                logger.info("Updated plume vector in GeoSetup of MeasGeometry")
                self.geo_setup.add_geo_vector(plume_vec, overwrite_existing=True)
            except BaseException:
                print_log.warning("Failed to compute plume GeoVector in GeoSetup of MeasGeometry")
                all_ok = False
            try:
                # horizontal intersection of plume and viewing direction
                offs = plume_vec.intersect_hor(cam_view_vec)
                # Geopoint at intersection
                intersect = source + offs
                intersect.name = "intersect"
                logger.info("Updated GeoPoint of intersection between camera CFOV "
                      "and plume vector in GeoSetup of MeasGeometry")
                self.geo_setup.add_geo_point(intersect, overwrite_existing=True)
            except BaseException:
                print_log.warning("Could not compute intersection point between camera CFOV"
                     " and plume vector in GeoSetup of MeasGeometry")
                all_ok = False
            try:
                self.geo_setup.set_borders_from_points(
                    extend_km=self._map_extend_km(),
                    to_square=True)
            except BaseException:
                pass
            if all_ok:
                logger.info("MeasGeometry was updated and fulfills all requirements")
                return True

        elif cam_ok:
            cam_view_vec = GeoVector3D(azimuth=self._cam["azim"],
                                       elevation=self._cam["elev"],
                                       dist_hor=mag, anchor=cam,
                                       name="cfov")
            self.geo_setup.add_geo_vector(cam_view_vec, overwrite_existing=True)
            logger.info("MeasGeometry was updated but misses source specifications")
        logger.info("MeasGeometry not (yet) ready for analysis")

        return False

    def get_coordinates_imgborders(self):
        """Get elev and azim angles corresponding to camera FOV."""
        det_width = self._cam["pix_width"] * self._cam["pixnum_x"]
        det_height = self._cam["pix_height"] * self._cam["pixnum_y"]
        del_az = rad2deg(arctan(det_width /
                                (2.0 * self._cam["focal_length"])))
        del_elev = rad2deg(arctan(det_height /
                                  (2.0 * self._cam["focal_length"])))

        return {"azim_left": self._cam["azim"] - del_az,
                "azim_right": self._cam["azim"] + del_az,
                "elev_bottom": self._cam["elev"] - del_elev,
                "elev_top": self._cam["elev"] + del_elev}

    def horizon_analysis(self, skip_cols=30):
        """Search pixel coordinates of horizon for image columns.

        The algorithm performs a topography analysis for a number of image
        columns. Elevation profiles are determined for each column (azimuth)
        and from those, the horizon elevation angle is searched. The retrieved
        values are returned in pixel coordinates.

        :param skip_cols: distance between pixel columns for which the analysis
            is performed

        .. note::

            This is a Beta version, please report any problems

        """
        cam = self.cam
        cols = arange(0, self._cam["pixnum_x"], skip_cols)
        rows = arange(0, self._cam["pixnum_y"], 1)
        azims, elevs = self.get_azim_elev(cols, rows)
        dist = self.geo_len_scale() * 1.2
        idx_x, idx_y, dists = [], [], []
        elev_min, elev_max = min(elevs), max(elevs)
        for k in range(len(azims)):
            azim = azims[k]
            elev_profile = cam.get_elevation_profile(azimuth=azim,
                                                     dist_hor=dist)
            (elev,
             elev_secs,
             dist_secs) = elev_profile.find_horizon_elev(
                elev_start=elev_min,
                elev_stop=elev_max,
                step_deg=0.1,
                view_above_topo_m=self._cam["alt_offset"]
            )

            idx_x.append(cols[k])
            idx_y.append(argmin(abs(elev - elevs)))
            try:
                dists.append(dist_secs[-1])
            except BaseException:
                logger.warning("Temporary solution, need a fix here...")
                dists.append(nan)
        return idx_x, idx_y, dists

    def get_viewing_directions_line(self, line: LineOnImage) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Determine viewing direction coords for a line in an image.

        Parameters
        ----------
        line : LineOnImage
            line on image object

        Returns
        -------
        tuple
            4-element tuple containing 4 arrays of same length:

            - 1-d array containing azimuth angles of pixels on line
            - 1-d array containing elevation angles of pixels on line
            - 1-d array containing corresponding x-pixel coordinates
            - 1-d array containing corresponding y-pixel coordinates

        """
        if not line.roi_abs_def == DEFAULT_ROI or line.pyrlevel_def > 0:
            print_log.warning("Input line is not in absolute detector coordinates "
                 "and will be converted to uncropped image coords at "
                 "pyrlevel 0")
            line = line.convert(to_pyrlevel=0, to_roi_abs=DEFAULT_ROI)
        line = line.to_list()
        f = self._cam["focal_length"]

        x0, y0, x1, y1 = line[0], line[1], line[2], line[3]
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0

        delx = abs(x1 - x0)
        dely = abs(y1 - y0)

        l = int(round(sqrt(delx ** 2 + dely ** 2)))
        x = linspace(x0, x1, l)
        y = linspace(y0, y1, l)
        dx = self._cam["pix_width"] * (x - self._cam["pixnum_x"] / 2)
        dy = self._cam["pix_height"] * (y - self._cam["pixnum_y"] / 2)
        azims = rad2deg(arctan(dx / f)) + self._cam["azim"]
        elevs = -rad2deg(arctan(dy / f)) + self._cam["elev"]
        return (azims, elevs, x, y)

    def get_topo_distance_pix(self, pos_x_abs, pos_y_abs, topo_res_m=5.,
                              min_slope_angle=5.):
        """Retrieve distance to topography for a certain image pixel.

        The computation of the distance is
        being done by retriving a elevation profile in the azimuthal viewing
        direction of tge pixel (i.e. pixel column) and then using this profile
        and the corresponding camera elevation (pixel row) to find the first
        intersection of the viewing direction (line) with the topography

        Parameters
        ----------
        pos_x_abs : int
            x-pixel position of point in image in absolute coordinate (i.e.
            pyramid level 0 and not cropped)
        pos_y_abs : int
            y-pixel position of point in image in absolute coordinate (i.e.
            pyramid level 0 and not cropped)
        topo_res_m : float
            desired resolution of topographic data (is interpolated)
        float min_slope_angle : float
            mininum required slope (steepness) of topography at pixel position
            (raises ValueError if topograpy is too flat)

        Returns
        -------
        tuple
            3-element tuple, containing

            - estimated distance to topography in m based on intersection of\
                pixel viewing direction with topographic data
            - corresponding uncertainty in m
            - :class:`GeoPoint` corresponding to intersection position

        """
        if "cam" not in self.geo_setup.points:
            raise AttributeError("Failed to retrieve distance to topo: geo "
                                 "location of camera is not available")
        if not isinstance(self.geo_setup.topo_data, TopoData):
            self.geo_setup.load_topo_data()
    
        azim = self.all_azimuths_camfov()[pos_x_abs]
        elev = self.all_elevs_camfov()[pos_y_abs]
        max_dist = self.geo_setup.vectors["source2cam"].magnitude * 1.10
        ep = self.get_elevation_profile(azim=azim,
                                        dist_hor=max_dist,
                                        topo_res_m=topo_res_m)

        (d, d_err, geo_point, _, _) = ep.get_first_intersection(
            elev,
            view_above_topo_m=self._cam["alt_offset"]
        )

        if d is None:
            raise ValueError("Distance to topography could not be retrieved")
        elif min_slope_angle > 0:
            slope = ep.slope_angle(d)
            if slope < min_slope_angle:
                raise ValueError("Topography at input point too flat")

        return (d, d_err, geo_point)

    def get_topo_distances_line(self, line, skip_pix=30, topo_res_m=5.,
                                min_slope_angle=5.):
        """Retrieve distances to topography for a line on an image.

        Calculates distances to topography based on pixels on the line. This is
        being done by retriving a elevation profile in the azimuthal viewing
        direction of each pixel (i.e. pixel column) and then using this profile
        and the corresponding camera elevation (pixel row) to find the first
        intersection of the viewing direction (line) with the topography

        :param list line: list with line coordinates: ``[x0, y0, x1, y1]``
            (can also be :class:`LineOnImage` object)
        :param int skip_pix: step width for retrieval along line
        :param float topo_res_m: desired resolution of topographic data
            (is interpolated)
        :param float min_slope_angle: mininum angle of slope, pixels
            pointing into flatter topographic areas are ignored
        """
        try:
            logger.info(self.cam)
        except BaseException:
            logger.info("Failed to retrieve distance to topo for line %s in "
                  "MeasGeometry: geo location of camera is not available"
                  % line)
            return False
        if not isinstance(self.geo_setup.topo_data, TopoData):
            try:
                self.geo_setup.load_topo_data()
            except BaseException:
                logger.info("Failed to retrieve distance to topo for line %s in "
                      "MeasGeometry: topo data could not be accessed..."
                      % line)
                return False
        azims, elevs, i_pos, j_pos = self.get_viewing_directions_line(line)
        cond = ~isnan(azims)
        # only consider points that are not nan
        (azims,
         elevs,
         i_pos,
         j_pos) = azims[cond], elevs[cond], i_pos[cond], j_pos[cond]
        if not len(azims) > 0:
            logger.info("Failed to retrieve distance to topo for line %s in "
                  "MeasGeometry: viewing directions (azim, elev) could not "
                  "be retrieved..." % line)
            return False
        # Take only every "skip_pix" pixel on the line
        azims, elevs = azims[::int(skip_pix)], elevs[::int(skip_pix)]
        i_pos, j_pos = i_pos[::int(skip_pix)], j_pos[::int(skip_pix)]

        max_dist = self.geo_setup.vectors["source2cam"].magnitude * 1.10
        # initiate results
        res = {"dists": [],
               "dists_err": [],
               "geo_points": [],
               "ok": []}

        for k in range(len(azims)):
            ep = None
            # try:
            ep = self.get_elevation_profile(azim=azims[k],
                                            dist_hor=max_dist,
                                            topo_res_m=topo_res_m)

            d, d_err, p, _, _ = ep.get_first_intersection(
                elevs[k],
                view_above_topo_m=self._cam["alt_offset"])

            if d is not None and min_slope_angle > 0:
                slope = ep.slope_angle(d)
                if slope < min_slope_angle:
                    logger.info("Slope angle too small, remove point at dist %.1f"
                          % d)
                    d = None
            ok = True
            if d is None:  # then, the intersection could be found
                ok = False
                d, d_err = nan, nan

            res["dists"].append(d)
            res["dists_err"].append(d_err)
            res["geo_points"].append(p)
            res["ok"].append(ok)

        res["azims"] = azims
        res["elevs"] = elevs
        res["i_pos"] = i_pos
        res["j_pos"] = j_pos
        for key in res:
            res[key] = asarray(res[key])
        return res

    def get_angular_displacement_pix_to_cfov(self, pos_x, pos_y):
        """Get the angular difference between pixel and detector center.

        :param int pos_x: x position on detector
        :param int pos_y: y position on detector
        """
        dx = self._cam["pix_width"] * (pos_x - self._cam["pixnum_x"] / 2)
        dy = self._cam["pix_height"] * (pos_y - self._cam["pixnum_y"] / 2)
        f = self._cam["focal_length"]
        del_az = rad2deg(arctan(dx / f))
        del_elev = rad2deg(arctan(dy / f))
        return del_az, del_elev

    def get_azim_elev(self, pos_x, pos_y):
        """Get values of azimuth and elevation in pixel (x|y).

        :param int pos_x: x position on detector
        :param int pos_y: y position on detector
        """
        del_az, del_elev = self.get_angular_displacement_pix_to_cfov(
            pos_x, pos_y)
        return self._cam["azim"] + del_az, self._cam["elev"] - del_elev

    def _check_topo(self):
        """Check if topo data can be accessed (returns True or False)."""
        if not isinstance(self.geo_setup.topoData, TopoData):
            try:
                self.geo_setup.load_topo_data()
                return True
            except Exception as e:
                logger.info("Failed to retrieve topo data in MeasGeometry..: %s"
                      % repr(e))
                return False
        return True

    def get_elevation_profile(self, col_num=None, azim=None, dist_hor=None,
                              topo_res_m=5.):
        """Retrieve elev profile from camera into a certain azim direction.

        :param int col_num: pixel column number of profile, if None or
            not in image detector range then try to use second input parameter
            azim
        :param float azim: is only used if input param col_num == None,
            then profile is retrieved from camera in direction of
            specified azimuth angle
        :param float dist_hor: horizontal distance (from camera, in km)
            up to witch the profile is determined. If None, then use 1.05 times
            the camera source distance
        :param float topo_res_m: desired horizontal grid resolution in m ()
        """
        az = azim
        if col_num and 0 <= col_num < self._cam["pixnum_x"]:
            az, _ = self.get_azim_elev(col_num, 0)

        if dist_hor is None:
            dist_hor = (self.cam - self._source).norm * 1.05
        p = self.cam.get_elevation_profile(
            azimuth=az, dist_hor=dist_hor, resolution=topo_res_m)
        logger.info("Succesfully determined elevation profile for az = %s" % az)
        return p

    def get_distance_to_topo(self, col_num=None, row_num=None, azim=None,
                             elev=None, min_dist=0.2, max_dist=None):
        """Determine distance to topography based on pixel coordinates.

        :param int col_num: pixel column number for elevation profile,
            from which the intersection with viewing direction is retrieved. If
            None or not in image detector range then try to use third input
            parameter (azim)
        :param int row_num: pixel row number for which the intersection
            with elevation profile is is retrieved. If None or not in image
            detector range then try to use 4th input parameter elev, row_num
            is only considerd if col_num is valid
        :param float azim: camera azimuth angle of intersection: is only
            used if input param col_num == None
        :param float elev: camera elevation angle for distance
            estimate: is only used if input param row_num == None
        :param float min_dist: minimum distance (in km) from camera for
            retrieval of first intersection. Intersections of viewing direction
            with topography closer than this distance are disregarded
            (default:  0.2)
        :param float max_dist: maximum distance (in km) from camera for
            which intersections with topography are searched

        """
        az, el = azim, elev
        try:
            # if input row and column are valid, use azimuth ane elevation
            # angles for the corresponding pixel
            if (0 <= col_num < self._cam["pixnum_x"]) and\
                    (0 <= row_num < self._cam["pixnum_y"]):
                az, el = self.get_azim_elev(col_num, row_num)
            # Check if azim and elev are valid numbers
            if not all([self._check_float(val) for val in [az, el]]):
                raise ValueError("Invalid value encounterd for azim, elev "
                                 "while trying to estimate cam to topo "
                                 "distance: %s,%s"
                                 % (az, el))
            # determine elevation profile
            p = self.get_elevation_profile(azim=az, dist_hor=max_dist)
            if not bool(p):
                raise TopoAccessError("Failed to retrieve topography profile")
            # Find first intersection
            d, d_err, pf = p.get_first_intersection(elev, min_dist)
            return d, d_err, pf
        except Exception as e:
            logger.info("Failed to retrieve distance to topo:" % repr(e))
            return False

    def _check_float(self, val):
        """Return bool."""
        if not isinstance(val, float) or isnan(val):
            return False
        return True

    def _correct_view_dir_lowlevel(self, pix_x, pix_y, obj_pos):
        # get the angular differnce of the object position to CFOV of camera
        del_az, del_elev = self.get_angular_displacement_pix_to_cfov(
            pix_x, pix_y)
        cam_pos = self.geo_setup.points["cam"]
        v = obj_pos - cam_pos

        az_obj = (v.azimuth + 360) % 360
        # rad2deg(arctan(delH/v.magnitude/1000))#the true elevation of the
        # object
        elev_obj = v.elevation
        elev_cam = elev_obj + del_elev
        az_cam = az_obj - del_az

        return elev_cam, az_cam

    def find_viewing_direction(self, pix_x, pix_y, pix_pos_err=10,
                               obj_id="", geo_point=None, lon_pt=None,
                               lat_pt=None, alt_pt=None, update=True,
                               draw_result=False):
        """Retrieve camera viewing direction from point in image.

        Uses the geo coordinates of a characteristic point in the image (e.g.
        the summit of a mountain) and the current position of the camera
        (Lon / Lat) to determine the viewing direction of the camera (azimuth,
        elevation).

        :param int pix_x: x position of object on camera detector
            (measured from left)
        :param int pix_y: y position of object on camera detector
            (measured from top)
        :param int pix_pos_err: radial uncertainty in pixel location (used to
            estimate and update
            ``self._cam["elev_err"], self._cam["azim_err"]``)
        :param bool update: if True current data will be updated and
            ``self.geo_setup`` will be updated accordingly
        :param str obj_id: string ID of object, if this object is available
            as :class:`GeoPoint` in ``self.geo_setup`` then the corresponding
            coordinates will be used, if not, please provide the position of
            the characteristic point either using :param:`geo_point` or by
            providing  its coordinates using params lat_pt, lon_pt, alt_pt
        :param GeoPoint geo_point: geo point object of characteristic point
        :param float lon_pt: longitude of characteristic point
        :param float lat_pt: latitude of characteristic point
        :param float alt_pt: altitude of characteristic point (unit m)
        :param bool update: if True, camera azim and elev are updated
            within this object
        :param bool draw_result: if True, a 2D map is drawn showing
            results

        :returns:
            - float, retrieved camera elevation
            - float, retrieved camera azimuth
            - MeasGeometry, initial state of this object, a deepcopy of\
            this class, before changes where applied (if they were applied,\
            see also `update`)

        """
        geom_old = deepcopy(self)
        if obj_id in self.geo_setup.points:
            obj_pos = self.geo_setup.points[obj_id]
        elif isinstance(geo_point, GeoPoint):
            obj_pos = geo_point
            self.geo_setup.add_geo_point(obj_pos)
        else:
            try:
                obj_pos = GeoPoint(lat_pt, lon_pt, alt_pt, name=obj_id)
                self.geo_setup.add_geo_point(obj_pos)
            except BaseException:
                raise IOError("Invalid input, characteristic point for "
                              "retrieval of viewing direction could not be "
                              "extracted from input params..")

        elev_cam, az_cam = self._correct_view_dir_lowlevel(pix_x, pix_y, obj_pos)

        pix_range_x = [pix_x - pix_pos_err,
                       pix_x - pix_pos_err]
        pix_range_y = [pix_y - pix_pos_err,
                       pix_y + pix_pos_err]
        elevs, azims = [], []
        for xpos in pix_range_x:
            for ypos in pix_range_y:
                elev, azim = self._correct_view_dir_lowlevel(xpos, ypos, obj_pos)
                elevs.append(elev), azims.append(azim)
        elev_err = max(abs(elev_cam - asarray(elevs)))
        azim_err = max(abs(az_cam - asarray(azims)))

        if update:
            elev_old, az_old = geom_old._cam["elev"], geom_old._cam["azim"]
            logger.info(f"Old Elev / Azim cam CFOV: {elev_old:.2f} / {az_old:.2f}")
            logger.info(f"New Elev / Azim cam CFOV: {elev_cam:.2f} / {az_cam:.2f}")
            self._cam["elev"] = elev_cam
            self._cam["azim"] = az_cam
            self._cam["elev_err"] = elev_err
            self._cam["azim_err"] = azim_err
            self.update_geosetup()

        map = None
        if draw_result:
            s = self.geo_setup
            nums = [int(255.0 / k) for k in range(1, len(s.vectors) + 3)]
            map = self.draw_map_2d(draw_fov=False)
            map.draw_geo_vector_2d(self.cam_view_vec,
                                   c=s.cmap_vecs(nums[1]),
                                   ls="-",
                                   label="cam cfov (corrected)")
            self.draw_azrange_fov_2d(map, poly_id="fov (corrected)")
            view_dir_vec_old = geom_old.geo_setup.vectors["cfov"]
            view_dir_vec_old.name = "cfov_old"

            map.draw_geo_vector_2d(view_dir_vec_old,
                                   c=s.cmap_vecs(nums[1]),
                                   ls="--",
                                   label="cam cfov (initial)")
            map.legend()

        return elev_cam, az_cam, geom_old, map

    def pix_dist_err(self, col_num, pyrlevel=0):
        """Get uncertainty measure for pixel distance of a pixel column.

        Parameters
        ----------
        colnum : int
           column number for which uncertainty in pix-to-pix distance is
           computed
        pyrlevel : int
            convert to pyramid level

        Returns
        -------
        float
            pix-to-pix distance in m corresponding to input column number and
            pyramid level

        """
        az = self.all_azimuths_camfov()[int(col_num)]
        return self.plume_dist_err(az) * self._cam["pix_width"] /\
            self._cam["focal_length"] * 2**pyrlevel

    def compute_all_integration_step_lengths(self, pyrlevel=0, roi_abs=None):
        """Determine images containing pixel and plume distances.

        Computes and returns three images where each pixel value corresponds
        to:

        1. the horizontal physical integration step length in units of m
        2. the vertical physical integration step length in units of m (is \
            the same as 1. for standard detectors where the vertical and \
            horizontal pixel pitch is the same)
        3. image where each pixel corresponds to the computed plume distance

        Parameters
        ----------
        pyrlevel : int
            returns images at a given gauss pyramid level
        roi_abs : list
            ROI ``[x0, y0, x1, y1]`` in absolute detector coordinates. If
            valid, then the images are cropped accordingly

        Returns
        -------
        tuple
            3-element tuple, containing

            - :obj:`Img`: image where each pixel corresponds to pixel\
                column distances in m
            - :obj:`Img`: image where each pixel corresponds to pixel\
                row distances in m (same as col_dist_img if pixel width and\
                height are equal)
            - :obj:`Img`: image where each pixel corresponds to plume\
                distance in m

        """
        ratio_hor = self._cam["pix_width"] / self._cam["focal_length"]  # in m
        
        azims = self.all_azimuths_camfov()
        elevs = self.all_elevs_camfov()

        plume_dists = self.plume_dist(azims, elevs)  # * 1000.0 #in m
        col_dists_m = plume_dists * ratio_hor

        # col_dists_m, plume_dists = self.calculate_pixel_col_distances()
        row_dists_m = (col_dists_m * self._cam["pix_height"] /
                       self._cam["pix_width"])

        col_dist_img = Img(col_dists_m)  # * ones(h).reshape((h, 1)))
        row_dist_img = Img(row_dists_m)  # * ones(h).reshape((h, 1)))
        plume_dist_img = Img(plume_dists)  # * ones(h).reshape((h, 1)))
        # the pix-to-pix distances need to be transformed based on pyrlevel
        col_dist_img.pyr_down(pyrlevel)
        col_dist_img = col_dist_img * 2**pyrlevel
        row_dist_img.pyr_down(pyrlevel)
        row_dist_img = row_dist_img * 2**pyrlevel
        plume_dist_img.pyr_down(pyrlevel)
        if check_roi(roi_abs):
            col_dist_img.crop(roi_abs)
            row_dist_img.crop(roi_abs)
            plume_dist_img.crop(roi_abs)
        return (col_dist_img, row_dist_img, plume_dist_img)

    def get_plume_direction(self):
        """Return the plume direction plus error based on wind direction."""
        return (self._wind["dir"] + 180) % 360, self._wind["dir_err"]

    """
    Plotting / visualisation etc
    """

    def plot_view_dir_pixel(self, col_num, row_num):
        """2D plot of viewing direction within elevation profile.

        Determines and plots elevation profile for azimuth angle of input pixel
        coordinate (column number). The viewing direction line is plotted based
        on the specified elevation angle (corresponding to detector row number)

        :param int col_num: column number of pixel on detector
        :param int row_num: row number of pixel on detector (measured from
            top)
        :returns: elevation profile

        """
        azim, elev = self.get_azim_elev(col_num, row_num)
        sc = self.geo_len_scale()
        ep = self.get_elevation_profile(azim=azim, dist_hor=sc * 1.10)
        if not bool(ep):
            raise TopoAccessError("Failed to retrieve topography profile")
        # Find first intersection
        d, d_err, pf = ep.get_first_intersection(elev, min_dist=sc * 0.05,
                                                 plot=True)
        return ep

    def draw_map_2d(self, draw_cam=True, draw_source=True, draw_plume=True,
                    draw_fov=True, draw_topo=True, draw_coastline=True,
                    draw_mapscale=True, draw_legend=True, *args, **kwargs):
        """Draw the current setup in a map.

        :param bool draw_cam: insert camera position into map
        :param bool draw_source: insert source position into map
        :param bool draw_plume: insert plume vector into map
        :param bool draw_fov: insert camera FOV (az range) into map
        :param bool draw_topo: plot topography
        :param bool draw_coastline: draw coastlines
        :param bool draw_mapscale: insert a map scale
        :param bool draw_legend: insert a legend
        :param *args: additional non-keyword arguments for setting up the base
            map (`see here <http://matplotlib.org/basemap/api/basemap_api.
            html#mpl_toolkits.basemap.Basemap>`_)
        :param **kwargs: additional keyword arguments for setting up the base
            map (`see here <http://matplotlib.org/basemap/api/basemap_api.html
            #mpl_toolkits.basemap.Basemap>`_)
        """
        s = self.geo_setup
        nums = [int(255.0 / k) for k in range(1, len(s.vectors) + 3)]
        m = s.plot_2d(0, 0, draw_topo, draw_coastline, draw_mapscale,
                      draw_legend=0, *args, **kwargs)

        if draw_cam:
            m.draw_geo_point_2d(self.cam)
            m.write_point_name_2d(self.cam,
                                  self.geo_setup.magnitude * .05, -45)
        if draw_source:
            m.draw_geo_point_2d(self.source)
            m.write_point_name_2d(self.source,
                                  self.geo_setup.magnitude * .05, -45)
        if draw_plume:
            m.draw_geo_vector_2d(self.plume_vec,
                                 c=s.cmap_vecs(nums[0]),
                                 ls="-",
                                 label="plume direction")
        if draw_fov:
            m.draw_geo_vector_2d(self.cam_view_vec,
                                 c=s.cmap_vecs(nums[1]),
                                 label="camera cfov")
            self.draw_azrange_fov_2d(m)
        if draw_legend:
            m.legend()
        return m

    def draw_azrange_fov_2d(self, m, fc="lime", ec="none", alpha=0.15,
                            poly_id="fov"):
        """Insert the camera FOV in a 2D map.

        :param geonum.mapping.Map m: the map object
        :param fc: face color of polygon
        :Param ec: edge color of polygon
        :param float alpha: alpha value of polygon
        """
        coords = self.get_coordinates_imgborders()
        l = self.geo_len_scale() * 1.5
        pl = self.cam.offset(azimuth=coords["azim_left"], dist_hor=l)
        pr = self.cam.offset(azimuth=coords["azim_right"], dist_hor=l)
        pts = [self.cam, pl, pr]
        m.add_polygon_2d(pts, poly_id=poly_id, fc=fc, ec=ec,
                         alpha=alpha)

    def draw_map_3d(self, draw_cam=True, draw_source=True, draw_plume=True,
                    draw_fov=True, cmap_topo="Oranges",
                    contour_color="#708090", contour_antialiased=True,
                    contour_lw=0.2, ax=None, **kwargs):
        """Draw the current setup in a 3D map.

        Parameters
        ----------
        draw_cam : bool
            insert camera position into map
        draw_source : bool
            insert source position into map
        draw_plume : bool
            insert plume vector into map
        draw_fov : bool
            insert camera FOV (az range) into map
        cmap_topo : str
            string ID of colormap for topography surface plot, defaults to
            "Oranges"
        contour_color : str
            string specifying color of contour lines colors of topo contour
            lines (default: "#708090")
        contour_antialiased : bool
            apply antialiasing to surface plot of topography, defaults to False
        contour_lw :
            width of drawn contour lines, defaults to 0.5, use 0 if you do not
            want contour lines inserted
        ax : Axes3D
            3D axes object (default: None -> creates new one)
        *args :
            non-keyword arguments for setting up the base map
            (`see here <http://matplotlib.org/basemap/api/basemap_api.
            html#mpl_toolkits.basemap.Basemap>`_)
        **kwargs: keyword arguments for setting up the basemap
            (`see here <http://matplotlib.org/basemap/api/basemap_api.html
            #mpl_toolkits.basemap.Basemap>`_)

        Returns
        -------
        Basemap
            plotted basemap

        """
        if ax is None:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = figure(figsize=(14, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        s = self.geo_setup
        m = s.plot_3d(False, False, cmap_topo=cmap_topo,
                      contour_color=contour_color,
                      contour_antialiased=contour_antialiased,
                      contour_lw=contour_lw, ax=ax, **kwargs)

        zr = self.geo_setup.topo_data.alt_range * 0.05
        if draw_cam:
            self.cam.plot_3d(m, add_name=True, dz_text=zr)
        if draw_source:
            self.source.plot_3d(m, add_name=True, dz_text=zr)
        if draw_fov:
            self.draw_azrange_fov_3d(m)
        if draw_plume:
            m.draw_geo_vector_3d(self.plume_vec)
        try:
            m.legend()
        except BaseException:
            pass
        return m

    def draw_azrange_fov_3d(self, m, fc="lime", ec="none", alpha=0.8):
        """Insert the camera FOV in a 2D map.

        :param geonum.mapping.Map m: the map object
        :param fc: face color of polygon ("lime")
        :Param ec: edge color of polygon ("none")
        :param float alpha: alpha value of polygon (0.8)
        """
        coords = self.get_coordinates_imgborders()
        v = self.geo_setup.points["intersect"] - self.cam
        pl = self.cam.offset(azimuth=coords["azim_left"],
                             dist_hor=v.dist_hor, dist_vert=v.dz)
        pr = self.cam.offset(azimuth=coords["azim_right"],
                             dist_hor=v.dist_hor, dist_vert=v.dz)
        pts = [self.cam, pl, pr]
        m.add_polygon_3d(pts, poly_id="fov",
                         facecolors=fc, edgecolors=ec,
                         alpha=alpha, zorder=1e8)

    """
    Helpers
    """
    @property
    def plume_dir(self):
        """Return current plume direction angle."""
        return self.get_plume_direction()[0]

    @property
    def plume_dir_err(self):
        """Return uncertainty in current plume direction angle."""
        return self.get_plume_direction()[1]

    @property
    def cam(self):
        """Camera location (:class:`geonum.GeoPoint`)."""
        return self.geo_setup.points["cam"]

    @property
    def source(self):
        """Return camera Geopoint."""
        return self.geo_setup.points["source"]

    @property
    def intersect_pos(self):
        """Return camera Geopoint."""
        return self.geo_setup.points["intersect"]

    @property
    def plume_vec(self):
        """Return the plume center vector."""
        return self.geo_setup.vectors["plume_vec"]

    @property
    def source2cam(self):
        """Return vector pointing camera to source."""
        return self.geo_setup.vectors["source2cam"]

    @property
    def cam_view_vec(self):
        """Return vector corresponding to CFOV azimuth of camera view dir."""
        return self.geo_setup.vectors["cfov"]

    def haversine(self, lon0, lat0, lon1, lat1, radius=6371.0):
        """Haversine formula to compute distances on a sphere

        Approximate horizontal distance between 2 points assuming a spherical
        earth.

        Parameters
        ----------
        lon0 : float
            longitude of first point in decimal degrees
        lat0 : float
            latitude of first point in decimal degrees
        lon1 : float
            longitude of second point in decimal degrees
        lat1 : float
            latitude of second point in decimal degrees
        radius : float
            average earth radius in km, defaults to 6371 km

        Returns
        -------
        float
            distance of both points in km
        """
        return haversine_formula(lon0,lat0,lon1,lat1,radius)

    def geo_len_scale(self):
        """Return the distance between cam and source in km.

        Uses haversine formula (:func:`haversine`) to determine the distance
        between source and cam to estimate the geoprahic dimension of this
        setup

        :returns: float, distance between source and camera
        """
        return self.haversine(self._cam["lon"], self._cam["lat"],
                              self._source["lon"], self._source["lat"])

    def _map_extend_km(self, fac=5.0):
        """Estimate the extend of map borders for plotting.

        :param float fac: fraction of geo length scale used to determine the
            extend
        """
        return self.geo_len_scale() / fac

    def del_az(self, pixcol1=0, pixcol2=1):
        """Determine the difference in azimuth angle between 2 pixel columns.

        Parameters
        ----------
        pixcol1 : int
            first pixel column
        pixcol2 : int
            second pixel column

        Returns
        -------
        float
            azimuth difference in degrees

        """
        delta = int(abs(pixcol1 - pixcol2))
        return rad2deg(arctan((delta * self._cam["pix_width"]) /
                              self._cam["focal_length"]))

    def del_elev(self, pixrow1=0, pixrow2=1):
        """Determine the difference in azimuth angle between 2 pixel columns.

        Parameters
        ----------
        pixrow1 : int
            first pixel row
        pixrow2 : int
            second pixel row

        Returns
        -------
        float
            elevation difference in degrees

        """
        delta = int(abs(pixrow1 - pixrow2))
        return rad2deg(arctan((delta * self._cam["pix_height"]) /
                              self._cam["focal_length"]))

    def plume_dist(self, az=None, elev=None):
        """Return plume distance for input azim and elev angles.

        Computes the distance to the plume for the whole image plane, assuming
        that the horizontal plume propagation direction is given by the
        meteorological wind direction. The vertical component of the plume
        distance for each pixel row and column is computed based on the
        corresponding elevation angle and horizontal distance to the plume.

        Parameters
        ----------
        az : :obj:`float` or :obj:`list`, optional
            azimuth value(s) (single val or array of values). If `None`, then
            the azimuth angle of the camera CFOV is used.

        elev : :obj:`float` or :obj:`list`, optional
            elevation angle(s) (single val or array of values). If `None`, then
            the elevation angle of the camera CFOV is used.

        Returns
        -------
        :obj:`float` or :obj:`list`
            plume distance(s) in m for input azimuth(s) and elevations

        """
        if az is None:
            az = [self.cam_azim]
        try:
            len(az)
        except TypeError:
            az = [az]
        az = array(az)
        if elev is None:
            elev = [self.cam_elev]
        try:
            len(elev)
        except TypeError:
            elev = [elev]
        # transpose elevation array so
        elev = deg2rad(array(elev)[newaxis].T)
        diff_vec = self.geo_setup.vectors["source2cam"]
        dv = array((diff_vec.dx, diff_vec.dy)).reshape(2, 1)
        pdrad = deg2rad(self.plume_dir)
        azrad = deg2rad(az)
        y = (diff_vec.dx - tan(azrad) * diff_vec.dy) /\
            (tan(pdrad) - tan(azrad))
        x = tan(pdrad) * y
        v = array((x, y)) - dv

        # 1D horizontal array containing horizontal plume distances
        dists_hor = linalg.norm(v, axis=0) * 1000.0

        # 2D array containing vertical distances
        dzs = tan(elev) * dists_hor

        dists = sqrt(dists_hor**2 + dzs**2)

        return dists

    def plume_dist_err(self, az=None):
        """Compute uncertainty in plume distances.

        The computation is based on uncertainties in the camera azimuth and
        the uncertainty in the horizontal wind direction (i.e. plume
        propagation direction).

        Parameters
        ----------
        az : :obj:`float`, optional
            camera azimuth angle for which the uncertainty is computed (if None
            then the CFOV azimuth is used).

        Returns
        -------
        float
            absolute uncertainty in plume distance in units of m

        """
        if az is None:
            az = self.cam_azim
        az = float(az)
        # the vector between camera and source (errors here are assumed
        # negligible)
        diff_vec = self.geo_setup.vectors["source2cam"]
        dv = array((diff_vec.dx, diff_vec.dy)).reshape(2, 1)
        pdir, pdir_err = self.plume_dir, self.plume_dir_err

        plume_dirs = [(pdir - pdir_err) % 360,
                      (pdir + pdir_err) % 360]

        cam_azims = [(az - self._cam["azim_err"]) % 360,
                     (az + self._cam["azim_err"]) % 360]

        azrads = deg2rad(cam_azims)
        pdrads = deg2rad(plume_dirs)
        dists = []
        for pdrad in pdrads:
            for azrad in azrads:
                y = (diff_vec.dx - tan(azrad) * diff_vec.dy) /\
                    (tan(pdrad) - tan(azrad))
                x = tan(pdrad) * y
                v = array((x, y)) - dv
            dists.append(linalg.norm(v, axis=0))
        dists = asarray(dists)
        return (dists.max() - dists.mean()) * 1000

    def all_elevs_camfov(self):
        """Return array containing elevation angles for each image row."""
        rownum = self._cam["pixnum_y"]
        if isnan(rownum):
            raise ValueError("Number of pixels of camera detector is not "
                             "available")
        rownum = int(rownum)
        daz = self.del_elev(0, 1)
        offs = -daz / 2.0 if rownum % 2 == 0 else 0.0

        angles_rel = linspace(rownum / 2, -rownum / 2, rownum) * daz
        return self.cam_elev + angles_rel + offs

    def all_azimuths_camfov(self):
        """Return array containing azimuth angles for each image row."""
        colnum = self._cam["pixnum_x"]
        if isnan(colnum):
            raise ValueError("Number of pixels of camera detector is not "
                             "available")
        colnum = int(colnum)
        daz = self.del_az(0, 1)
        offs = -daz / 2.0 if colnum % 2 == 0 else 0.0
# =============================================================================
#         if colnum%2 == 0: #even number of pixels
#             offs = -daz/2.0
# =============================================================================
        angles_rel = linspace(-colnum / 2, colnum / 2, colnum) * daz
        return self.cam_azim + angles_rel + offs

    def col_to_az(self, colnum):
        r"""Convert pixel column number (in absolute coords) into azimuth angle.

        Note
        ----
        - See also :func:`az_to_col` for the inverse operation
        - Not super efficient, just convenience function which should not\
            be used if performance is required

        Parameters
        ----------
        colnum : int
            pixel column number (left column corresponds to 0)

        Returns
        -------
        float
            corresponding azimuth angle

        """
        return self.all_azimuths_camfov()[colnum]

    def az_to_col(self, azim):
        """Convert azimuth into pixel number.

        Note
        ----
        The pixel number is calculated relative to the leftmost column of the
        image

        Parameters
        ----------
        azim : float
            azimuth angle which is supposed to be converted into column
            number

        Returns
        -------
        int
            column number

        Raises
        ------
        IndexError
            if input azimuth is not within camera FOV

        """
        azs = self.all_azimuths_camfov()
        if azim < azs[0] or azim > azs[-1]:
            raise IndexError("Input azimuth is out of camera FOV")
        return argmin(abs(azs - azim))

    def __str__(self):
        s = "pyplis MeasGeometry object\n##################################\n"
        s += "\nCamera specifications\n-------------------\n"
        for k, v in self._cam.items():
            s += "%s: %s\n" % (k, v)
        s += "\nSource specifications\n-------------------\n"
        for k, v in self._source.items():
            s += "%s: %s\n" % (k, v)
        s += "\nWind specifications\n-------------------\n"
        for k, v in self._wind.items():
            s += "%s: %s\n" % (k, v)
        return s

    def __call__(self, item):
        """Return class attribute with a specific name.

        :param item: name of attribute
        """
        for key, val in self.__dict__.items():
            try:
                if item in val:
                    return val[item]
            except BaseException:
                pass
