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
"""Image list objects.

Image list objects (e.g. :class:`BaseImgList`, :class:`ImgList`,
:class:`DarkImgList`, :class:`CellImgList`) contain a list of image file paths
and are central for the data analysis. Images are loaded as :class:`Img`
objects and are loaded and processed iteratively. Typically one list contains all
images of a certain type (e.g. onband, offband, see :class:`Dataset` object).
:class:`ImgList` objects (inherited from :class:`BaseImgList`) contain powerful
pre-processing modes (e.g. load images as dark corrected and calibrated images,
compute optical flow between current and next image).
"""
from __future__ import (absolute_import, division)
from numpy import (asarray, zeros, argmin, arange, ndarray, float32, isnan,
                   logical_or, uint8, exp, ones)
from numpy.ma import nomask
from datetime import timedelta, datetime, date

import pandas as pd
from pandas import Series, DataFrame
from matplotlib.pyplot import figure, draw, ion, ioff, close
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter

from os.path import exists, abspath, dirname, join, basename
from os import mkdir
from collections import OrderedDict as od

from traceback import format_exc
from pyplis import logger, print_log
from .glob import DEFAULT_ROI
from .image import Img, model_dark_image
from .exceptions import ImgMetaError
from .setupclasses import Camera
from .geometry import MeasGeometry
from .processing import ImgStack, PixelMeanTimeSeries
from .utils import LineOnImage
# from .optimisation import PolySurfaceFit
from .plumebackground import PlumeBackgroundModel
from .plumespeed import OptflowFarneback, LocalPlumeProperties
from .helpers import check_roi, map_roi, _print_list, closest_index, exponent,\
    isnum, get_pyr_factor_rel
from .calib_base import CalibData

from numpy import size, array
from pandas import to_datetime, concat
from astropy.io import fits
from .custom_image_import import _read_binary_timestamp
import six


class BaseImgList(object):
    """Basic image list object.

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
    files : list, optional
        list with image file paths
    list_id : str, optional
        a string used to identify this list (e.g. "second_onband")
    list_type : str, optional
        type of images in list (please use "on" or "off")
    camera : Camera, optional
        camera specifications
    init : bool
        if True, list will be initiated and files loaded (given that image
        files are provided on input)
    **img_prep_settings
        additional keyword args specifying image preparation settings applied
        on image load
    """

    def __init__(self, files=None, list_id=None, list_type=None,
                 camera=None, geometry=None, init=True, **img_prep_settings):
        # this list will be filled with filepaths
        self.files = []
        # id of this list
        self.list_id = list_id
        self.list_type = list_type

        self.filter = None  # can be used to store filter information
        self._meas_geometry = None

        # these variables can be accessed using corresponding @property
        # attributes
        self._integration_step_lengths = None
        self._plume_dists = None

        self.set_camera(camera=camera, cam_id=None)

        self._update_cam_geodata = False
        self._edit_active = True

        # the following dictionary contains settings for image preparation
        # applied on image load
        self.img_prep = {"blurring": 0,  # width of gauss filter
                         "median": 0,  # width of median filter
                         "crop": False,
                         "pyrlevel": 0,  # int, gauss pyramide level
                         "8bit": 0}  # to 8bit

        self._roi_abs = DEFAULT_ROI  # in original img resolution
        self._auto_reload = True

        self._list_modes = {}  # init for :class:`ImgList` object

        self._vign_mask = None  # vignetting correction mask can be stored here
        self.__sky_mask = nomask  # mask for invalid pixel
        self.loaded_images = {"this": None}

        # used to store the img edit state on load
        self._load_edit = {"this": {},
                           "next": {}}

        self.index = 0
        self._skip_files = 0  # if 0, no files are skipped
        self.next_index = 0
        self.prev_index = 0

        # Other image lists can be linked to this and are automatically updated
        self.linked_lists = {}
        # this dict (linked_indices) is filled in :func:`link_imglist` to
        # increase the linked reload image performance
        self._linked_indices = {}
        # contains info about the always_reload option of linked image lists
        # is updated whenever a new list is linked to this one
        self._always_reload = {}

        # update image preparation settings (if applicable)
        for key, val in six.iteritems(img_prep_settings):
            if key in self.img_prep:
                self.img_prep[key] = val

        if bool(files):
            self.add_files(files, load=False)

        if isinstance(geometry, MeasGeometry):
            self.meas_geometry = geometry

        if self.data_available and init:
            self.load()

    """ATTRIBUTES / DECORATORS"""

    @property
    def start(self):
        """Acquisistion time of first image."""
        try:
            return self.start_acq[0]
        except IndexError:
            raise IndexError("No data available")

    @property
    def stop(self):
        """Start acqusition time of last image."""
        try:
            return self.start_acq[-1]
        except IndexError:
            raise IndexError("No data available")

    @property
    def this(self):
        """Return current image."""
        return self.current_img()

    @property
    def edit_active(self):
        """Define whether images are edited on image load or not.

        If False, images will be loaded as raw, i.e. without any editing or
        further calculations (e.g. determination of optical flow, or updates of
        linked image lists). Images will be reloaded.
        """
        return self._edit_active

    @edit_active.setter
    def edit_active(self, value):
        if value == self._edit_active:
            return
        self._edit_active = value
        self.load()

    @property
    def skip_files(self):
        """Integer specifying the image iter step in the file list.

        Defaults to 1: every file is used, 2 means, that every second file is
        used.
        """
        return self._skip_files

    @skip_files.setter
    def skip_files(self, val):
        if not val >= 0:
            raise ValueError("Value must be 0 or positive")
        self._skip_files = int(val)
        self.iter_indices(self.index)
        self.load()

    @property
    def meas_geometry(self):
        """Return measurement geometry."""
        return self._meas_geometry

    @meas_geometry.setter
    def meas_geometry(self, val):
        if not isinstance(val, MeasGeometry):
            raise TypeError("Could not set meas_geometry, need MeasGeometry "
                            "object")
        self._meas_geometry = val

    @property
    def update_cam_geodata(self):
        """Update measurement geometry whenever list index is changed."""
        return self._update_cam_geodata

    @update_cam_geodata.setter
    def update_cam_geodata(self, value):
        try:
            self._update_cam_geodata = bool(value)
        except BaseException:
            raise

    @property
    def plume_dists(self):
        """Distance to plume.

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
            raise TypeError(
                "Need Img or numerical data type (e.g. float, int)")
        if isinstance(value, Img):
            value = value  # .to_pyrlevel(self.pyrlevel)
# =============================================================================
#             if not value.shape == self.this.shape:
#                 raise ValueError("Cannot set plume distance image: shape "
#                                  "mismatch between input and images in list")
# =============================================================================
        self._plume_dists = value

    @property
    def vign_mask(self):
        """Return current vignetting correction mask."""
        if not any([isinstance(self._vign_mask, x) for x in (Img, ndarray)]):
            raise AttributeError("Vignetting mask is not available in list")
        return self._vign_mask

    @vign_mask.setter
    def vign_mask(self, value):
        if not any([isinstance(value, x) for x in (Img, ndarray)]):
            raise AttributeError("Invalid input for vignetting mask, need "
                                 "Img object or numpy ndarray")
        try:
            value = Img(value)
        except BaseException:
            pass
        pyrlevel_rel = get_pyr_factor_rel(self.this.img, value.img)
        if pyrlevel_rel != 0:
            if pyrlevel_rel < 0:
                value.pyr_down(pyrlevel_rel)
            else:
                value.pyr_up(pyrlevel_rel)
        self._vign_mask = value

    @property
    def sky_mask(self):
        """Return sky access mask.

        0 for sky,
        1 for non-sky (=invalid)
        (in masked arrays, entries marked with 1 are invalid)
        """
        return self.__sky_mask

    @sky_mask.setter
    def sky_mask(self, value):
        # TODO: Check if the mask has the same dimension as the images
        # TODO: maybe load as pyplis img
        if not isinstance(value, ndarray):
            raise TypeError("Could not set sky_mask, need MeasGeometry "
                            "object")
        self.__sky_mask = deepcopy(value)

    @property
    def integration_step_length(self):
        """Return integration step length for emission-rate analyses.

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
            raise TypeError(
                "Need Img or numerical data type (e.g. float, int)")
        if isinstance(value, Img):
            value = value.to_pyrlevel(self.pyrlevel)
            if not value.shape == self.this.shape:
                raise ValueError("Cannot set plume distance image: shape "
                                 "mismatch between input and images in list")
        self._integration_step_lengths = value

    @property
    def auto_reload(self):
        """Activate / deactivate automatic reload of images."""
        return self._auto_reload

    @auto_reload.setter
    def auto_reload(self, val):
        self._auto_reload = val
        if bool(val):
            logger.info("Reloading images...")
            self.load()

    @property
    def crop(self):
        """Activate / deactivate crop mode."""
        return self.img_prep["crop"]

    @crop.setter
    def crop(self, value):
        """Set crop."""
        self.img_prep["crop"] = bool(value)
        self.load()

    @property
    def pyrlevel(self):
        """Return current Gauss pyramid level.

        Note
        ----
        images are reloaded on change
        """
        return self.img_prep["pyrlevel"]

    @pyrlevel.setter
    def pyrlevel(self, value):
        logger.info("Updating pyrlevel and reloading")
        if value != self.pyrlevel:
            self.img_prep["pyrlevel"] = int(value)
            self.load()

    @property
    def gaussian_blurring(self):
        """Return current blurring level.

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
            print_log.warning("Activate gaussian blurring with kernel size exceeding 10, "
                 "this might significantly slow down things..")
        self.img_prep["blurring"] = val
        self.load()

    @property
    def roi(self):
        """Return current ROI (in relative coordinates).

        The ROI is returned with respect to the current :attr:`pyrlevel`
        """
        return map_roi(self._roi_abs, self.pyrlevel)

    @roi.setter
    def roi(self):
        raise AttributeError("Please use roi_abs to set the current ROI in "
                             "absolute image coordinates. :func:`roi` is used "
                             "to access the current ROI for the actual "
                             "pyramide level.")

    @property
    def roi_abs(self):
        """Return current roi in absolute detector coords (cf. :attr:`roi`)."""
        # return map_roi(self._roi_abs, self.img_prep["pyrlevel"])
        return self._roi_abs

    @roi_abs.setter
    def roi_abs(self, val):
        if check_roi(val):
            self._roi_abs = val
            self.load()

    @property
    def cfn(self):
        """Return current index (file number in ``files``)."""
        return self.index

    @property
    def nof(self):
        """Return number of files in this list."""
        return len(self.files)

    @property
    def last_index(self):
        """Return index of last image."""
        return len(self.files) - 1

    @property
    def data_available(self):
        """Return wrapper for :func:`has_files`."""
        return self.has_files()

    @property
    def has_images(self):
        """Return wrapper for :func:`has_files`."""
        return self.has_files()

    @property
    def start_acq(self):
        """Array containing all image acq. time stamps of this list.

        Note
        ----
        The time stamps are extracted from the file names
        """
        ts = self.get_img_meta_all_filenames()[0]
        return ts

    def timestamp_to_index(self, val=datetime(1900, 1, 1)):
        """Convert a datetime to the list index.

        Returns the list index that is closest in time to the input time
        stamp.

        Parameters
        ----------
        val : datetime
            time stamp

        Raises
        ------
        AttributeError
            if time stamps of images in list cannot be accessed from their
            file names

        Returns
        -------
        int
            corresponding list index

        """
        times = self.start_acq
        if not len(times) == self.nof:
            raise AttributeError("Failed to access all acq. time stamps could "
                                 "not be accessed")
        return argmin(abs(val - times))

    def index_to_timestamp(self, val=0):
        """Get timestamp of input list index.

        Parameters
        ----------
        val : index
            time stamp

        Raises
        ------
        AttributeError
            if time stamps of images in list cannot be accessed from their
            file names

        Returns
        -------
        int
            corresponding list index

        """
        times = self.start_acq
        if not len(times) == self.nof:
            raise AttributeError("Acq. time stamps could not be accessed")
        if not 0 <= val <= self.last_index:
            raise IndexError("List index out of range")
        return times[val]

    def add_files(self, files, load=True):
        """Add images to this list.

        Parameters
        ----------
        file_list : list
            list with file paths

        Returns
        -------
        bool
            success / failed

        """
        if files is None:
            files = []
        elif isinstance(files, str):
            files = [files]
        if not isinstance(files, list):
            raise TypeError("Error: file paths could not be added to image "
                            "list, wrong input type %s" % type(files))

        self.files.extend(files)
        self.init_filelist(at_index=self.index)
        if load and self.data_available:
            logger.info("Added %d files in list %s, load %s" % (len(files),
                                                       self.list_id, load))
            self.load()

    def init_filelist(self, at_index=0):
        """Initialize the filelist.

        Sets current list index and resets loaded images

        Parameters
        ----------
        at_index : int
            desired image index, defaults to 0

        """
        self.iter_indices(to_index=at_index)
        for key, val in six.iteritems(self.loaded_images):
            self.loaded_images[key] = None

        if self.nof > 0:
            logger.info("\nInit ImgList %s" % self.list_id)
            logger.info("-------------------------")
            logger.info("Number of files: " + str(self.nof))
            logger.info("-----------------------------------------")

    def iter_indices(self, to_index):
        """Change the current image indices for previous, this and next img.

        Note
        ----
        This method only updates the actual list indices and does not perform
        a reload.
        """
        try:
            self.index = to_index % self.nof
            self.next_index = (self.index + self.skip_files + 1) % self.nof
            self.prev_index = (self.index - self.skip_files - 1) % self.nof

        except:
            self.index, self.prev_index, self.next_index = 0, 0, 0

    def load(self):
        """Load current image.

        Try to load the current file ``self.files[self.cfn]`` and if remove the
        file from the list if the import fails

        Returns
        -------
        bool
            if True, image was loaded, if False not

        """
        if not self._auto_reload:
            print_log.info("Automatic image reload deactivated in image list %s"
                  % self.list_id)
            return False
        try:
            img = self._load_image(self.index)
            self._load_edit["this"].update(img.edit_log)
            self.loaded_images["this"] = img
            if img.vign_mask is not None:
                self.vign_mask = img.vign_mask

            if self.update_cam_geodata:
                self.meas_geometry.update_cam_specs(**self.this.meta)

            self._apply_edit("this")

        except IOError:
            print_log.warning("Invalid file encountered at list index %s, file will"
                 " be removed from list" % self.index)
            self.pop()
            if self.nof == 0:
                raise IndexError("No filepaths left in image list...")
            self.load()

        except IndexError:
            try:
                self.init_filelist()
                self.load()
            except BaseException:
                raise IndexError("Could not load image in list %s: file list "
                                 " is empty" % (self.list_id))

        return True

    def goto_next(self):
        """Goto next index in list."""
        if self.nof < 2:
            print_log.warning("Only one image available, no index change or "
                 "reload performed")
            return self.this
        self.iter_indices(to_index=self.next_index)
        self.load()
        return self.this

    def goto_prev(self):
        """Load previous image in list."""
        if self.nof < 2:
            print_log.warning("Only one image available, no index change or "
                 "reload performed")
            return self.this
        self.iter_indices(to_index=self.prev_index)
        self.load()
        return self.this

    def goto_img(self, to_index, reload_here=False):
        """Change the index of the list, load and prepare images at new index.

        Parameters
        ----------
        to_index : float
             new list index
        reload_here : bool
            applies only if :param:`to_index` is the current list index. If
            True, then the current images are reloaded, if False, nothing is
            done.

        """
        if not -1 < to_index < self.nof:
            raise IndexError("Invalid index %d. List contains only %d files"
                             % (to_index, self.nof))

        elif to_index == self.index:
            if reload_here:
                self.load()
            return self.this
        elif to_index == self.next_index:
            self.goto_next()
        elif to_index == self.prev_index:
            self.goto_prev()
        else:
            self.iter_indices(to_index)
            self.load()

        return self.loaded_images["this"]

    def pop(self, idx=None):
        """Remove one file from this list."""
        raise NotImplementedError("pop method of ImgList is currently not "
                                  "available...")
        logger.warning("Removing image at index %n from image list")
        if idx is None:
            idx = self.index
        self.files.pop(idx)

    def has_files(self):
        """Return boolean whether or not images are available in list."""
        return bool(self.nof)

    def plume_dist_access(self):
        """Check if measurement geometry is available."""
        if not isinstance(self.meas_geometry, MeasGeometry):
            return False
        try:
            plume_dist_img = self.meas_geometry.\
                compute_all_integration_step_lengths()[2]
            logger.info("Plume distances available, dist_avg = %.2f"
                  % plume_dist_img.mean())
        except BaseException:
            return False

    def update_img_prep(self, **settings):
        """Update image preparation settings and reload.

        Parameters
        ----------
        **settings
            key word args specifying settings to be updated (see keys of
            ``img_prep`` dictionary)

        """
        for key, val in six.iteritems(settings):
            if key in self.img_prep and\
                    isinstance(val, type(self.img_prep[key])):
                self.img_prep[key] = val
        try:
            self.load()
        except IndexError:
            pass

    def clear(self):
        """Empty this list (i.e. :attr:`files`)."""
        self.files = []

    def separate_by_substr_filename(self, sub_str, sub_str_pos, delim="_"):
        """Separate this list by filename specifications.

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

    def set_camera(self, camera=None, cam_id=None):
        """Set the current camera.

        Two options:

            1. set :obj:`Camera` directly
            2. provide one of the default camera IDs (e.g. "ecII", "hdcam")

        Parameters
        ----------
        camera : Camera
            the camera used
        cam_id : str
            one of the default cameras (use
            :func:`pyplis.inout.get_all_valid_cam_ids` to get the default
            camera IDs)

        """
        if camera is not None:
            if not isinstance(camera, Camera):
                raise TypeError("Camera argument for image list was not "
                                "correctly initialised with an object of type "
                                "pyplis.Camera")

            self.camera = camera

        else:
            if cam_id is not None:
                self.camera = Camera(cam_id)

        # if not isinstance(camera, Camera):
        #    camera = Camera(cam_id)
        # self.camera = camera

    def reset_img_prep(self):
        """Init image pre-edit settings."""
        self.img_prep = dict.fromkeys(self.img_prep, 0)
        self._roi_abs = DEFAULT_ROI
        if self.nof > 0:
            self.load()

    def get_img_meta_from_filename(self, file_path):
        """Load and prepare img meta input dict for Img object.

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
        return {"start_acq": info[0], "texp": info[3]}

    def get_img_meta_all_filenames(self):
        """Try to load acquisition and exposure times from filenames.

        Note
        ----
        Only works if relevant information is specified in ``self.camera`` and
        can be accessed from the file names

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
            except BaseException:
                pass
        try:
            if times[0].date() == date(1900, 1, 1):
                d = self.this.meta["start_acq"].date()
                print_log.warning("Warning accessing acq. time stamps from file names in "
                     "ImgList: date information could not be accessed, using "
                     "date of currently loaded image meta info: %s" % d)
                times = asarray([datetime(d.year, d.month, d.day, x.hour,
                                          x.minute, x.second, x.microsecond)
                                 for x in times])
        except BaseException:
            pass
        return times, texps

    def assign_indices_linked_list(self, lst):
        """Create a look up table for fast indexing between image lists.

        Parameters
        ----------
        lst : BaseImgList
            image list supposed to be linked

        Returns
        -------
        array
            array contining linked indices

        """
        idx_array = zeros(self.nof, dtype=int)
        times, _ = self.get_img_meta_all_filenames()
        times_lst, _ = lst.get_img_meta_all_filenames()
        if lst.nof == 1:
            logger.warning("Other list contains only one file, assign all indices to "
                 "the corresponding image")
        elif (any([x is None for x in times]) or
              any([x is None for x in times_lst])):
            print_log.warning("Image acquisition times could not be accessed from file "
                 "names, assigning by indices")
            lst_idx = arange(lst.nof)
            for k in range(self.nof):
                idx_array[k] = abs(k - lst_idx).argmin()
        else:
            for k in range(self.nof):
                idx = abs(times[k] - times_lst).argmin()
                idx_array[k] = idx

        return idx_array

    def same_preedit_settings(self, settings_dict):
        """Compare input settings dictionary with self.img_prep.

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
        for key, val in six.iteritems(settings_dict):
            if key in sd:
                if not sd[key] == val:
                    return False
        return True

    def make_stack(self, stack_id=None, pyrlevel=None, roi_abs=None,
                   start_idx=0, stop_idx=None, ref_check_roi_abs=None,
                   ref_check_min_val=None, ref_check_max_val=None,
                   dtype=float32):
        """Stack all images in this list.

        The stacking is performed using the current image preparation
        settings (blurring, dark correction etc). Only stack ROI and pyrlevel
        can be set explicitely.

        Note
        ----
        In case of ``MemoryError`` try stacking less images (specifying
        start / stop index) or reduce the size setting a different Gauss
        pyramid level.

        Parameters
        ----------
        stack_id : :obj:`str`, optional
            identification string of the image stack
        pyrlevel : :obj:`int`, optional
            Gauss pyramid level of stack
        roi_abs : list
            build stack of images cropped in ROI
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`, optional
            index of last considered image (if None, the last image in this
            list is used). Note that the timestamp option only works if acq.
            times can be accessed from filenames for all files in the list
            (using method :func:`timestamp_to_index`)
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
            image stack containing stacked images

        """
        self.edit_active = True
        cfn = self.cfn
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        num = self._iter_num(start_idx, stop_idx)
        # remember last image shape settings
        _roi = deepcopy(self._roi_abs)
        _pyrlevel = deepcopy(self.pyrlevel)
        _crop = self.crop

        self.auto_reload = False
        if pyrlevel is not None and pyrlevel != _pyrlevel:
            logger.info("Changing image list pyrlevel from %d to %d"
                  % (_pyrlevel, pyrlevel))
            self.pyrlevel = pyrlevel
        if check_roi(roi_abs):
            logger.info("Activate cropping in ROI %s (absolute coordinates)"
                  % roi_abs)
            self.roi_abs = roi_abs
            self.crop = True

        if stack_id is None:
            stack_id = self.list_id

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
        except BaseException:
            ref_check = False
        try:
            ref_check_max_val = float(ref_check_max_val)
        except BaseException:
            ref_check = False
        exp = int(10**exponent(num) / 4.0)
        if not exp:
            exp = 1
        for k in range(num):
            if k % exp == 0:
                print_log.info("Building img-stack from list %s, progress: (%s | %s)"
                      % (lid, k, num - 1))
            img = self.loaded_images["this"]
            append = True
            if ref_check:
                sub_val = img.crop(roi_abs=ref_check_roi_abs, new_img=1).mean()
                if not ref_check_min_val <= sub_val <= ref_check_max_val:
                    print_log.warning("Exclude image no. %d from stack, got value=%.2f in "
                          "ref check ROI (out of specified range)"
                          % (k, sub_val))
                append = False
            if append:
                stack.add_img(img.img, img.meta["start_acq"],
                              img.meta["texp"])
            self.goto_next()
            k += 1
        stack.start_acq = asarray(stack.start_acq)
        stack.texps = asarray(stack.texps)
        stack.roi_abs = self._roi_abs

        print_log.info("Img stack calculation finished, rolling back to intial list"
              "state:\npyrlevel: %d\ncrop modus: %s\nroi (abs coords): %s "
              % (_pyrlevel, _crop, _roi))
        self.auto_reload = False
        self.pyrlevel = _pyrlevel
        self.crop = _crop
        self.roi_abs = _roi
        self.goto_img(cfn)
        self.auto_reload = True
        if not sum(stack._access_mask) > 0:
            raise ValueError("Failed to build stack, stack is empty...")
        return stack

    def get_mean_img(self, start_idx=0, stop_idx=None):
        """Determine an average image from a number of list images.

        Parameters
        ----------
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`, optional
            index of last considered image (if None, the last image in this
            list is used). Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)

        Returns
        -------
        Img
            average image

        """
        cfn = self.index
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        self.goto_img(start_idx)
        num = self._iter_num(start_idx, stop_idx)
        img = Img(zeros(self.current_img().shape))
        img.edit_log = self.current_img().edit_log
        img.meta["start_acq"] = self.current_time()
        added = 0
        texps = []
        for k in range(num):
            try:
                cim = self.current_img()
                img.img += cim.img
                try:
                    texps.append(cim.texp)
                except BaseException:
                    pass
                self.goto_next()
                added += 1
            except BaseException:
                print_log.warning("Failed to add image at index %d" % k)
        img.img = img.img / added
        img.meta["stop_acq"] = self.current_time()
        if len(texps) == added:
            img.meta["texp"] = asarray(texps).mean()
        self.goto_img(cfn)
        return img

    def get_mean_tseries_rects(self, start_idx, stop_idx, *rois):
        """Similar to :func:`get_mean_value` but for multiple rects.

        Parameters
        ----------
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`
            index of last considered image (if None, the last image in this
            list is used). Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
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
        if num_rois == 0:
            raise ValueError("No ROIs provided...")
        for roi in rois:
            dat.append([[], [], [], []])
        cfn = self.cfn
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        self.goto_img(start_idx)
        num = self._iter_num(start_idx, stop_idx)

        lid = self.list_id
        pnum = int(10**exponent(num) / 2.0)
        for k in range(num):
            try:
                if k % pnum == 0:
                    print_log.info("Calc pixel mean t-series in list %s (%d | %d)"
                          % (lid, (k + 1), num))
            except BaseException:
                pass
            img = self.loaded_images["this"]
            for i in range(num_rois):
                roi = rois[i]
                d = dat[i]
                d[0].append(img.meta["texp"])
                d[1].append(img.meta["start_acq"])
                sub = img.img[roi[1]:roi[3], roi[0]:roi[2]]
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

    def get_mean_value(self, start_idx=0, stop_idx=None, roi=DEFAULT_ROI,
                       apply_img_prep=True):
        """Determine pixel mean value time series in ROI.

        Determines the mean pixel value (and standard deviation) for all images
        in this list. Default ROI is the whole image and can be set via
        input param roi, image preparation can be turned on or off.

        Parameters
        ----------
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`
            index of last considered image (if None, the last image in this
            list is used). Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
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
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        self.edit_active = apply_img_prep
        self.goto_img(start_idx)
        num = self._iter_num(start_idx, stop_idx)

        cfn = self.cfn
        vals, stds, texps, acq_times = [], [], [], []
        lid = self.list_id
        pnum = int(10**exponent(num) / 4.0)
        for k in range(num):
            try:
                if k % pnum == 0:
                    print_log.info("Calc pixel mean t-series in list %s (%d | %d)"
                          % (lid, (k + 1), num))
            except BaseException:
                pass
            img = self.loaded_images["this"]
            texps.append(img.meta["texp"])
            acq_times.append(img.meta["start_acq"])
            sub = img.img[roi[1]:roi[3], roi[0]:roi[2]]
            vals.append(sub.mean())
            stds.append(sub.std())

            self.goto_next()

        self.goto_img(cfn)

        return PixelMeanTimeSeries(vals, acq_times, stds, texps, roi,
                                   img.edit_log)

    def current_edit(self):
        """Return :attr:`edit_log` of current image."""
        return self.current_img().edit_log

    def edit_info(self):
        """Print the current image preparation settings."""
        d = self.current_img().edit_log
        print_log.info("\nImgList %s, image edit info\n----------------------------"
              % self.list_id)
        for key, val in six.iteritems(d):
            print_log.info("%s: %s" % (key, val))

    """
    Functions related to image editing and edit management
    """

    def add_gaussian_blurring(self, sigma=1):
        """Increase amount of gaussian blurring on image load.

        :param int sigma (1): Add width gaussian blurring kernel
        """
        self.img_prep["blurring"] += sigma
        self.load()

    def cam_id(self):
        """Get the current camera ID (if camera is available)."""
        return self.camera.cam_id

    def current_time(self):
        """Get the acquisition time of the current image from image meta data.

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
        """Return a string of the current acq time."""
        return self.current_img().meta["start_acq"].strftime(format)

    def current_img(self, key="this"):
        """Get the current image object.

        Parameters
        ----------
        key : str
            this" or "next"

        Returns
        -------
        Img
            currently loaded image in list

        """
        img = self.loaded_images[key]
        if not isinstance(img, Img):
            logger.info("CALLING LOAD IN CURRENT_IMG %s, list %s"
                  % (key, self.list_id))
            self.load()
            img = self.loaded_images[key]
        return img

    def show_current(self, **kwargs):
        """Show the current image."""
        return self.current_img().show(**kwargs)

    def append(self, file_path):
        """Append image file to list.

        :param str file_path: valid file path
        """
        if not exists(file_path):
            raise IOError("Image file path does not exist %s" % file_path)

        self.files.append(file_path)

# ==============================================================================
#     """GUI features
#     """
#     def open_in_imageviewer(self):
#         from .gui.ImgViewer import ImgViewer
#         app = QApplication(argv)
#         widget = ImgViewer(self.list_id, self)
#         widget.show()
#         exit(app.exec_())
# ==============================================================================

    """
    Plotting etc
    """

    def plot_mean_value(self, roi=DEFAULT_ROI, yerr=False, ax=None):
        """Plot mean value of image time series.

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
            fig = figure()  # figsize=(16, 6))
            ax = fig.add_subplot(1, 1, 1)

        mean = self.get_mean_value()
        ax = mean.plot(yerr=yerr, ax=ax)
        return ax

    def plot_tseries_vert_profile(self, pos_x, start_y=0, stop_y=None,
                                  step_size=0.1, blur=4):
        """Plot the temporal evolution of a line profile.

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
        # fig,axes=plt.subplots(1,2,sharey=True,figsize=(width,height))
        cidx = 0
        img_arr = self.loaded_images["this"].img
        rad = gaussian_filter(l.get_line_profile(img_arr), blur)
        del_x = int((rad.max() - rad.min()) * step_size)
        y_arr = arange(start_y, stop_y, 1)
        ax1 = fig.add_axes([0.1, 0.1, 0.35, 0.8])
        times = self.get_img_meta_all_filenames()[0]
        if any([x is None for x in times]):
            raise ValueError("Cannot access all image acq. times")
        idx = []
        idx.append(cidx)
        for k in range(1, self.nof):
            rad = rad - rad.min() + cidx
            ax1.plot(rad, y_arr, "-b")
            img_arr = self.goto_next().img
            rad = gaussian_filter(l.get_line_profile(img_arr), blur)
            cidx = cidx + del_x
            idx.append(cidx)
        idx = asarray(idx)
        ax1.set_ylim([0, h])
        ax1.invert_yaxis()
        draw()
        new_labels = []
    # ==============================================================================
    #     labelNums=[int(a.get_text()) for a in ax1.get_xticklabels()]
    #     print labelNums
    # ==============================================================================
        ticks = ax1.get_xticklabels()
        new_labels.append("")
        for k in range(1, len(ticks) - 1):
            tick = ticks[k]
            index = argmin(abs(idx - int(tick.get_text())))
            new_labels.append(times[index].strftime("%H:%M:%S"))
        new_labels.append("")
        ax1.set_xticklabels(new_labels)
        ax1.grid()
        self.goto_img(cfn)
        ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.8])
        l.plot_line_on_grid(self.loaded_images["this"].img, ax=ax2)
        ax2.set_title(self.loaded_images["this"].meta["start_acq"].strftime(
            "%d.%m.%Y %H:%M:%S"))
        return fig

    """
    Private methods
    """
    def _this_raw_fromfile(self):
        """Reload and return current image.

        This method is used for test purposes and does not change the list
        state. See for instance :func:`activate_dilution_corr` in
        :class:`ImgList`

        Returns
        -------
        Img
            the current image loaded and unmodified from file

        """
        return self._load_image(self.index)

    def _load_image(self, list_index):
        """Load the actual image data for a given index.

        Parameters
        ----------
        list_index : int
            Index of image in file list ``self.files``

        Returns
        -------
        Img
            the loaded image data (unmodified)

        """
        file_path = self.files[list_index]
        try:
            meta = self.get_img_meta_from_filename(file_path)
        except:
            print_log.warning("Failed to retrieve image meta information from file path %s"
                 % file_path)
            meta = {}
        meta["filter_id"] = self.list_id
        return Img(file_path,
                   import_method=self.camera.image_import_method,
                   **meta)

    def _apply_edit(self, key):
        """Apply the current image edit settings to image.

        :param str key: image id (e.g. this)
        """
        if not self.edit_active:
            logger.warning("Edit not active in img_list " + self.list_id + ": no image "
                  "preparation will be performed")
            return
        img = self.loaded_images[key]
        img.to_pyrlevel(self.img_prep["pyrlevel"])
        if self.img_prep["crop"]:
            img.crop(self.roi_abs)
        img.add_gaussian_blurring(self.img_prep["blurring"])
        img.apply_median_filter(self.img_prep["median"])
        if self.img_prep["8bit"]:
            img._to_8bit_int(new_im=False)
        self.loaded_images[key] = img

    def _iter_num(self, start_idx, stop_idx):
        """Return the number of iterations for a loop.

        The number of iterations is based on the current attribute
        ``skip_files``.

        Parameters
        ----------
        start_idx : int
            start index of loop
        stop_idx : int
            stop index of loop

        Returns
        -------
        int
            number of required iterations

        """
        # the int(x) function rounds down, so no floor(x) needed
        return int((stop_idx - start_idx) / (self.skip_files + 1.0))

    def _first_file(self):
        """Get first file path of image list."""
        if not bool(self.files):
            raise IndexError('ImgList is empty...')
        return self.files[0]

    def _last_file(self):
        """Get last file path of image list."""
        if not bool(self.files):
            raise IndexError('ImgList is empty...')
        return self.files[self.nof - 1]

    def _make_header(self):
        """Make header for current image (based on image meta information)."""
        try:
            im = self.current_img()
            if not isinstance(im, Img):
                raise Exception("Current image not accessible in ImgList...")

            s = ("%s (Img %s of %s), read_gain %s, texp %.2f s"
                 % (self.current_time().strftime('%d/%m/%Y %H:%M:%S'),
                    self.index + 1, self.nof, im.meta["read_gain"],
                    im.meta["texp"]))
            return s

        except Exception as e:
            logger.warning(repr(e))
            return "Creating img header failed..."

    def _get_and_set_geometry_info(self):
        """Compute and write plume and pix-to-pix distances from MeasGeometry.
        """
        try:
            (int_steps, _,
             dists) = self.meas_geometry.compute_all_integration_step_lengths()
            self._plume_dists = dists  # .to_pyrlevel(0)
            self._integration_step_lengths = int_steps
            logger.info("Computed and updated list attributes plume_dist and "
                  "integration_step_length in ImgList from MeasGeometry")
        except BaseException:
            raise ValueError("Measurement geometry not ready for access "
                             "of plume distances and integration steps in "
                             "image list %s."
                             % self.list_id)
    """
    Magic methods
    """

    def __str__(self):
        s = "\npyplis ImgList\n----------------------------------\n"
        s += "ID: %s\nType: %s\n" % (self.list_id, self.list_type)
        s += "Number of files (imgs): %s\n\n" % self.nof
        s += "Current image prep settings\n.................................\n"
        if not self.has_files():
            return s
        try:
            for k, v in six.iteritems(self.current_img().edit_log):
                s += "%s: %s\n" % (k, v)
            if self.crop is True:
                s += "Cropped in ROI\t[x0, y0, x1, y1]:\n"
                s += "  Absolute coords:\t%s\n" % self.roi_abs
                s += "  @pyrlevel %d:\t%s\n" % (self.pyrlevel, self.roi)
        except BaseException:
            s += "FATAL: Image access failed, msg\n: %s" % format_exc()
        return s

    def __call__(self, num=0):
        """Change current file number, load and return image.

        :param int num: file number
        """
        return self.goto_img(num)

    def __getitem__(self, name):
        """Get item method."""
        if name in self.__dict__:
            return self.__dict__[name]
        for k, v in six.iteritems(self.__dict__):
            try:
                if name in v:
                    return v[name]
            except BaseException:
                pass


class DarkImgList(BaseImgList):
    """A :class:`BaseImgList`object only extended by read_gain value.

    This class is meant for storage of dark and offset images.

    Note
    ----
    It is recommended to perform the dark and offset correction using
    non-edited raw dark offset images. Therefore, the default edit state
    of these list (:attr:`edit_active`) is set to False. This means, if
    you have such a list and want to add blurring, cropping, etc., you
    first have to activate the image edit on image load via the
    :attr:`edit_active`.
    """

    def __init__(self, files=None, list_id=None, list_type=None, read_gain=0,
                 camera=None, init=True):

        super(DarkImgList, self).__init__(files, list_id, list_type, camera,
                                          init=False)
        self.read_gain = read_gain
        if init:
            self.add_files(files, load=False)
        self._edit_active = False
        if self.data_available:
            self.load()


class AutoDilcorrSettings(object):
    """Store settings for automatic dilution correction in ImgLists.

    Attributes
    ----------
    tau_thresh : float
        OD threshold for computation of plume pixel mask
    erosion_kernel_size : int
        size of erosion kernel applied to plume pixel mask
    dilation_kernel_size : int
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
    erosion_kernel_size : int
        size of erosion kernel applied to plume pixel mask
    dilation_kernel_size : int
        size of dilation kernel applied to plume pixel mask after
        erosion was applied

    """

    def __init__(self, tau_thresh=0.05, erosion_kernel_size=0,
                 dilation_kernel_size=0):
        self.tau_thresh = tau_thresh
        self.erosion_kernel_size = erosion_kernel_size
        self.dilation_kernel_size = dilation_kernel_size
        self.bg_model = PlumeBackgroundModel(mode=99)

    def __str__(self):
        return self.__dict__.__str__()


class _LinkedLists:
    """Management class for linked image lists.

    Attributes
    ----------
    lists : dict
        dictionary containing linked image lists and their IDs (keys)
    indices : dict
        dictionary containing lists that contain mapped indices for fast
        access for each linked list (keys are list IDs)
    always_reload : dict
        dictionary containing information (bool) for each linked list (keys)
        that specifies whether images are supposed to be reloaded whenever the
        index of the parent list is changed (regardless whether the actual
        index in the linked list remains the same or not).

    """

    def __init__(self):
        self.lists = od()
        self.indices = od()
        self.always_reload = od()

    @property
    def all_list_ids(self):
        return self.lists.keys()

    def add(self):
        raise NotImplementedError


class ImgList(BaseImgList):
    u"""Image list object with expanded functionality (cf. :class:`BaseImgList`).

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

    def __init__(self, files=None, list_id=None, list_type=None, camera=None,
                 geometry=None, init=True, **dilcorr_settings):

        super(ImgList, self).__init__(files, list_id, list_type, camera,
                                      geometry, init=False)

        self.loaded_images.update({"next": None})

        #: List modes (currently only tau) are flags for different list states
        #: and need to be activated / deactivated using the corresponding
        #: method (e.g. :func:`activate_tau_mode`) to be changed, dont change
        #: them directly via this private dictionary
        self._list_modes.update({
            "darkcorr": False,  # dark correction
            "optflow": False,  # compute optical flow
            "vigncorr": False,  # load vignetting corrected images
            "dilcorr": False,  # load as dilution corrected images
            "tau": False,  # load as OD images
            "aa": False,  # load as AA images
            "senscorr": False,  # correct for cross-detector sensitivity
                                # variations
            "gascalib": False,
            "shift": False
        })  # load as calibrated SO2 images

        self._ext_coeffs = None

        # a constant background image that may be assigned
        # to the list using attr. bg_img
        self._bg_img = None

        self._bg_list_id = None  # ID of linked background list

        self.bg_model = PlumeBackgroundModel()

        self._senscorr_mask = None
        self._calib_data = None

        # these two images can be set manually, if desired
        self.master_dark = None
        self.master_offset = None

        # These dicitonaries contain lists with dark and offset images
        self.dark_lists = od()
        self.offset_lists = od()
        self._dark_corr_opt = self.camera.darkcorr_opt

        self._last_dark = None

        # Dark images will be updated every 10 minutes (i.e. before an image is
        # dark and offset corrected it will be checked if the currently loaded
        # images match the time interval (+-10 min) of this image and if not
        # a new one will be searched).
        self.update_dark_ival = 10  # mins
        self.time_last_dark_check = datetime(1900, 1, 1)

        # tau threshold for calculation of plume pixel mask fro dilution
        # correction
        self.dilcorr_settings = AutoDilcorrSettings(**dilcorr_settings)
        # self.currentMaxI=None

        # Optical flow engine
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
    def darkcorr_opt(self):
        """Return the current dark correction mode.

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

    @darkcorr_opt.setter
    def darkcorr_opt(self, val):
        val = int(val)
        if val not in [0, 1, 2]:
            raise ValueError
        self._dark_corr_opt = val

    @property
    def darkcorr_mode(self):
        """Return current list darkcorr mode."""
        return self._list_modes["darkcorr"]

    @darkcorr_mode.setter
    def darkcorr_mode(self, value):
        """Change current list darkcorr mode.

        Wrapper for :func:`activate_darkcorr`
        """
        return self.activate_darkcorr(value)

    @property
    def optflow_mode(self):
        """Activate / deactivate optical flow calc on image load."""
        return self._list_modes["optflow"]

    @optflow_mode.setter
    def optflow_mode(self, val):
        self.activate_optflow_mode(val)

    @property
    def vigncorr_mode(self):
        """Activate / deactivate vignetting correction on image load."""
        return int(self._list_modes["vigncorr"])

    @vigncorr_mode.setter
    def vigncorr_mode(self, val):
        self.activate_vigncorr(val)

    @property
    def dilcorr_mode(self):
        """Activate / deactivate dilution correction on image load."""
        return int(self._list_modes["dilcorr"])

    @dilcorr_mode.setter
    def dilcorr_mode(self, val):
        self.activate_dilcorr_mode(val)

    @property
    def sensitivity_corr_mode(self):
        """Activate / deactivate AA sensitivity correction mode."""
        return self._list_modes["senscorr"]

    @sensitivity_corr_mode.setter
    def sensitivity_corr_mode(self, val):
        """Activate / deactivate AA sensitivity correction mode."""
        if val == self._list_modes["senscorr"]:
            return
        if val:
            self.senscorr_mask  # raise AttributeError if mask is not available
            if not self.aa_mode:
                raise AttributeError("AA sensitivity correction mode can only "
                                     "be activated in list aa_mode, please "
                                     "activate aa_mode first...")

        self._list_modes["senscorr"] = val
        self.load()

    @property
    def tau_mode(self):
        """Return current list tau mode."""
        return self._list_modes["tau"]

    @tau_mode.setter
    def tau_mode(self, value):
        """Change current list tau mode.

        Wrapper for :func:`activate_tau_mode`
        """
        self.activate_tau_mode(value)

    @property
    def shift_mode(self):
        """Return current list registration shift mode."""
        return self._list_modes["shift"]

    @shift_mode.setter
    def shift_mode(self, value):
        """Change current list registration shift mode.

        Wrapper for :func:`activate_shift_mode`
        """
        self.activate_shift_mode(value)

    @property
    def aa_mode(self):
        """Return current list AA mode."""
        return self._list_modes["aa"]

    @aa_mode.setter
    def aa_mode(self, value):
        """Change current list AA mode.

        Wrapper for :func:`activate_aa_mode`
        """
        self.activate_aa_mode(value)

    @property
    def calib_mode(self):
        """Acitivate / deactivate current list gas calibration mode."""
        return self._list_modes["gascalib"]

    @calib_mode.setter
    def calib_mode(self, value):
        """Change current list calibration mode."""
        self.activate_calib_mode(value)

    @property
    def ext_coeff(self):
        """Return current extinction coefficient."""
        if not isinstance(self.ext_coeffs, Series):
            raise AttributeError("Extinction coefficients not available in "
                                 "image list %s" % self.list_id)
        elif len(self.ext_coeffs) == self.nof:
            # assuming that time stamps correspond to list time stamps
            return self.ext_coeffs[self.cfn]
        else:
            idx = closest_index(self.current_time(), self.ext_coeffs.index)
            return self.ext_coeffs[idx]

    @ext_coeff.setter
    def ext_coeff(self, val):
        self.ext_coeffs = val

    @property
    def ext_coeffs(self):
        """Dilution extinction coefficients."""
        return self._ext_coeffs

    @ext_coeffs.setter
    def ext_coeffs(self, val):
        if isinstance(val, float):
            val = Series(val, [self.start_acq[0]])
        if not isinstance(val, Series):
            raise ValueError("Need pandas Series object")
        self._ext_coeffs = val

    @property
    def bg_img(self):
        """Return background image based on current vignetting corr setting."""
        img = self._bg_img
        if isinstance(img, Img):
            return img

        lst = self.bg_list
        if isinstance(lst.this, Img):
            return lst.this

        raise AttributeError("No background image found in image list")

    @bg_img.setter
    def bg_img(self, val):
        self.set_bg_img(val)

    @property
    def dark_img(self):
        """Return current dark image."""
        return self.get_dark_image()

    @property
    def bg_list(self):
        """Return background image list (if assigned)."""
        try:
            return self.linked_lists[self._bg_list_id]
        except KeyError:
            raise AttributeError("No linked background list found with ID %s "
                                 "found in ImgList %s. "
                                 % (self._bg_list_id, self.list_id))

    @bg_list.setter
    def bg_list(self, val):
        self.set_bg_list(val)

    @property
    def senscorr_mask(self):
        """Get / set AA correction mask."""
        if isinstance(self._senscorr_mask, ndarray):
            logger.warning("AA correction mask in list %s is numpy array and"
                 "will be converted into Img object" % self.list_id)
            self._senscorr_mask = Img(self._senscorr_mask)
        if not isinstance(self._senscorr_mask, Img):
            try:
                if not isinstance(self.calib_data, CalibData):
                    raise AttributeError("No CalibData object assigned to "
                                         "list from which a sensitivity  "
                                         "correction mask might be loaded...")
                mask = self.calib_data.senscorr_mask
                mask.to_pyrlevel(0)
                if mask.is_cropped:
                    raise ValueError("Sensitivity correction mask assigned to "
                                     "calibration data object is cropped and "
                                     "cannot be used")
                if not mask.shape == self._load_image().shape:
                    raise ValueError("Shape mismatch...")
                self._senscorr_mask = mask
            except Exception as e:
                raise AttributeError("AA correction mask could not be "
                                     "accessed. Traceback:\n%s"
                                     % format_exc(e))
        return self._senscorr_mask

    @senscorr_mask.setter
    def senscorr_mask(self, val):
        """Set AA correction mask."""
        if isinstance(val, ndarray):
            logger.warning("Input for AA correction mask in list %s is numpy array and"
                 "will be converted into Img object" % self.list_id)
            val = Img(val)
        if not isinstance(val, Img):
            raise TypeError("Invalid input for AA correction mask: need Img"
                            " object (or numpy array)")
        if not val.pyrlevel == 0:
            logger.warning("AA correction mask is required to be at pyramid level 0 and "
                 "will be converted")
            val.to_pyrlevel(0)
        img_temp = self._load_image(self.index)
        if val.shape != img_temp.shape:
            try:
                val = val.to_pyrlevel(img_temp.pyrlevel)
                if val.shape != img_temp.shape:
                    raise ValueError
            except BaseException:
                raise ValueError("Img shape mismatch between AA correction "
                                 "mask and list images")

        self._senscorr_mask = val

    @property
    def calib_data(self):
        """Get set object to perform calibration."""
        if not isinstance(self._calib_data, CalibData):
            print_log.warning("No calibration data available in imglist %s" % self.list_id)
        return self._calib_data

    @calib_data.setter
    def calib_data(self, val):
        if not isinstance(val, CalibData):
            raise TypeError("Could not set calibration data in imglist %s: "
                            "need CellCalibData obj or DoasCalibData obj"
                            % self.list_id)
        try:
            val(0.1)  # try converting a fake tau value into a gas column
        except ValueError:
            raise ValueError("Cannot set calibration data in image list, "
                             "calibration object is not ready")
        self._calib_data = val

    @property
    def doas_fov(self):
        """Try access DOAS FOV info (in case cailbration data is available)."""
        try:
            return self.calib_data.fov
        except BaseException:
            raise ValueError("No DOAS FOV information available")

    """RESETTING AND INIT METHODS"""
    def init_bg_model(self, **kwargs):
        """Init clear sky reference areas in background model."""
        self.bg_model.update(**kwargs)
        self.bg_model.set_missing_ref_areas(self.current_img())

    """LIST MODE MANAGEMENT METHODS"""
    def activate_darkcorr(self, value=True):
        """Activate or deactivate dark and offset correction of images.

        If dark correction turned on, dark image access is attempted, if that
        fails, Exception is raised including information what did not work
        out.

        Parameters
        ----------
        val : bool
            new mode

        """
        if value is self.darkcorr_mode:  # do nothing
            return
        if not value and self._load_edit["this"]["darkcorr"]:
            raise ImgMetaError("Cannot deactivate dark correction, original"
                               "image file was already dark corrected")
        if value:
            if self.this.edit_log["darkcorr"]:
                print_log.warning("Cannot activate dark correction in image list %s: "
                     "current image is already corrected for dark current"
                     % self.list_id)
                return
            self.get_dark_image()
            # self.update_index_dark_offset_lists()

        self._list_modes["darkcorr"] = value
        self.load()

    def activate_vigncorr(self, value=True):
        """Activate / deactivate vignetting correction on image load.

        Note
        ----

        Requires ``self.vign_mask`` to be set or an background image
        to be available (from which ``self.vign_mask`` is then determined)

        Parameters
        ----------
        value : bool
            new mode

        """
        if value is self.vigncorr_mode:  # do nothing
            return
        elif value:
            if self.this.edit_log["vigncorr"]:
                print_log.warning("Cannot activate vignetting correction in image list %s: "
                     "current image is already corrected for vignetting"
                     % self.list_id)
                return
            try:
                self.vign_mask
            except BaseException:
                self.det_vign_mask_from_bg_img()
            sh = self._load_image(self.index).img.shape
            if not self.vign_mask.shape == sh:
                raise ValueError("Shape of vignetting mask %s deviates from "
                                 "raw img shape %s"
                                 % (list(self.vign_mask.shape), list(sh)))
        self._list_modes["vigncorr"] = value
        self.load()

    def activate_shift_mode(self, value=True):
        """Activate / deactivate image shift on load.

        The shift that is set in the assigned Camera class is used

        Parameters
        ----------
        value : bool
            new mode

        """
        if value is self.shift_mode:
            return
        if value:
            if not self.list_type == "off":
                raise TypeError("Automatic shift can only be activated in "
                                "offband lists")
            if all([x == 0 for x in self.camera.reg_shift_off]):
                raise ValueError("Camera %s has no image registration "
                                 "shift defined" % self.camera.cam_id)
            dx, dy = self.camera.reg_shift_off
            img = self._this_raw_fromfile()
            if img.pyrlevel != 0:
                raise AttributeError("Loaded raw images have non-zero "
                                     "pyramid level, cannot apply shift")
            img.shift(dx, dy)
        self._list_modes["shift"] = value
        self._check_shift_others()
        self.load()

    def activate_tau_mode(self, value=True):
        """Activate tau mode.

        In tau mode, images will be loaded as tau images (if background image
        data is available).

        Parameters
        ----------
        value : bool
            new mode

        """
        if value is self.tau_mode:  # do nothing
            return
        if value:
            if self.this.edit_log["is_tau"]:
                print_log.warning("Cannot activate tau mode in image list %s: "
                     "current image is already a tau image"
                     % self.list_id)
                return
            cim = self._load_image(self.index)
            bg_img = None
            self.bg_model.set_missing_ref_areas(cim)
            if self.bg_model.mode == 0:
                logger.info("Background correction mode is 0, initiating "
                            "settings for poly surface fit")
                # self.calc_sky_background_mask()
                try:
                    self.calc_sky_background_mask()
                except Exception as e:
                    raise ValueError('Could not activate tau mode in BG model '
                                     'mode 0: Failed to compute sky background '
                                     'mask: Error: {}'.repr(e))
            else:
                if not self.has_bg_img():
                    raise AttributeError(
                        "no background image available in "
                        "list %s, please set a suitable background image "
                        "using method set_bg_img, or change current bg "
                        "modelling mode to 0 using self.bg_model.mode=0)"
                        % self.list_id)
                bg_img = self.bg_img
                if bg_img.is_vigncorr and not cim.is_vigncorr:
                    raise AttributeError("Background image in bg_list is "
                                         "corrected for vignetting but raw "
                                         "plume image is not. Please "
                                         "check")
            self.bg_model.get_tau_image(cim, bg_img, check_state=False)
        self._list_modes["tau"] = value
        self.load()

    def activate_aa_mode(self, value=True):
        """Activates AA mode (i.e. images are loaded as AA images).

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
            raise TypeError("AA mode could not be activated: This list is "
                            "not an onband list")
        aa_test = None
        if value:
            if self.this.edit_log["is_aa"]:
                raise AttributeError("AA mode cannot be activated in image "
                                     "list %s: current image is already AA "
                                     "image" % self.list_id)

            offlist = self.get_off_list()
            logger.info("Activation of AA mode in ImgList %s. Updating settings for "
                  "plume background modelling and the following image "
                  "preparation settings in linked off-band list:\n"
                  "pyrlevel = 0 (prev: %d)\n"
                  "darkcorr_mode = %s (prev: %s)\n"
                  "roi_abs (cropping) = %s (prev: %s)"
                  % (self.list_id, offlist.pyrlevel, self.darkcorr_mode,
                     offlist.darkcorr_mode, DEFAULT_ROI, offlist.roi_abs))
            ed = offlist.edit_active
            offlist.edit_active = False
            offlist.bg_model.update(**self.bg_model.settings_dict())
            offlist.pyrlevel = offlist._load_edit["this"]["pyrlevel"]
            offlist.darkcorr_mode = self.darkcorr_mode
            offlist.roi_abs = DEFAULT_ROI
            offlist.edit_active = ed

            if not isinstance(offlist, BaseImgList):
                raise Exception("Linked off band list could not be found")
            if not offlist.nof / float(self.nof) > 0.25:
                raise IndexError(
                    "Off band list does not have enough images...")
            if self.bg_model.mode != 0:
                if not self.has_bg_img():
                    raise AttributeError("no background image available, "
                                         "please set suitable background "
                                         "image using method set_bg_img "
                                         "or set background modelling mode=0")
                if not offlist.has_bg_img():
                    raise AttributeError("no background image available in "
                                         "off band list. Please set suitable "
                                         "background image using method "
                                         "set_bg_img or set background "
                                         "modelling mode = 0")
            # offlist.update_img_prep(**self.img_prep)
            # offlist.init_bg_model(mode = self.bg_model.mode)
            self._list_modes["tau"] = False
            # updated on 12/1/17 (i.e. the current image in the offband list
            # needs to be reloaded in case the list is in tau mode)
            # offlist._list_modes["tau"] = False
            offlist.tau_mode = False
            aa_test = self._aa_test_img(offlist)
        self._list_modes["aa"] = value
        self.load()
        return aa_test

    def activate_calib_mode(self, value=True):
        """Activate calibration mode."""
        if value == self._list_modes["gascalib"]:
            return

        if value:
            if not self.aa_mode:
                self._list_modes["aa"] = True
                print_log.warning("List is not in AA mode")

            if not self.sensitivity_corr_mode:
                print_log.warning("AA sensitivity correction mode is deactivated. This "
                     "may yield erroneous results at the image edges")
            try:
                self.calib_data(self.current_img())
            except TypeError:
                raise AttributeError("Calibration data is not available "
                                     "in image list")

        self._list_modes["gascalib"] = value
        self.load()

    def activate_dilcorr_mode(self, value=True):
        """Activate dilution correction mode.

        Please see :func:`correct_dilution` for details.

        Parameters
        ----------
        value : bool
            New mode: True or False

        """
        logger.info("Activating dilution correction mode in list %s" % self.list_id)
        if value == self._list_modes["dilcorr"]:
            return
        if value:
            try:
                img = self._this_raw_fromfile()
                mask = self.correct_dilution(img,
                                             img_check_plumemask=False)[2]
            except Exception as e:
                raise AttributeError("Failed to activate dilution correction "
                                     "mode in list {}. Error: {}"
                                     .format(self.list_id, repr(e)))
            # now make sure that in case an off-band list is assigned, it can
            # also be used to perform a dilution correction (i.e. bg_model
            # ready)
            if self.list_type == "on":
                lid = "No offlist found..."
                try:
                    off_list = self.get_off_list()
                    lid = off_list.list_id
                    off_list.bg_model.update(**self.bg_model.settings_dict())
                    off_img = off_list._this_raw_fromfile().to_pyrlevel(
                        off_list.pyrlevel)
                    mask = mask.to_pyrlevel(off_list.pyrlevel)
                    off_list.correct_dilution(off_img, plume_pix_mask=mask)
                except AttributeError:
                    pass
                except Exception as e:
                    msg = ("Failed to apply dilution correction "
                           "in linked offband list {}. Error: {}"
                           .format(lid, repr(e)))
                    if self.aa_mode:
                        raise ValueError(msg)
                    else:
                        print_log.warning(msg)

        self._list_modes["dilcorr"] = value
        self.load()

    def activate_optflow_mode(self, value=True, draw=False):
        """Activate / deactivate optical flow calculation on image load.

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
                                 "image list %s: list is at last index, "
                                 "please change list index and retry")
            self.optflow.calc_flow()
            if draw:
                self.optflow.draw_flow()
        self._list_modes["optflow"] = value

    """GETTERS"""

    def get_dark_image(self, key="this"):
        """Prepare the current dark image dependent on ``darkcorr_opt``.

        The code checks current dark correction mode and, if applicable,
        prepares the dark image.

            1. ``self.darkcorr_opt == 0`` (no dark correction)
                return False

            2. ``self.darkcorr_opt == 1`` (model dark image from a sample dark
                and offset image)
                Try to access current dark and offset image from
                ``self.dark_lists`` and ``self.offset_lists`` (so these must
                exist). If this fails for some reason, set
                ``self.darkcorr_opt = 2``, else model dark image using
                :func:`model_dark_image` and return this image

            3. ``self.darkcorr_opt == 2`` (subtract dark image if exposure
                times of current image does not deviate by more than 20% to
                current dark image)
                Try access current dark image in ``self.dark_lists``, if this
                fails, try to access current dark image in ``self.darkImg``
                (which can be set manually using :func:`set_dark_image`). If
                this also fails, set ``self.darkcorr_opt = 0`` and return
                False. If a dark image could be found and the exposure time
                differs by more than 20%, set ``self.darkcorr_opt = 0`` and
                raise ValueError. Else, return this dark image.

        """
        if self.darkcorr_opt == 0:
            raise ValueError(
                "Dark image could not be accessed in list %s: "
                "darkcorr_opt is zero, please set darkcorr_opt according "
                "to your data type")
        # this was changed on 8/1/2017
        texp = self.this.meta["texp"]
        # img = self.current_img(key)
        read_gain = self.this.meta["read_gain"]
        val = self.update_index_dark_offset_lists()
        last_dark = self._last_dark
        if not val and isinstance(last_dark, Img):
            if (last_dark.texp == texp and
                    read_gain == last_dark.meta["read_gain"]):
                return last_dark
        if self.darkcorr_opt == 1:
            try:
                dark = self.dark_lists[read_gain]["list"].current_img()
                offset = self.offset_lists[read_gain]["list"].current_img()
                dark = model_dark_image(texp, dark, offset)
            except KeyError as e:
                msg = format_exc(e.args[0])
                try:
                    dark = model_dark_image(texp, self.master_dark,
                                            self.master_offset)
                    logger.info("Using master dark and offset image")
                except BaseException:
                    raise ValueError(
                        "Dark image could not be accessed in "
                        "image list %s (darkcorr_opt=1), traceback: %s"
                        % (self.list_id, msg))

        if self.darkcorr_opt == 2:
            try:
                dark = self.dark_lists[read_gain]["list"].current_img()
                if not isinstance(dark, Img):
                    raise ValueError
            except BaseException:
                dark = self.master_dark
                if not isinstance(dark, Img):
                    raise ValueError(
                        "Dark image could not be accessed in "
                        "image list %s (darkcorr_opt=2)" % self.list_id)
        try:
            texp_ratio = texp / dark.meta["texp"]
            if not 0.8 <= texp_ratio <= 1.2:
                print_log.warning("Exposure time of current dark image in list %s "
                     "deviates by more than 20% from list image %s "
                     "(current list index: %d)"
                     % (self.list_id, key, self.cfn))
        except BaseException:
            pass
        self._last_dark = dark
        return dark

    def get_off_list(self, list_id=None):
        """Search off band list in linked lists.

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
                # print "Found default off band key %s" %list_id
            except BaseException:
                pass
        for lst in self.linked_lists.values():
            if lst.list_type == "off":
                if list_id is None or list_id == lst.list_id:
                    return lst
        raise AttributeError("No linked offband list was found")

    """SETTERS: ATTRIBUTE ASSIGNMENT METHODS"""

    def set_bg_img(self, bg_img):
        """Update the current background image object.

        Check input background image and, in case a vignetting mask is not
        available in this list, determine a vignetting mask from the
        background image. Furthermore, if the input image is not blurred it
        is blurred using current list blurring factor and in case the
        latter is 0, then it is blurred with a Gaussian filter of width 1.


        Parameters
        ----------
        bg_img : Img
            the background image object used for plume background modelling
            (modes 1 - 6 in :class:`PlumeBackgroundModel`)

        """
        if not isinstance(bg_img, Img):
            print_log.warning("Could not set background image in ImgList %s: "
                  ": wrong input type, need Img object" % self.list_id)
            return False
        vc_raw = self._this_raw_fromfile().is_vigncorr
        if bg_img.is_vigncorr:
            if not vc_raw:
                raise AttributeError("Cannot set vignetting corrected BG-img "
                                     "in image list that does not contain "
                                     "files that are not initially corrected "
                                     "for vignetting")
            self._bg_img = bg_img
        else:  # Input image is not vignetting corrected
            self._bg_img = bg_img
            try:
                self.vign_mask  # raises AttributeError if not available
            except AttributeError:
                self.det_vign_mask_from_bg_img()
        self._check_shift_others()

    def _check_shift_others(self):
        """Ensure consistent background and vignetting mask shift.

        Checks if background and vignetting mask are shifted according to
        current list mode.
        """
        if self.shift_mode:
            dx, dy = self.camera.reg_shift_off
            try:
                if not self.bg_img.is_shifted:
                    self.bg_img.shift(dx, dy)
            except AttributeError:
                logger.warning("No BG img available")
            try:
                if not self.vign_mask.is_shifted:
                    self.vign_mask.shift(dx, dy)
            except AttributeError:
                logger.warning("No vignetting mask available")
        else:
            try:
                if self.bg_img.is_shifted:
                    raise ImgMetaError("Current BG image is shifted...")
            except ImgMetaError as e:
                raise e
            except:
                pass
            try:
                if self.vign_mask.is_shifted:
                    raise ImgMetaError("Current vignetting mask is shifted...")
            except ImgMetaError as e:
                raise e
            except:
                pass

    def set_bg_list(self, lst, always_reload=False):
        """Assign background image list to this list.

        Assigns and links an image list containing background images to this
        list. Similar to other linked lists, the index of the current BG image
        is automatically updated such that the current BG image is closest in
        time to the current image in this list. Please note also, that a single
        master BG image can be assigned using :attr:`bg_img`.

        Parameters
        ----------
        lst : ImgList
            image list containing background images. Note that the input can
            also be a string specifying the list_id of an image list that is
            already linked to this list.
        always_reload : bool
            if True, the current BG image is always reloaded, whenever the
            index in this list is changed (not recommended since it is slow).
            If False, the state of the background list is only changed, if the
            actual background image index is altered.

        """
        if isinstance(lst, str):
            if lst not in self.linked_lists:
                raise AttributeError("No linked list with ID %s found in image"
                                     " list %s" % (self.list_id, lst))
            self._bg_list_id = lst
            self.det_vign_mask_from_bg_img()
        elif isinstance(lst, ImgList):
            lid = "bg_" + self.list_id
            self.link_imglist(lst, list_id=lid, always_reload=always_reload)
            self._bg_list_id = lid
            self.det_vign_mask_from_bg_img()
        else:
            raise ValueError(
                "Invalid input for assignment of background image "
                "list. Please provide either a string of one of the image "
                "lists already linked to this list or provide an ImgList "
                "object containing BG images")

    def set_flow_images(self):
        """Update images for optical flow determination.

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
                             " print_log.warning( image ..." % self.list_id)

        self.optflow.set_images(self.loaded_images["this"],
                                self.loaded_images["next"])

    def set_optical_flow(self, optflow):
        """Set the current optical flow object.

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
        """Update dark correction mode.

        :param int mode (1): new mode
        """
        if 0 <= mode <= 2:
            self.camera.darkcorr_opt = mode
            return True
        return False

    def add_master_dark_image(self, dark, acq_time=datetime(1900, 1, 1),
                              texp=0.0, read_gain=0):
        """Add a (master) dark image data to list.

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
            if texp is None:
                raise ValueError("Could not add dark image in image list, "
                                 "missing input for texp")
            dark = Img(dark, texp=texp)

        if (acq_time != datetime(1900, 1, 1) and
                dark.meta["start_acq"] == datetime(1900, 1, 1)):
            dark.meta["start_acq"] = acq_time
        dark.meta["read_gain"] = read_gain

        self.master_dark = dark

    def add_master_offset_image(self, offset, acq_time=datetime(1900, 1, 1),
                                texp=0.0, read_gain=0):
        """Add a (master) offset image to list.

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
            if texp is None:
                raise ValueError("Could not add offset image in image list, "
                                 "missing input for texp")
            offset = Img(offset, texp=texp)

        if (acq_time != datetime(1900, 1, 1) and
                offset.meta["start_acq"] == datetime(1900, 1, 1)):
            offset.meta["start_acq"] = acq_time
        offset.meta["read_gain"] = read_gain
        self.master_offset = offset

    def set_closest_dark_offset(self):
        """Update the index of the current dark and offset images.

        The index is updated in all existing dark and offset lists.
        """
        updated = False
        try:
            num = self.index
            for read_gain, info in six.iteritems(self.dark_lists):
                darknum = info["idx"][num]
                if darknum != info["list"].index:
                    updated = True
                    logger.info("Dark image index (read_gain %s) was changed in "
                          "list %s from %s to %s"
                          % (read_gain, self.list_id,
                             info["list"].index, darknum))
                    info["list"].goto_img(darknum)

            if self.darkcorr_opt == 1 and updated:
                for read_gain, info in six.iteritems(self.offset_lists):
                    offsnum = info["idx"][num]
                    if offsnum != info["list"].index:
                        logger.info("Offset image index (read_gain {}) was changed "
                                    "in list %s from {} to {}"
                                    .format(read_gain, self.list_id, info["list"].index,
                                    offsnum))
                        info["list"].goto_img(offsnum)
        except Exception:
            print_log.warning("Failed to update index of dark and offset lists")
            return False
        return updated

    """LINKING OF OTHER IMAGE LIST OBJECTS"""
    def link_imglist(self, other_list, list_id=None, always_reload=True):
        """Link another image list to this list.

        Parameters
        ----------
        other_list : ImgList
            image list object that is supposed to be linked to this one
        always_reload : bool
            if True, the current image in the linked list is always reloaded,
            whenever the index in this list is changed. This is useful in case
            an offband list is linked to an onband list, not so much if a
            list containing BG images is linked to an oband list (see also
            :func:`set_bg_list`)

        """
        logger.info("Linking list %s to list %s" % (other_list.list_id,
                                              self.list_id))
        if list_id is None:
            list_id = other_list.list_id

        if list_id in self.linked_lists:
            raise AttributeError("ImgList %s has already linked an ImgList "
                                 "with list_id %s. "
                                 "Please choose a different ID.")
        self.linked_lists[list_id] = other_list
        # self._linked_indices[list_id] = {}
        self._always_reload[list_id] = always_reload
        idx_array = self.assign_indices_linked_list(other_list)
        self._linked_indices[list_id] = idx_array
        # self.change_index_linked_lists()
        other_list.bg_model.update(**self.bg_model.settings_dict())

        self.load()

    def disconnect_linked_imglist(self, list_id):
        """Disconnect a linked list from this object.

        :param str list_id: string id of linked list
        """
        if list_id not in self.linked_lists.keys():
            print_log.warning("Error: no linked list found with ID " + str(list_id))
            return 0
        del self.linked_lists[list_id]
        del self._linked_indices[list_id]
        del self._always_reload[list_id]

    def link_dark_offset_lists(self, *lists):
        """Assign dark and offset image lists to this object.

        Assign dark and offset image lists: get "closest-in-time" indices of
        dark list with respect to the capture times of the images in this list.
        Then get "closest-in-time" indices of offset list with respect to dark
        list. The latter is done to ensure, that dark and offset set used for
        image correction are recorded subsequently and not individual from each
        other (i.e. only closest in time to the current image)
        """
        dark_assigned = False
        offset_assigned = False
        try:
            texp = self.current_img().texp
            if texp == 0 or isnan(texp):
                raise ValueError
        except BaseException:
            print_log.warning("Exposure time could not be accessed in ImgList %s"
                 % self.list_id)

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
                                or lst.read_gain not in self.dark_lists:
                            self.dark_lists[lst.read_gain] = od()
                            self.dark_lists[lst.read_gain]["list"] = lst
                            dtexp_dark = dt
                            dark_assigned = True
                    except BaseException:
                        self.dark_lists[lst.read_gain] = od()
                        self.dark_lists[lst.read_gain]["list"] = lst
                        dark_assigned = True

                elif lst.list_type == "offset":
                    try:
                        dt = abs(texp - lst.current_img().texp)
                        if (dt < dtexp_offset or
                                lst.read_gain not in self.offset_lists):
                            self.offset_lists[lst.read_gain] = od()
                            self.offset_lists[lst.read_gain]["list"] = lst
                            dtexp_offset = dt
                            offset_assigned = True
                    except BaseException:
                        self.offset_lists[lst.read_gain] = od()
                        self.offset_lists[lst.read_gain]["list"] = lst
                        offset_assigned = True

                else:

                    warnings.append("List %s, type %s could not be linked "
                                    % (lst.list_id, lst.list_type))
            else:
                warnings.append("Obj of type %s could not be linked, need "
                                " DarkImgList " % type(lst))

        for gain, value in six.iteritems(self.dark_lists):
            value["idx"] = self.assign_indices_linked_list(value["list"])
        for gain, value in six.iteritems(self.offset_lists):
            value["idx"] = self.assign_indices_linked_list(value["list"])
        _print_list(warnings)
        return dark_assigned, offset_assigned

    """INDEX AND IMAGE LOAD MANAGEMENT"""

    def change_index_linked_lists(self):
        """Update current index in all linked lists based on ``cfn``."""
        for key, lst in six.iteritems(self.linked_lists):
            lst.goto_img(self._linked_indices[key][self.index],
                         reload_here=self._always_reload[key])

    def load(self):
        """Try load current and next image."""
        self.change_index_linked_lists()  # based on current index in this list
        if not super(ImgList, self).load():
            print_log.warning("Image load aborted...")
            return False
        if self.nof > 1:
            next_img = self._load_image(self.next_index)
            self.loaded_images["next"] = next_img
            self._load_edit["next"].update(next_img.edit_log)
            self._apply_edit("next")
        else:
            print_log.warning("Image list contains only one image. Setting this image both "
                 "in <this> and <next> attr.")
            self.loaded_images["next"] = self.loaded_images["this"]
            self._load_edit["next"].update(self._load_edit["this"])

        if self.optflow_mode:
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                print_log.warning("Reached last index in image list, optflow_mode will be "
                     "deactivated")
                self.optflow_mode = 0
        return True

    def goto_next(self):
        """Load next image in list."""
        if self.nof < 2 or not self._auto_reload:
            logger.warning("Could not load next image, number of files in list: " +
                  str(self.nof))
            return False

        self.iter_indices(to_index=self.next_index)
        self.change_index_linked_lists()  # load new images in all linked lists

        this_img = self.loaded_images["next"]
        self.loaded_images["this"] = this_img
        self._load_edit["this"].update(self._load_edit["next"])

        if this_img.vign_mask is not None:
            self.vign_mask = this_img.vign_mask

        if self.update_cam_geodata:
            self.meas_geometry.update_cam_specs(**this_img.meta)

        next_img = self._load_image(self.next_index)
        self.loaded_images["next"] = next_img
        self._load_edit["next"].update(next_img.edit_log)
        self._apply_edit("next")
        if self.optflow_mode:
            try:
                self.set_flow_images()
                self.optflow.calc_flow()
            except IndexError:
                print_log.warning("Reached last index in image list, optflow_mode will be "
                     "deactivated")
                self.optflow_mode = 0
        return True

    """PROCESSING AND ANALYSIS METHODS"""

    def optflow_histo_analysis(self, lines=None, start_idx=0, stop_idx=None,
                               intensity_thresh=0, **optflow_settings):
        """Perform optical flow histogram analysis for list images.

        The analysis is performed for all list images within the specified
        index (or time) range and for an arbitraty number of PCS lines.

        Parameters
        ----------
        lines : list
            list containing :class:`LineOnImage` instances
        start_idx : :obj:`int` or :obj:`datetime`
            index or timestamp of first considered image. Note that the
            timestamp option only works if acq. times can be accessed from
            filenames for all files in the list (using method
            :func:`timestamp_to_index`)
        stop_idx : :obj:`int` or :obj:`datetime`, optional
            index of last considered image (if None, the last image in this
            list is used). Note that the timestamp option only works if acq.
            times can be accessed from filenames for all files in the list
            (using method :func:`timestamp_to_index`)
        intensity_thresh : float
            additional intensity threshold that may, e.g. be used to identify
            plume pixels (e.g. if list is in ``tau_mode``).
        **optflow_settings
            additional keyword args passed to :class:`OptflowFarneback`

        Returns
        -------
        list
            list containing the computed time series of optical flow
            histogram parameters (:class:`LocalPlumeProperties` instances) for
            each of the provided input :class:`LineOnImage` objects.

        """
        if lines is None:
            lines = []
        cfn_tmp = self.cfn
        if isinstance(start_idx, datetime):
            start_idx = self.timestamp_to_index(start_idx)
        if isinstance(stop_idx, datetime):
            stop_idx = self.timestamp_to_index(stop_idx)
        if stop_idx is None or stop_idx > self.nof:
            stop_idx = self.nof

        num = self._iter_num(start_idx, stop_idx)
        flm = self.optflow_mode
        self.goto_img(start_idx)
        self.optflow.settings.update(**optflow_settings)
        props = []
        for line in lines:
            if isinstance(line, LineOnImage):
                props.append(LocalPlumeProperties(line.line_id,
                                                  color=line.color))

        if len(props) == 0:
            lines = [None]
            props.append(
                LocalPlumeProperties("thresh_%.1f" % intensity_thresh))

        self.optflow_mode = True
        for k in range(num):
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
        """Get bool mask based on intensity threshold.

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
        mask = self.this.get_thresh_mask(thresh)
        if this_and_next and not self.cfn == self.nof - 1:
            next_mask = self.loaded_images["next"].get_thresh_mask(thresh)
            mask = logical_or(mask, next_mask).astype(uint8)
        return mask

    def det_vign_mask_from_bg_img(self):
        r"""Determine vignetting mask from current background image.

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
        bg_img = self.bg_img
        if bg_img.is_vigncorr:
            raise AttributeError("Cannot compute vignetting correction mask "
                                 "from current BG image in ImgList %s: BG "
                                 "image is already vignetting corrected"
                                 % self.list_id)
        mask = bg_img.duplicate()
        if mask.edit_log["blurring"] < 3:
            mask.add_gaussian_blurring(3)
        mask.img = mask.img / mask.img.max()
        self.vign_mask = Img(mask)
        return self.vign_mask

    def calc_sky_background_mask(self, lower_thresh=None,
                                 apply_movement_search=True,
                                 **settings_movement_search):
        """Retrieve and set background mask for 2D poly surface fit.

        Calculates mask specifying sky radiance pixels for background
        modelling mode 0. The mask is updated in the background model
        (class attribute :attr:`bg_model`).

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
        return self.bg_model.calc_sky_background_mask(self.this,
                                     self.loaded_images["next"],
                                     lower_thresh,
                                     apply_movement_search,
                                     **settings_movement_search)

# =============================================================================
#     def prepare_bg_fit_mask(self, **kwargs):
#         """Calculate mask specifying sky-reference pixels in current image
#
#         Note
#         ----
#
#         1. The method was redefined and renamed, please see (and use)
#             :func:`calc_sky_background_mask` instead
#         2. This is a beta version
#
#         """
#         logger.warning("Old name (wrapper) for method calc_sky_background_mask")
#
#         return self.calc_sky_background_mask(**kwargs)
# =============================================================================

# =============================================================================
#     def prep_data_dilutioncorr_old(self, tau_thresh=0.05,
#                                    plume_pix_mask=None, plume_dists=None,
#                                    ext_coeff=None):
#         """Get parameters relevant for dilution correction
#
#         Relevant parameters are:
#
#             1. Current plume background
#             #. Plume distance estimate (either global or on a pixel basis)
#             #. Plume pixel mask (only plume pixels are corrected)
#
#         Note
#         ----
#         This method changes the current image preparation state such that tau
#         mode is deactivated and vigncorr mode is activated.
#
#         Parameters
#         ----------
#         tau_thresh : float
#             tau threshold for retrieval of plume pixel mask. Is only used in
#             case next :param:`plume_mask` is unspecified or invalid. In this
#             case the plume mask is retrieved using :func:`get_thresh_mask`.
#         plume_pix_mask : :obj:`array`, :obj:`Img`, optional
#             mask specifying plume pixels. If valid, it will be passed through
#             and no threshold mask will retrieved (see :param:`tau_thresh`)
#         plume_dists : :obj:`array`, :obj:`Img`, :obj:`float`, optional
#             plume distance(s) in m. If input is numpy array or :class:`Img`
#             then, it must have the same shape as the current image
#         ext_coeff : :obj:`float`, optional
#             atmospheric extinction coefficient. If unspecified, try access
#             via :attr:`ext_coeff` which returns the current extinction
#             coefficient and raise :obj:`AttributeError` in case, no coeffs
#             are assigned to this list
#
#         Returns
#         -------
#         tuple
#             5-element tuple containing input for dilution correction
#
#             - :obj:`Img`, current vignetting corrected image
#             - :obj:`float`, current extinction coefficient
#             - :obj:`Img`, current plume background
#             - (:obj:`array`, :obj:`float`), plume distance(s)
#             - :obj:`array`, mask specifying plume pixels
#         """
#         # check input distance and if invalid try retrieve using measurement
#         # geometry
#         try:
#             try:
#                 plume_pix_mask = plume_pix_mask.img
#             except:
#                 pass
#
#             if plume_pix_mask.shape == self.this.shape:
#                 mask_ok = True
#             else:
#                 mask_ok = False
#         except:
#             mask_ok = False
#
#
#         dists = plume_dists
#
#         if dists is None:
#             try:
#                 (_,
#                  _,
#                  dists)=\
#                  self.meas_geometry.compute_all_integration_step_lengths(
#                          pyrlevel=self.pyrlevel)
#                 dists = dists.img
#             except:
#                 raise ValueError("Measurement geometry not ready for access "
#                     "of plume distances in image list %s. Please provide "
#                     "plume distance using input parameter plume_dist_m"
#                     %self.list_id)
#         # get current extinction coefficient, raises AttributeError if not
#         # available
#         try:
#             ext_coeff = float(ext_coeff)
#         except:
#             ext_coeff = self.ext_coeff
#         self.vigncorr_mode = False
#         self.tau_mode = True
#         tau0 = self.current_img().duplicate()
#         self.vigncorr_mode = True
#         #bg = self.bg_model.current_plume_background
#         #bg.edit_log["vigncorr"] = True
#         if not mask_ok:
#             #print "Retrieving plume pixel mask in list %s" %self.list_id
#             plume_pix_mask = self.get_thresh_mask(tau_thresh)
#         self.tau_mode = False
#         bg = self.current_img() * exp(tau0.img)
#         return (self.current_img(), ext_coeff, bg, dists, plume_pix_mask)
# =============================================================================

    def calc_plumepix_mask(self, od_img, tau_thresh=0.05,
                           erosion_kernel_size=0,
                           dilation_kernel_size=0):
        """Calculate plume pixel mask from an OD image using a OD thrshold.

        The method further allows to close gaps using a suitable combination
        of an erosion
        """
        if not od_img.is_tau:
            raise ValueError("Input image must be optical density image "
                             "(or similar, e.g. calibrated CD image)")

        mask = od_img.to_binary(threshold=tau_thresh,
                                new_img=True)
        if erosion_kernel_size > 0:
            mask.erode(ones((erosion_kernel_size,
                             erosion_kernel_size), dtype=uint8))
        if dilation_kernel_size > 0:
            mask.dilate(ones((dilation_kernel_size,
                              dilation_kernel_size), dtype=uint8))
        return mask

    def correct_dilution(self, img, plume_bg_vigncorr=None,
                         plume_pix_mask=None, plume_dists=None,
                         ext_coeff=None, tau_thresh=0.05, vigncorr_mask=None,
                         erosion_kernel_size=0, dilation_kernel_size=0,
                         img_check_plumemask=True):
        r"""Correct a plume image for the signal dilution effect.

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
        (in the best case on the pixel-level) and a binary mask specifying
        plume image pixels. If the latter is not provided on input, it is
        computed within this function by calculating an OD (tau or AA) image
        (based on current list state) and by applying a specified OD threshold.
        Thus, in case no mask is provided, it must be possible to
        compute optical density images in this list, hence the :attr:`bg_model`
        (instance of :class:`PlumeBackgroundModel`) needs to be ready for
        tau image computation. In addition, a vignetting correction mask must
        be available.

        Parameters
        ----------
        img : Img
            the plume image object
        plume_bg_vigncorr : :obj:`Img`, optional
            vignetting corrected plume background image used for dilution
            correction
        plume_pix_mask : :obj:`Img`, optional
            binary mask specifying plume pixels in the image, is retrieved
            automatically if input is None
        plume_dists : :obj:`ndarray` or :obj:`Img`, optional
            2D array containing pixel based plume distances. If None, this
            mask will be attempted to be retrieved from the
            :class:`MeasGeometry` instance assigned to this list
        ext_coeff : :obj:`float`, optional
            atmospheric extinction coefficient. If unspecified, try access
            via :attr:`ext_coeff` which returns the current extinction
            coefficient and raises :obj:`AttributeError` in case, no coeffs are
            assigned to this list
        tau_thresh : float
            OD (tau) threshold to compute plume pixel mask (irrelevant if
            next :param:`plume_pix_mask` is provided)
        vigncorr_mask : :obj:`ndarray` or :obj:`Img`, optional
            mask used for vignetting correction
        erosion_kernel_size : int
            if not zero, the morphological operation erosion is applied
            to the plume pixel mask (e.g. to remove noise outliers) using
            an appropriate quadratic kernel corresponding to the input size
        dilation_kernel_size : int
            if not zero, the morphological operation dilation is applied
            to the plume pixel mask (e.g. to slightly extend the borders of
            the detected plume) using an appropriate quadratic kernel
            corresponding to the input size
        img_check_plumemask : bool
            if True, the current dark and vignetting correction states of
            plume and BG images are checked before computation of the plume
            background and, if applicable, the plume pixel mask

        Returns
        -------
        tuple
            3-element tuple containing

            - :obj:`Img`, dilution corrected input image (vignetting corrected)
            - :obj:`Img`, vignetting corrected plume background used for the\
            correction (is computed within this function body if not provided\
            on input
            - :obj:`array`, mask specifying plume pixels

        """
        if img.is_tau or img.is_aa or img.is_calibrated:
            raise ValueError("Img must not be an OD, AA or calibrated CD img")
        if vigncorr_mask is not None:
            self.vign_mask = vigncorr_mask
        vign_mask = self.vign_mask  # raises Exception if not available
        calc_mask = True if plume_pix_mask is None else False
        compute_bg = False if isinstance(plume_bg_vigncorr, Img) else True
        if plume_dists is None:
            plume_dists = self.plume_dists
        # get current extinction coefficient, raises AttributeError if not
        # available

        ext_coeff = self.ext_coeff if ext_coeff is None else float(ext_coeff)

        # initiate variable that may be filled with the uncorrected plume
        # background (if this is an onband list in AA mode, see below)
        if calc_mask or compute_bg:
            # model OD image for computation of plume pixel mask and current
            # sky background
            bg_raw = self.bg_img.to_pyrlevel(img.pyrlevel)
            tau_uncorr = self.bg_model.get_tau_image(
                img, bg_raw, check_state=img_check_plumemask)

        if not img.is_vigncorr:
            img.correct_vignetting(vign_mask, new_state=True)

        # compute plume background in image
        if compute_bg:
            plume_bg_vigncorr = img * exp(tau_uncorr.img)

        if calc_mask:
            # print "Retrieving plume pixel mask in list %s" %self.list_id
            plume_pix_mask = self.calc_plumepix_mask(tau_uncorr, tau_thresh,
                                                     erosion_kernel_size,
                                                     dilation_kernel_size)

        from .dilutioncorr import correct_img
        corr = correct_img(img, ext_coeff,
                           plume_bg_vigncorr, plume_dists, plume_pix_mask)

        bad_pix = corr.img <= 0
        corr.img[bad_pix] = img.img[bad_pix]

        return (corr, plume_bg_vigncorr, plume_pix_mask)

    def correct_dilution_all(self, tau_thresh=0.05, ext_on=None, ext_off=None,
                             add_off_list=True, save_dir=None,
                             save_masks=False, save_bg_imgs=False,
                             save_tau_prev=False, vmin_tau_prev=None,
                             vmax_tau_prev=None, **kwargs):
        """Correct all images for signal dilution.

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
        self.darkcorr_mode = True
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
        num = self._iter_num(0, self.nof)
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
        """Import extinction coefficients from csv.

        The text file requires datetime information in the first column and
        a header which can be used to identify the column. The import is
        performed using :func:`pandas.read_csv`

        Parameters
        ----------
        file_path : str
          the csv data file
        header_id : str
          header string for column containing ext. coeffs
        **kwargs :
          additionald keyword args passed to :func:`pandas.read_csv`

        Returns
        -------
        Series
            pandas Series containing extinction coeffs

        Todo
        ----
        This is a Beta version, insert try / except block after testing

        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            s = df[header_id]
        except BaseException:
            s = pd.read_csv(file_path, header=None, index_col=0, squeeze=True,
                            parse_dates=True, **kwargs)
        self.ext_coeffs = s
        return self.ext_coeffs

    """HELPERS"""

    def has_bg_img(self):
        """Return boolean whether or not background image is available."""
        if not isinstance(self.bg_img, Img):
            return False
        return True

    def update_index_dark_offset_lists(self):
        """Check and update current dark image (if possible / applicable)."""
        if self.darkcorr_opt == 0:
            return False
        t_last = self.time_last_dark_check
        ctime = self.current_time()
        if not (
            (t_last - timedelta(minutes=self.update_dark_ival)) < ctime <
                (t_last + timedelta(minutes=self.update_dark_ival))):
            if self.set_closest_dark_offset():
                logger.info("Updated dark / offset in img_list %s at %s"
                      % (self.list_id, ctime))
                self.time_last_dark_check = ctime
                return True
        return False

    """Private methods"""

    def _apply_edit(self, key):
        """Apply the current image edit settings to image.

        Parameters
        ----------
        key : str
            image identifier (use "this" or "next")

        """
        if not self.edit_active:
            logger.warning("Edit not active in img_list %s: no image preparation will "
                 "be performed" % self.list_id)
            return
        img = self.loaded_images[key]
        # apply dark correction
        if self.darkcorr_mode:
            dark = self.get_dark_image(key).to_pyrlevel(img.pyrlevel)
            img.subtract_dark_image(dark)
        if self.shift_mode:
            if img.pyrlevel != 0:
                raise AttributeError("Shift cannot be applied for images that "
                                     "are not on pyramid level 0 on load")
            img.shift(*self.camera.reg_shift_off)
        # things are more complicated if dilution correction mode is active,
        # since this requires the retrieval of a plume pixel mask, the
        # retrieval of the actual pixel dependent plume background and a
        # correction for vignetting. This is even more complicated, if the
        # list is in AA mode, since then, these things need to be done both
        # for off and onband. Finally, the (vignetting and dilution corrected)
        # raw images can be used to compute the dilution corrected optical
        # densities
        if self.dilcorr_mode:
            s = self.dilcorr_settings
            # init variable plume_pix_mask
            plume_pix_mask = None
            if self.aa_mode:
                # hold a copy of the uncorrected onband plume image that is
                # needed below to compute the offband background
                img_uncorr = img.duplicate()
                # if this is an onband list that is in AA mode, the plume pixel
                # mask is retrieved from a precomputed AA image which is done
                # below. This allows in one go to also retrieve the actual
                # plume background in the offband from the already computed
                # onband background
                off_list = self.get_off_list()
                off_img = off_list.loaded_images[key].to_pyrlevel(img.pyrlevel)
                if off_img.is_vigncorr:
                    print_log.warning("List %s is in AA mode: it appears that "
                         "vigncorr_mode is active in either or both of the "
                         "lists %s and %s. You might want to consider to turn "
                         "off vigncorr_mode as this is irrelevant for the "
                         "computation of OD and AA images and only slows "
                         "things down a little")
                    off_img = off_img.duplicate().\
                        correct_vignetting(off_list.vign_mask,
                                           new_state=False)
                if off_img.is_tau:
                    raise AttributeError("Off-band image is in tau mode, "
                                         "please deactivate tau_mode in "
                                         "offband list...")
                bg_on = self.bg_img.to_pyrlevel(img.pyrlevel)
                bg_off = off_list.bg_img.to_pyrlevel(img.pyrlevel)
                aa_uncorr = self.bg_model.get_aa_image(img, off_img,
                                                       bg_on, bg_off)
                plume_pix_mask = self.calc_plumepix_mask(
                    aa_uncorr,
                    s.tau_thresh,
                    s.erosion_kernel_size,
                    s.dilation_kernel_size)

            (img,  # vignetting corrected and dilution corrected image
             bg_vigncorr,
             plume_pix_mask) = self.correct_dilution(
                img,
                plume_pix_mask=plume_pix_mask,
                tau_thresh=s.tau_thresh,
                erosion_kernel_size=s.erosion_kernel_size,
                dilation_kernel_size=s.dilation_kernel_size)
            if self.tau_mode:
                img = s.bg_model.get_tau_image(plume_img=img,
                                               bg_img=bg_vigncorr)
            elif self.aa_mode:
                if off_img.is_dilcorr:
                    raise AttributeError("Current offband image in linked "
                                         "off-band list is corrected for "
                                         "signal dilution. Please "
                                         "deactivate.")

                bg_on = bg_vigncorr.duplicate().\
                    correct_vignetting(self.vign_mask, 0)
                bg_off = (bg_on * off_img / (img_uncorr * exp(aa_uncorr.img)))

                bg_off_vigncorr = bg_off.correct_vignetting(off_list.vign_mask)

                (off_img, _, _) = off_list.correct_dilution(
                    off_img,
                    plume_bg_vigncorr=bg_off_vigncorr,
                    plume_pix_mask=plume_pix_mask)

                img = s.bg_model.get_aa_image(plume_on=img,
                                              plume_off=off_img,
                                              bg_on=bg_vigncorr,
                                              bg_off=bg_off_vigncorr)

        else:
            bg = None
            if self.tau_mode:
                if self.bg_model.mode > 0:  # dilution_corr is not active
                    bg = self.bg_img.to_pyrlevel(img.pyrlevel)
                img = self.bg_model.get_tau_image(plume_img=img,
                                                  bg_img=bg)
            elif self.aa_mode:
                off_list = self.get_off_list()
                off_img = off_list.loaded_images[key].to_pyrlevel(img.pyrlevel)
                if off_img.is_vigncorr:
                    print_log.warning("List %s is in AA mode: it appears that "
                         "vigncorr_mode is active in either or both of the "
                         "lists %s and %s. You might want to consider to turn "
                         "off vigncorr_mode as this is irrelevant for the "
                         "computation of OD and AA images and only slows "
                         "things down a little")
                    off_img = off_img.duplicate()
                    off_img.correct_vignetting(off_list.vign_mask,
                                               new_state=False)
                if off_list.dilcorr_mode:
                    raise AttributeError("Linked off-band list has dilution "
                                         "correction mode activated. Please "
                                         "deactivate.")
                elif off_img.is_tau:
                    raise AttributeError(
                        "Linked off-band list is in tau mode. "
                        "Please deactivate...")

                if self.bg_model.mode > 0:  # dilution_corr is not active
                    bg = self.bg_img.to_pyrlevel(img.pyrlevel)

                # make sure, the dilution correction mode is activated in the
                # off list if it is activated here
                bg_off = off_list.bg_img.to_pyrlevel(img.pyrlevel)

                img = self.bg_model.get_aa_image(plume_on=img,
                                                 plume_off=off_img,
                                                 bg_on=bg,
                                                 bg_off=bg_off)
        # if image is tau or AA
        if img.is_tau:
            if self.sensitivity_corr_mode:
                img = img / self.senscorr_mask
                img.edit_log["senscorr"] = 1
            if self.calib_mode:
                img.img = self.calib_data(img.img)
                img.edit_log["gascalib"] = True

        # apply vignetting correction only to images in intensity space
        if not img.is_tau and self.vigncorr_mode:
            img.correct_vignetting(self.vign_mask, new_state=True)

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
        """Try to compute an AA test-image."""
        on = self._load_image(self.index)
        off = off_list._load_image(off_list.index)
        bg_on = self.bg_img.to_pyrlevel(on.pyrlevel)
        bg_off = off_list.bg_img.to_pyrlevel(off.pyrlevel)
        return self.bg_model.get_aa_image(on, off, bg_on, bg_off,
                                          check_state=False)


class CellImgList(ImgList):
    """Image list object for cell images.

    Whenever cell calibration is performed, one calibration cell is put in
    front of the lense for a certain time and the camera takes one (or ideally)
    a certain amount of images.

    This image list corresponds to such a list of images with one specific
    cell in the camera FOV. It is a :class:`BaseImgList` only extended by
    the variable ``self.gas_cd`` specifying the amount of gas (column
    density) in this cell.
    """

    def __init__(self, files=None, list_id=None, list_type=None, camera=None,
                 geometry=None, cell_id="", gas_cd=0.0, gas_cd_err=0.0):

        super(CellImgList, self).__init__(files, list_id, list_type, camera,
                                          geometry)
        self.cell_id = cell_id
        self.gas_cd = gas_cd
        self.gas_cd_err = gas_cd_err

    def update_cell_info(self, cell_id, gas_cd, gas_cd_err):
        """Update cell_id and gas_cd amount."""
        self.cell_id = cell_id
        self.gas_cd = gas_cd
        self.gas_cd_err = gas_cd_err


class ImgListLayered(ImgList):
    """Image list object able to deal with multi layered fits files.

    Additional features:

            1. Indexing using double index: Filename and image layer
            2. Function which returns a DataFrame of all available data

    Parameters
    ----------
    files : list
        list containing image file paths, defaults to ``[]`` (i.e. empty list)
    meta : DataFrame
        meta data from a previous load of the same image list. Can be used
        alternatively for a faster initialisation
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

    Note
    ----
    Initialise with a list of n files each containing a (variable)
    number of image layers) m_n. The file header needs to be read in for
    every file in order to get the right amount of total images. The
    attribute `self.files` will contain m_n copies of the file n.
    If the list has been loaded before, the ImgListLayered can also be
    initialised with a DataFrame containing all meta information.

    """

    def __init__(self, files=None, meta=None, list_id=None, list_type=None,
                 camera=None, geometry=None, init=True):
        # uses the init method from ImgList but does not load the files!
        if files is None:
            files = []
        super(ImgListLayered, self).__init__(files, list_id, list_type, camera,
                                             geometry, init=False)

        self.fitsfiles = files

        if isinstance(meta, DataFrame):
            try:
                self.metaData = meta
            except:
                self.metaData = self.get_img_meta_all()
        else:
            self.metaData = self.get_img_meta_all()

        # Image referencing by two information: file and image layer
        # filename subindex (file is repeated m_n times)
        self.files = self.metaData['file'].values
        # image layer inside fits file
        self.hdu_nr = self.metaData['hdu_nr'].values

        if self.data_available and init:
            self.load()

    def get_img_meta_from_filename(self, file_path):
        """Load and prepare img meta input dict for Img object.

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
        print_log.warning('This method does not make a lot of sense for the ImgListLayered!'
             ' Returns the meta data of the first image in file_path.'
             ' Use metaData attribute to access meta information instead.')

        hdulist = fits.open(file_path)
        # Load the image
        image = hdulist[0].data
        time = _read_binary_timestamp(image)
        texp = float(hdulist[0].header['EXP']) / 1000.  # in s
        return {"start_acq": time, "texp": texp}

    def get_img_meta_all_filenames(self):
        """Return the same data as ImgList.get_img_meta_all_filenames.

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
        """Load all meta data from all image layers of one fits file.

        In this form it is custom for the comtessa files
        TODO: Make general for multilayered fits
        """
        # temporary lists of parameters
        imgFileStart = []
        imgFileStop = []
        imgFileMin = []
        imgFileMax = []
        imgFileMean = []
        imgFileExp = []
        imgFileTemp = []

        # open the file, returning a list containg Header-Data Units (HDU)
        hdulist = fits.open(file_path)
        imgPerFile = size(hdulist)
#            hdulist.close()
        for hdu in range(imgPerFile):
            # Info from image
            image = hdulist[hdu].data
            imgFileStop.append(_read_binary_timestamp(image))
            image[0, 0:14] = image[1, 0:14]  # replace binary timestamp
            imgFileMin.append(image.min())
            imgFileMax.append(image.max())
            imgFileMean.append(image.mean())
            # Info from header
            imageHeader = hdulist[hdu].header
            ms = int(imageHeader['EXP']) * 1000
            imgFileStart.append(imgFileStop[-1] -
                                timedelta(microseconds=ms))
            imgFileExp.append(float(imageHeader['EXP']) / 1000.)  # in s
            imgFileTemp.append(float(imageHeader['TCAM']))

        # Combine the temporary lists to a dataFrame and return it
        meta = DataFrame(data={'file': [file_path] * imgPerFile,
                               'hdu_nr': array(range(imgPerFile), dtype=int),
                               'start_acq': imgFileStart,
                               'stop_acq': imgFileStop,
                               'exposure': imgFileExp,
                               'temperature': imgFileTemp,
                               'min': imgFileMin,
                               'max': imgFileMax,
                               'mean': imgFileMean},
                         index=imgFileStart)
        meta.index = to_datetime(meta.index)
        return meta

    def get_img_meta_all(self):
        """Load all available meta data from fits files.

        Returns
        -------
        dataFrame
            containing all metadata

        """
        if self.fitsfiles == []:
            logger.warning("ImgListLayered was intialised without providing the "
                  "fitsfile (e.g. only by meta file). self.get_img_meta_all "
                  "will return the existing metaData.")
            return self.metaData

        # Exatract information of every image in every fits file in fitsFile
        meta_single = [self.get_img_meta_one_fitsfile(file_path)
                       for file_path in self.fitsfiles]
        meta = concat(meta_single)
        meta['img_id'] = arange(0, len(meta), 1)
        meta.index = to_datetime(meta.index)
        return meta

    def _load_image(self, list_index):
        """Load a single image.

        Parameters
        ----------
        index : int
            index of image which should be loaded
        Returns
        -------
        Img
            loaded image including meta data

        """
        img_file = self.files[list_index]
        img_hdu = self.hdu_nr[list_index]
        meta = {}
        meta["fits_idx"] = img_hdu
        meta["filter_id"] = self.list_id

        # Comment jgliss (24.5.19): removed try/except block that would just give warning
        # but would then throw error anyways because `image` is not initialised...
        # Developers: please only insert try / except blocks where you can actually
        # catch the exception, everything else may lead to silent and hard to debug
        # errors.
        image = Img(input=img_file,
                    import_method=self.camera.image_import_method,
                    **meta)
        return image

# OLD version of ImgList before major changes (stamp: 3/3/2018)
# =============================================================================
# class ImgList(BaseImgList):
#     """Image list object with expanded functionality
#        (cf. :class:`BaseImgList`)
#
#     Additional features:
#
#             1. Optical flow determination
#             #. Linking of lists (e.g. on and offband lists)
#             #. Dark and offset image correction
#             #. Plume background modelling and tau image determination
#             #. Methods for dilution correction
#             #. Automatic vignetting correction
#             #. Assignment of calibration data and automatic image calibration
#
#     Parameters
#     ----------
#     files : list
#         list containing image file paths, defaults to ``[]``
#         (i.e. empty list)
#     list_id : :obj:`str`, optional
#         string ID of this list, defaults to None
#     list_type : :obj:`str`, optional
#         string specifying type of image data in this list (e.g. on, off)
#     camera : :obj:`Camera`, optional
#         camera specifications, defaults to None
#     geometry : :obj:`MeasGeometry`, optional
#         measurement geometry
#     init : bool
#         if True, the first two images in list ``files`` are loaded
#     **dilcorr_settings
#         additional keyword args corresponding to settings for automatic
#         dilution correction passed to __init__ of
#         :class:´AutoDilcorrSettings`
#
#     """
#     def __init__(self, files=[], list_id=None, list_type=None, camera=None,
#                  geometry=None, init=True, **dilcorr_settings):
#
#         super(ImgList, self).__init__(files, list_id, list_type, camera,
#                                       geometry, init=False)
#
#         self.loaded_images.update({"next": None})
#
#         # List modes (currently only tau) are flags for different list states
#         # and need to be activated / deactivated using the corresponding
#         # method (e.g. :func:`activate_tau_mode`) to be changed, dont change
#         # them directly via this private dictionary
#         self._list_modes.update({
#             "darkcorr": False,  # dark correction
#             "optflow": False,  # compute optical flow
#             "vigncorr": False,  # load vignetting corrected images
#             "dilcorr": False,  # load as dilution corrected images
#             "tau": False,  # load as OD images
#             "aa": False,  # load as AA images
#             "senscorr": False,  # correct for cross-detector sensitivity
#                                 # variations
#             "gascalib": False})  # load as calibrated SO2 images
#
#         self._ext_coeffs = None
#
#         self._bg_imgs = [None, None]  # sets bg images
#         self._bg_list_id = None  # ID of linked background list
#         self._which_bg = "img"  # change using :attr:`which_bg` either,
#                                 # img or list
#
#         self.bg_model = PlumeBackgroundModel()
#
#         self._senscorr_mask = None
#         self._calib_data = None
#
#         # these two images can be set manually, if desired
#         self.master_dark = None
#         self.master_offset = None
#
#         # These dicitonaries contain lists with dark and offset images
#         self.dark_lists = od()
#         self.offset_lists = od()
#         self._dark_corr_opt = self.camera.darkcorr_opt
#
#         # Dark images will be updated every 10 minutes (i.e. before an image
#         # is dark and offset corrected it will be checked if the currently
#         # loaded images match the time interval (+-10 min) of this image and
#         # if nota new one will be searched).
#         self.update_dark_ival = 10 #mins
#         self.time_last_dark_check = datetime(1900, 1, 1)
#
#         # tau threshold for calculation of plume pixel mask fro dilution
#         # correction
#         self.dilcorr_settings = AutoDilcorrSettings(**dilcorr_settings)
#         #self.currentMaxI=None
#
#         #Optical flow engine
#         self.optflow = OptflowFarneback(name=self.list_id)
#
#
#         if self.data_available and init:
#             self.load()
#
# # ===========================================================================
# #     @property
# #     def next(self):
# #         """Next image"""
# #         return self.loaded_images["next"]
# # ===========================================================================
#
#
#     @property
#     def darkcorr_opt(self):
#         """Return the current dark correction mode
#
#         The following modes are available:
#
#             0   =>  no dark correction possible (is e.g. set if camera is
#                     unspecified)
#             1   =>  individual correction with separate dark and offset
#                     (e.g. ECII data)
#             2   =>  one dark image which is subtracted (including the offset,
#                     e.g. HD cam data)
#
#         For details see documentation of :class:`CameraBaseInfo`
#         """
#         return self._dark_corr_opt
#
#
#     @darkcorr_opt.setter
#     def darkcorr_opt(self, val):
#         try:
#             val = int(val)
#             if not val in [0, 1, 2]:
#                 raise ValueError
#             self._dark_corr_opt = val
#         except:
#             logger.warning("Failed to update dark correction option")
#
#     @property
#     def BG_MODEL_MODE(self):
#         """Current background image modelling mode"""
#         return self.bg_model.mode
#
#     @BG_MODEL_MODE.setter
#     def BG_MODEL_MODE(self, val):
#         self.bg_model.mode = val
#         self.load()
#
#     @property
#     def darkcorr_mode(self):
#         """Returns current list darkcorr mode"""
#         return self._list_modes["darkcorr"]
#
#     @darkcorr_mode.setter
#     def darkcorr_mode(self, value):
#         """Change current list darkcorr mode
#
#         Wrapper for :func:`activate_darkcorr`
#         """
#         return self.activate_darkcorr(value)
#
#     @property
#     def optflow_mode(self):
#         """Activate / deactivate optical flow calc on image load"""
#         return self._list_modes["optflow"]
#
#     @optflow_mode.setter
#     def optflow_mode(self, val):
#         self.activate_optflow_mode(val)
#
#     @property
#     def vigncorr_mode(self):
#         """Activate / deactivate vignetting correction on image load"""
#         return int(self._list_modes["vigncorr"])
#
#     @vigncorr_mode.setter
#     def vigncorr_mode(self, val):
#         self.activate_vigncorr(val)
#
#     @property
#     def dilcorr_mode(self):
#         """Activate / deactivate dilution correction on image load"""
#         return int(self._list_modes["dilcorr"])
#
#     @dilcorr_mode.setter
#     def dilcorr_mode(self, val):
#         self.activate_dilcorr_mode(val)
#
#     @property
#     def sensitivity_corr_mode(self):
#         """Activate / deactivate AA sensitivity correction mode"""
#         return self._list_modes["senscorr"]
#
#     @sensitivity_corr_mode.setter
#     def sensitivity_corr_mode(self, val):
#         """Activate / deactivate AA sensitivity correction mode"""
#         if val == self._list_modes["senscorr"]:
#             return
#         if val:
#             self.senscorr_mask #raise AttributeError if mask is not available
#             if not self.aa_mode:
#                 raise AttributeError(
#                     "AA sensitivity correction mode can only "
#                     "be activated in list aa_mode, please activate aa_mode "
#                     "first...")
#
#         self._list_modes["senscorr"] = val
#         self.load()
#
#     @property
#     def tau_mode(self):
#         """Returns current list tau mode"""
#         return self._list_modes["tau"]
#
#     @tau_mode.setter
#     def tau_mode(self, value):
#         """Change current list tau mode
#
#         Wrapper for :func:`activate_tau_mode`
#         """
#         self.activate_tau_mode(value)
#
#     @property
#     def aa_mode(self):
#         """Returns current list AA mode"""
#         return self._list_modes["aa"]
#
#     @aa_mode.setter
#     def aa_mode(self, value):
#         """Change current list AA mode
#
#         Wrapper for :func:`activate_aa_mode`
#         """
#         self.activate_aa_mode(value)
#
#     @property
#     def calib_mode(self):
#         """Acitivate / deactivate current list gas calibration mode"""
#         return self._list_modes["gascalib"]
#
#     @calib_mode.setter
#     def calib_mode(self, value):
#         """Change current list calibration mode"""
#         self.activate_calib_mode(value)
#
#     @property
#     def ext_coeff(self):
#         """Current extinction coefficient"""
#         if not isinstance(self.ext_coeffs, Series):
#             raise AttributeError("Extinction coefficients not available in "
#                 "image list %s" %self.list_id)
#         elif len(self.ext_coeffs) == self.nof:
#             #assuming that time stamps correspond to list time stamps
#             return self.ext_coeffs[self.cfn]
#         else:
#             idx = closest_index(self.current_time(), self.ext_coeffs.index)
#             return self.ext_coeffs[idx]
#
#     @property
#     def ext_coeffs(self):
#         """Dilution extinction coefficients"""
#         return self._ext_coeffs
#
#     @ext_coeffs.setter
#     def ext_coeffs(self, val):
#         if isinstance(val, float):
#             val = Series(val, [self.start_acq[0]])
#         if not isinstance(val, Series):
#             raise ValueError("Need pandas Series object")
#         self._ext_coeffs = val
#
#     @property
#     def bg_img(self):
#         """Return background image based on current vignetting corr
#            setting"""
#         img = None
#         if self.which_bg == "img":
#             img = self._bg_imgs[self.loaded_images["this"].\
#                     edit_log["vigncorr"]]
#         elif self.which_bg == "list":
#             lst = self.bg_list
#             try:
#                 img = lst.current_img()
#             except:
#                 raise AttributeError(
#                    "Background image list not assigned or "
#                    "does not contain images %s" % self.list_id)
#         else:
#             raise ValueError("Invalid bg-img access mode: %s (check attr "
#                              "which_bg")
#         if not isinstance(img, Img):
#             raise AttributeError("No background image found in image list")
#         return img
#
#     @bg_img.setter
#     def bg_img(self, val):
#         self.set_bg_img(val)
#
#     @property
#     def dark_img(self):
#         """Current dark image"""
#         return self.get_dark_image()
#
#     @property
#     def bg_list(self):
#         """Returns background image list (if assigned)"""
#         try:
#             return self.linked_lists[self._bg_list_id]
#         except KeyError:
#             raise AttributeError("No linked background list found with "
#                                  "ID %s found in ImgList %s. "
#                                  %(self._bg_list_id, self.list_id))
#
#     @bg_list.setter
#     def bg_list(self, val):
#         self.set_bg_list(val)
#
#     @property
#     def which_bg(self):
#         """Specifies from where background images are accessed"""
#         return self._which_bg
#
#     @which_bg.setter
#     def which_bg(self, val):
#         if val in ["img", "list"]:
#             if val == "list" and self.vigncorr_mode:
#                 raise AttributeError("Cannot set bg-img access mode to list "
#                                      "since vigncorr_mode is active...")
#             self._which_bg = val
#             #warns if no background image is available for this access method
#             self.bg_img
#         else:
#             raise ValueError("Invalid input: choose from img or list")
#
#     @property
#     def senscorr_mask(self):
#         """Get / set AA correction mask"""
#         if isinstance(self._senscorr_mask, ndarray):
#             logger.warning("AA correction mask in list %s is numpy array and"
#             "will be converted into Img object" %self.list_id)
#             self._senscorr_mask = Img(self._senscorr_mask)
#         if not isinstance(self._senscorr_mask, Img):
#             raise AttributeError("AA correction mask is not available...")
#         return self._senscorr_mask
#
#     @senscorr_mask.setter
#     def senscorr_mask(self, val):
#         """Setter for AA correction mask"""
#         if isinstance(val, ndarray):
#             logger.warning("Input for AA correction mask in list %s is numpy array and"
#                     "will be converted into Img object" %self.list_id)
#             val = Img(val)
#         if not isinstance(val, Img):
#             raise TypeError("Invalid input for AA correction mask: need Img"
#                 " object (or numpy array)")
#         if not val.pyrlevel == 0:
#             logger.warning("AA correction mask is required to be at pyramid level 0 "
#                 "and will be converted")
#             val.to_pyrlevel(0)
#         img_temp = self._load_image(self.index)
#         if val.shape != img_temp.shape:
#             try:
#                 val = val.to_pyrlevel(img_temp.pyrlevel)
#                 if val.shape != img_temp.shape:
#                     raise ValueError
#             except:
#                 raise ValueError("Img shape mismatch between AA correction "
#                     "mask and list images")
#
#         self._senscorr_mask = val
#
#     @property
#     def calib_data(self):
#         """Get set object to perform calibration"""
#         from pyplis.cellcalib import CellCalibData as cc
#         from pyplis.doascalib import DoasCalibData as dc
#         if not any([isinstance(self._calib_data, x) for x in [cc, dc]]):
#             logger.warning("No calibration data available in imglist %s" %self.list_id)
#         return self._calib_data
#
#     @calib_data.setter
#     def calib_data(self, val):
#         from pyplis.cellcalib import CellCalibData as cc
#         from pyplis.doascalib import DoasCalibData as dc
#         if not any([isinstance(val, x) for x in [cc, dc]]):
#             raise TypeError("Could not set calibration data in imglist %s: "
#             "need CellCalibData obj or DoasCalibData obj" %self.list_id)
#         try:
#             val(0.1) #try converting a fake tau value into a gas column
#         except ValueError:
#             raise ValueError("Cannot set calibration data in image list, "
#                 "calibration object is not ready")
#         self._calib_data = val
#
#     @property
#     def doas_fov(self):
#         """Try access DOAS FOV info (in case cailbration data is
#            available)"""
#         try:
#             return self.calib_data.fov
#         except:
#             logger.warning("No DOAS FOV information available")
#
#     @property
#     def img_mode(self):
#         """Checks and returns current img mode (tau, aa, or raw)
#
#         :return:
#             - "tau", if ``self._list_modes["tau"] == True``
#             - "aa", if ``self._list_modes["aa"] == True``
#             - "raw", else
#         """
#         if self._list_modes["tau"] == True:
#             return "tau"
#         elif self._list_modes["aa"] == True:
#             return "aa"
#         else:
#             return "raw"
#
#     """RESETTING AND INIT METHODS"""
# # ===========================================================================
# #     def init_filelist(self):
# #         """Adding functionality to filelist init"""
# #         super(ImgList, self).init_filelist()
# # ===========================================================================
#
#     def init_bg_model(self, **kwargs):
#         """Init clear sky reference areas in background model"""
#         self.bg_model.update(**kwargs)
#         self.bg_model.set_missing_ref_areas(self.current_img())
#
#     """LIST MODE MANAGEMENT METHODS"""
#     def activate_darkcorr(self, value=True):
#         """Activate or deactivate dark and offset correction of images
#
#         If dark correction turned on, dark image access is attempted, if that
#         fails, Exception is raised including information what did not work
#         out.
#
#         Parameters
#         ----------
#         val : bool
#             new mode
#         """
#         if value is self.darkcorr_mode: #do nothing
#             return
#         if not value and self._load_edit["this"]["darkcorr"]:
#             raise ImgMetaError("Cannot deactivate dark correction, original"
#                 "image file was already dark corrected")
#         if value:
#             if self.this.edit_log["darkcorr"]:
#                 logger.warning("Cannot activate dark correction in image list %s: "
#                      "current image is already corrected for dark current"
#                      %self.list_id)
#                 return
#             self.get_dark_image()
#             self.update_index_dark_offset_lists()
#
#         self._list_modes["darkcorr"] = value
#         self.load()
#
#     def activate_vigncorr(self, value=True):
#         """Activate / deactivate vignetting correction on image load
#
#         Note
#         ----
#
#         Requires ``self.vign_mask`` to be set or an background image
#         to be available (from which ``self.vign_mask`` is then determined)
#
#         Parameters
#         ----------
#         value : bool
#             new mode
#         """
#         if value is self.vigncorr_mode: #do nothing
#             return
#         elif self.which_bg == "list":
#             raise AttributeError("Feature not yet available: vigncorr mode "
#                                  "cannot be activated if bg-img access mode "
#                                  "is set to <list>. Please update using "
#                                  "attr. which_bg='img'.")
#         elif value:
#             if self.this.edit_log["vigncorr"]:
#                 logger.warning("Cannot activate vignetting correction in image list "
#                      "%s: current image is already corrected for vignetting"
#                      % self.list_id)
#                 return
#             try:
#                 self.vign_mask
#             except:
#                 self.det_vign_mask_from_bg_img()
#             sh = self._load_image(self.index).img.shape
#             if not self.vign_mask.shape == sh:
#                 raise ValueError("Shape of vignetting mask %s deviates from "
#                             "raw img shape %s" %(list(self.vign_mask.shape),
#                             list(sh)))
#         self._list_modes["vigncorr"] = value
#         self.load()
#
#     def activate_tau_mode(self, value=True):
#         """Activate tau mode
#
#         In tau mode, images will be loaded as tau images (if background image
#         data is available).
#
#         Parameters
#         ----------
#         value : bool
#             new mode
#
#         """
#         if value is self.tau_mode: #do nothing
#             return
#         if value:
#             if self.this.edit_log["is_tau"]:
#                 logger.warning("Cannot activate tau mode in image list %s: "
#                      "current image is already a tau image"
#                      %self.list_id)
#                 return
#             cim = self._load_image(self.index)
#             try:
#                 dark = self.get_dark_image("this")
#                 cim.subtract_dark_image(dark)
#             except:
#                 logger.warning("Dark images not available")
#             bg_img = None
#             self.bg_model.set_missing_ref_areas(cim)
#             if self.bg_model.mode == 0:
#                 print ("Background correction mode is 0, initiating "
#                        "settings for poly surface fit")
#                 #self.calc_sky_background_mask()
#                 try:
#                     self.calc_sky_background_mask()
#                 except:
#                     logger.warning("Background access mask could not be retrieved for "
#                         "PolySurfaceFit in background model of image list %s"
#                         %self.list_id)
#
#             else:
#                 if not self.has_bg_img():
#                     raise AttributeError("no background image available in "
#                         "list %s, please set a suitable background image "
#                         "using method set_bg_img, or change current bg "
#                         "modelling mode to 0 using self.bg_model.mode=0)"
#                         %self.list_id)
#                 if self.which_bg == "img":
#                     bg_img = self._bg_imgs[0]
#                 else:
#                     bg_img = self.bg_list.this
#                     if bg_img.is_vigncorr:
#                         raise AttributeError(
#                             "Background image in bg_list is "
#                             "corrected for vignetting. Please check")
#             self.bg_model.get_tau_image(cim, bg_img)
#         self._list_modes["tau"] = value
#         self.load()
#
#     def activate_aa_mode(self, value=True):
#         """Activates AA mode (i.e. images are loaded as AA images)
#
#         In order for this to work, the following prerequisites need to be
#         fulfilled:
#
#             1. This list needs to be an on band list
#             (``self.list_type = "on"``)
#             #. At least one offband list must be linked to this list (if more
#             offband lists are linked and input param off_id is unspecified,
#             then the first offband list found is used)
#             #. The number of images in the off band list must exceed a
#             minimum of 50% of the images in this list
#
#         Parameters
#         ----------
#         val : bool
#             Activate / deactivate AA mode
#
#         """
#         if value is self.aa_mode:
#             return
#         if not self.list_type == "on":
#             raise TypeError("AA mode could not be activated: This list is "
#                             "not an onband list")
#         aa_test = None
#         if value:
#             if self.this.edit_log["is_aa"]:
#                 logger.warning("Cannot activate AA mode in image list %s: "
#                      "current image is already AA image"
#                      %self.list_id)
#                 return
#
#             offlist = self.get_off_list()
#             if not isinstance(offlist, ImgList):
#                 raise Exception("Linked off band list could not be found")
#             if not offlist.nof / float(self.nof) > 0.25:
#                 raise IndexError("Off band list does not have enough images")
#             if self.bg_model.mode != 0:
#                 if not self.has_bg_img():
#                     raise AttributeError("no background image available, "
#                         "please set suitable background image using method "
#                         "set_bg_img or set background modelling mode = 0")
#                 if not offlist.has_bg_img():
#                     raise AttributeError("no background image available in "
#                         "off band list. Please set suitable background "
#                         "image using method set_bg_img or set background "
#                         "modelling mode = 0")
#             #offlist.update_img_prep(**self.img_prep)
#             #offlist.init_bg_model(mode = self.bg_model.mode)
#             self._list_modes["tau"] = False
#             #updated on 12/1/17 (i.e. the current image in the offband list
#             #needs to be reloaded in case the list is in tau mode)
#             #offlist._list_modes["tau"] = False
#             offlist.tau_mode=False
#             aa_test = self._aa_test_img(offlist)
#         self._list_modes["aa"] = value
#
#
#         self.load()
#         return aa_test
#
#     def activate_calib_mode(self, value=True):
#         """Activate calibration mode"""
#         if value == self._list_modes["gascalib"]:
#             return
#
#         if value:
#             if not self.aa_mode:
#                 self._list_modes["aa"] = True
#                 logger.warning("List is not in AA mode")
#
#             if not self.sensitivity_corr_mode:
#                 logger.warning("AA sensitivity correction mode is deactivated. This "
#                     "may yield erroneous results at the image edges")
#             try:
#                 self.calib_data(self.current_img())
#             except TypeError:
#                 raise AttributeError("Calibration data is not available "
#                                      "in image list")
#
#         self._list_modes["gascalib"] = value
#         self.load()
#
#     def activate_dilcorr_mode(self, value=True):
#         """Activate dilution correction mode
#
#         Please see :func:`correct_dilution` for details.
#
#         Parameters
#         ----------
#         value : bool
#             New mode: True or False
#         """
#         if value == self._list_modes["dilcorr"]:
#             return
#         if value:
#             img = self._this_raw_fromfile()
#             _,_,mask = self.correct_dilution(img)
#             # now make sure that in case and off-band list is assigned, it
#             # can also be used to perform a dilution correction (i.e.
#             # bg_model ready)
#             try:
#                 off_list = self.get_off_list()
#                 off_img = off_list._this_raw_fromfile().to_pyrlevel(
#                                   off_list.pyrlevel)
#                 mask = mask.to_pyrlevel(off_list.pyrlevel)
#                 try:
#                     off_list.correct_dilution(off_img, plume_pix_mask=mask)
#                 except:
#                     off_list.bg_model.update(**self.bg_model.settings_dict())
#             except AttributeError as e:
#                 print repr(e)
#
#         self._list_modes["dilcorr"] = value
#         self.load()
#
#     def activate_optflow_mode(self, value=True, draw=False):
#         """Activate / deactivate optical flow calculation on image load
#
#         Parameters
#         ----------
#         val : bool
#             activate / deactivate
#         draw : bool
#             if True, flow field is plotted into current image
#
#
#         """
#         if value is self.optflow_mode:
#             return
#         if value:
#             try:
#                 self.set_flow_images()
#             except IndexError:
#                 raise IndexError("Optical flow mode cannot be activated in "
#                     "image list %s: list is at last index, please change "
#                     "list index and retry")
#             self.optflow.calc_flow()
#             if draw:
#                 self.optflow.draw_flow()
#         self._list_modes["optflow"] = value
#
#     """GETTERS"""
#     def get_dark_image(self, key="this"):
#         """Prepares the current dark image dependent on ``darkcorr_opt``
#
#         The code checks current dark correction mode and, if applicable,
#         prepares the dark image.
#
#             1. ``self.darkcorr_opt == 0`` (no dark correction)
#                 return False
#
#             2. ``self.darkcorr_opt == 1`` (model dark image from a sample
#                 dark and offset image)
#                 Try to access current dark and offset image from
#                 ``self.dark_lists`` and ``self.offset_lists`` (so these must
#                 exist). If this fails for some reason, set
#                 ``self.darkcorr_opt = 2``, else model dark image using
#                 :func:`model_dark_image` and return this image
#
#             3. ``self.darkcorr_opt == 2`` (subtract dark image if exposure
#                 times of current image does not deviate by more than 20% to
#                 current dark image)
#                 Try access current dark image in ``self.dark_lists``, if this
#                 fails, try to access current dark image in ``self.darkImg``
#                 (which can be set manually using :func:`set_dark_image`). If
#                 this also fails, set ``self.darkcorr_opt = 0`` and return
#                 False. If a dark image could be found and the exposure time
#                 differs by more than 20%, set ``self.darkcorr_opt = 0`` and
#                 raise ValueError. Else, return this dark image.
#
#         """
#         if self.darkcorr_opt == 0:
#             raise ValueError("Dark image could not be accessed in list %s: "
#                 "darkcorr_opt is zero, please set darkcorr_opt according "
#                 "to your data type")
#         # this was changed on 8/1/2017
#         texp = self.this.meta["texp"]
#         #img = self.current_img(key)
#         read_gain = self.this.meta["read_gain"]
#         self.update_index_dark_offset_lists()
#         if self.darkcorr_opt == 1:
#             try:
#                 dark = self.dark_lists[read_gain]["list"].current_img()
#                 offset = self.offset_lists[read_gain]["list"].current_img()
#                 dark = model_dark_image(texp, dark, offset)
#             except Exception as e:
#                 msg = format_exc(e)
#                 try:
#                     dark = model_dark_image(texp, self.master_dark,
#                                             self.master_offset)
#                     print "Using master dark and offset image"
#                 except:
#                     raise ValueError("Dark image could not be accessed in "
#                             "image list %s (darkcorr_opt=1), traceback: %s"
#                             %(self.list_id, msg))
#
#         if self.darkcorr_opt == 2:
#             try:
#                 dark = self.dark_lists[read_gain]["list"].current_img()
#                 if not isinstance(dark, Img):
#                     raise ValueError
#             except:
#                 dark = self.master_dark
#                 if not isinstance(dark, Img):
#                     raise ValueError("Dark image could not be accessed in "
#                             "image list %s (darkcorr_opt=2)" %self.list_id)
#         try:
#             texp_ratio = texp / dark.meta["texp"]
#             if not 0.8 <= texp_ratio <= 1.2:
#                 logger.warning("Exposure time of current dark image in list %s "
#                      "deviates by more than 20% from list image %s "
#                      "(current list index: %d)"
#                      %(self.list_id, key, self.cfn))
#         except:
#             pass
#
#         return dark
#
#     def get_off_list(self, list_id=None):
#         """Search off band list in linked lists
#
#         Parameters
#         ----------
#         list_id : :obj:`str`, optional
#             ID of the list. If unspecified (None), the default off band
#             filter key is attempted to be accessed
#             (``self.camera.filter_setup.default_key_off``) and if this fails,
#             the first off band list found is returned.
#
#         Raises
#         ------
#         AttributeError
#             if not offband list can be assigned
#
#         Returns
#         -------
#         ImgList
#             the corresponding off-band list
#         """
#         if list_id is None:
#             try:
#                 list_id = self.camera.filter_setup.default_key_off
#                 #print "Found default off band key %s" %list_id
#             except:
#                 pass
#         for lst in self.linked_lists.values():
#             if lst.list_type == "off":
#                 if list_id is None or list_id == lst.list_id:
#                     return lst
#         raise AttributeError("No linked offband list was found")
#
#     """SETTERS: ATTRIBUTE ASSIGNMENT METHODS"""
#     def set_bg_img(self, bg_img):
#         """Update the current background image object
#
#         Check input background image and, in case a vignetting mask is not
#         available in this list, determine a vignetting mask from the
#         background image. Furthermore, if the input image is not blurred it
#         is blurred using current list blurring factor and in case the
#         latter is 0, then it is blurred with a Gaussian filter of width 1.
#
#         The image is then stored twice, 1. as is and 2. corrected for
#         vignetting.
#
#         Parameters
#         ----------
#         bg_img : Img
#             the background image object used for plume background modelling
#             (modes 1 - 6 in :class:`PlumeBackgroundModel`)
#         """
#         if not isinstance(bg_img, Img):
#             print ("Could not set background image in ImgList %s: "
#                 ": wrong input type, need Img object" %self.list_id)
#             return False
#         try:
#             vign_mask = self.vign_mask
#         except:
#             if bg_img.edit_log["vigncorr"]:
#                 raise AttributeError("Input background image is vignetting "
#                     "corrected and cannot be used to calculate vignetting "
#                     "corr mask.")
#             self._bg_imgs[0] = bg_img
#             vign_mask = self.det_vign_mask_from_bg_img()
#             self._bg_imgs[1] = bg_img.duplicate().correct_vignetting(
#                 vign_mask)
#         else:
#             if not bg_img.edit_log["vigncorr"]:
#                 bg = bg_img
#                 bg_vigncorr = bg_img.duplicate().correct_vignetting(
#                                   vign_mask)
#             else:
#                 bg_vigncorr = bg_img
#                 bg = bg_img.duplicate().correct_vignetting(vign_mask,
#                                                            new_state=0)
#             self._bg_imgs = [bg, bg_vigncorr]
#
#     def set_bg_list(self, lst):
#         """Assign background image list to this list
#
#         Assigns and links an image list containing background images to this
#         list. Similar to other linked lists, the index of the current BG
#         image is automatically updated such that the current BG image is
#         closest intime to the current image in this list. Please note also,
#         that a single master BG image can be assigned using :attr:`bg_img`.
#
#         Parameters
#         ----------
#         lst : ImgList
#             image list containing background images. Note that the input can
#             also be a string specifying the list_id of an image list that is
#             already linked to this list.
#         """
#         if isinstance(lst, str):
#             if not self.linked_lists.has_key(lst):
#                 raise AttributeError("No linked list with ID %s found in "
#                     "image list %s" %(self.list_id, lst))
#             self._bg_list_id = lst
#         elif isinstance(lst, ImgList):
#             lid = "bg_" + self.list_id
#             self.link_imglist(lst, list_id=lid)
#             self._bg_list_id = lid
#         else:
#             raise ValueError("Invalid input for assignment of background "
#                 "image list. Please provide either a string of one of the "
#                 "image lists already linked to this list or provide an "
#                 "ImgList object containing BG images")
#         self.which_bg = "list"
#
#     def set_bg_corr_mode(self, mode=1):
#         """Update the current background correction mode in ``self.bg_model``
#
#         Parameters
#         ----------
#         mode : int
#             valid bakground modelling mode
#         """
#         self.BG_MODEL_MODE = mode
#
#     def set_flow_images(self):
#         """Update images for optical flow determination
#
#         The images are updated in :attr:`optflow`
#         (:class:`OptflowFarneback` object) using method :func:`set_images`
#
#         Raises
#         ------
#         IndexError
#             object, i.e. `self.loaded_images["this"]` and
#             `self.loaded_images["next"]`
#         """
#         if self.cfn == self.nof - 1:
#             self.optflow.reset_flow()
#             raise IndexError("Optical flow images cannot be set in ImgList "
#                 "%s: reached last image ..." %self.list_id)
#
#         self.optflow.set_images(self.loaded_images["this"],
#                                 self.loaded_images["next"])
#
#     def set_optical_flow(self, optflow):
#         """Set the current optical flow object
#
#         Currently only support for type :class:`OptflowFarneback`
#
#         Parameters
#         ----------
#         optflow : OptflowFarneback
#             the optical flow engine
#         """
#         if not isinstance(optflow, OptflowFarneback):
#             raise ValueError("Need class OptflowFarneback")
#         self.optflow = optflow
#
#     def set_darkcorr_mode(self, mode):
#         """Update dark correction mode
#
#         :param int mode (1): new mode
#         """
#         if 0 <= mode <= 2:
#             self.camera.darkcorr_opt = mode
#             return True
#         return False
#
#     def add_master_dark_image(self, dark, acq_time=datetime(1900, 1, 1),
#                               texp=0.0, read_gain=0):
#         """Add a (master) dark image data to list
#
#         Sets a dark image, which is used for dark correction in case,
#         no dark / offset image lists are linked to this object or the data
#         extraction from these lists does not work for some reason.
#
#         :param (Img, ndarray) dark: dark image data
#         :param datetime acq_time: image acquisition time (only updated if
#             input image is numpy array or if acqtime in Img object is
#              default), default: (1900, 1, 1)
#         :param float texp: optional input for exposure time in units of
#             s (i.e. is used if img input is ndarray or if exposure time is
#             not set in the input img)
#
#         The image is stored at::
#
#             stored at self.master_dark
#
#         """
#         if not any([isinstance(dark, x) for x in [Img, ndarray]]):
#             raise TypeError("Could not set dark image in image list, invalid"
#                 " input type")
#         elif isinstance(dark, Img):
#             if dark.meta["texp"] == 0.0:
#                 if texp == 0.0:
#                     raise ValueError("Could not set dark image in image "
#                             "list, missing input for texp")
#                 dark.meta["texp"] = texp
#
#         elif isinstance(dark, ndarray):
#             if texp == None:
#                 raise ValueError("Could not add dark image in image list, "
#                     "missing input for texp")
#             dark = Img(dark, texp=texp)
#
#         if (acq_time != datetime(1900,1,1) and
#             dark.meta["start_acq"] == datetime(1900,1,1)):
#             dark.meta["start_acq"] = acq_time
#         dark.meta["read_gain"] = read_gain
#
#         self.master_dark = dark
#
#
#     def add_master_offset_image(self, offset, acq_time=datetime(1900, 1, 1),
#                                 texp=0.0, read_gain=0):
#         """Add a (master) offset image to list
#
#         Sets a offset image, which is used for dark correction in case,
#         no dark / offset image lists are linked to this object or the data
#         extraction from these lists does not work for some reason.
#
#         :param (Img, ndarray) offset: offset image data
#         :param datetime acq_time: image acquisition time (only used if input
#             image is numpy array or if acqtime in Img object is default)
#         :param float texp: optional input for exposure time in units of
#             s (i.e. is used if img input is ndarray or if exposure time is
#             not set in the input img)
#
#         The image is stored at::
#
#             self.master_offset
#
#         """
#         if not any([isinstance(offset, x) for x in [Img, ndarray]]):
#             raise TypeError("Could not set offset image in image list, "
#                 "invalid input type")
#         elif isinstance(offset, Img):
#             if offset.meta["texp"] == 0.0:
#                 if texp == 0.0:
#                     raise ValueError("Could not set offset image in image "
#                             "list, missing input for texp")
#                 offset.meta["texp"] = texp
#
#         elif isinstance(offset, ndarray):
#             if texp == None:
#                 raise ValueError("Could not add offset image in image list, "
#                     "missing input for texp")
#             offset = Img(offset, texp=texp)
#
#         if (acq_time != datetime(1900,1,1)
#                 and offset.meta["start_acq"] == datetime(1900,1,1)):
#             offset.meta["start_acq"] = acq_time
#         offset.meta["read_gain"] = read_gain
#         self.master_offset = offset
#
# # this method was commented out on 9/1/2018
# # ===========================================================================
# #     def set_bg_img_from_polyfit(self, mask=None, **kwargs):
# #         """Sets background image from results of a poly surface fit
# #
# #         Parameters
# #         ----------
# #         mask : array
# #             mask specifying sky background pixels, if None (default) then
# #             this mask is determined automatically using
# #             :func:`prepare_bg_fit_mask`
# #         **kwargs:
# #             additional keyword arguments for :class:`PolySurfaceFit
# #         Returns
# #         -------
# #         Img
# #             fitted background image
# #         """
# #         if mask is None:
# #             mask = self.prepare_bg_fit_mask(dilation=True)
# #         fit = PolySurfaceFit(self.current_img(), mask, **kwargs)
# #         bg = fit.model
# #         try:
# #             low = self.get_dark_image().mean()
# #         except:
# #             low = finfo(float).eps
# #         print "LOW: %s" %low
# #         bg [bg <= low] = low
# #         self.bg_img = Img(bg)
# # ===========================================================================
#
#     def set_closest_dark_offset(self):
#         """Updates the index of the current dark and offset images
#
#         The index is updated in all existing dark and offset lists.
#         """
#         try:
#             num = self.index
#             for read_gain, info in self.dark_lists.iteritems():
#                 darknum = info["idx"][num]
#                 if darknum != info["list"].index:
#                     print ("Dark image index (read_gain %s) was changed in "
#                            "list %s from %s to %s"
#                            % (read_gain, self.list_id,
#                               info["list"].index, darknum))
#                     info["list"].goto_img(darknum)
#
#             if self.darkcorr_opt == 1:
#                 for read_gain, info in self.offset_lists.iteritems():
#                     offsnum = info["idx"][num]
#                     if offsnum != info["list"].index:
#                         print ("Offset image index (read_gain %s) was "
#                             "changed in list %s from %s to %s" %(read_gain,
#                                 self.list_id, info["list"].index, offsnum))
#                         info["list"].goto_img(offsnum)
#         except Exception:
#             print ("Failed to update index of dark and offset lists")
#             return False
#         return True
#
#     """LINKING OF OTHER IMAGE LIST OBJECTS"""
#     def link_imglist(self, other_list, list_id=None):
#         """Link another image list to this list
#
#         :param other_list: another image list object
#
#         """
#         if list_id is None:
#             list_id = other_list.list_id
#         self.current_img(), other_list.current_img()
#         if self.linked_lists.has_key(list_id):
#             raise AttributeError(
#                 "ImgList %s has already linked an ImgList "
#                 "with list_id %s. Please choose a different ID")
#         self.linked_lists[list_id] = other_list
#         self._linked_indices[list_id] = {}
#         idx_array = self.assign_indices_linked_list(other_list)
#         self._linked_indices[list_id] = idx_array
#         self.change_index_linked_lists()
#         self.load()
#
#     def disconnect_linked_imglist(self, list_id):
#         """Disconnect a linked list from this object
#
#         :param str list_id: string id of linked list
#         """
#         if not list_id in self.linked_lists.keys():
#             print ("Error: no linked list found with ID " + str(list_id))
#             return 0
#         del self.linked_lists[list_id]
#         del self._linked_indices[list_id]
#
#     def link_dark_offset_lists(self, *lists):
#         """Assign dark and offset image lists to this object
#
#         Assign dark and offset image lists: get "closest-in-time" indices of
#         dark list with respect to the capture times of the images in this
#         list. Then get "closest-in-time" indices of offset list with respect
#         to dark list. The latter is done to ensure, that dark and offset set
#         used for imagecorrection are recorded subsequently and not individual
#         from each other (i.e. only closest in time to the current image)
#         """
#         dark_assigned = False
#         offset_assigned = False
#         try:
#             texp = self.current_img().texp
#             if texp == 0 or isnan(texp):
#                 raise ValueError
#         except:
#             logger.warning("Exposure time could not be accessed in ImgList %s"
#                 %self.list_id)
#
#         warnings = []
#         # if input contains multiple lists for one of the two types (e.g. 2
#         # type "dark" lists), then try to assign dark list with the smallest
#         # difference in image exposure time. Here two helpers are initiated
#         # for logging the difference in exposure (this method is for instance
#         # relevant for the HD cam), requires flag: texp_access = True (see
#         # above)
#         dtexp_dark, dtexp_offset = 999999, 999999
#         for lst in lists:
#             if isinstance(lst, DarkImgList):
#                 if lst.list_type == "dark":
#                     try:
#                         dt = abs(texp - lst.current_img().texp)
#                         if isnan(dt):
#                             raise ValueError
#                         elif dt < dtexp_dark\
#                                 or lst.read_gain not in self.dark_lists:
#                             self.dark_lists[lst.read_gain] = od()
#                             self.dark_lists[lst.read_gain]["list"] = lst
#                             dtexp_dark = dt
#                             dark_assigned = True
#                     except:
#                         self.dark_lists[lst.read_gain] = od()
#                         self.dark_lists[lst.read_gain]["list"] = lst
#                         dark_assigned = True
#
#                 elif lst.list_type == "offset":
#                     try:
#                         dt = abs(texp - lst.current_img().texp)
#                         if dt < dtexp_offset or not\
#                                 self.offset_lists.has_key(lst.read_gain):
#                             self.offset_lists[lst.read_gain] = od()
#                             self.offset_lists[lst.read_gain]["list"] = lst
#                             dtexp_offset = dt
#                             offset_assigned = True
#                     except:
#                         self.offset_lists[lst.read_gain] = od()
#                         self.offset_lists[lst.read_gain]["list"] = lst
#                         offset_assigned = True
#
#                 else:
#
#                     warnings.append("List %s, type %s could not be linked "
#                         %(lst.list_id, lst.list_type))
#             else:
#                 warnings.append("Obj of type %s could not be linked, need "
#                                         " DarkImgList " %type(lst))
#
#         for gain, value in self.dark_lists.iteritems():
#             value["idx"] = self.assign_indices_linked_list(value["list"])
#         for gain, value in self.offset_lists.iteritems():
#             value["idx"] = self.assign_indices_linked_list(value["list"])
#         _print_list(warnings)
#         return dark_assigned, offset_assigned
#
#     """INDEX AND IMAGE LOAD MANAGEMENT"""
#     def change_index_linked_lists(self):
#         """Update current index in all linked lists based on ``cfn``"""
#         for key, lst in self.linked_lists.iteritems():
#             lst.goto_img(self._linked_indices[key][self.index],
#                          reload_here=True)
#
#     def load(self):
#         """Try load current and next image"""
#         self.change_index_linked_lists() #based on current index in this list
#         if not super(ImgList, self).load():
#             print ("Image load aborted...")
#             return False
#         if self.nof > 1:
#             next_img = self._load_image(self.next_index)
#             self.loaded_images["next"] = next_img
#             self._load_edit["next"].update(next_img.edit_log)
#             self._apply_edit("next")
#         else:
#             logger.warning("Image list contains only one image. Setting this image "
#                  "both in <this> and <next> attr.")
#             self.loaded_images["next"] = self.loaded_images["this"]
#             self._load_edit["next"].update(self._load_edit["this"])
#
#         if self.optflow_mode:
#             try:
#                 self.set_flow_images()
#                 self.optflow.calc_flow()
#             except IndexError:
#                 logger.warning("Reached last index in image list, optflow_mode will "
#                      "be deactivated")
#                 self.optflow_mode = 0
#         return True
#
#     def goto_next(self):
#         """Load next image in list"""
#         if self.nof < 2 or not self._auto_reload:
#             print ("Could not load next image, number of files in list: " +
#                 str(self.nof))
#             return False
#
#         self.iter_indices(to_index=self.next_index)
#
#         # load new images in all linked lists
#         self.change_index_linked_lists()
#
#         this_img = self.loaded_images["next"]
#         self.loaded_images["this"] = this_img
#         self._load_edit["this"].update(self._load_edit["next"])
#
#         if this_img.vign_mask is not None:
#             self.vign_mask = this_img.vign_mask
#
#         if self.update_cam_geodata:
#             self.meas_geometry.update_cam_specs(**this_img.meta)
#
#         next_img = self._load_image(self.next_index)
#         self.loaded_images["next"] = next_img
#         self._load_edit["next"].update(next_img.edit_log)
#         self._apply_edit("next")
#         if self.optflow_mode:
#             try:
#                 self.set_flow_images()
#                 self.optflow.calc_flow()
#             except IndexError:
#                 logger.warning("Reached last index in image list, optflow_mode will "
#                      "be deactivated")
#                 self.optflow_mode = 0
#         return True
#
#     """PROCESSING AND ANALYSIS METHODS"""
#     def optflow_histo_analysis(self, lines=[], start_idx=0, stop_idx=None,
#                                intensity_thresh=0, **optflow_settings):
#         """Performs optical flow histogram analysis for list images
#
#         The analysis is performed for all list images within the specified
#         index (or time) range and for an arbitraty number of PCS lines.
#
#         Parameters
#         ----------
#         lines : list
#             list containing :class:`LineOnImage` instances
#         start_idx : :obj:`int` or :obj:`datetime`
#             index or timestamp of first considered image. Note that the
#             timestamp option only works if acq. times can be accessed from
#             filenames for all files in the list (using method
#             :func:`timestamp_to_index`)
#         stop_idx : :obj:`int` or :obj:`datetime`, optional
#             index of last considered image (if None, the last image in this
#             list is used). Note that the timestamp option only works if acq.
#             times can be accessed from filenames for all files in the list
#             (using method :func:`timestamp_to_index`)
#         intensity_thresh : float
#             additional intensity threshold that may, e.g. be used to identify
#             plume pixels (e.g. if list is in ``tau_mode``).
#         **optflow_settings
#             additional keyword args passed to :class:`OptflowFarneback`
#
#         Returns
#         -------
#         list
#             list containing the computed time series of optical flow
#             histogram parameters (:class:`LocalPlumeProperties` instances)
#             for each of the provided input :class:`LineOnImage` objects.
#         """
#         cfn_tmp = self.cfn
#         if isinstance(start_idx, datetime):
#             start_idx = self.timestamp_to_index(start_idx)
#         if isinstance(stop_idx, datetime):
#             stop_idx = self.timestamp_to_index(stop_idx)
#         if stop_idx is None or stop_idx > self.nof:
#             stop_idx = self.nof
#
#         num = self._iter_num(start_idx, stop_idx)
#         flm = self.optflow_mode
#         self.goto_img(start_idx)
#         self.optflow.settings.update(**optflow_settings)
#         props = []
#         for line in lines:
#             if isinstance(line, LineOnImage):
#                 props.append(LocalPlumeProperties(line.line_id,
#                                                   color=line.color))
#
#         if len(props) == 0:
#             lines=[None]
#             props.append(LocalPlumeProperties("thresh_%.1f"
#                                               % intensity_thresh))
#
#         self.optflow_mode = True
#         for k in range(num):
#             plume_mask = self.get_thresh_mask(intensity_thresh)
#             for i in range(len(props)):
#                 props[i].get_and_append_from_farneback(self.optflow,
#                                                       line=lines[i],
#                                                       pix_mask=plume_mask)
#
#             self.goto_next()
#         self.goto_img(cfn_tmp)
#         self.optflow_mode = flm
#         return props
#
#     def get_thresh_mask(self, thresh=None, this_and_next=True):
#         """Get bool mask based on intensity threshold
#
#         Parameters
#         ----------
#         thresh : :obj:`float`, optional
#             intensity threshold
#         this_and_next : bool
#             if True, uses the current AND next image to determine mask
#
#         Returns
#         -------
#         array
#             mask specifying pixels that exceed the threshold
#         """
#         mask = self.this.duplicate().to_binary(thresh).img
#         if this_and_next and not self.cfn == self.nof - 1:
#             mask = logical_or(mask,
#                               self.loaded_images["next"].duplicate().to_binary(thresh).img)
#         return mask
#
#     def det_vign_mask_from_bg_img(self):
#         """Determine vignetting mask from current background image
#
#         The mask is determined using a blurred (:math:`\sigma = 3`)
#         background image which is normalised to one.
#
#         The mask is stored in ``self.vign_mask``
#
#         Returns
#         -------
#         Img
#             vignetting mask
#         """
#         if not self.has_bg_img():
#             raise AttributeError("Please set a background image first")
#         mask = self._bg_imgs[0].duplicate()
#         if mask.edit_log["blurring"] < 3:
#             mask.add_gaussian_blurring(3)
#         mask.img = mask.img / mask.img.max()
#         self.vign_mask = Img(mask)
#         return self.vign_mask
#
#     def calc_sky_background_mask(self, lower_thresh=None,
#                                 apply_movement_search=True,
#                                 **settings_movement_search):
#         """Retrieve and set background mask for 2D poly surface fit
#
#         Calculates mask specifying sky radiance pixels for background
#         modelling mode 0. The mask is updated in the background model
#         (class attribute :attr:`bg_model`).
#
#         Parameters
#         ----------
#         lower_thresh : :obj:`float`, optional
#             lower intensity threshold. If provided, this value is used,
#             else, the minimum value is derived from the minimum intensity
#             in the plume image within the current 3 sky reference
#             rectangles
#         **settings_movement_search
#             additional keyword arguments passed to :func:`find_movement`.
#             Note that these may include settings for the optical flow
#             calculation which are further passed to the
#             initiation of the :class:`FarnebackSettings` class
#
#         Returns
#         -------
#         array
#             2D-numpy boolean numpy array specifying sky background pixels
#         """
#         return self.bg_model.\
#             calc_sky_background_mask(self.this,
#                                      self.loaded_images["next"],
#                                      lower_thresh,
#                                      apply_movement_search,
#                                      **settings_movement_search)
#
# # ===========================================================================
# #     def prepare_bg_fit_mask(self, **kwargs):
# #         """Calculate mask specifying sky-reference pixels in current image
# #
# #         Note
# #         ----
# #
# #         1. The method was redefined and renamed, please see (and use)
# #             :func:`calc_sky_background_mask` instead
# #         2. This is a beta version
# #
# #         """
# #         logger.warning("Old name (wrapper) for method calc_sky_background_mask")
# #
# #         return self.calc_sky_background_mask(**kwargs)
# # ===========================================================================
#
#     def prep_data_dilutioncorr_old(self, tau_thresh=0.05,
#                                    plume_pix_mask=None,
#                                    plume_dists=None, ext_coeff=None):
#         """Get parameters relevant for dilution correction
#
#         Relevant parameters are:
#
#             1. Current plume background
#             #. Plume distance estimate (either global or on a pixel basis)
#             #. Plume pixel mask (only plume pixels are corrected)
#
#         Note
#         ----
#         This method changes the current image preparation state such that tau
#         mode is deactivated and vigncorr mode is activated.
#
#         Parameters
#         ----------
#         tau_thresh : float
#             tau threshold for retrieval of plume pixel mask. Is only used in
#             case next :param:`plume_mask` is unspecified or invalid. In this
#             case the plume mask is retrieved using :func:`get_thresh_mask`.
#         plume_pix_mask : :obj:`array`, :obj:`Img`, optional
#             mask specifying plume pixels. If valid, it will be passed through
#             and no threshold mask will retrieved (see :param:`tau_thresh`)
#         plume_dists : :obj:`array`, :obj:`Img`, :obj:`float`, optional
#             plume distance(s) in m. If input is numpy array or :class:`Img`
#             then, it must have the same shape as the current image
#         ext_coeff : :obj:`float`, optional
#             atmospheric extinction coefficient. If unspecified, try access
#             via :attr:`ext_coeff` which returns the current extinction
#             coefficient and raise :obj:`AttributeError` in case, no coeffs
#             are assigned to this list
#
#         Returns
#         -------
#         tuple
#             5-element tuple containing input for dilution correction
#
#             - :obj:`Img`, current vignetting corrected image
#             - :obj:`float`, current extinction coefficient
#             - :obj:`Img`, current plume background
#             - (:obj:`array`, :obj:`float`), plume distance(s)
#             - :obj:`array`, mask specifying plume pixels
#         """
#         # check input distance and if invalid try retrieve using measurement
#         # geometry
#         try:
#             try:
#                 plume_pix_mask = plume_pix_mask.img
#             except:
#                 pass
#
#             if plume_pix_mask.shape == self.this.shape:
#                 mask_ok = True
#             else:
#                 mask_ok = False
#         except:
#             mask_ok = False
#
#
#         dists = plume_dists
#
#         if dists is None:
#             try:
#                 (_,
#                  _,
#                  dists)=\
#                  self.meas_geometry.compute_all_integration_step_lengths(
#                          pyrlevel=self.pyrlevel)
#                 dists = dists.img
#             except:
#                 raise ValueError("Measurement geometry not ready for access "
#                     "of plume distances in image list %s. Please provide "
#                     "plume distance using input parameter plume_dist_m"
#                     %self.list_id)
#         # get current extinction coefficient, raises AttributeError if not
#         # available
#         try:
#             ext_coeff = float(ext_coeff)
#         except:
#             ext_coeff = self.ext_coeff
#         self.vigncorr_mode = False
#         self.tau_mode = True
#         tau0 = self.current_img().duplicate()
#         self.vigncorr_mode = True
#         #bg = self.bg_model.current_plume_background
#         #bg.edit_log["vigncorr"] = True
#         if not mask_ok:
#             #print "Retrieving plume pixel mask in list %s" %self.list_id
#             plume_pix_mask = self.get_thresh_mask(tau_thresh)
#         self.tau_mode = False
#         bg = self.current_img() * exp(tau0.img)
#         return (self.current_img(), ext_coeff, bg, dists, plume_pix_mask)
#
#     def correct_dilution(self, img, tau_thresh=0.10, ext_coeff=None,
#                          plume_pix_mask=None, plume_dists=None,
#                          vigncorr_mask=None, erosion_kernel_size=0,
#                          dilation_kernel_size=0):
#         """Correct a plume image for the signal dilution effect
#
#         The provided plume image needs to be in intensity space, meaning the
#         pixel values need to be intensities and not optical densities or
#         calibrated gas-CDs. The correction is based on Campion et al., 2015
#         and requires knowledge of the atmospheric scattering extinction
#         coefficients (``ext_coeff``) in the viewing direction of the camera.
#         These can be provided using the corresponding input parameter
#         ``ext_coeff`` or can be assigned to the list beforehand (up to you).
#         See example script no. 11 to check out how you can retrieve the
#         extinction coefficients using dark terrain features in the plume
#         image. The correction furthermore requires knowledge of the plume
#         distance (in the best case on the pixel-level) and it must be
#         possible to compute optical density images, hence the
#         :attr:`bg_model` (instance of :class:`PlumeBackgroundModel`) needs
#         to be ready for tau image computation. In addition, a vignetting
#         correction mask must be available.
#
#         Parameters
#         ----------
#         img : Img
#             the plume image object
#         tau_thresh : float
#             OD (tau) threshold to compute plume pixel mask (irrelevant if
#             next :param:`plume_pix_mask` is provided)
#         ext_coeff : :obj:`float`, optional
#             atmospheric extinction coefficient. If unspecified, try access
#             via :attr:`ext_coeff` which returns the current extinction
#             coefficient and raises :obj:`AttributeError` in case, no coeffs
#             are assigned to this list
#         vigncorr_mask : :obj:`ndarray` or :obj:`Img`, optional
#             mask used for vignetting correction
#         plume_pix_mask : :obj:`Img`, optional
#             binary mask specifying plume pixels in the image, is retrieved
#             automatically if input is None
#         erosion_kernel_size : int
#             if not zero, the morphological operation erosion is applied
#             to the plume pixel mask (e.g. to remove noise outliers) using
#             an appropriate quadratic kernel corresponding to the input size
#         dilation_kernel_size : int
#             if not zero, the morphological operation dilation is applied
#             to the plume pixel mask (e.g. to slightly extend the borders of
#             the detected plume) using an appropriate quadratic kernel
#             corresponding to the input size
#
#         Returns
#         -------
#         tuple
#             3-element tuple containing
#
#             - :obj:`Img`, dilution corrected image (vignetting corrected)
#             - :obj:`Img`, corresponding vignetting corrected plume background
#             - :obj:`array`, mask specifying plume pixels
#         """
#         if img.is_tau or img.is_aa or img.is_calibrated:
#             raise ValueError(
#                 "Img must not be an OD, AA or calibrated CD img")
#         try:
#             self.vign_mask = vigncorr_mask
#         except:
#             pass
#         vign_mask = self.vign_mask #raises Exception if not available
#         try:
#             try:
#                 plume_pix_mask = plume_pix_mask.img
#             except:
#                 pass
#             if plume_pix_mask.shape == self.this.shape:
#                 mask_ok = True
#             else:
#                 mask_ok = False
#         except:
#             mask_ok = False
#         if plume_dists is None:
#             plume_dists = self.plume_dists
#         # get current extinction coefficient, raises AttributeError if not
#         # available
#         try:
#             ext_coeff = float(ext_coeff)
#         except:
#             ext_coeff = self.ext_coeff
#         if img.is_vignetting_corrected:
#             idx=1
#         else:
#             idx=0
#         tau0 = self.bg_model.get_tau_image(img, self._bg_imgs[idx])
#         if not idx:
#             img.correct_vignetting(vign_mask, new_state=True)
#         #bg = self.bg_model.current_plume_background
#         #bg.edit_log["vigncorr"] = True
#         if not mask_ok:
#             #print "Retrieving plume pixel mask in list %s" %self.list_id
#             plume_pix_mask = tau0.to_binary(threshold=tau_thresh,
#                                             new_img=True)
#             if erosion_kernel_size > 0:
#                 plume_pix_mask.erode(ones((erosion_kernel_size,
#                                            erosion_kernel_size),dtype=uint8))
#             if dilation_kernel_size > 0:
#                 plume_pix_mask.dilate(ones((dilation_kernel_size,
#                                             dilation_kernel_size),dtype=uint8))
#         bg = img * exp(tau0.img)
#         from .dilutioncorr import correct_img
#         corr = correct_img(img, ext_coeff, bg, plume_dists, plume_pix_mask)
#
#         bad_pix = corr.img <= 0
#         corr.img[bad_pix] = img.img[bad_pix]
#
#         return (corr, bg, plume_pix_mask)
#
#     def correct_dilution_all(self, tau_thresh=0.05, ext_on=None,
#                              ext_off=None, add_off_list=True, save_dir=None,
#                              save_masks=False, save_bg_imgs=False,
#                              save_tau_prev=False, vmin_tau_prev=None,
#                              vmax_tau_prev=None, **kwargs):
#         """Correct all images for signal dilution
#
#         Correct and save all images in this list for the signal dilution
#         effect. See :func:`correct_dilution` and
#         :func:`prep_data_dilutioncorr` for details about requirements and
#         additional input options.
#
#         Note
#         ----
#         The vignetting and dilution corrected images are stored with all
#         additional image preparation settings applied (e.g. dark correction,
#         blurring)
#
#         Parameters
#         ----------
#         tau_thresh : :obj:`float`, optional
#             tau threshold applied to determine plume pixel mask (retrieved
#             using :attr:`tau_mode`, not :attr:`aa_mode`)
#         ext_on : :obj:`float`, optional
#             atmospheric extinction coefficient at on-band wavelength, if None
#             (default), try access via :attr:`ext_coeff`
#         ext_off : :obj:`float`, optional
#             atmospheric extinction coefficient at off-band wavelength. Only
#             relevant if input param ``add_off_list`` is True. If None
#             (default) and ``add_off_list=True`` try access via
#             :attr:`ext_coeff` in off band list.
#         add_off_list : bool
#             if True, also the images in a linked off-band image list
#             (using :func:`get_off_list`) are corrected as well. For the
#             correction of the off-band images, the current plume pixel mask
#             of this list is used.
#         save_dir : :obj:`str`, optional
#             base directory for saving the corrected images. If None
#             (default), then a new directory ``dilcorr`` is created at the
#             storage location of the first image in this list
#         save_masks : bool
#             if True,  a folder *plume_pix_masks* is created within
#             :param:`save_dir` in which all plume pixel masks are stored as
#             FITS
#         save_bg_imgs : bool
#             if True, a folder *bg_imgs* is created which is used to store
#             modelled plume background images for each image in this list.
#             This folder can be used on re-import of the data in order to save
#             background modelling time using background modelling mode 99.
#         save_tau_prev : bool
#             if True, png previews of dilution corrected tau images are saved
#         vmin_tau_prev : :obj:`float`, optional
#             lower tau value for tau image preview plots
#         vmax_tau_prev : :obj:`float`, optional
#             upper tau value for tau image preview plots
#         **kwargs
#             additional keyword args for dilution correction functions
#             :func:`correct_dilution` and :func:`prep_data_dilutioncorr`
#         """
#         ioff()
#         if self.calib_mode or self.aa_mode or self.tau_mode:
#             raise AttributeError("List must not be in tau, AA or calib mode")
#         self.darkcorr_mode=True
#         if save_dir is None or not exists(save_dir):
#             save_dir = abspath(join(dirname(self.files[0]), ".."))
#         save_dir = join(save_dir, "dilutioncorr")
#         if not exists(save_dir):
#             mkdir(save_dir)
#         if save_masks:
#             mask_dir = join(save_dir, "plume_pix_masks")
#             if not exists(mask_dir):
#                 mkdir(mask_dir)
#         if save_bg_imgs:
#             bg_dir = join(save_dir, "bg_imgs")
#             if not exists(bg_dir):
#                 mkdir(bg_dir)
#         if save_tau_prev:
#             tau_dir = join(save_dir, "tau_prev")
#             if not exists(tau_dir):
#                 mkdir(tau_dir)
#
#         self.goto_img(0)
#         saved_off = []
#         num = self._iter_num(0, self.nof)
#         if add_off_list:
#             off = self.get_off_list()
#             off.bg_model.update(**self.bg_model.settings_dict())
#         for k in range(num):
#             (corr,
#              bg,
#              plume_pix_mask) = self.correct_dilution(self.this,
#                                                      tau_thresh=tau_thresh,
#                                                      ext_coeff=ext_on,
#                                                      **kwargs)
#             corr.save_as_fits(save_dir)
#             fname = corr.meta["file_name"]
#             if save_masks:
#                 Img(plume_pix_mask.img, dtype=uint8,
#                     file_name=fname).save_as_fits(mask_dir)
#             if save_bg_imgs:
#                 bg.save_as_fits(bg_dir, fname)
#             if save_tau_prev:
#                 tau = corr.to_tau(bg)
#                 fig = self.bg_model.plot_tau_result(tau,
#                                                     tau_min=vmin_tau_prev,
#                                                     tau_max=vmax_tau_prev)
#                 name = fname.split(".")[0] + ".png"
#                 fig.savefig(join(tau_dir, name))
#                 close("all")
#                 del fig
#             if add_off_list:
#                 if not off.current_img().meta["file_name"] in saved_off:
#                     # use on band plume pixel mask
#                     (corr_off,
#                      bg_off,
#                      _) = off.correct_dilution(off.this, ext_coeff=ext_off,
#                                                plume_pix_mask=plume_pix_mask,
#                                                **kwargs)
#                     saved_off.append(corr_off.save_as_fits(save_dir))
#                     if save_bg_imgs:
#                         bg_off.save_as_fits(bg_dir,
#                                             corr_off.meta["file_name"])
#             self.goto_next()
#         ion()
#
#     """I/O"""
#     def import_ext_coeffs_csv(self, file_path, header_id=None, **kwargs):
#         """Import extinction coefficients from csv
#
#         The text file requires datetime information in the first column and
#         a header which can be used to identify the column. The import is
#         performed using :func:`pandas.DataFrame.from_csv`
#
#         Parameters
#         ----------
#         file_path : str
#             the csv data file
#         header_id : str
#             header string for column containing ext. coeffs
#         **kwargs :
#             additionald keyword args passed to
#             :func:`pandas.DataFrame.from_csv`
#
#         Returns
#         -------
#         Series
#             pandas Series containing extinction coeffs
#
#         Todo
#         ----
#
#         This is a Beta version, insert try / except block after testing
#
#         """
#         try:
#             df = DataFrame.from_csv(file_path, **kwargs)
#             s=df[header_id]
#         except:
#             s = Series.from_csv(file_path, **kwargs)
#         self.ext_coeffs = s#
#         return self.ext_coeffs
#
#     """HELPERS"""
#     def has_bg_img(self):
#         """Returns boolean whether or not background image is available"""
#         if not isinstance(self.bg_img, Img):
#             return False
#         return True
#
#     def update_index_dark_offset_lists(self):
#         """Check and update current dark image (if possible / applicable)"""
#         if self.darkcorr_opt == 0:
#             return
#         t_last = self.time_last_dark_check
#
#         ctime = self.current_time()
#
#         if not (t_last - timedelta(minutes=self.update_dark_ival)) < ctime <\
#                       (t_last + timedelta(minutes = self.update_dark_ival)):
#             if self.set_closest_dark_offset():
#                 print ("Updated dark / offset in img_list %s at %s"
#                         %(self.list_id, ctime))
#                 self.time_last_dark_check = ctime
#
#
#     """Private methods"""
#     def _apply_edit(self, key):
#         """Applies the current image edit settings to image
#
#         :param str key: image id (e.g. this)
#         """
#         if not self.edit_active:
#             logger.warning("Edit not active in img_list %s: no image preparation will "
#                 "be performed" %self.list_id)
#             return
#         if key == "this":
#             upd_bgmodel = True
#         else:
#             upd_bgmodel = False
#         img = self.loaded_images[key]
#         bg = None
#         if self.darkcorr_mode:
#             dark = self.get_dark_image(key).to_pyrlevel(img.pyrlevel)
#             img.subtract_dark_image(dark)
#         shift = self.camera.reg_shift_off
#         if self.list_id=="off" and not all([x==0 for x in shift]):
#             logger.info("BLAAAAA: Image shift")
#             img.apply_registration_shift(dx_abs=shift[0], dy_abs=shift[1])
#         bg_model = self.bg_model
#         if self.dilcorr_mode:
#             s = self.dilcorr_settings
#             # update bg_model in case dilution correction is active, the
#             # model stored in the settings class is set at mode 99,
#             # i.e. no modelling is performed
#             bg_model = s.bg_model
#             (img,
#              bg,
#              mask) = self.correct_dilution(img,
#                                            s.tau_thresh,
#                                            erosion_kernel_size=s.erosion_kernel_size,
#                                            dilation_kernel_size=s.dilation_kernel_size)
#         elif self.vigncorr_mode:
#             # elif because if dilcorr is active the image is already
#             # vign corrected
#             img.correct_vignetting(self.vign_mask, new_state=True)
#         if self.tau_mode:
#             if bg is None and self.bg_model.mode > 0:
#                 # dilution_corr is not active
#                 bg = self.bg_img.to_pyrlevel(img.pyrlevel)
#             img = bg_model.get_tau_image(plume_img=img,
#                                          bg_img=bg,
#                                          update_imgs=upd_bgmodel)
#         elif self.aa_mode:
#             off_list = self.get_off_list()
#             if off_list.dilcorr_mode:
#                 raise AttributeError("Linked off-band list has dilution "
#                                      "correction mode activated. Please "
#                                      "deactivate.")
#             elif off_list.this.is_tau:
#                 raise AttributeError("Linked off-band list is in tau mode. "
#                                      "Please deactivate...")
#             #off_list.dilcorr_mode = self.dilcorr_mode
#             if bg is None:
#                 bg = self.bg_img.to_pyrlevel(img.pyrlevel)
#             img_off = off_list.this
#             # make sure, the dilution correction mode is activated in the off
#             # list if it is activated here
#             if self.dilcorr_mode:
#                 mask = mask.to_pyrlevel(off_list.pyrlevel)
#                 (img_off,
#                  bg_off,
#                  _)=off_list.correct_dilution(img_off,
#                                               plume_pix_mask=mask)
#             else:
#                 bg_off = off_list.bg_img
#             img_off.to_pyrlevel(img.pyrlevel)
#             bg_off.to_pyrlevel(img.pyrlevel)
#
#             img = bg_model.get_aa_image(plume_on=img,
#                                         plume_off=img_off,
#                                         bg_on=bg,
#                                         bg_off=bg_off)
#             if self.sensitivity_corr_mode:
#                 img = img / self.senscorr_mask
#                 img.edit_log["senscorr"] = 1
#
#         if self.calib_mode:
#             img.img = self.calib_data(img.img)
#             img.edit_log["gascalib"] = True
#
#         img.to_pyrlevel(self.img_prep["pyrlevel"])
#         if self.img_prep["crop"]:
#             img.crop(self.roi_abs)
#         if self.img_prep["8bit"]:
#             img._to_8bit_int(new_im=False)
#         # do this at last, since it can be time consuming and is therefore
#         # much faster in case pyrlevel > 0 or crop applied
#         img.add_gaussian_blurring(self.img_prep["blurring"])
#         img.apply_median_filter(self.img_prep["median"])
#         self.loaded_images[key] = img
#
#     def _aa_test_img(self, off_list):
#         """Try to compute an AA test-image"""
#         on = self._load_image(self.index)
#         off = off_list._load_image(off_list.index)
#         if self.which_bg == "img":
#             # the stored images may be vignetting corrected, then also a
#             # vignetting corrected BG image is required. The attribute
#             # _bg_imgs is a list that contains two images: one that is not
#             # corrected for vignetting (index 0), and one that is corrected
#             # for vignetting (index 1). Thus, the right bg image can simply
#             # be accessed passing the img state variable "is_vigncorr"
#             bg_on = self._bg_imgs[on.is_vigncorr].to_pyrlevel(on.pyrlevel)
#             bg_off = off_list._bg_imgs[off.is_vigncorr].to_pyrlevel(
#                 off.pyrlevel)
#         else:
#             bg_on = self.bg_list.this.to_pyrlevel(on.pyrlevel)
#             bg_off = off_list.bg_list.this.to_pyrlevel(off.pyrlevel)
#         return self.bg_model.get_aa_image(on, off, bg_on, bg_off,
#                                           check_state=False)
# =============================================================================
