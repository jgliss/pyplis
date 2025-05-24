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
"""Pyplis module containing features related to cell calibration."""
from pyplis import logger, print_log
from matplotlib.pyplot import subplots

from numpy import (log, arange, linspace, isnan, diff, mean, argmin, asarray)
from matplotlib.pyplot import Figure
from matplotlib.cm import get_cmap
from datetime import timedelta
from os.path import exists
from collections import OrderedDict as od
from pyplis.dataset import Dataset
from pyplis.setupclasses import MeasSetup
from pyplis.processing import ImgStack, PixelMeanTimeSeries
from pyplis.imagelists import ImgList, CellImgList
from pyplis.exceptions import CellSearchError, ImgMetaError
from pyplis.image import Img
from pyplis.helpers import subimg_shape, isnum
from pyplis.optimisation import PolySurfaceFit
from pyplis.calib_base import CalibData
from pyplis.glob import SPECIES_ID


class CellSearchInfo(object):
    """Class for for storage cell search from automatic cell search engine.

    Parameters
    ----------
    filter_id : str
        string ID of filter / Image type
    add_id : str
        additional identifier (e.g. "bg", "cell1")
    y_max : float
        a reference intensity

    """

    def __init__(self, filter_id, add_id, y_max):
        self.filter_id = filter_id
        self.add_id = add_id
        # self.abbr = None
        self.y_max = y_max
        self.mean_vals = []
        self.mean_vals_err = []
        self.file_paths = []
        self.start_acq = []
        self.texps = []

        self.img_list = None

    @property
    def start(self):
        """Get time stamp of first detected image.

        Returns
        -------
        datetime
            First time stamp, i.e. ``start_acq[0]``

        Raises
        ------
        IndexError
            if ``start_acq`` is empty

        """
        return self.start_acq[0]

    @property
    def stop(self):
        """Get time stamp of last detected image.

        Returns
        -------
        datetime
            Last time stamp, i.e. ``self.start_acq[-1]``

        Raises
        ------
        IndexError
            if ``start_acq`` is empty

        """
        return self.start_acq[-1]

    @property
    def tot_num(self):
        """Get total number of detected images.

        Returns
        -------
        int
            length of ``self.mean_vals``

        """
        return len(self.mean_vals)

    def from_img_list(self, img_list):
        """Fill values using all images from a specific image list.

        Note
        ----

        Old beta version, not tested, currently not in use

        Parameters
        ----------
        img_list : ImgList
            image list from which pixel mean value time series is supposed
            to be determined

        """
        if not isinstance(img_list, ImgList):
            raise TypeError("Wrong input type")
        dat = img_list.get_mean_value()

        self.file_paths = img_list.files
        self.mean_vals = asarray(dat.values)
        self.mean_vals_err = asarray(dat.std)
        self.start_acq = asarray(dat.index)
        self.y_max = max(dat.values)

    @property
    def mean_err(self):
        """Return average std of mean value time series.

        Returns
        -------
        float
            Error of mean value

        """
        return mean(self.mean_vals_err)

    @property
    def mid_point_val(self):
        """Get mean value of middle image of the time series.

        Returns
        -------
        float
            Mean intensity of middle image

        """
        num = len(self.mean_vals)
        if num < 1:
            raise Exception("No data available in CellSearchInfo")
        elif num == 1:
            return self.mean_vals[0]
        elif num == 2:
            return mean(self.mean_vals)
        else:
            mid_index = int(num / 2)
            return self.mean_vals[mid_index]

    def point_ok(self, idx):
        """Check data point at given index.

        Checks if intensity value at given index is within acceptance
        intensity range with respect to the middle intensity value of
        the time series

        Parameters
        ----------
        idx : int
            index of datapoint

        Returns
        -------
        bool
            True, if ok, False if not

        """
        try:
            val = self.mean_vals[idx]
            if abs(self.mid_point_val - val) < self.mean_err:
                return True
            return False
        except IndexError as e:
            logger.warning(repr(e))
            return False
        except BaseException:
            raise

    def create_image_list(self, camera):
        """Create image list containing all valid cell images.

        Creates a :class:`CellImgList` which includes all detected data
        points that fulfill condition :func:`point_ok`.

        Note
        ----
        If successful, the list is assigned to :attr:`img_list`

        Parameters
        ----------
        camera : Camera
            the camera used

        Returns
        -------
        CellImgList
            image list containing all (valid) cell images

        """
        lst = CellImgList(list_id=self.filter_id, cell_id=self.add_id,
                          camera=camera)
        paths = self.file_paths
        for idx in range(len(self.mean_vals)):
            if self.point_ok(idx):
                lst.files.append(paths[idx])
        lst.init_filelist()
        self.img_list = lst
        if lst.nof < 1:
            raise CellSearchError("No suitable %s images found on creation of "
                                  "image list for cell %s"
                                  % (self.filter_id, self.add_id))
        logger.info("Succesfully created image list %s for cell with ID %s from "
                    "cell search results" % (self.filter_id, self.add_id))
        return lst

    @property
    def offs(self):
        """Get array containing offset values.

        The offset values are determined from :attr:`mean_vals`` with
        respect to :attr:`y_max`.

        Returns
        -------
        ndarray
            array containing offset values for each intensity in
            :attr:`mean_vals`

        """
        return self.y_max - self.mean_vals


class CellAutoSearchResults(object):
    """Helper class collecting results from auto-cell detection algorithm.

    This object is included in :class:`CellCalibEngine` object and will be
    filled with :class:`CellSearchInfo` objects if the cell autodetection
    is used (:func:`find_cells`)

    Note
    ----
    This class is normally not intended to be used directly

    Attributes
    ----------
    cell_info : OrderedDict
        Ordered dictionary containing dictionaries for all filter_ids for
        which cell search was performed (e.g. on, off). These dictionaries
        contain :class:`CellSearchInfo` ordered based on the acq. time of
        the detected cell dip
    bg_info : OrderedDict
        Ordered dictionary containing :class:`CellSearchInfo` objects
        that include detected background images for each filter (e.g. on
        off)
    rest_info : OrderedDict
        Ordered dictionary containing all images for each filter (e.g. on
        off) that could not be identified as Cell or BG image

    """

    def __init__(self):
        self.cell_info = od()
        self.bg_info = od()
        self.rest_info = od()

    def add_cell_search_result(self, filter_id, cell_info, bg_info,
                               rest_info):
        """Add a collection of :class:`CellSearchInfo` objects.

        Parameters
        ----------
        filter_id : str
            image type ID (e.g. on, off)
        cell_info : dictlike
            dictonary containing :class:`CellSearchInfo` objects containing
            information about images belonging to one cell
        bg_info : CellSearchInfo
            object containing information about detected background images
            for image type specified by ``filter_id``
        rest_info : CellSearchInfo
            object containing information about images that could not be
            assigned to a detected cell nor to the background images

        """
        self.cell_info[filter_id] = od()
        for cell_id, res in cell_info.items():
            if isinstance(res, CellSearchInfo):
                self.cell_info[filter_id][cell_id] = res
        if isinstance(bg_info, CellSearchInfo):
            self.bg_info[filter_id] = bg_info
        if isinstance(rest_info, CellSearchInfo):
            self.rest_info[filter_id] = rest_info


class CellCalibData(CalibData):
    """Class representing cell calibration data.

    This class inherits from the :class:`CalibData` base class.

    Parameters
    ----------
    tau_vec : ndarray
        tau data vector for calibration data
    cd_vec : ndarray
        DOAS-CD data vector for calibration data
    cd_vec_err : ndarray
        Fit errors of DOAS-CDs
    time_stamps : ndarray
        array with datetime objects containing time stamps
        (e.g. start acquisition) of calibration data
    calib_fun : function
        optimisation function used for fitting of calibration data
    calib_coeffs : list, optional
        optimisation parameters for calibration curve.
    senscorr_mask : :obj:`ndarray`or :obj:`Img`, optional
        sensitivity correction mask that was normalised relative to the
        pixel position where the calibration data was retrieved (i.e.
        position of DOAS FOV in case of DOAS calibration data, or image pixel
        position, where cell calibration data was retrieved)
    calib_id : str
        calibration ID (e.g. "aa", "tau_on", "tau_off")
    camera : Camera
        camera object (not necessarily required). A camera can be assigned
        in order to convert the FOV extend from pixel coordinates into
        decimal degrees
    pos_x_abs : int
        x-position of image pixel for which the data was retrieved
    pos_y_abs : int
        y-position of image pixel for which the data was retrieved

    """

    def __init__(
            self, 
            tau_vec=None, 
            cd_vec=None, 
            cd_vec_err=None, 
            time_stamps=None,
            calib_fun=None, 
            calib_coeffs=None, 
            senscorr_mask=None,
            polyorder=1, 
            calib_id=None, 
            camera=None,
            pos_x_abs=None, 
            pos_y_abs=None
            ):
        super().__init__(
            tau_vec=tau_vec,
            cd_vec=cd_vec,
            cd_vec_err=cd_vec_err,
            time_stamps=time_stamps,
            calib_fun=calib_fun,
            calib_coeffs=calib_coeffs,
            senscorr_mask=senscorr_mask,
            polyorder=polyorder,
            calib_id=calib_id,
            camera=camera
        )
        self.type = "cell"
        self.pos_x_abs = pos_x_abs
        self.pos_y_abs = pos_y_abs

    def _prep_fits_save(self):
        """Prepare FITS HDU list for storing calibration data.

        Note
        ----
        See baseclass :class:`CalibData` for more detailed information

        Returns
        -------
        HDUList
            hdu list containing sensitivity correction mask and table with
            calib data

        """
        hdu = super(CellCalibData, self)._prep_fits_save()
        hdu[0].header["pos_x_abs"] = self.pos_x_abs
        hdu[0].header["pos_y_abs"] = self.pos_y_abs
        return hdu

    def load_from_fits(self, file_path):
        """Load calibration data from FITS file.

        Note
        ----
        See methods of base class
        :func:`pyplis.calib_base.CalibData.save_as_fits` and
        :func:`pyplis.calib_base.CalibData.load_from_fits` for more details

        Parameter
        ---------
        file_path : str
            path of FITS file
        """
        hdu = super(CellCalibData, self).load_from_fits(file_path)
        self.pos_x_abs = hdu[0].header["pos_x_abs"]
        self.pos_y_abs = hdu[0].header["pos_y_abs"]
        hdu.close()


class CellCalibEngine(Dataset):
    """Class for performing automatic cell calibration.

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

    Is initialised as :class:`pyplis.Datasets.Dataset` object, i.e. normal
    setup is like plume data using a :class:`MeasSetup` object (make sure
    that ``cell_info_dict`` is set in the setup class).

    Parameters
    ----------
    setup : MeasSetup
        see :class:`Dataset` for details
    init : bool
        if True, the image lists are initiated and filled (if possible)

    """

    def __init__(self, setup=None, init=True):
        logger.info('\n')
        logger.info("INIT CALIB DATASET OBJECT")
        logger.info('\n')
        super(CellCalibEngine, self).__init__(setup, lst_type=CellImgList,
                                              init=init)

        self.type = "cell_calib"

        self.cell_search_performed = False
        self._cell_info_auto_search = od()

        if isinstance(self.setup, MeasSetup):
            self.set_cell_info_dict_autosearch(self.setup.cell_info_dict)

        self.cell_lists = od()
        self.bg_lists = od()

        self.search_results = CellAutoSearchResults()

        self.pix_mean_tseries = od()
        self.bg_tseries = od()

        self.tau_stacks = od()
        self.calib_data = od()
        logger.info('\n')
        logger.info("FILELISTS IN CALIB DATASET OBJECT INITIALISED")
        logger.info('\n')

    @property
    def cell_lists_ready(self):
        """Call :func:`check_all_lists``."""
        return self.check_all_lists()

    def set_cell_images(self, img_paths, cell_gas_cd, cell_id, filter_id,
                        img_import_method=None):
        """Add cell images corresponding to one cell and image type.

        Creates :class:`CellImgList` containing input cell images and
        adds them calling :func:`add_cell_img_list`

        Parameters
        ----------
        img_paths : list
            list containing image file paths (can also be a single image
            path)
        cell_gas_cd : float
            column amount of gas in cell
        cell_id : str
            string identification of cell
        filter_id : str
            filter ID for images (e.g. "on", "off")

        """
        try:
            # input is not a list but a valid path to (hopefully) an image
            if exists(img_paths):
                img_paths = [img_paths, ]
        except BaseException:
            pass

        paths = [p for p in img_paths if exists(p)]
        if not len(paths) > 0:
            raise TypeError("No valid filepaths could be identified")

        lst = CellImgList(files=paths, list_id=filter_id,
                          camera=self.camera,
                          cell_id=cell_id, gas_cd=cell_gas_cd)
        self.add_cell_img_list(lst)

    def set_bg_images(self, img_paths, filter_id):
        """Set background images for a certain filter type.

        Creates :class:`CellImgList` containing input background images and
        adds them calling :func:`add_bg_img_list`

        Parameters
        ----------
        img_paths : list
            list containing image file paths of bg images (can also be a
            single image file path)
        filter_id : str
            image type (e.g. "on", "off")

        """
        try:
            # input is not a list but a valid path to (hopefully) an image
            if exists(img_paths):
                img_paths = [img_paths, ]
        except BaseException:
            pass

        paths = [p for p in img_paths if exists(p)]
        if not len(paths) > 0:
            raise TypeError("No valid filepaths could be identified")

        lst = CellImgList(files=paths, list_id=filter_id,
                          camera=self.camera)
        self.add_bg_img_list(lst)

    def add_cell_img_list(self, lst):
        """Add a cell image list for calibration.

        Parameters
        ----------
        lst :  CellImgList
            if, valid, the list is added to :attr:`cell_lists` using it's
            ID (``lst.list_id``) as first key and ``lst.cell_id`` as
            second, e.g. ``self.cell_lists["on"]["a53"]``

        """
        if not isinstance(lst, CellImgList):
            raise TypeError("Error adding cell image list, need CellImgList "
                            "object, got %s" % type(lst))
        elif not lst.nof > 0:
            raise IOError("No files available in cell ImgList %s, %s"
                          % (lst.list_id, lst.cell_id))
        elif any([lst.gas_cd == x for x in [0, None]]) or isnan(lst.gas_cd):
            raise ValueError("Error adding cell image list, invalid value "
                             "encountered for attribute gas_cd: %s"
                             % lst.gas_cd)
        if lst.list_id not in self.cell_lists:
            self.cell_lists[lst.list_id] = od()
        self.cell_lists[lst.list_id][lst.cell_id] = lst

    def add_bg_img_list(self, lst):
        """Add an image list containing background images for calibration.

        Parameters
        ----------
        lst :  CellImgList
            if valid input, the list is added to dictionary
            ``self.bg_lists`` using ``lst.list_id`` as key

        """
        if not isinstance(lst, CellImgList):
            raise TypeError("Error adding bg image list, background image "
                            "list needs to be initiated as CellImgList in "
                            "CellCalibEngine, got %s" % type(lst))
        elif not lst.nof > 0:
            raise IOError("No files available in bg ImgList %s, %s"
                          % (lst.list_id, lst.cell_id))
        self.bg_lists[lst.list_id] = lst

    def det_bg_mean_pix_timeseries(self, filter_id):
        """Determine (or get) pixel mean values of background image list.

        Gets the average pixel intenisty (considering the whole image) for
        all images in specified background image list and stores it within
        a :class:`PixelMeanTimeSeries` object. The latter is then stored in
        :attr:`bg_tseries` and can be used to interpolate background
        intensities for cell image time stamps (this might be important for
        large SZA measurements where the background radiance changes
        fastly, cf. :func:`prepare_tau_calib`).

        Parameters
        ----------
        filter_id : str
            ID of background image list (must be valid key of dict
            :attr:`bg_lists`

        Returns
        -------
        PixelMeanTimeSeries
            time series object containing background mean intensities

        """
        ts = self.bg_lists[filter_id].get_mean_value()
        ts.fit_polynomial()
        ts.img_prep.update(self.bg_lists[filter_id].current_img().edit_log)
        self.bg_tseries[filter_id] = ts
        return ts

    def find_cells(self, filter_id="on", threshold=0.10,
                   accept_last_in_dip=False):
        """Autodetection of cell images and bg images using mean value series.

        This algorithm tries to separate individual cell images and
        background images by analysing the 1st derivative of the mean pixel
        intensity of each image in the time span specified in this object
        (``self.start``, ``self.stop``).

        The separation of the individual cell images is performed by
        identifying dips in the mean intensity evolution and assignment of
        all image files belonging to each dip.

        Parameters
        ----------
        filter_id : str
            filter ID (e.g. on, off, uses :func:`get_list` with
            ``filter_id`` as input)
        threshold : float
            threshold in percent by which intensity decreases are
            identified
        accept_last_in_dip : bool
            if True, also the last image in one of the Cell intensity dips
            is considered a valid cell image (by default, the first and the
            last images of a dip are not considered)

        """
        logger.info('\n')
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info("++++SEARCHING CELL TIME WINDOWS ", filter_id, " ++++++++++++++")
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info('\n')
        l = self.get_list(filter_id)
        ts = l.get_mean_value()
        ts.name = filter_id
        x, y, yerr, texps = ts.index, ts.values, ts.std, ts.texps
        # this will be overwritten in the loop to find the BG image with the
        # lowest standard deviation, which is then set as current bg image
        # yerrCurrentBG = 9999
        ydiff = diff(y)  # 1st derivative (finite differences)
        y_max = max(y)
        bg_info = CellSearchInfo(filter_id, "bg", y_max)
        rest = CellSearchInfo(filter_id, "rest", y_max)
        cell_info = od()  # will be filled with CellSearchInfo objects
        #                   in the loop
        cell_count = 0  # counter for number of cells detected
        on_cell = 0  # flag which is set when cell entry_cond is fulfilled
        for k in range(len(y) - 2):
            # Look for dip in intensity => candidate for calib cell time stamp

            # Define entry and leave acceptance condition for detection of time
            # window
            entry_cond = ((1 - abs(y[k + 1] / y_max)) > threshold and
                          abs(ydiff[k]) / y_max < threshold and
                          abs(ydiff[k - 1]) / y_max > threshold)

            leave_cond = ((1 - abs(y[k] / y_max)) > threshold and
                          abs(ydiff[k]) / y_max > threshold and
                          abs(ydiff[k - 1]) / y_max < threshold)

            # Condition for background image candidates
            bg_cond = ((1 - abs(y[k] / y_max)) < threshold and
                       abs(ydiff[k]) / y_max < threshold and
                       abs(ydiff[k - 1]) / y_max < threshold)

            if not accept_last_in_dip:
                # adapt entry and leave condition for cell time window"
                entry_cond = (entry_cond and
                              abs(ydiff[k + 1]) / y_max < threshold)

                leave_cond = ((1 - abs(y[k] / y_max)) > threshold and
                              abs(ydiff[k]) / y_max < threshold and
                              abs(ydiff[k - 1]) / y_max < threshold and
                              abs(ydiff[k + 1]) / y_max > threshold)

            if entry_cond:
                if on_cell:
                    ts.plot()
                    raise Exception("Fatal: found cell dip within cell dip"
                                    "plotted time series...")

                logger.info("Found cell at %s, %s" % (k, x[k]))
                on_cell = 1
                cell_count += 1
                cell_id = "Cell%s" % cell_count
                result = CellSearchInfo(filter_id, cell_id, y_max)

            # Look for increase in intensity => candidate for removal of calib
            # cell
            elif leave_cond and on_cell:
                # and onFilter:
                logger.info("Reached end of cell DIP at %s, %s" % (k, x[k]))
                # result.stop = x[k]
                on_cell = 0
                result.mean_vals.append(y[k])
                result.mean_vals_err.append(yerr[k])
                result.file_paths.append(l.files[k])
                result.start_acq.append(x[k])
                result.texps.append(texps[k])
                cell_info[result.add_id] = result
                # onFilter=0
            elif bg_cond:
                logger.info("Found BG candidate at %s, %s" % (k, x[k]))
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

        if not len(self._cell_info_auto_search) == len(cell_info):
            raise CellSearchError("Number of detected cells (%s) is "
                                  "different from number of cells "
                                  "specified in cellSpecInfo (%s) "
                                  % (len(self._cell_info_auto_search),
                                     len(cell_info)))

        # Create new image lists from search results for background images
        # and one list for each cell that was detected
        bg_info.create_image_list(self.camera)
        bg_info.img_list.update_cell_info("bg", 0.0, 0.0)
        self.assign_dark_offset_lists(into_list=bg_info.img_list)
        for cell_id, info in cell_info.items():
            info.create_image_list(self.camera)
            self.assign_dark_offset_lists(into_list=info.img_list)

        self.search_results.add_cell_search_result(filter_id, cell_info,
                                                   bg_info, rest)
        self.pix_mean_tseries["%s_auto_search" % filter_id] = ts

    def _assign_calib_specs(self, filter_id=None):
        """Assign the gas CD amounts to search results for all filter lists.

        This function assigns gas CD amounts (stored in
        ``self._cell_info_auto_search`` to the individual cells detected
        using the automatic cell search routine (:func:`find_cells`)
        and which are stored in ``self.search_results``. The latter object
        is of type :class:`CellAutoSearchResults` and contains image lists
        for all *filter_ids* for which the search was performed (e.g. "on",
        "off"). These are seperated by detected background images
        (``self.search_results.bg_info``) and individual cells
        (``self.search_results.cell_info``).

        The gas amounts (in ``self._cell_info_auto_search``) are assigned
        based on the magnitude of the corrseponding intensity decrease of
        the cell (in the pixel mean time series).

        Parameters
        ----------
        filter_id : str
            ID of filter which is supposed used for assignment (e.g. "on").
            Uses default on-band filter if input is unspecified (None) or
            if the imagelist for this filter key does not exist

        Note
        ----

            1. In order for this to work, the automatic cell search must
            have been performed
            2. This function does not change class attributes which are
            actually used for calibration. The latter are stored in
            ``self.cell_lists`` and ``self.bg_lists`` and have to be
            assigned specifically

        """
        # check input list ID and set default if invalid
        if filter_id not in self.lists_access_info:
            filter_id = self.filters.default_key_on
        # the info about columns in the cells
        cell_info = self._cell_info_auto_search
        # init temporary dicts (will be filled below)
        offs_dict = {}
        cell_cd_dict = {}
        # the results of the cell search
        res = self.search_results.cell_info
        for val in res[filter_id].values():
            offs_dict[val.add_id] = val.offs.mean()

        # read the gas column amounts
        for key, val in cell_info.items():
            cell_cd_dict[key] = val[0]
        # sort the dicionaries by column amount or intensity decrease
        s0 = sorted(offs_dict, key=offs_dict.get)
        s1 = sorted(cell_cd_dict, key=cell_cd_dict.get)
        logger.info("Cell search keys sorted by depth of Dip: %s" % s0)
        logger.info("Cell amounts sorted by depth of Dip: %s" % s1)
        filter_ids = res.keys()
        for k in range(len(s0)):
            cell_id = s1[k]
            gas_cd, gas_cd_err = cell_info[s1[k]][0], cell_info[s1[k]][1]
            logger.info("Search key: %s\nDel I: %s\nCell abbr: %s\nGasCol %s +/- %s"
                        % (s0[k], offs_dict[s0[k]], cell_id, gas_cd, gas_cd_err))
            # now add gas column to corresponding list in search result object
            for filter_id in filter_ids:
                res[filter_id][s0[k]].img_list.update_cell_info(cell_id,
                                                                gas_cd,
                                                                gas_cd_err)

    def add_search_results(self):
        """Add results from automatic cell detection to calibration.

        This method analyses ``self.search_results`` for valid cell image
        lists (i.e. lists that contain images and have the gas column
        assigned)

        """
        # Add all cell image lists that were found for each filter
        for filter_id, cell_dict in self.search_results.cell_info.items():
            for cell_id, cell in cell_dict.items():
                lst = cell.img_list
                if lst.has_files() and lst.gas_cd > 0:
                    self.add_cell_img_list(lst)
        # Add all BG image lists found for each filter
        for filter_id, info in self.search_results.bg_info.items():
            self.add_bg_img_list(info.img_list)
        # set background images closest to first detected cell list for each
        # filter
        self.set_bg_closest()

    def set_bg_closest(self, cell_id=None):
        """Set the current background image closest to one of the cells.

        Parameters
        ----------
        cell_id : str
            cell ID supposed to be used, if None, then the first cell list
            in :attr:`cell_lists` is used

        """
        if cell_id is None:
            cell_id = list(
                self.cell_lists[self.filters.default_key_on].keys())[0]

        for filter_id, lst in self.bg_lists.items():
            self.cell_lists[filter_id][cell_id].link_imglist(lst)

    def find_and_assign_cells_all_filter_lists(self, threshold=0.10):
        """High level function for automatic cell and BG image search.

        This method basically calls the following functions:

            1. :func:`find_cells` (for all filter IDs, e.g. on/off)
            #. :func:`_assign_calib_specs`
            #. :func:`add_search_results`
            #. :func:`check_all_lists`

        and sets flag ``cell_search_performed=True``.

        Parameters
        ----------
        threshold : float
            percentage threshold for identification of regions of
            decreased intensity in time series

        """
        for filter_id in self.filters.filters.keys():
            try:
                self.find_cells(filter_id, threshold, False)
            except BaseException:
                self.find_cells(filter_id, threshold, True)

        self._assign_calib_specs()
        self.add_search_results()
        self.check_all_lists()
        self.cell_search_performed = True

    def bg_img_available(self, filter_id):
        """Check if a background image is available.

        Parameters
        ----------
        filter_id : str
            filter ID of image list (e.g. on / off)

        """
        try:
            if isinstance(self.bg_lists[filter_id], Img):
                return True
            raise Exception
        except BaseException:
            self.check_all_lists(filter_id)
            if isinstance(self.bg_lists[filter_id], Img):
                return True
            return False

    def check_image_list(self, lst):
        """Check if image list contains files and has images ready (loaded).

        Parameters
        ----------
        lst : ImgList
            image list object

        Raises
        ------
        IndexError
            If list does not contain images
        Exception
            If images cannot be loaded in list (unexpected error) or if
            ``lst.gas_cd`` is not a float

        """
        if not lst.nof > 0:
            raise IndexError("Error, image list %s does not contain images"
                             % lst.list_id)
        if not isinstance(lst.current_img(), Img):
            if not lst.load():
                raise Exception("Unexpected error...")
        # raises Exception is gas column is not a number
        float(lst.gas_cd)

    def check_all_lists(self):
        """Check if all image lists are ready for analysis.

        Returns
        -------
        bool
            True (if it makes it to the return statement)

        """
        filter_ids = list(self.cell_lists.keys())
        cell_ids = self.cell_lists[filter_ids[0]].keys()
        # get number of cells for first filter ID
        first_cell_num = len(self.cell_lists[filter_ids[0]])

        for filter_id in filter_ids:
            if not len(self.cell_lists[filter_id]) == first_cell_num:
                raise Exception("Mismatch in number of cells in "
                                "self.cell_lists between filter list %s and %s"
                                % (filter_ids[0], filter_id))
            for cell_id in cell_ids:
                self.check_image_list(self.cell_lists[filter_id][cell_id])
            if filter_id not in self.bg_lists:
                raise KeyError("Error: BG image data (list) for filter ID %s "
                               "is not available" % filter_id)
            else:
                self.check_image_list(self.bg_lists[filter_id])

        return True

    def check_cell_info_dict_autosearch(self, cell_info_dict):
        """Check if dict including cell gas column info is right format.

        Parameters
        ----------
        cell_info_dict : dict
            keys: cell ids (e.g. "a57"),
            values: list of gas column density and uncertainty in cm-2,
            format: ``[value, error]``

        Raises
        ------
        Exception
            If any of the specs in ``cell_info_dict`` is invalid

        """
        for key, val in cell_info_dict.items():
            if (not isinstance(key, str) and not isinstance(key, str)):
                raise KeyError("Invalid key: %s" % key)
            if not isinstance(val, list):
                raise ValueError("Invalid cell column specification, need "
                                 "list containing [value, uncertainty] of gas "
                                 "column with id %s, got %s" % (key, val))
            else:
                if len(val) != 2:
                    raise ValueError("Invalid cell column specification, need "
                                     "list containing [value, uncertainty] of "
                                     "gas column with "
                                     "id %s, got %s" % (key, val))
                for k in range(len(val)):
                    if not isinstance(val[k], (int, float, complex)):
                        raise ValueError("Invalid data type for cell gas "
                                         "column specification %s" % val)

    def set_cell_info_dict_autosearch(self, cell_info_dict):
        """Set attribute ``self._cell_info_auto_search`` (dictionary).

        Parameters
        ----------
        cell_info_dict : dict
            dictionary containing cell information

        """
        self.check_cell_info_dict_autosearch(cell_info_dict)
        self._cell_info_auto_search = cell_info_dict

    def _prep_tau_stack(self, filter_id="on", darkcorr=True, blurring=2):
        """Prepare a stack containing cell tau images of a certain type.

        The number of images in the stack corresponds to the number of cells
        that are available in :attr:`cell_lists`. The images in the stack are
        optical density (OD) images for each of the cells, determined based on
        the corresponding sky background that is derived from the background
        image list.

        Parameters
        ----------
        filter_id : str
            filter ID for which the stack is computed (must be a valid list
            ID in :attr:`cell_lists` as well as :attr:`bg_list`)
        darkcorr : bool
            if True, the images will be dark corrected before the OD image

        """
        if filter_id not in self.bg_lists:
            raise AttributeError("No background images for filter ID %s "
                                 "available in cell calibration engine"
                                 % filter_id)
        elif filter_id not in self.cell_lists:
            raise AttributeError("No cell images for filter ID %s "
                                 "available in cell calibration engine"
                                 % filter_id)

        bg_list = self.bg_lists[filter_id]
        bg_list.update_img_prep(blurring=blurring)

        bg_list.darkcorr_mode = darkcorr

        bg_img = bg_list.current_img()
        bg_mean = bg_img.img.mean()

        h, w = subimg_shape(bg_list.current_img().img.shape)

        num = len(self.cell_lists[filter_id])

        tau_stack = ImgStack(h, w, num, stack_id=filter_id)
        # TEMPORARY SOLUTION
        tau_stack.add_data_err = []
        try:
            bg_mean_tseries = self.det_bg_mean_pix_timeseries(filter_id)
        except BaseException:
            pass
        for cell_id, lst in self.cell_lists[filter_id].items():
            lst.update_img_prep(blurring=blurring)
            lst.darkcorr_mode = darkcorr
            cell_img = lst.current_img()
            try:
                bg_mean_now = \
                    bg_mean_tseries.get_poly_vals(cell_img.meta["start_acq"])
                offset = bg_mean - bg_mean_now
            except BaseException:
                print_log.warning("Warning in tau image stack calculation for filter "
                                  " %s: Time series data for background list (background "
                                  "poly) is not available. Calculating tau image for cell "
                                  " image  %s, %s based on unchanged background image "
                                  " recorded at %s"
                                  % (filter_id, cell_id, cell_img.meta["start_acq"],
                                     bg_img.meta["start_acq"]))

                offset = 0.0

            bg_img = bg_img - offset
            tau_img = cell_img.duplicate()
            if bg_img.edit_log["darkcorr"] != cell_img.edit_log["darkcorr"]:
                raise ImgMetaError("Fatal: cannot determine tau stack, bg "
                                   "image and cell image have different "
                                   "darkcorr modes")
            tau_img.img = log(bg_img.img / cell_img.img)

            # tau_img.to_pyrlevel(pyrlevel)
            tau_stack.add_img(tau_img.img,
                              start_acq=cell_img.meta["start_acq"],
                              texp=cell_img.meta["texp"],
                              add_data=lst.gas_cd)
            tau_stack.add_data_err.append(lst.gas_cd_err)

        tau_stack.img_prep.update(tau_img.edit_log)
        self.tau_stacks[filter_id] = tau_stack
        return tau_stack

    def prep_tau_stacks(self, on_id="on", off_id="off", darkcorr=True,
                        blurring=2):
        """Prepare image stacks for on, off and AA calibration data.

        Parameters
        ----------
        on_id : str
            ID of onband filter
        off_id : str
            ID of offband filter
        darkcorr : bool
            Use dark corrected images
        blurring : int
            sigma of Gaussian blurring kernel
        pyrlevel : int
            pyramid level of calibration stack

        """
        on_stack = self._prep_tau_stack(on_id, darkcorr, blurring)
        off_stack = self._prep_tau_stack(off_id, darkcorr, blurring)
        self.tau_stacks["aa"] = on_stack - off_stack

    def prepare_calib_data(self, pos_x_abs=None, pos_y_abs=None,
                           radius_abs=1, on_id="on", off_id="off",
                           darkcorr=True, blurring=1, **kwargs):
        """Prepare calib data for onband, offband and AA.

        This function creates 3 :class:`CellCalibData` objects for each OD
        type (on, off and from that, AA). If not differently specified using
        the input parameters ``pos_x_abs`` and ``pos_y_abs`` the corresponding
        cell optical densities are retrieved at the image center coordinate.

        The 3 :class:`CellCalibData` instances for each type (on, off, AA) can
        be accessed via the :attr:`calib_data` attribute of this class.

        Parameters
        ----------
        pos_x_abs : :obj:`int`, optional
            x-position for which the calibration data is retrieved
        pos_y_abs : :obj:`int`, optional
            y-position for which the calibration data is retrieved
        radius_abs : int
            radius specifying the disk size around ``pos_x_abs`` and
            ``pos_y_abs`` used to retrieve the cell-ODs
        on_id : str
            ID of onband filter used to determine calib curve
        off_id : str
            ID of offband filter used for calibration
        darkcorr : bool
            perform dark correction before determining cell tau images
        blurring : int
            apply gaussian blurring to cell tau images
        pyrlevel : int
            downscale factor (Gauss pyramid)

        """
        self.check_all_lists()
        self.prep_tau_stacks(on_id, off_id, darkcorr, blurring)

        for calib_id, stack in self.tau_stacks.items():
            if any([x is None for x in [pos_x_abs, pos_y_abs]]):
                logger.warning("Using image center coordinates for retrieval of cell "
                               "calibration polynomial")
                h, w = stack.shape[1:]
                pos_x_abs, pos_y_abs = int(w / 2.0), int(h / 2.0)
            tau_series = stack.get_time_series(pos_x=pos_x_abs,
                                               pos_y=pos_y_abs,
                                               radius=radius_abs)[0]

            c = CellCalibData(tau_vec=tau_series.values, cd_vec=stack.add_data,
                              cd_vec_err=stack.add_data_err,
                              time_stamps=tau_series.index,
                              pos_x_abs=pos_x_abs, pos_y_abs=pos_y_abs,
                              calib_id=calib_id)
            try:
                c.fit_calib_data()
            except:
                print_log.warning("Failed to fit calibration data for calib_id %s"
                                  % calib_id)
            self.calib_data[calib_id] = c

    def get_sensitivity_corr_mask(self, calib_id="aa", pos_x_abs=None,
                                  pos_y_abs=None, radius_abs=1,
                                  cell_cd_closest=0,
                                  surface_fit_pyrlevel=2):
        """Get sensitivity correction mask.

        Prepares a sensitivity correction mask to correct for filter
        transmission shifts. These shifts result in increasing optical
        densities towards the image edges for a given gas column density.

        The mask is determined for original image resolution, i.e. pyramid
        level 0 and for a specific cell optical density image
        (aa, tau_on, tau_off). The latter is normalised with respect
        to the input pixel position (e.g. center position of DOAS FOV or
        pixel position where cell calibration data was retrieved).

        Plume AA (or tau_on, tau_off) images can then be corrected for
        sensitivity variations by division with the mask. If DOAS
        calibration is used, the calibration function can then be used
        for all image pixels. If only cell calibration is used, the mask is
        normalised with respect to the image center, the corresponding cell
        calibration polynomial should then be retrieved in the center
        coordinate which is the default calibration position when using
        creating calibration data if not explicitely specified. You may then
        calibrate a given aa image (``aa_img``) as follows with using a
        :class:`CellCalibData` object (denoted with ``cellcalib``)::

            mask = cellcalib.get_sensitivity_corr_mask()
            aa_corr = aa_img.duplicate()
            aa_corr.img = aa_img.img / mask
            #this is retrieved in the image center if not other specified
            gas_cd_img = cellcalib(aa_corr)
            gas_cd_img.show()

        Parameters
        ----------
        calib_id : str
            the mask is determined from the corresponding calib data
            (e.g. "on", "off", "aa")
        pos_x_abs : int
            x-pixel position of normalisation mask, if None the image center
            position is used (which is also the default pixel used to retrieve
            the vector of calibration optical densities from the cell OD
            images)
        pos_y_abs : int
            y-pixel position of normalisation mask, if None the image center
            position is used (which is also the default pixel used to retrieve
            the vector of calibration optical densities from the cell OD
            images)
        radius_abs : int
            radius specifying the disk size around ``pos_x_abs`` and
            ``pos_y_abs`` used to normalise the mask (i.e. uses average OD of
            cell image in this OD)
        filter_id : str
            mask is determined from the corresponding calib data (e.g.
            "on", "off", "aa")
        cell_cd_closest : float
            use the cell which is closest to the provided column density
        surface_fit_pyrlevel : int
            additional downscaling factor for 2D polynomial surface fit

        Raises
        ------
        ValueError
            if the corresponding :class:`ImgStack` is cropped, from which the
            cell OD image is supposed to be retrieved
        Returns
        -------
        Img
            the sensitivity correction mask

        Note
        ----

        This function was only tested for AA images and not for on / off
        cell tau images


        """
        if calib_id not in self.tau_stacks.keys():
            raise ValueError("Tau image is not available for calib ID %s"
                             % calib_id)

        stack = self.tau_stacks[calib_id]
        # make sure the stack is at pyramid level 0
        stack = stack.to_pyrlevel(0)
        try:
            if stack.img_prep["crop"]:
                raise ValueError("Stack is cropped: sensitivity mask can only"
                                 "be determined for uncropped images")
        except:
            pass
        idx = argmin(abs(stack.add_data - cell_cd_closest))
        cell_img, cell_cd_closest = stack.stack[idx], stack.add_data[idx]
        try:
            pos_x_abs = int(pos_x_abs)
            pos_y_abs = int(pos_y_abs)
            if not isnum(pos_x_abs) * isnum(pos_y_abs) == 1:
                raise ValueError
        except:
            print_log.warning("Using image center coordinate for normalisation position "
                              "of sensitivity correction mask")

            h, w = stack.shape[1:]
            pos_x_abs, pos_y_abs = int(w / 2.0), int(h / 2.0)

        fov_mask = stack.make_circular_access_mask(pos_x_abs,
                                                   pos_y_abs,
                                                   radius_abs)
        try:
            cell_img = PolySurfaceFit(cell_img,
                                      pyrlevel=surface_fit_pyrlevel).model
        except:
            print_log.warning("2D polyfit failed while determination of sensitivity "
                              "correction mask, using original cell tau image for mask "
                              "determination")
        mean = (cell_img * fov_mask).sum() / fov_mask.sum()
        mask = Img(cell_img / mean)
        if calib_id in self.calib_data:
            c = self.calib_data[calib_id]
            if c.pos_x_abs == pos_x_abs and c.pos_y_abs == pos_y_abs:
                logger.info("Assigning sensitivity correction for calibration ID "
                            "%s to corresponding CellCalibData object" % calib_id)
                c.senscorr_mask = mask
        return mask

    """
    Redefinitions from base class (:class:`Dataset`)
    """

    def get_list(self, list_id, cell_id=None):
        """Expand functionality of this method from :class:`Dataset`.

        Parameters
        ----------
        list_id : str
            filter ID of list (e.g. on, off). If parameter
            ``cell_id`` is None, then this function returns the initial
            Dataset list (containing all images, not the ones separated by
            cells / background).
        cell_id : str
            if input is specified (type str) and valid (available
            cell img list), then the corresponding list is returned which
            only contains images from this cell. The string "bg" might be
            used to access the background image list of the filter
            specified with parameter ``list_id``

        Returns
        -------
        ImgList
            the actual list object

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

    def plot_cell_search_result(self, filter_id="on", for_app=False,
                                include_tit=True, cell_cmap="Oranges",
                                ax=None):
        """High level plotting function for results from auto-cell search.

        Parameters
        ----------
        filter_id : str
            filter ID (e.g. "on", "off")
        for_app : bool
            currently irrelevant (default is False)
        include_tit : bool
            if True, include default title
        cell_cmap : str
            string specifying matplotlib colormap used to plot cell time
            windows
        ax :
            matplotlib axes object

        Returns
        -------
        axes
            matplotlib axes object

        """
        try:
            cmap = get_cmap(cell_cmap)
        except BaseException:
            logger.warning("Invalid input for cell_cmap, using Oranges")
            cmap = get_cmap("Oranges")
        # get stored time series (was automatically saved in
        # :func:`find_cells`)
        ts_all = self.pix_mean_tseries[("%s_auto_search" % filter_id)]
        # get cell search results
        res = self.search_results
        if filter_id not in res.cell_info or \
                len(res.cell_info[filter_id]) < 1:
            logger.info("Error plotting cell search results: no results found...")
            return 0
        if for_app:
            fig = Figure(figsize=(14, 8))
            ax = fig.add_subplot(111)
        else:
            if ax is None:
                fig, ax = subplots(1, 1, figsize=(14, 8))

        info = res.cell_info[filter_id]
        num = len(info)
        nums = [int(255.0 / k) for k in range(1, num + 3)]
        ts_all.plot(include_tit=include_tit, ax=ax, ls="--",
                    c=cmap(nums[0]),
                    label="Avg. pix intensities (%s)" % filter_id)

        ts = ts_all.index
        dt = timedelta(
            0, (ts[-1] - ts[0]).total_seconds() / (len(ts_all) * 10))

        k = 2
        for cell in info.values():
            lbl = (r"Cell %s: $S_{%s}$=%.2e cm$^{-2}$"
                   % (cell.img_list.cell_id, SPECIES_ID,
                      cell.img_list.gas_cd))
            p = ax.plot(cell.start_acq, cell.mean_vals, ' o',
                        color=cmap(nums[k]),
                        ms=8, label=lbl, markeredgecolor="None",
                        markeredgewidth=1)
            c = p[0].get_color()
            ax.fill_betweenx(arange(0, ts_all.max() * 1.05, 1),
                             cell.start - dt,
                             cell.stop + dt, facecolor=c, alpha=0.15,
                             edgecolor=c)
            k += 1
        if filter_id in res.bg_info.keys():
            bg_info = res.bg_info[filter_id]
            c = cmap(nums[1])
            ax.plot(bg_info.start_acq, bg_info.mean_vals, ' o', color=c,
                    ms=10, markerfacecolor="None", markeredgecolor=c,
                    mew=2, label='BG image candidates')
            ts = PixelMeanTimeSeries(bg_info.mean_vals, bg_info.start_acq)
            ts.fit_polynomial(2)
            bg_poly_vals = ts.get_poly_vals(bg_info.start_acq,
                                            ext_border_secs=30)

            ax.plot(bg_info.start_acq, bg_poly_vals, '-', color=c, lw=2,
                    ls="--", label='Fitted BG polynomial')

            cfn = bg_info.img_list.cfn
            ax.plot(bg_info.start_acq[cfn], bg_info.mean_vals[cfn],
                    marker="+", color=cmap(nums[0]), ms=12, mew=2,
                    label='Current BG image')

        ax.legend(loc="best", fancybox=True, framealpha=0.5)
        ax.set_ylabel(r"$\mu_{pix}$")
        return ax

    def plot_calib_curve(self, calib_id, **kwargs):
        """Plot calibration curve.

        Parameters
        ----------
        filter_id : str
            image type ID (e.g. "aa")
        **kwargs :
            additional keyword arguments for plot passed to :func:`plot` of
            corresponding :class:`CellCalibData` object

        Returns
        -------
        axes
            matplotlib axes object

        """
        return self.calib_data[calib_id].plot(**kwargs)

    def plot_all_calib_curves(self, ax=None, **kwargs):
        """Plot all available calibration curves in a certain pixel region.

        Parameters
        ----------
        ax : axes
            matplotlib axes instance

        **kwargs :
            additional keyword arguments passed to
            :func:`get_calibration_polynomial` of corresponding
            :class:`CellCalibData` objects

        Returns
        -------
        axes
            matplotlib axes object

        """
        if ax is None:
            fig, ax = subplots(1, 1)
        tau_max = -10
        y_min = 1e20
        for calib_id, calib in self.calib_data.items():
            tau = calib.tau_vec
            gas_cd, gas_cd_errs = calib.cd_vec, calib.cd_vec_err
            fun, coeffs = calib.calib_fun, calib.calib_coeffs
            if coeffs is None:
                raise ValueError("Calibration coefficients not available "
                                 "for calib_id %s. Please call fit_calib_data "
                                 "first" % calib_id)
            taus = linspace(0, tau.max() * 1.2, 100)
            # plot data points
            pl = ax.plot(tau, gas_cd, " ^",
                         label="Data %s (pix" % calib_id)
            # try adding error bars
            ax.errorbar(tau, gas_cd, gas_cd_errs, linestyle="none",
                        color="#6E6E6E")

            ax.plot(taus, fun(taus, *coeffs), "--",
                    color=pl[0].get_color(),
                    label="Fit result")

            tm = tau.max()
            if tm > tau_max:
                tau_max = tm
            offs = fun(0, *coeffs)
            if offs < y_min:
                y_min = offs

        ax.set_ylabel(r"$S_{%s}$ [cm$^{-2}$]" % SPECIES_ID)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylim([y_min - gas_cd.min() * 0.1, gas_cd.max() * 1.05])
        ax.set_xlim([0, tau_max * 1.05])
        ax.grid()
        ax.legend(loc="best", fancybox=True, framealpha=0.5)
        return ax

    def __call__(self, value, calib_id="aa", **kwargs):
        """Apply calibration to input value (i.e. convert into gas CD).

        Parameters
        ----------
        value : float
            tau or AA value
        calib_id : str
            ID of calibration data supposed to be used
        **kwargs :
            additional keyword arguments to extract calibration information
            (e.g. pos_x_abs, pos_y_abs, radius_abs)

        Returns
        -------
        float
            corresponding column density

        """
        return self.calib_data[calib_id](value, **kwargs)
