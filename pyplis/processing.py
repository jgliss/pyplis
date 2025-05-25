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
r"""Pyplis module contains the following processing classes and methods.

1. :class:`ImgStack`: Object for storage of 3D image data
#. :class:`PixelMeanTimeSeries`: storage and post analysis of time\
series of average pixel intensities
"""
from numpy import (vstack, empty, ones, asarray, sum, dstack, float32, zeros,
                   poly1d, polyfit, argmin, where, logical_and, rollaxis,
                   delete, hstack)

from scipy.ndimage import gaussian_filter1d, median_filter


from copy import deepcopy
from datetime import datetime, timedelta
from matplotlib.pyplot import subplots
from matplotlib.dates import date2num, DateFormatter

from pandas import Series, concat, DatetimeIndex
from cv2 import pyrDown, pyrUp
from os.path import join, exists, abspath
from astropy.io import fits
from pydoas.analysis import DoasResults
from pyplis import logger
from pyplis.image import Img
from pyplis.setupclasses import Camera
from pyplis.helpers import to_datetime, make_circular_mask
from pyplis.glob import DEFAULT_ROI


class ImgStack:
    """Image stack object.

    The images are stacked into a 3D numpy array, note, that for large datasets
    this may cause MemoryErrors. This object is for instance used to perform
    a DOAS field of view search (see also :mod:`doascalib`).

    It provides basic image processing functionality, for instance changing
    the pyramid level, time merging with other time series data (e.g. DOAS
    CD time series vector).

    The most important attributes (data objects) are:

        1. ``self.stack``: 3D numpy array containing stacked images. The first
        axis corresponds to the time axis, allowing for easy image access,
        e.g. ``self.stack[10]`` would yield the 11th image in the time series.

        2. ``self.start_acq``: 1D array containing acquisition time stamps
        (datetime objects)

        3. ``self.texps``: 1D array conaining exposure times in s for each
        image

        4. ``self.add_data``: 1D array which can be used to store additional
        data for each image (e.g. DOAS CD vector)

    Todo
    ----
    1. Include optical flow routine for emission rate retrieval

    Parameters
    ----------
    height : int
        height of images to be stacked
    width : int
        width of images to be stacked
    num : int
        number of images to be stacked
    dtype :
        numerical data type (e.g. uint8, makes the necessary space smaller,
            default: float32)
    stack_id : str
        string ID of this object ("")
    img_prep : dict
        additional information about the preparation state of the images
        (e.g. roi, gauss pyramid level, dark corrected?, blurred?)
    **stack_data
        can be used to pass stack data directly

    """

    def __init__(self, height=0, width=0, img_num=0, dtype=float32,
                 stack_id=None, img_prep=None, camera=None, **stack_data):
        if stack_id is None:
            stack_id = ""
        self.stack_id = stack_id
        self.dtype = dtype
        self.current_index = 0

        self.stack = None
        self.start_acq = None
        self.texps = None
        self.add_data = None
        self._access_mask = None

        if img_prep is None:
            img_prep = {"pyrlevel": 0}
        self.img_prep = img_prep

        self.roi_abs = DEFAULT_ROI

        self._cam = Camera()

        self.init_stack_array(height, width, img_num)
        if "stack" in stack_data:
            self.set_stack_data(**stack_data)

        if isinstance(camera, Camera):
            self.camera = camera

    def init_stack_array(self, height=0, width=0, img_num=0):
        """Initialize the actual stack data array.

        Note
        ----
        All current data stored in :attr:`stack`, :attr:`start_acq`,
        :attr:`texps`, :attr:`add_data` will be deleted.

        Parameters
        ----------
        height : int
            height of images to be stacked
        width : int
            width of images to be stacked
        num : int
            number of images to be stacked

        """
        try:
            self.stack = empty((int(img_num), int(height), int(width))).\
                astype(self.dtype)
        except MemoryError:
            raise MemoryError("Could not initiate empty 3D numpy array "
                              "(d, h, w): (%s, %s, %s)" % (img_num, height,
                                                           width))
        self.start_acq = asarray([datetime(1900, 1, 1)] * img_num)
        self.texps = zeros(img_num, dtype=float)
        self.add_data = zeros(img_num, dtype=float)

        self._access_mask = zeros(img_num, dtype=bool)
        self.current_index = 0

    @property
    def last_index(self):
        """Return last index."""
        return self.num_of_imgs - 1

    @property
    def start(self):
        """Return start time stamp of first image."""
        try:
            i = self.start_acq[self._access_mask][0]
            add = timedelta(self.texps[self._access_mask][0] / 86400.)
            return i + add
        except IndexError:
            raise IndexError("Stack is empty...")
        except BaseException:
            raise ValueError("Start acquisition time could accessed in stack")

    @property
    def stop(self):
        """Return start time stamp of first image."""
        try:
            i = self.start_acq[self._access_mask][-1]
            add = timedelta(self.texps[self._access_mask][-1] / 86400.)
            return i + add
        except IndexError:
            raise IndexError("Stack is empty...")
        except Exception as e:
            raise ValueError(f"Unexpected error {e}")

    @property
    def time_stamps(self):
        """Acq. time stamps of all images."""
        try:
            dts = ([timedelta(x / (2 * 86400.)) for x in self.texps])
            return self.start_acq + asarray(dts)
        except BaseException:
            raise ValueError("Failed to access information about acquisition "
                             "time stamps and / or exposure times")

    @property
    def pyrlevel(self):
        """Gauss pyramid level of images in stack."""
        return self.img_prep["pyrlevel"]

    @property
    def camera(self):
        """Camera object assigned to stack."""
        return self._cam

    @camera.setter
    def camera(self, value):
        if isinstance(value, Camera):
            self._cam = value
        else:
            raise TypeError("Need Camera object...")

    @property
    def num_of_imgs(self):
        """Depth of stack."""
        return self.stack.shape[0]

    def check_index(self, idx=0):
        if 0 <= idx <= self.last_index:
            return
        elif idx == self.num_of_imgs:
            self._extend_stack_array()
        else:
            raise IndexError("Invalid index %d for inserting image in stack "
                             "with current depth %d" % (idx, self.num_of_imgs))

    def _extend_stack_array(self):
        """Extend the first index of the stack array."""
        h, w = self.shape[1:]
        try:
            self.stack = vstack((self.stack, empty((1, h, w))))
        except MemoryError:
            raise MemoryError("Cannot add more data to stack due to memory "
                              "overflow...")
        self.start_acq = hstack((self.start_acq, [datetime(1900, 1, 1)]))
        self.texps = hstack((self.texps, [0.0]))
        self.add_data = hstack((self.add_data, [0.0]))
        self._access_mask = hstack((self._access_mask, [False]))

    def insert_img(self, pos, img_arr, start_acq=datetime(1900, 1, 1),
                   texp=0.0, add_data=0.0):
        """Insert an image into the stack at provided index.

        Parameters
        ----------
        pos : int
            Insert position of img in stack
        img_arr : array
            image data (must have same dimension than ``self.stack.shape[:2]``,
            can also be of type :obj:`Img`)
        start_acq : datetime
            acquisition time stamp of image, defaults to datetime(1900, 1, 1)
        texp : float
            exposure time of image (in units of s), defaults to 0.0
        add_data
            arbitrary additional data appended to list :attr:`add_data`

        """
        try:
            img_arr = img_arr.img
        except BaseException:
            pass
        if sum(self.shape) == 0:
            h, w = img_arr.shape
            self.init_stack_array(height=h, width=w, img_num=1)
        self.check_index(pos)
        self.stack[pos] = img_arr
        self.start_acq[pos] = to_datetime(start_acq)
        self.texps[pos] = texp
        self.add_data[pos] = add_data
        self._access_mask[pos] = True

    def add_img(self, img_arr, start_acq=datetime(1900, 1, 1), texp=0.0,
                add_data=0.0):
        """Add image at current index position.

        The image is inserted at the current index position ``current_index``
        which is increased by 1 afterwards. If the latter exceeds the dimension
        of the actual stack data array :attr:`stack`, the stack shape will be
        extended by 1.

        Parameters
        ----------
        img_arr : array
            image data (must have same dimension than ``self.stack.shape[:2]``)
        start_acq : datetime
            acquisition time stamp of image, defaults to datetime(1900, 1, 1)
        texp : float
            exposure time of image (in units of s), defaults to 0.0
        add_data
            arbitrary additional data appended to list :attr:`add_data`

        """
# ==============================================================================
#         if self.current_index >= self.last_index:
#             print self.last_index
#             raise IndexError("Last stack index reached...")
# ==============================================================================
        self.insert_img(self.current_index, img_arr, start_acq, texp, add_data)
        self.current_index += 1

    def make_circular_access_mask(self, cx, cy, radius):
        """Create a circular mask for stack.

        Parameters
        ----------
        cx : int
            x position of centre
        cy : nint
            y position of centre
        radius : int
            radius

        Returns
        -------
        array
            circular mask (use e.g. like ``img[mask]`` which will return a
            1D vector containing all pixel values of ``img`` that fall into
            the mask)

        """
        # cx, cy = self.img_prep.map_coordinates(pos_x_abs, pos_y_abs)
        h, w = self.stack.shape[1:]
        return make_circular_mask(h, w, cx, cy, radius)

    def set_stack_data(self, stack, start_acq=None, texps=None):
        """Set the current data based on input.

        Parameters
        ----------
        stack : array
            3D numpy array containing the image stack data
        start_acq : :obj:`array`, optional
            array containing acquisition time stamps
        texps : obj:`array`, optional
            array containing exposure times

        """
        num = stack.shape[0]
        self.stack = stack
        if start_acq is None:
            start_acq = asarray([datetime(1900, 1, 1)] * num)
        self.start_acq = start_acq
        if texps is None:
            texps = zeros(num, dtype=float32)
        self.texps = texps
        self._access_mask = ones(num, dtype=bool)

    def get_data(self):
        """Get stack data (containing of stack, acq. and exp. times).

        Returns
        -------
        tuple
            3-element tuple containing

            - :obj:`array`: stack data
            - :obj:`array`: acq. time stamps
            - :obj:`array`: exposure times

        """
        m = self._access_mask
        return (self.stack[m], asarray(self.time_stamps)[m],
                asarray(self.texps)[m])

    def apply_mask(self, mask):
        """Convolves the stack data with a input mask along time axis.

        Parameter
        ---------
        mask : array
            2D bool mask for image pixel access

        Returns
        -------
        tuple
            3-element tuple containing

            - :obj:`array`: 3D numpy array containing convolved stack data
            - :obj:`array`: acq. time stamps
            - :obj:`array`: exposure times

        """
        # mask_norm = boolMask.astype(float32)/sum(boolMask)
        d = self.get_data()
        # [:, :, newaxis])#, d[1], d[2])
        data_conv = (d[0] * mask.astype(float32))
        return (data_conv, d[1], d[2])

    def get_time_series(self, pos_x=None, pos_y=None, radius=1, mask=None):
        """Get time series in a ROI.

        Retrieve time series at a given pixel position *in stack
        coordinates* in a circular pixel neighbourhood.

        Parameters
        ----------
        pos_x : int
            x position of center pixel on detector
        pos_y : int
            y position of center pixel on detector
        radius : float
            radius of pixel disk on detector (centered around pos_x, pos_y,
            default: 1)
        mask : array
            mask for image pixel access, default is None, if the mask is
            specified and valid (i.e. same shape than images in stack) then
            the other three input parameter are ignored

        Returns
        -------
        tuple
            2-element tuple containing

            - :obj:`Series`: time series data
            - :obj:`array`: pixel access mask used to convolve stack images

        """
        d = self.get_data()
        try:
            data_mask, start_acq, texps = self.apply_mask(mask)
        except BaseException:
            if not radius > 0:
                raise ValueError("Invalid input for param radius (3. pos): "
                                 "value must be larger than 0, got %d"
                                 % radius)
            if radius == 1:
                mask = zeros(self.shape[1:]).astype(bool)
                mask[pos_y, pos_x] = True
                s = Series(d[0][self._access_mask, pos_y, pos_x], d[1])
                return s, mask
            mask = self.make_circular_access_mask(pos_x, pos_y, radius)
            data_mask, start_acq, texps = self.apply_mask(mask)
        values = data_mask.sum((1, 2)) / float(sum(mask))
        return Series(values, start_acq), mask

    def merge_with_time_series(self, time_series, method="average", **kwargs):
        """High level wrapper for data merging.

        Choose from either of three methods to perform an index merging based
        on time stamps of stack and of other time series data (provided on
        input).

        Parameters
        ----------
        time_series : Series
            time series data supposed to be merged with stack data
        method : str
            merge method, currently available methods are:

                - average: determine new stack containing images averaged based
                  on start / stop time stamps of each datapoint in input
                  ``time_series`` (requires corresponding data to be available
                  in input, i.e. ``time_series`` must be of type
                  :class:`DoasResults` of ``pydoas`` library).
                - nearest: perform merging based on nearest datapoint per image
                - interpolation: perform cross interpolation onto unified time
                  index array from stack and time series data
        **kwargs
            additional keyword args specifying additional merge settings (e.g.
            ``itp_type=quadratic`` in case ``method=interpolation`` is used)

        Returns
        -------
        tuple
            2-element tuple containing

            - :obj:`ImgStack`: new stack containing merged data
            - :obj:`Series`: merged time series data

        """
        if not isinstance(time_series, DoasResults):
            raise ValueError(f"Input time series data must be of type DoasResults, received {time_series}")

        if method == "average":
            try:
                return self._merge_tseries_average(time_series, **kwargs)
            except Exception as e:
                logger.info(f"Failed to merge data using method average, trying method nearest instead (Reason: {e})")
                method = "nearest"
        if method == "nearest":
            return self._merge_tseries_nearest(time_series, **kwargs)
        elif method == "interpolation":
            return self._merge_tseries_cross_interpolation(time_series, **kwargs)
        else:
            raise TypeError(f"Unkown merge type: {method}. Choose from [nearest, average, interpolation]")

    def _merge_tseries_nearest(self, time_series: DoasResults, **kwargs):
        """Find nearest in time image for each time stamp in input series.

        Find indices (and time differences) in input time series of nearest
        data point for each image in this stack. Then, get rid of all indices
        showing double occurences using time delta information.
        """
        nearest_idxs, del_ts = self.get_nearest_indices(time_series.index)
        img_idxs = []
        spec_idxs_final = []
        del_ts_abs = []
        for idx in range(min(nearest_idxs), max(nearest_idxs) + 1):
            logger.info("Current tseries index %s" % idx)
            matches = where(nearest_idxs == idx)[0]
            if len(matches) > 0:
                del_ts_temp = del_ts[matches]
                spec_idxs_final.append(idx)
                del_ts_abs.append(min(del_ts_temp))
                img_idxs.append(matches[argmin(del_ts_temp)])

        series_new = DoasResults(time_series.iloc[spec_idxs_final])
        if time_series.fit_errs is not None and len(time_series.fit_errs) == len(time_series):
            series_new.fit_errs = time_series.fit_errs[spec_idxs_final]
        
        stack_new = self.stack[img_idxs]
        texps_new = asarray(self.texps[img_idxs])
        start_acq_new = asarray(self.start_acq[img_idxs])
        stack_obj_new = ImgStack(
            stack_id=self.stack_id,
            img_prep=self.img_prep, 
            stack=stack_new,
            start_acq=start_acq_new, 
            texps=texps_new
        )
        stack_obj_new.roi_abs = self.roi_abs
        stack_obj_new.add_data = series_new
        return (stack_obj_new, series_new)

    def _merge_tseries_cross_interpolation(self, time_series: DoasResults, itp_type="linear"):
        """Merge this stack with input data using interpolation.

        :param Series time_series_data: pandas Series object containing time
            series data (e.g. DOAS column densities)
        :param str itp_type: interpolation type (passed to
            :class:`pandas.DataFrame` which does the interpolation, default is
            linear)
        """
        h, w = self.shape[1:]
        stack, time_stamps, _ = self.get_data()

        # first crop time series data based on start / stop time stamps
        time_series = self.crop_other_tseries(time_series)
        time_series.name = None
        if not len(time_series) > 0:
            raise IndexError("Time merging failed, data does not overlap")

        # interpolate exposure times
        s0 = Series(self.texps, time_stamps)
        if time_series.fit_errs is not None and len(time_series.fit_errs) == len(time_series):
            errs = Series(time_series.fit_errs, time_series.index)
            df0 = concat([s0, time_series, errs], axis=1).interpolate(itp_type).dropna()
            fit_errs_avail = True
        else:
            df0 = concat([s0, time_series], axis=1).interpolate(itp_type).dropna()
            fit_errs_avail = False
        new_num = len(df0[0])
        if not new_num >= self.num_of_imgs:
            raise ValueError("Unexpected error, length of merged data "
                             "array does not exceed length of inital image "
                             "stack...")
        # create new arrays for the merged stack
        new_stack = empty((new_num, h, w))
        new_acq_times = df0[0].index
        new_texps = df0[0].values
        for i in range(h):
            for j in range(w):
                logger.info(f"Stack interpolation active...: current img row (y): {i} ({j})")
                # get series from stack at current pixel
                series_stack = Series(stack[:, i, j], time_stamps)
                # create a dataframe
                df = concat([series_stack, df0[1]], axis=1).interpolate(itp_type).dropna()
                # throw all N/A values
                # df = df.dropna()
                new_stack[:, i, j] = df[0].values

        stack_obj = ImgStack(new_num, h, w,
                             stack_id=self.stack_id,
                             img_prep=self.img_prep)
        stack_obj.roi_abs = self.roi_abs
        
        stack_obj.set_stack_data(new_stack, new_acq_times, new_texps)

        new_series = DoasResults(df[1])
        if fit_errs_avail:
            new_series.fit_errs = df0[2].values
        return (stack_obj, new_series)

    def _merge_tseries_average(self, time_series, **kwargs):
        """Make new stack of averaged images based on input start / stop arrays.

        The averaging is based on the start / stop time stamps (e.g. of
        measured spectra) specified by two input arrays.
        These arrays must have the same length.
        The method loops over these arrays indices and at each iteration step
        k, all images (wihin this stack) falling into the corresponding
        start / stop interval are averaged and added to a new stack of averaged
        images. Indices k (of the input arrays) for which
        no images can be found are added to the list ``bad_indices`` (second
        return parameter) and have to be removed from the corresponding data
        in case, these data (e.g. DOAS SO2 CD time series) is supposed to be
        compared with the averaged stack.

        Parameters
        ----------
        time_series : DoasResults
            Time series containing DOAS results, including arrays
            for start / stop acquisition time stamps (required for averaging)

        Returns
        -------
        tuple
            2-element tuple containing
            - :class:`ImgStack`: new stack object with averaged images matching input DOAS timeseries meas intervals
            - :class:`DoasResults`: new DOAS timeseries matching timestamps in output ImgStack

        """
        if not time_series.has_start_stop_acqtamps():
            raise ValueError("No start / stop acquisition time stamps "
                                "available in input data...")
        start_acq = asarray(time_series.start_acq)
        stop_acq = asarray(time_series.stop_acq)
    
        stack, times, _ = self.get_data()
        h, w = stack.shape[1:]
        num = len(start_acq)

        # new_stack = empty((h, w, self.num_of_imgs))
        new_acq_times = []
        new_texps = []
        bad_indices = []
        counter = 0
        for k in range(num):
            i = start_acq[k]
            f = stop_acq[k]
            texp = (f - i).total_seconds()
            cond = (times >= i) & (times < f)
            if sum(cond) > 0:
                im = stack[cond].mean(axis=0)
                if counter == 0:
                    new_stack = im
                else:
                    new_stack = dstack((new_stack, im))
                new_acq_times.append(i + (f - i) / 2)
                new_texps.append(texp)
                counter += 1
            else:
                bad_indices.append(k)
        new_stack = rollaxis(new_stack, 2)
        stack_obj = ImgStack(len(new_texps), h, w,
                             stack_id=self.stack_id,
                             img_prep=self.img_prep)
        stack_obj.roi_abs = self.roi_abs
        stack_obj.set_stack_data(new_stack, asarray(new_acq_times),
                                 asarray(new_texps))

        tseries = DoasResults(time_series.drop(time_series.index[bad_indices]))
        if time_series.fit_errs is not None and len(time_series.fit_errs) == len(time_series):
            errs = delete(time_series.fit_errs, bad_indices)
            tseries.fit_errs = errs
        return (stack_obj, tseries)

    def crop_other_tseries(self, time_series: DoasResults) -> DoasResults:
        """Crops other time series object based on start / stop time stamps."""
        cond = logical_and(time_series.index >= self.start, time_series.index <= self.stop)
        new = DoasResults(time_series[cond])
        if time_series.fit_errs is not None and len(time_series.fit_errs) == len(time_series):
            new.fit_errs = time_series.fit_errs[cond]
        return new

    def total_time_period_in_seconds(self):
        """Return start time stamp of first image."""
        return (self.stop - self.start).total_seconds()

    def get_nearest_indices(self, tstamps_other):
        """Find indices of time stamps nearest to img acq. time stamps.

        Parameters
        ----------
        tstamps_other :
            datetime, or datetime array of other time series for which closest
            index / indices are searched

        """
        idx = []
        delt = []
        img_stamps = self.time_stamps[self._access_mask]
        for tstamp in img_stamps:
            diff = [x.total_seconds() for x in abs(tstamps_other - tstamp)]
            delt.append(min(diff))
            idx.append(argmin(diff))
        return asarray(idx), asarray(delt)

    def has_data(self):
        """Return bool."""  # fixme: improve this doc
        return bool(sum(self._access_mask))

    def sum(self, *args, **kwargs):
        """Sum over all pixels of stack.

        Parameters
        ----------
        *args
            non-keyword arguments passed to :func:`sum` of numpy array
        **kwargs
            keyword arguments passed to :func:`sum` of numpy array

        Returns
        -------
        float
            result of summation operation

        """
        return self.stack.sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        """Apply numpy.mean function to stack data.

        :param *args: non keyword arguments passed to :func:`numpy.mean`
            applied to stack data
        :param **kwargs: keyword arguments passed to :func:`numpy.mean`
            applied to stack data
        """
        return self.stack.mean(*args, **kwargs)

    def std(self, *args, **kwargs):
        """Apply numpy.std function to stack data.

        :param *args: non keyword arguments passed to :func:`numpy.std`
            applied to stack data
        :param **kwargs: keyword arguments passed to :func:`numpy.std`
            applied to stack data
        """
        return self.stack.std(*args, **kwargs)

    @property
    def shape(self):
        """Return stack shape."""
        return self.stack.shape

    @property
    def ndim(self):
        """Return stack dimension."""
        return self.stack.ndim

    def show_img(self, index=0):
        """Show image at input index.

        Parameters
        ----------
        index : int
            index of image in stack

        """
        stack, ts, _ = self.get_data()
        im = Img(stack[index], start_acq=ts[index], texp=self.texps[index])
        im.edit_log.update(self.img_prep)
        im.roi_abs = self.roi_abs
        return im.show()

    def pyr_down(self, steps=0):
        """Reduce the stack image size using gaussian pyramid.

        Parameters
        ----------
        steps : int
            steps down in the pyramide

        Returns
        -------
        ImgStack
            new, downscaled image stack object

        """
        if not steps:
            return
        h, w = Img(self.stack[0]).pyr_down(steps).shape
        prep = deepcopy(self.img_prep)
        new_stack = ImgStack(height=h, width=w, img_num=self.num_of_imgs,
                             stack_id=self.stack_id, img_prep=prep)
        for i in range(self.shape[0]):
            im = self.stack[i]
            for k in range(steps):
                im = pyrDown(im)
            new_stack.add_img(img_arr=im, start_acq=self.start_acq[i],
                              texp=self.texps[i], add_data=self.add_data[i])
        new_stack._format_check()
        new_stack.img_prep["pyrlevel"] += steps
        return new_stack

    def pyr_up(self, steps):
        """Increasing the image size using gaussian pyramide.

        :param int steps: steps down in the pyramide

        Algorithm used: :func:`cv2.pyrUp`
        """
        if not steps:
            return

        h, w = Img(self.stack[0]).pyr_up(steps).shape
        prep = deepcopy(self.img_prep)
        new_stack = ImgStack(height=h, width=w, img_num=self.num_of_imgs,
                             stack_id=self.stack_id, img_prep=prep)
        for i in range(self.shape[0]):
            im = self.stack[i]
            for k in range(steps):
                im = pyrUp(im)
            new_stack.add_img(img_arr=im, start_acq=self.start_acq[i],
                              texp=self.texps[i], add_data=self.add_data[i])
        new_stack._format_check()
        new_stack.img_prep["pyrlevel"] -= steps
        return new_stack

    def to_pyrlevel(self, final_state=0):
        """Down / upscale image to a given pyramide level."""
        steps = final_state - self.img_prep["pyrlevel"]
        if steps > 0:
            return self.pyr_down(steps)
        elif steps < 0:
            return self.pyr_up(-steps)
        else:
            return self

    def duplicate(self):
        """Return deepcopy of this object."""
        return deepcopy(self)

    def _format_check(self):
        """Check if all relevant data arrays have the same length."""
        if not all([len(x) == self.num_of_imgs for x in [self.add_data,
                                                         self.texps,
                                                         self._access_mask,
                                                         self.start_acq]]):
            raise ValueError("Mismatch in array lengths of stack data, check"
                             "add_data, texps, start_acq, _access_mask")

    def load_stack_fits(self, file_path):
        """Load stack object (fits).

        Note
        ----
        FITS stores in Big-endian and needs to be converted into little-endian
        (see `this issue <https://github.com/astropy/astropy/issues/1156>`__).
        We follow the suggested fix and use::

            byteswap().newbyteorder()

        on any loaded data array.

        Parameters
        ----------
        file_path : str
            file path of stack

        """
        if not exists(file_path):
            raise IOError("ImgStack could not be loaded, path does not exist")
        hdu = fits.open(file_path)
        self.set_stack_data(hdu[0].data.astype(self.dtype))
        prep = Img().edit_log
        for key, val in hdu[0].header.items():
            if key.lower() in prep.keys():
                self.img_prep[key.lower()] = val
        self.stack_id = hdu[0].header["stack_id"]
        times = hdu[1].data["start_acq"]
        self.start_acq = asarray([datetime.strptime(x, "%Y%m%d%H%M%S%f") for x in times])
        self.texps = asarray(hdu[1].data["texps"])
        self._access_mask = asarray(hdu[1].data["_access_mask"])
        self.add_data = asarray(hdu[1].data["add_data"])
        self.roi_abs = hdu[2].data["roi_abs"]
        self._format_check()

    def save_as_fits(self, save_dir, save_name=None,
                     overwrite_existing=True) -> str:
        """Save stack as FITS file.
        
        Args:
            save_dir (str): directory to save the FITS file
            save_name (str): name of the FITS file (optional)
            overwrite_existing (bool): whether to overwrite existing files
                (default: True)
        
        Returns:
            output path
        """
        self._format_check()
        save_dir = abspath(save_dir)
        if save_name is None:
            save_name = ("pyplis_imgstack_id_%s_%s_%s_%s.fts"
                         % (self.stack_id,
                            self.start.strftime("%Y%m%d"),
                            self.start.strftime("%H%M"),
                            self.stop.strftime("%H%M")))
        
        logger.info(f"DIR: {save_dir}")
        logger.info(f"Name: {save_name}")
        hdu = fits.PrimaryHDU()
        start_acq_str = [x.strftime("%Y%m%d%H%M%S%f") for x in self.start_acq]
        col1 = fits.Column(name="start_acq", format="25A", array=start_acq_str)
        col2 = fits.Column(name="texps", format="D", array=self.texps)
        col3 = fits.Column(name="_access_mask", format="L", array=self._access_mask)
        col4 = fits.Column(name="add_data", format="D", array=self.add_data)
        cols = fits.ColDefs([col1, col2, col3, col4])
        arrays = fits.BinTableHDU.from_columns(cols)

        col5 = fits.Column(name="roi_abs", format="I", array=self.roi_abs)

        roi_abs = fits.BinTableHDU.from_columns([col5])
        hdu.data = self.stack
        hdu.header.update(self.img_prep)
        hdu.header["stack_id"] = self.stack_id
        hdu.header.append()
        hdulist = fits.HDUList([hdu, arrays, roi_abs])
        path = join(save_dir, save_name)
        if exists(path):
            logger.info(f"Stack already exists at {path} and will be overwritten")
        hdulist.writeto(path, overwrite=overwrite_existing)
        return path

    def __sub__(self, other):
        """Subtract data.

        :param other: data to be subtracted object (e.g. offband stack)

        """
        new = self.duplicate()
        try:
            new.stack = self.stack - other.stack
            new.stack_id = "%s - %s" % (self.stack_id, other.stack_id)
        except BaseException:
            new.stack = self.stack - other
            new.stack_id = "%s - %s" % (self.stack_id, other)
        return new


def find_registration_shift_optflow(on_img, off_img,
                                    roi_abs=DEFAULT_ROI, **flow_settings):
    """Search average shift between two images using optical flow.

    Computes optical flow between two input images and determines the
    registration shift based on peaks in two histograms of the orientation
    angle distribution and vector magnitued distribution of the retrieved
    flow field. The histogram analysis may be reduced to a certain ROI in the
    images.

    The default settings used here correspond to the settings suggested by
    Peters et al., Use of motion estimation algorithms for improved flux
    measurements using SO2 cameras, JVGR, 2015.

    Parameters
    ----------
    on_img : Img
        onband image containing (preferably fixed) objects in the scene that
        can be tracked
    off_img : Img
        corresponding offband image (ideally recorded at the same time)
    roi_abs : list
        if specified, the optical flow histogram parameters are retrieved from
        the flow field within this ROI (else, the whole image is used)
    **flow_settings
        additional keyword args specifying the optical flow computation and
        post analysis settings (see
        :class:`pyplis.plumespeed.FarnebackSettings` for details)

    Returns
    -------
    tuple
        2-element tuple containing

        - float: shift in x-direction
        - float: shift in y-direction

    """
    if not on_img.shape == off_img.shape:
        raise ValueError("Shape mismatch between input images")
    if on_img.pyrlevel != 0:
        logger.warning("Input images are at pyramid level %d and registration shift "
             "will be computed for this pyramid level")
    # from pyplis import OptflowFarneback
    # flow = OptflowFarneback(on_img, off_img, **flow_settings)
    raise NotImplementedError("Under development")


class PixelMeanTimeSeries(Series):
    """A time series of mean pixel values.

    This class implements a ``pandas.Series`` object with extended
    functionality representing time series data of pixel mean values in a
    certain image region.

    .. note::
        This object is only used to store results of a mean series analysis
        in a certain ROI, it does not include any algorithms for actually
        calculating the series

    """

    std = None
    texps = None
    img_prep = {}
    roi_abs = None
    poly_model = None

    def __init__(self, data, start_acq, std=None, texps=None, roi_abs=None,
                 img_prep=None, **kwargs):
        """Initialize pixel mean time series.

        :param ndarray data: data array
            (is passed into pandas Series init -> ``self.values``)
        :param ndarray start_acq: array containing acquisition time stamps
            (is passed into pandas Series init -> ``self.index``)
        :param ndarray std: array containing standard deviations
        :param ndarray texps: array containing exposure times
        :param list roi_abs: image area from which data was extracted, list of
            shape: ``[x0, y0, x1, y1]``
        :param dict img_prep: dictionary containing information about image
            preparation settings (e.g. blurring, etc..) or other
            important information which may need to be stored
        :param **kwargs: additional keyword parameters which are passed to
            the initiation of the :class:`pandas.Series` object

        """
        super(PixelMeanTimeSeries, self).__init__(data, start_acq, **kwargs)
        if img_prep is None:
            img_prep = {}
        try:
            if len(texps) == len(data):
                self.texps = texps
        except BaseException:
            self.texps = zeros(len(data), dtype=float32)
        try:
            if len(std) == len(data):
                self.std = std
        except BaseException:
            self.std = zeros(len(data), dtype=float32)

        self.img_prep = img_prep
        self.roi_abs = roi_abs

        for key, val in kwargs.items():
            self[key] = val

    @property
    def start(self):
        return self.index[0]

    @property
    def stop(self):
        return self.index[-1]

    def get_data_normalised(self, texp=None):
        """Normalise the mean value to a given exposure time.

        :param float texp (None): the exposure time to which all deviating
            times will be normalised. If None, the values will be normalised
            to the largest available exposure time
        :return: A new :class:`PixelMeanTimeSeries`instance with normalised
            data
        """
        try:
            if texp is None:
                texp = self.texps.max()
            facs = texp / self.texps
            ts = self.texps * facs

            return PixelMeanTimeSeries(self.values * facs, self.index,
                                       self.std, ts, self.roi_abs,
                                       self.img_prep)

        except Exception as e:
            logger.info("Failed to normalise data bases on exposure times:\n%s\n\n"
                  % repr(e))

    def fit_polynomial(self, order=2):
        """Fit polynomial to data series.

        :param int order: order of polynomial
        :returns:
            - poly1d, the fitted polynomial
        """
        s = self.dropna()
        num = len(s)
        if num == 1:
            raise ValueError("Could not fit polynomial to PixelMeanTimeSeries"
                             " object: only one data point available")
        elif num == 2:
            logger.warning("PixelMeanTimeSeries object only contains 2 data points, "
                 "setting polyfit order to one (default is 2)")
            order = 1
        x = [date2num(idx) for idx in s.index]
        y = s.values
        p = poly1d(polyfit(x, y, deg=order))
        self.poly_model = p
        return p

    def includes_timestamp(self, time_stamp, ext_border_secs=0.0):
        """Check if input time stamp is included in this dataset.

        :param datetime time_stamp: the time stamp to be checked
        :param float ext_border_secs: extend start / stop range (default 0 s)
        :return:
            - bool, True / False (timestamp is within interval)
        """
        i = self.start - timedelta(ext_border_secs / 86400.0)
        f = self.stop + timedelta(ext_border_secs / 86400.0)
        if i <= to_datetime(time_stamp) <= f:
            return True
        return False

    def get_poly_vals(self, time_stamps, ext_border_secs=0.0):
        """Get value of polynomial at input time stamp.

        :param datetime time_stamp: poly input value
        """
        if not isinstance(self.poly_model, poly1d):
            raise AttributeError("No polynomial available, please call"
                                 "function fit_polynomial first")
        if isinstance(time_stamps, datetime):
            time_stamps = [time_stamps, ]
        if not any([isinstance(time_stamps, x)
                    for x in [list, DatetimeIndex]]):
            raise ValueError("Invalid input for time stamps, need list")
        if not all([self.includes_timestamp(x, ext_border_secs)
                    for x in time_stamps]):
            raise IndexError("At least one of the time stamps is not included "
                             "in this series: %s - %s"
                             % (self.start, self.stop))
        values = []
        for time_stamp in time_stamps:
            values.append(self.poly_model(date2num(time_stamp)))

        return asarray(values)

    def estimate_noise_amplitude(self, sigma_gauss=1, median_size=3, plot=0):
        """Estimate the amplitude of the noise in the data.

        Steps:

            1. Determines high frequency variations by applying binomial
                filter (sigma = 3) to data and subtract this from data,
                resulting in a residual
            2. Median filtering of residual signal to get rid of narrow peaks
                (i.e. where the original data shows abrupt changes)
            3. subtract both signals and determine std

        ..note::

            Beta version: no guarantee it works for all cases

        """
        # make bool array of indices considered (initally all)
        y0 = median_filter(self.values, 3)
        y1 = gaussian_filter1d(y0, sigma_gauss)
        res0 = y0 - y1
        res1 = median_filter(res0, median_size)
        diff = res1 - res0
        if plot:
            fig, ax = subplots(2, 1)
            ax[0].plot(y0, "-c", label="y0")
            ax[0].plot(y1, "--xr", label="y1: Smoothed y0")
            ax[0].legend(
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=10)
            ax[1].plot(res0, "--c", label="res0: y0 - y1")
            ax[1].plot(res1, "--r", label="res1: Median(res0)")
            ax[1].plot(diff, "--b", label="diff: res1 - res0")
            ax[1].legend(
                loc='best',
                fancybox=True,
                framealpha=0.5,
                fontsize=10)
        return diff.std()

    def plot(self, include_tit=True, date_fmt=None, **kwargs):
        """Plot time series.

        Parameters
        ----------
        include_tit : bool
            Include a title
        date_fmt : str
            Date / time formatting string for x labels, passed to
            :class:`DateFormatter` instance (optional)
        **kwargs
            Additional keyword arguments passed to pandas Series plot method

        Returns
        -------
        axes
            matplotlib axes instance

        """
        try:
            self.index = self.index.to_pydatetime()
        except BaseException:
            pass
        try:
            if "style" not in kwargs:
                kwargs["style"] = "--x"

            ax = super(PixelMeanTimeSeries, self).plot(**kwargs)
            try:
                if date_fmt is not None:
                    ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
            except BaseException:
                pass
            if include_tit:
                ax.set_title("Mean value (%s), roi_abs: %s"
                             % (self.name, self.roi_abs))
            ax.grid()

            return ax
        except Exception as e:
            logger.info(repr(e))
            fig, ax = subplots(1, 1)
            ax.text(.1, .1, "Plot of PixelMeanTimeSeries failed...")
            fig.canvas.draw()
            return ax

    def __setitem__(self, key, value):
        """Update class item."""
        logger.info("%s : %s" % (key, value))
        if key in self.__dict__:
            logger.info("Writing...")
            self.__dict__[key] = value

    def __call__(self, normalised=False):
        """Return the current data arrays (mean, std)."""
        if normalised:
            return self.get_data_normalised()
        return self.get_data()