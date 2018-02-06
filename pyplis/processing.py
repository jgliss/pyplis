# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliß (jonasgliss@gmail.com)
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
"""
This Pyplis module contains the following processing classes and methods:

    1. :class:`PixelMeanTimeSeries`: storage and post analysis of time\
    series of average pixel intensities
    #. :class:`LineOnImage`: data access along a line on a 2D discrete\
    grid using interpolation
    #. :class:`ProfileTimeSeriesImg`: expanded :class:`Img` object, where\
    y axis corresponds to 1D profile data (e.g. line profile from an\
    image) and x corresponds to the time axis.
    #. :class:`ImgStack`: Object for storage of 3D image data
    #. :func:`model_dark_image`: method to model a dark image from a long\
    and short exposure dark
        
"""
from numpy import vstack, ogrid, empty, ones, asarray, ndim, round, hypot,\
    linspace, sum, dstack, float32, zeros, poly1d, polyfit, argmin, where,\
    logical_and, rollaxis, complex, angle, array, ndarray, cos, sin,\
    arctan, dot, int32, pi, isnan, nan, delete, mean, hstack
    
from numpy.linalg import norm
from scipy.ndimage import map_coordinates 
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from warnings import warn
from json import loads, dumps
from copy import deepcopy
from datetime import datetime, timedelta
from matplotlib.pyplot import subplot, subplots, tight_layout, draw
from matplotlib.dates import date2num, DateFormatter
from matplotlib.patches import Polygon, Rectangle

from pandas import Series, concat, DatetimeIndex
from cv2 import cvtColor, COLOR_BGR2GRAY, pyrDown, pyrUp, fillPoly
from os import remove
from os.path import join, exists, dirname, basename, isdir, abspath
from astropy.io import fits
    
from .image import Img
from .setupclasses import Camera
from .exceptions import ImgMetaError, ImgModifiedError
from .helpers import map_coordinates_sub_img, same_roi, map_roi, to_datetime,\
    roi2rect

class PixelMeanTimeSeries(Series):
    """A ``pandas.Series`` object with extended functionality representing time
    series data of pixel mean values in a certain image region
    
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
                 img_prep={}, **kwargs):
        """
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
        try:
            if len(texps) == len(data):
                self.texps = texps
        except:
            self.texps = zeros(len(data), dtype = float32)
        try:
            if len(std) == len(data):
                self.std = std
        except:
            self.std = zeros(len(data), dtype = float32)
            
        self.img_prep = img_prep
        self.roi_abs = roi_abs
        
        for key, val in kwargs.iteritems():
            self[key] = val
        
    def get_data_normalised(self, texp=None):
        """Normalise the mean value to a given exposure time
        
        :param float texp (None): the exposure time to which all deviating times
            will be normalised. If None, the values will be normalised to the 
            largest available exposure time
       :return: A new :class:`PixelMeanTimeSeries`instance with normalised data
        """
        try:
            if texp is None:
                texp = self.texps.max()
            facs = texp / self.texps
            ts = self.texps * facs
            
            return PixelMeanTimeSeries(self.values*facs, self.index, self.std, 
                                       ts, self.roi_abs, self.img_prep)
            
        except Exception as e:
            print ("Failed to normalise data bases on exposure times:\n%s\n\n"
                        %repr(e))
                      
    
    def fit_polynomial(self, order=2):
        """Fit polynomial to data series
        
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
            warn("PixelMeanTimeSeries object only contains 2 data points, "
                "setting polyfit order to one (default is 2)")
            order = 1
        x = [date2num(idx) for idx in s.index] 
        y = s.values
        p = poly1d(polyfit(x, y, deg = order))
        self.poly_model = p
        return p
    
    def includes_timestamp(self, time_stamp, ext_border_secs=0.0):
        """Checks if input time stamp is included in this dataset
        
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
        """Get value of polynomial at input time stamp
        
        :param datetime time_stamp: poly input value
        """
        if not isinstance(self.poly_model, poly1d):
            raise AttributeError("No polynomial available, please call"
                "function fit_polynomial first")
        if isinstance(time_stamps, datetime):
            time_stamps = [time_stamps,]
        if not any([isinstance(time_stamps, x) for x in [list, DatetimeIndex]]):
            raise ValueError("Invalid input for time stamps, need list")
        if not all([self.includes_timestamp(x, ext_border_secs)\
                                                    for x in time_stamps]):
            raise IndexError("At least one of the time stamps is not included "
                "in this series: %s - %s" %(self.start, self.stop))
        values = []
        for time_stamp in time_stamps:
            values.append(self.poly_model(date2num(time_stamp)))
            
        return asarray(values)
        
    def estimate_noise_amplitude(self, sigma_gauss=1, median_size=3, plot=0):
        """Estimate the amplitude of the noise in the data 
        
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
        #make bool array of indices considered (initally all)
        y0 = median_filter(self.values, 3)
        y1 = gaussian_filter1d(y0, sigma_gauss)
        res0 = y0 - y1
        res1 = median_filter(res0, median_size)
        diff = res1 - res0
        if plot:
            fig, ax = subplots(2,1)
            ax[0].plot(y0, "-c",label="y0")
            ax[0].plot(y1, "--xr",label="y1: Smoothed y0")
            ax[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
            ax[1].plot(res0, "--c",label="res0: y0 - y1")
            ax[1].plot(res1, "--r",label="res1: Median(res0)")
            ax[1].plot(diff, "--b",label="diff: res1 - res0")
            ax[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
        return diff.std()
    
    def plot(self, include_tit=True, date_fmt=None, **kwargs):
        """Plot time series
        
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
        except:
            pass
        try:
            if not "style" in kwargs:
                kwargs["style"] = "--x"  
        
            ax = super(PixelMeanTimeSeries, self).plot(**kwargs)
            try:
                if date_fmt is not None:
                    ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
            except:
                pass
            if include_tit:
                ax.set_title("Mean value (%s), roi_abs: %s" 
                                %(self.name, self.roi_abs))
            ax.grid()
            
            return ax
        except Exception as e:
            print repr(e)
            fig, ax = subplots(1,1)
            ax.text(.1,.1, "Plot of PixelMeanTimeSeries failed...")
            fig.canvas.draw()
            return ax
            
    @property
    def start(self):
        return self.index[0]
    
    @property
    def stop(self):
        return self.index[-1]

    def __setitem__(self, key, value):
        """Update class item"""
        print "%s : %s" %(key, value)
        if self.__dict__.has_key(key):
            print "Writing..."
            self.__dict__[key]=value
            
    def __call__(self, normalised = False):
        """Returns the current data arrays (mean, std)"""
        if normalised:
            return self.get_data_normalised()
        return self.get_data()
        
class LineOnImage(object):
    """Class representing a line on an image
    
    Main purpose is data extraction along this line on a discrete image grid.
    This is done using spline interpolation.
    
    Parameters
    ----------    
    x0 : int
        start x coordinate
    y0 : int
        start y coordinate
    x1 : int
        stop x coordinate
    y1 : int 
        stop y coordinate
    normal_orientation : str
        orientation of normal vector, choose from left or right (left means in 
        negative x direction for a vertical line)
    roi_abs_def : list
        ROI specifying image sub coordinate system in which the line
        coordinates are defined (is used to convert to other image shape 
        settings)
    pyrlevel_def : int
        pyramid level of image for which start /stop coordinates are defined
    line_id : str
        string for identification (optional)
        
    Note
    ----
    The input coordinates correspond to relative image coordinates 
    with respect to the input ROI (``roi_def``) and pyramid level
    (``pyrlevel_def``)
        
            
    """
    def __init__(self, x0=0, y0=0, x1=1, y1=1,  normal_orientation="right",
                 roi_abs_def=[0, 0, 9999, 9999], pyrlevel_def=0, line_id = "",
                 color="lime", linestyle="-"):
                     
        self.line_id = line_id # string ID of line
        self.color = color
        self.linestyle = linestyle
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        elif x0 == x1 and y0 > y1:
            y0, y1 = y1, y0
        self.x0 = x0 #start x coordinate
        self.y0 = y0 #start y coordinate
        self.x1 = x1 #stop x coordinate
        self.y1 = y1 #stop y coordinate
        
        self._roi_abs_def = roi_abs_def
        self._pyrlevel_def = pyrlevel_def
        self._rect_roi_rot = None
        self._line_roi_abs = [0, 0, 9999, 9999]
        self._last_rot_roi_mask = None
        
        self.profile_coords = None
        
        self._dir_idx = {"left"   :   0,
                         "right"  :   1}
                        
        self.normal_vecs = [None, None]
        
        self._velo_glob = nan
        self._velo_glob_err = nan
        self._plume_props = None
                         
        self.check_coordinates()
        self.normal_orientation = normal_orientation
                                       
        self.prepare_coords()
            
    @property
    def start(self):
        """x, y coordinates of start point (``[x0, y0]``)"""
        return [self.x0, self.y0]

    @start.setter
    def start(self, val):
        try:
            if len(val) == 2:
                self.x0 = val[0]
                self.y0 = val[1]
        except:
            warn("Start coordinates could not be set")

    @property
    def stop(self):
        """x, y coordinates of stop point (``[x1, y1]``)"""
        return [self.x1, self.y1]

    @stop.setter
    def stop(self, val):
        try:
            if len(val) == 2:
                self.x1 = val[0]
                self.y1 = val[1]
        except:
            warn("Stop coordinates could not be set")

    @property
    def center_pix(self):
        """Returns coordinate of center pixel"""
        dx, dy = self._delx_dely()
        xm = self.x0 + dx / 2.
        ym = self.y0 + dy / 2.
        return xm, ym
        
    
    @property
    def normal_orientation(self):
        """Get / set value for orientation of normal vector"""
        return self._normal_orientation
        
    @normal_orientation.setter
    def normal_orientation(self, val):
        if not val in ["left", "right"]:
            raise ValueError("Invalid input for attribute orientation, please"
                " choose from left or right")
        dx, dy = self._delx_dely()
        if dx*dy < 0:
            self._dir_idx["left"] =1
            self._dir_idx["right"] = 0
        self._normal_orientation = val
    
    @property
    def line_frame(self):
        """ROI framing the line (in line coordinate system)"""
        return map_roi(self._line_roi_abs, self.pyrlevel_def)
    
    @property
    def line_frame_abs(self):
        """ROI framing the line (in absolute coordinate system)"""
        return self._line_roi_abs
        
    @property
    def roi_def(self):
        """ROI in which line is defined (at current ``pyrlevel``)"""
        return map_roi(self.roi_abs_def, pyrlevel_rel=self.pyrlevel_def)
        
    @property
    def roi_abs_def(self):
        """Current ROI (in absolute detector coordinates)"""
        return self._roi_abs_def
    
    @roi_abs_def.setter
    def roi_abs_def(self):
        raise AttributeError("This attribute is not supposed to be changed, "
            "please use method convert() to create a new LineOnImage object "
            "corresponding to other image shape settings")
    
    # Redundancy (after renaming attribute in v0.10)        
    @property
    def pyrlevel(self):
        """Pyramid level at which line coords are defined"""
        warn("This method was renamed in version 0.10. Please use pyrlevel_def")
        return self._pyrlevel_def
        
    @pyrlevel.setter
    def pyrlevel(self):
        raise AttributeError("This attribute is not supposed to be changed, "
            "please use method convert() to create a new LineOnImage object "
            "corresponding to other image shape settings")
    
    @property
    def roi_abs(self):
        """Current ROI (in absolute detector coordinates)"""
        warn("This method was renamed in version 0.10. Please use roi_abs_def")
        return self._roi_abs_def
    
    @roi_abs.setter
    def roi_abs(self):
        raise AttributeError("This attribute is not supposed to be changed, "
            "please use method convert() to create a new LineOnImage object "
            "corresponding to other image shape settings")
    
    @property
    def pyrlevel_def(self):
        """Pyramid level at which line coords are defined"""
        return self._pyrlevel_def
        
    @pyrlevel_def.setter
    def pyrlevel_def(self):
        """Raises AttributeError"""
        raise AttributeError("This attribute is not supposed to be changed, "
            "please use method convert() to create a new LineOnImage object "
            "corresponding to other image shape settings")
            
    @property
    def coords(self):
        """Return coordinates as ROI list"""
        return [self.x0, self.y0, self.x1, self.y1]
    
    @property
    def rect_roi_rot(self):
        """Rectangle specifying coordinates of ROI aligned with line normal"""
        try:
            if not self._rect_roi_rot.shape == (5,2):
                raise Exception
        except:
            print("Rectangle for rotated ROI was not set and is not being "
                "set to default depth of +/- 30 pix around line. Use "
                "method set_rect_roi_rot to change the rectangle")
            self.set_rect_roi_rot()
        return self._rect_roi_rot
    
    @property
    def velo_glob(self):
        """Global velocity in m/s, assigned to this line
        
        Raises
        ------
        AttributeError
            if current value is not of type float
        """
        if not isinstance(self._velo_glob, float) or isnan(self._velo_glob):
            raise AttributeError("Global velocity not assigned to line")
        return self._velo_glob
        
    @velo_glob.setter
    def velo_glob(self, val):
        try:
            val = float(val)
            if isnan(val):
                raise Exception
        except:
            raise ValueError("Invalid input, need float or int...")
        if val < 0:
            raise ValueError("Velocity must be larger than 0")
        elif val > 40:
            warn("Large value warning: input velocity exceeds 40 m/s")
        self._velo_glob = val
        if self._velo_glob_err is None or isnan(self._velo_glob_err):
            warn("Global velocity error not assigned, assuming 50% of "
                 "velocity")
            self.velo_glob_err = val * 0.50
    
    @property
    def velo_glob_err(self):
        """Error of global velocity in m/s, assigned to this line
        
        Raises
        ------
        AttributeError
            if current value is not of type float
        """
        if not isinstance(self._velo_glob_err, float) or\
                                        isnan(self._velo_glob_err):
            raise AttributeError("Global velocity error not assigned to line")
        return self._velo_glob_err
        
    @velo_glob_err.setter
    def velo_glob_err(self, val):
        try:
            val = float(val)
            if isnan(val):
                raise Exception
        except:
            raise ValueError("Invalid input, need float or int...")
        self._velo_glob_err = val
        
    @property
    def plume_props(self):
        """:class:`LocalPlumeProperties` object assigned to this list"""
        from pyplis import LocalPlumeProperties
        if not isinstance(self._plume_props, LocalPlumeProperties):
            raise AttributeError("Local plume properties not assigned to line")
        return self._plume_props
        
    @plume_props.setter
    def plume_props(self, val):
        from pyplis import LocalPlumeProperties
        if not isinstance(val, LocalPlumeProperties):
            raise ValueError("Invalid input, need class LocalPlumeProperties")
        self._plume_props = val
        
    def dist_other(self, other):
        """Determines the distance to another line
        
        
        Note
        ----
        
            1. The offset is applied in relative coordinates, i.e. it does not
            consider the pyramide level or ROI.
            
            #. The two lines need to be parallel
        
        Parameters
        ----------
        other : LineOnImage
            the line to which the distance is retrieved
        
        Returns
        -------
        float
            retrieved distance in pixel coordinates
            
        Raises
        ------
        ValueError
            if the two lines are not parallel
        """
        dx0, dy0 = other.x0 - self.x0, other.y0 - self.y0
        dx1, dy1 = other.x1 - self.x1, other.y1 - self.y1
        if dx1 != dx0 or dy1 != dy0:
            warn("Lines are not parallel...")
        return mean([norm([dx0, dy0]), norm([dx1, dy1])])
        
    def offset(self, pixel_num=20, line_id=None):
        """Returns a new line shifted within normal direction
        
        Note
        ----
                
            1. The offset is applied in relative coordinates, i.e. it does not
                consider the pyramide level or ROI
            
            2. The determined required displacement (dx, dy) is converted into
                integers                
                
        Parameters
        ----------
        pixel_num : int
            shift length in pixels
        line_id : str
            string ID of new line, if None (default) it is set automatically
        
        Returns
        -------
        LineOnImage
            shifted line        
            
        """
        if line_id is None:
            line_id = self.line_id + "_shifted_%dpix" %pixel_num
        dx, dy = self.normal_vector * pixel_num
        x0, x1 = self.x0 + int(dx), self.x1 + int(dx)
        y0, y1 = self.y0 + int(dy), self.y1 + int(dy)
        return LineOnImage(x0, y0, x1, y1,
                           normal_orientation=self.normal_orientation,
                           line_id=line_id,
                           color=self.color,
                           linestyle=self.linestyle,
                           pyrlevel_def=self.pyrlevel_def)
        
        
    def convert(self, to_pyrlevel=0, to_roi_abs=[0, 0, 9999, 9999]):
        """Convert to other image preparation settings"""
        if to_pyrlevel == self.pyrlevel_def and same_roi(self.roi_abs_def, 
                                                         to_roi_abs):
            print("Same shape settings, returning current line object""")
            return self
        # first convert to absolute coordinates
        ((x0, x1), 
         (y0, y1)) = map_coordinates_sub_img([self.x0, self.x1],
                                             [self.y0, self.y1],
                                             roi_abs=self._roi_abs_def,
                                             pyrlevel=self._pyrlevel_def,
                                             inverse=True)
        # now convert from absolute into specified coords
        (x0, x1), (y0, y1) = map_coordinates_sub_img([x0, x1], [y0, y1],
                                                     roi_abs=to_roi_abs,
                                                     pyrlevel=to_pyrlevel,
                                                     inverse=False)
        
        new_line = LineOnImage(x0, y0, x1, y1, roi_abs_def=to_roi_abs, 
                               pyrlevel_def=to_pyrlevel, 
                               normal_orientation=self.normal_orientation,
                               line_id=self.line_id,
                               color=self.color, linestyle=self.linestyle)
        try:
            new_line.velo_glob = self.velo_glob
        except:
            pass
        try:
            new_line.velo_glob_err = self.velo_glob_err
        except:
            pass
        try:
            new_line.plume_props = self.plume_props
        except:
            pass
                
        return new_line
        
    def check_coordinates(self):
        """Check line coordinates
        
        Checks if coordinates are in the right order and exchanges start / stop
        points if not
        
        Raises
        ------
        ValueError
            if any of the current coordinates is smaller than zero
        """
        if any([x < 0 for x in self.coords]):
            raise ValueError("Invalid value encountered, all coordinates of "
                "line must exceed zero, current coords: %s" %self.coords)
        if self.start[0] > self.stop[0]:
            print ("x coordinate of start point is larger than of stop point: "
                    "start and stop will be exchanged")
            self.start, self.stop = self.stop, self.start
     
    def in_image(self, img_array):
        """Check if this line is within the coordinates of an image array
        
        Parameters        
        ----------
        img_array : array
            image data
            
        Returns
        -------
        bool
            True if point is in image, False if not
        """
        if not all(self.point_in_image(p, img_array)\
                                    for p in [self.start, self.stop]):
            return False
        return True
        
    def point_in_image(self, x, y, img_array):
        """Check if a given coordinate is within image 
        
        Parameters
        ----------
        x : int
            x coordinate of point
        y : int
            y coordinate of point
        img_array : array
            image data
        
        Returns
        -------
        bool
            True if point is in image, False if not
        
        """
        h, w = img_array.shape[:2]
        if not 0 < x < w:
            print "x coordinate out of image"
            return False
        if not 0 < y < h:
            print "y coordinate out of image"
            return False
        return True
    
    def get_roi_abs_coords(self, img_array, add_left=5, add_right=5,
                           add_bottom=5, add_top=5):
        """Get a rectangular ROI covering this line 
        
        Parameters
        ----------
        add_left : int
            expand range to left of line (in pix)
        add_right : int
            expand range to right of line (in pix)
        add_bottom : int
            expand range to bottom of line (in pix)
        add_top : int
            expand range to top of line (in pix)
        
        Returns
        -------
        list
            ROI around this line
        
        """
        
        x0, x1 = self.start[0] - add_left, self.stop[0] + add_right
        #y start must not be smaller than y stop
        y_arr = [self.start[1], self.stop[1]]
        y_min, y_max = min(y_arr), max(y_arr)
        y0, y1 = y_min - add_top, y_max + add_bottom
        roi = self.check_roi_borders([x0, y0, x1, y1], img_array)
        roi_abs = map_roi(roi, pyrlevel_rel=-self.pyrlevel_def)
        self._line_roi_abs= roi_abs
        return roi_abs
    
    def _roi_from_rot_rect(self):
        """Set current ROI from current rotated rectangle coords"""
        r = self._rect_roi_rot
        xc = asarray([x[0] for x in r])
        xc[xc < 0] = 0
        yc = asarray([x[1] for x in r])
        yc[yc < 0] = 0
        roi = [xc.min(), yc.min(), xc.max(), yc.max()]
        self._line_roi_abs = map_roi(roi, pyrlevel_rel=-self.pyrlevel_def)
        return roi
    
    def set_rect_roi_rot(self, depth=None):
        """Get rectangle for rotated ROI based on current tilting
        
        Note
        ----
        This function also changes the current ``roi_abs`` attribute
        
        Parameters
        ----------
        depth : int
            depth of rotated ROI (in normal direction of line) in pixels
            
        Returns
        -------
        list 
            rectangle coordinates
        """
        dx, dy = self._delx_dely()
        if depth is None:
            depth  = norm((dx, dy)) * 0.10
                
        n = self.normal_vecs[1]
        dx0, dy0 = n * depth / 2.0
        
#==============================================================================
#     
#         if sign(dx0) == sign(dy0):
#             dx0 = -dx0
#             dy0 = -dy0
#==============================================================================
        x0 = self.x0 + int(dx0)    
        y0 = self.y0 + int(dy0)
        offs = array([x0, y0])
        
        w = self.length()
        r = array([(0, 0), (w, 0), (w, depth), (0, depth), (0, 0)])

        dx, dy = self._delx_dely()
        try:
            theta = arctan(dy / dx)
        except ZeroDivisionError:
            theta = pi / 2
        #rotation matrix (account for neg. y direction)
        m_rot = array([[cos(theta), sin(theta)],
                        [-sin(theta), cos(theta)]])
        r = dot(r, m_rot) + offs
        self._rect_roi_rot = r
        self._roi_from_rot_rect()
        return r
        
#==============================================================================
#     def set_rect_roi_rot_v0(self, depth=None):
#         """Get rectangle for rotated ROI based on current tilting
#         
#         Note
#         ----
#         This function also changes the current ``roi_abs`` attribute
#         
#         Parameters
#         ----------
#         depth : int
#             depth of rotated ROI (in normal direction of line) in pixels
#             
#         Returns
#         -------
#         list 
#             rectangle coordinates
#         """
#         if depth is None:
#             depth  = self.length() * 0.10
#         dx0, dy0 = self.normal_vector * depth / 2.0
#     
#         if sign(dx0) == sign(dy0):
#             dx0 = -dx0
#             dy0 = -dy0
#         x0 = self.x0 + int(dx0)    
#         y0 = self.y0 + int(dy0)
#         offs = array([x0, y0])
#         
#         w = self.length()
#         r = array([(0, 0), (w, 0), (w, depth), (0, depth), (0, 0)])
# 
#         dx, dy = self._delx_dely()
#         try:
#             theta = arctan(dy / dx)
#         except ZeroDivisionError:
#             theta = pi / 2
#         #rotation matrix (account for neg. y direction)
#         m_rot = array([[cos(theta), sin(theta)],
#                         [-sin(theta), cos(theta)]])
#         r = dot(r, m_rot) + offs
#         self._rect_roi_rot = r
#         self._roi_from_rot_rect()
#         return r
#==============================================================================
    
    
    def get_rotated_roi_mask(self, shape):
        """Returns pixel access mask for rotated ROI
        
        Parameters
        ----------
        shape : tuple
            shape of image for which the mask is supposed to be used
            
        Returns
        -------
        array
            bool array that can be used to access pixels within the ROI
        """
        try:
            if not self._last_rot_roi_mask.shape == shape:
                raise Exception
            mask = self._last_rot_roi_mask
        except:
            mask = zeros(shape)
            rect = self.rect_roi_rot
            poly = array([rect], dtype=int32)
            fillPoly(mask, poly, 1)
            mask = mask.astype(bool)
        self._last_rot_roi_mask = mask
        return mask
    
    def check_roi_borders(self, roi, img_array):
        """ Check if all points of ROI are within image borders
        
        Parameters
        ----------
        roi : list 
            ROI rectangle ``[x0,y0,x1,y1]``
        img_array : array
            exemplary image data for which the ROI is checked
            
        Returns
        -------
        list        
            roi within image coordinates (unchanged, if input is ok, else image 
            borders)
        """
        x0, y0 = roi[0], roi[1]
        x1, y1 = roi[2], roi[3]
        h, w = img_array.shape
        if not x0 >= 0:
            x0 = 1
        if not x1 < w:
            x1 = w - 2
        if not y0 >= 0:
            y0 = 1
        if not y1 < h:
            y1 = h - 2
        return [x0, y0, x1, y1]
        
    def prepare_coords(self):
        """Prepare the analysis mesh
        
        Note
        ----
        
        The number of analysis points on this object correponds to the physical 
        length of this line in pixel coordinates. 
        """
        length = self.length()
        x0, y0 = self.start     
        x1, y1 = self.stop

        x = linspace(x0, x1, length)
        y = linspace(y0, y1, length)
        self.profile_coords = vstack((y, x))
        self.det_normal_vecs()
        self.set_rect_roi_rot()
        
    def length(self):
        """Determine the length in pixel coordinates"""
        return int(round(hypot(*self._delx_dely())))
        
    def get_line_profile(self, array):
        """Retrieve the line profile along pixels in input array
         
        Parameters
        ----------
        array : array
            2D data array (e.g. image data). Color images are converted into
            gray scale using :func:`cv2.cvtColor`.
        
        Returns
        -------
        array
            profile
            
        """
        try:
            array = array.img #if input is Img object
        except:
            pass
        if ndim(array) != 2:
            if ndim(array) != 3:
                print ("Error retrieving line profile, invalid dimension of "  
                        "input array: %s" %(ndim(array)))
                return
            if array.shape[2] != 3:
                print ("Error retrieving line profile, invalid dimension of "
                         "input array: %s" %(ndim(array)))
                return
            "Input in BGR, conversion into gray image"
            array = cvtColor(array, COLOR_BGR2GRAY)

        # Extract the values along the line, using interpolation
        zi = map_coordinates(array, self.profile_coords)
        return zi
        
    """Plotting / visualisation etc...
    """
    def plot_line_on_grid(self, img_arr=None, ax=None, include_normal=False,
                          include_roi_rot=False, include_roi=False, 
                          annotate_normal=False, **kwargs):
        """Draw this line on the image
        
        Parameters
        ----------
        
        img_arr : ndarray 
            if specified, the array is plotted using :func:`imshow` and onto 
            that axes, the line is drawn
        ax : 
            matplotlib axes object. Is created if unspecified. Leave 
            :param:`img_arr` empty if you want the line to be drawn onto an
            already existing image (plotted in ax)
        include_normal : bool
            if True, the line normal vector is drawn
        include_roi_rot : bool
            if True, a line-orientation specific ROI is drawn 
        include_roi : bool
            if True, an ROI is drawn which spans the i,j range of the image
            covered by the line
        annotate_normal : bool
            if True, the normal vector is annotated (only if include_normal is
            set True)
        **kwargs : 
            additional keyword arguments for plotting of line (please use 
            following keys: marker for marker style, mec for marker 
            edge color, c for line color and ls for line style)
            
        Returns
        -------
        Axes
            matplotlib axes instance
            
        """
        new_ax = False
        keys = kwargs.keys()
        if not "mec" in keys:
            kwargs["mec"] = "none"
        if not "color" in keys:
            kwargs["color"] = self.color
        if not "ls" in keys:
            kwargs["ls"] = self.linestyle
        if not "marker" in keys:
            kwargs["marker"] = "o"
        if not "label" in keys:
            kwargs["label"] = self.line_id
        if ax is None:
            new_ax = True
            ax = subplot(111)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        c = kwargs["color"]
        if img_arr is not None:
            ax.imshow(img_arr, cmap = "gray")
        p = ax.plot([self.start[0],self.stop[0]], [self.start[1],
                     self.stop[1]], **kwargs)
        if img_arr is not None:
            ax.set_xlim([0,img_arr.shape[1]])
            ax.set_ylim([img_arr.shape[0],0])
        if include_normal:
            mag = self.norm * 0.06 #3 % of length of line
            n = self.normal_vector * mag
            xm, ym = self.center_pix
            epx, epy = n[0], n[1]
            c = p[0].get_color()
            
            ax.arrow(xm, ym, epx, epy, head_width=mag/2, head_length=mag,
                     fc=c, ec=c)
            if annotate_normal:
                ax.text(xm + epx*2, ym + epy*3, r'$\hat{n}$', 
                        color=c,
                        fontweight='bold',
                        fontsize=18)
                
        if include_roi:
            x0, y0, w, h = roi2rect(self.roi)
            ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=c))
        if include_roi_rot:
            self.plot_rotated_roi(color=c, ax=ax)
        #axis('image')
        if new_ax:
            ax.set_title("Line " + str(self.line_id))
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        draw()
        return ax
    
    def plot_rotated_roi(self, color=None, ax=None):
        """Plot current rotated ROI into axes
        
        Parameters
        ----------
        color
            optional, color information. If None (default) then the current 
            line color is used
        ax : :obj:`Axes`, optional
            matplotlib axes object, if None, a figure with one subplot will
            be created
        
        Returns
        -------
        Axes
            axes instance
        """     
        if ax is None:
            ax = subplot(111)
        if color is None:
            color = self.color
        r = self.rect_roi_rot
        p = Polygon(r, fc=color, alpha=0.2)
        ax.add_patch(p)
        return ax
        
    def plot_line_profile(self, img_arr, ax=None):
        """Plots the line profile"""
        if ax is None:
            ax=subplot(111)
        p = self.get_line_profile(img_arr)
        ax.set_xlim([0,self.length()])
        ax.grid()
        ax.plot(p, label=self.line_id)
        ax.set_title("Profile")
        return ax
    
    def plot(self, img_arr):
        """Creates two subplots showing line on image and corresponding profile
        
        Parameters
        ----------
        img_arr : array
            the image data
        
        Returns
        -------
        Figure
            figure containing the supblots
            
        """
        fig, axes = subplots(1,2)
        self.plot_line_on_grid(img_arr,axes[0])
        self.plot_line_profile(img_arr,axes[1])
        tight_layout()
        return fig
        
    def _delx_dely(self):
        """Length of x and y range covered by line"""
        return float(self.x1) - float(self.x0), float(self.y1) - float(self.y0)
    
    @property
    def norm(self):
        """Return length of line in pixels"""
        dx, dy = self._delx_dely()
        return norm([dx, dy])
    
    def det_normal_vecs(self):
        """Get both normal vectors"""
        dx, dy = self._delx_dely()
        v1, v2 = array([-dy, dx]), array([dy, -dx])
        self.normal_vecs = [v1 / norm(v1), v2 / norm(v2)]
        return self.normal_vecs
            
    @property
    def normal_vector(self):
        """Get normal vector corresponding to current orientation setting"""
        return self.normal_vecs[self._dir_idx[self.normal_orientation]]
        
    @property
    def complex_normal(self):
        """Return current normal vector as complex number"""
        dx, dy = self.normal_vector
        return complex(-dy, dx)
    
    @property
    def normal_theta(self):
        """Returns orientation of normal vector in degrees
        
        The angles correspond to:
        
            1. 0    =>  to the top (neg. y direction)
            2. 90   =>  to the right (pos. x direction)
            3. 180  =>  to the bottom (pos. y direction)
            4. 270  =>  to the left (neg. x direction)
        """
        return angle(self.complex_normal, True)%360
        
    def to_list(self):
        """Returns line coordinates as 4-list"""
        return [self.x0, self.y0, self.x1, self.y1]
        
    def to_dict(self):
        """Writes relevant parameters to dictionary"""
        return {"class"                 :   "LineOnImage",
                "line_id"               :   self.line_id,
                "color"                 :   self.color,
                "linestyle"             :   self.linestyle,
                "x0"                    :   self.x0,
                "y0"                    :   self.y0,
                "x1"                    :   self.x1, 
                "y1"                    :   self.y1,
                "_normal_orientation"   :   self._normal_orientation,
                "_pyrlevel_def"         :   self._pyrlevel_def, 
                "_roi_abs_def"          :   self._roi_abs_def}
    
    def from_dict(self, settings_dict):
        """Load line parameters from dictionary
        
        Parameters
        ----------
        settings_dict : dict
            dictionary containing line parameters (cf. :func:`to_dict`)
        """
        for k, v in settings_dict.iteritems():
            if self.__dict__.has_key(k):
                self.__dict__[k] = v
        
        self.check_coordinates()
        self.prepare_coords()
    
    @property
    def orientation_info(self):
        """Returns string about orientation of line and normal"""
        dx, dy = self._delx_dely()
        s = "delx, dely = %s, %s\n" %(dx, dy)
        s += "normal orientation: %s\n" %self.normal_orientation
        s += "normal vector: %s\n" %self.normal_vector
        s += "Theta normal: %s\n" %self.normal_theta
        return s
        
    """Magic methods
    """
    def __str__(self):
        """String representation"""
        s = ("Line %s: [%d, %d, %d, %d], @pyrlevel %d, @ROI: %s" 
             %(self.line_id, self.x0, self.y0, self.x1, self.y1, 
               self.pyrlevel_def, self.roi_abs_def))
        return s

class ProfileTimeSeriesImg(Img):
    """Image representing time series of line profiles
    
    The y axis of the profile image corresponds to the actual profiles 
    (retrieved from the individual images) and the x axis corresponds to the 
    image time axis (i.e. the individual frames). Time stamps (mapping of 
    x indices) can also be stored in this object.
    
    Example usage is, for instance to represent ICA time series retrieved
    along a profile (e.g. using :class:`LineOnImage`) for plume speed cross 
    correlation
    """
    def __init__(self, img_data=None, time_stamps=asarray([]), img_id="",
                 dtype=float32, profile_info_dict={}, **meta_info):
        self.img_id = img_id
        self.time_stamps = asarray(time_stamps)
        self.profile_info = {}
        if isinstance(profile_info_dict, dict):
            self.profile_info = profile_info_dict
        #Initiate object as Img object
        super(ProfileTimeSeriesImg, self).__init__(input=img_data,
                                                   dtype=dtype, **meta_info)
                                                
    @property
    def img(self):
        """Get / set image data"""
        return self._img
    
    @img.setter
    def img(self, val):
        """Setter for image data"""
        if not isinstance(val, ndarray) or val.ndim != 2:
            raise ValueError("Could not set image data, need 2 dimensional"
                " numpy array as input")
        self._img = val
        num = val.shape[1]
        if not len(self.time_stamps) == num:
            self.time_stamps = asarray([datetime(1900,1,1)] * num)
        
    def _format_check(self):
        """Checks if current data is of right format"""
        if not all([isinstance(x, ndarray) for x in [self._img,\
                                                self.time_stamps]]):
            raise TypeError("self.img and self.time_stamps must be numpy "
                "arrays")
        if not len(self.time_stamps) == self.shape[1]:
            raise ValueError("Mismatch in array lengths")

    @property
    def start(self):
        """Returns first datetime from ``self.time_stamps``"""
        try:
            return self.time_stamps[0]
        except:
            print "no time information available, return 1/1/1900"
            return datetime(1900,1,1)
    
    @property
    def stop(self):
        """Returns first datetime from ``self.time_stamps``"""
        try:
            return self.time_stamps[-1]
        except:
            print "no time information available, return 1/1/1900"
            return datetime(1900,1,1)
              
    def save_as_fits(self, save_dir=None, save_name=None):
        """Save stack as FITS file"""
        self._format_check()
        save_dir = abspath(save_dir) #returns abspath of current wkdir if None
        if not isdir(save_dir): #save_dir is a file path
            save_name = basename(save_dir)
            save_dir = dirname(save_dir)
        if save_name is None:
            save_name = "pyplis_profile_tseries_id_%s_%s_%s_%s.fts"\
                %(self.img_id, self.start.strftime("%Y%m%d"),\
                self.start.strftime("%H%M"), self.stop.strftime("%H%M"))
        else:
            save_name = save_name.split(".")[0] + ".fts"
    
        hdu = fits.PrimaryHDU()
        time_strings = [x.strftime("%Y%m%d%H%M%S%f") for x in self.time_stamps]
        col1 = fits.Column(name = "time_stamps", format = "25A", array =\
            time_strings)
    
        cols = fits.ColDefs([col1])
        arrays = fits.BinTableHDU.from_columns(cols)
        
        hdu.data = self._img
        hdu.header.update(self.edit_log)
        hdu.header["img_id"] = self.img_id
        for key, val in self.profile_info.iteritems():
            if key == "_roi_abs_def":
                try:
                    hdu.header["_roi_abs_def"] = dumps(val)
                except:
                    warn("Failed to write roi_abs_def")
            else:
                hdu.header[key] = val
    
        hdu.header.append()
        hdulist = fits.HDUList([hdu, arrays])
        path = join(save_dir, save_name)
        if exists(path):
            try:
                print "Image already exists at %s and will be overwritten" %path
                remove(path)
            except:
                warn("Failed to delete existing file...")
        try:
            hdulist.writeto(path)
        except:
            warn("Failed to save FITS File (check previous warnings)")
    
    def _profile_dict_keys(self, profile_type = "LineOnImage"):
        """Returns profile dictionary keys for input profile type"""
        d = {"LineOnImage"  :   LineOnImage().to_dict().keys()}
        return d[profile_type]
        
    def load_fits(self, file_path, profile_type="LineOnImage"):
        """Load stack object (fits)
        
        :param str file_path: file path of fits image
        """
        
        if not exists(file_path):
            raise IOError("Img could not be loaded, path %s does not "
                          "exist" %file_path)
        hdu = fits.open(file_path)
        self.img = asarray(hdu[0].data)
        prep = Img().edit_log
        try:
            profile_keys = self._profile_dict_keys(profile_type)
        except:
            profile_keys = []
            print "Failed to load profile info dictionary"
        
        for key, val in hdu[0].header.iteritems():
            k = key.lower()
            if k in prep.keys():
                self.edit_log[k] = val
            elif k in profile_keys:
                if k == "_roi_abs_def":
                    self.profile_info[k] = loads(val)
                else:
                    self.profile_info[k] = val
        self.img_id = hdu[0].header["img_id"]
        
        try:
            self.time_stamps = asarray([datetime.strptime(x, "%Y%m%d%H%M%S%f")\
                for x in hdu[1].data["time_stamps"]])
        except:
            print "Failed to import time stamps"
        self._format_check()
        
class ImgStack(object):
    """Image stack object 
    
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
                 stack_id="", img_prep=None, camera=None, **stack_data):
        self.stack_id = stack_id
        self.dtype = dtype
        self.current_index = 0
        
        self.stack = None
        self.start_acq = None
        self.texps = None
        self.add_data = None
        self._access_mask = None
        
        if img_prep is None:
            img_prep = {"pyrlevel"  :   0}
        self.img_prep = img_prep 
        
        self.roi_abs = [0, 0, 9999, 9999]
        
        self._cam = Camera()
        
        self.init_stack_array(height, width, img_num)
        if stack_data.has_key("stack"):
            self.set_stack_data(**stack_data)
        
        if isinstance(camera, Camera):
            self.camera = camera
       
    def init_stack_array(self, height=0, width=0, img_num=0):
        """Initiate the actual stack data array
        
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
            number of images to be stacked (can be zero, then, whenever an 
            image is added to stack, the method :func:`append_img` is used)
        """
        try:
            self.stack = empty((int(img_num), int(height), int(width))).\
                astype(self.dtype)
        except MemoryError:
            raise MemoryError("Could not initiate empty 3D numpy array "
                "(d, h, w): (%s, %s, %s)" %(img_num, height, width))
        self.start_acq = asarray([datetime(1900,1,1)] * img_num)
        self.texps = zeros(img_num, dtype=float32)
        self.add_data = zeros(img_num, dtype=float32)
        
        self._access_mask = zeros(img_num, dtype=bool)
        self.current_index = 0
    
    @property
    def last_index(self):
        """Returns last index"""
        return self.num_of_imgs - 1
    
    @property
    def start(self):
        """Returns start time stamp of first image"""
        try:
            i = self.start_acq[self._access_mask][0]
            add = timedelta(self.texps[self._access_mask][0] / 86400.)
            return i + add
        except IndexError:
            raise IndexError("Stack is empty...")
        except:
            raise ValueError("Start acquisition time could accessed in stack")
                
    @property
    def stop(self):
        """Returns start time stamp of first image"""
        try:
            i = self.start_acq[self._access_mask][-1]
            add = timedelta(self.texps[self._access_mask][-1] / 86400.)
            return i + add
        except IndexError:
            raise IndexError("Stack is empty...")
        except:
            raise ValueError("Start acquisition time could accessed in stack")
        
    @property
    def time_stamps(self):
        """Acq. time stamps of all images"""
        try:
            dts = ([timedelta(x /(2 * 86400.)) for x in self.texps])
            return self.start_acq + asarray(dts)
        except:
            raise ValueError("Failed to access information about acquisition "
                "time stamps and / or exposure times")
    
    @property
    def pyrlevel(self):
        """Gauss pyramid level of images in stack"""
        return self.img_prep["pyrlevel"]
    
    @property
    def camera(self):
        """Camera object assigned to stack"""
        return self._cam
    
    @camera.setter
    def camera(self, value):
        if isinstance(value, Camera):
            self._cam = value
        else:
            raise TypeError("Need Camera object...")

    @property
    def num_of_imgs(self):
        """Depth of stack"""
        return self.stack.shape[0]
     
    def check_index(self, idx=0):
        if 0 <= idx <= self.last_index:
            return
        elif idx == self.num_of_imgs:
            self._extend_stack_array()
        else:
            raise IndexError("Invalid index %d for inserting image in stack "
                             "with current depth %d" %(idx, self.num_of_imgs))
    
    def _extend_stack_array(self):
        """Extend the first index of the stack array"""
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
        """Insert an image into the stack at provided index
        
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
        except:
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
        """Add image at current index position
        
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
#==============================================================================
#         if self.current_index >= self.last_index:
#             print self.last_index
#             raise IndexError("Last stack index reached...")
#==============================================================================
        self.insert_img(self.current_index, img_arr, start_acq, texp, add_data)
        self.current_index += 1
        
    def make_circular_access_mask(self, cx, cy, radius):
        """Create a circular mask for stack 
        
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
        #cx, cy = self.img_prep.map_coordinates(pos_x_abs, pos_y_abs)
        h, w = self.stack.shape[1:]
        y, x = ogrid[:h, :w]
        m = (x - cx)**2 + (y - cy)**2 < radius**2
        return m
        
    def set_stack_data(self, stack, start_acq=None, texps=None):
        """Sets the current data based on input
        
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
        """Get stack data (containing of stack, acq. and exp. times) 
        
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
        """Convolves the stack data with a input mask along time axis
        
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
        #mask_norm = boolMask.astype(float32)/sum(boolMask)
        d = self.get_data()
        data_conv = (d[0] * mask.astype(float32))#[:, :, newaxis])#, d[1], d[2])
        return (data_conv, d[1], d[2])
    
    def get_time_series(self, pos_x=None, pos_y=None, radius=1, mask=None):
        """Get time series in a ROI
        
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
        except:
            if not radius > 0:
                raise ValueError("Invalid input for param radius (3. pos): "
                    "value must be larger than 0, got %d" %radius)
            if radius == 1:
                mask = zeros(self.shape[1:]).astype(bool)
                mask[pos_y, pos_x] = True
                s = Series(d[0][self._access_mask, pos_y, pos_x], d[1])
                return s, mask
            mask = self.make_circular_access_mask(pos_x, pos_y, radius)
            data_mask, start_acq, texps = self.apply_mask(mask)
        values = data_mask.sum((1, 2)) / float(sum(mask))
        return Series(values, start_acq), mask
    
    def merge_with_time_series(self, time_series, method="average", 
                               **kwargs):
        """High level wrapper for data merging
        
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
        if not isinstance(time_series, Series):
            raise TypeError("Could not merge stack data with input data: "
                "wrong type: %s" %type(time_series))
        
        if method == "average":
            try:
                return self._merge_tseries_average(time_series, **kwargs)
            except:
                print ("Failed to merge data using method average, trying "
                       "method nearest instead")
                method = "nearest"
        if method == "nearest":
            return self._merge_tseries_nearest(time_series, **kwargs)
        elif method == "interpolation":
            return self._merge_tseries_cross_interpolation(time_series,
                                                           **kwargs)
        else:
            raise TypeError("Unkown merge type: %s. Choose from "
                    "[nearest, average, interpolation]")
        
    def _merge_tseries_nearest(self, time_series):
        """Find nearest in time image for each time stamp in input series
        
        Find indices (and time differences) in input time series of nearest 
        data point for each image in this stack. Then, get rid of all indices
        showing double occurences using time delta information. 
        
            
        """
        stack, time_stamps, texps = self.get_data()
        nearest_idxs, del_ts = self.get_nearest_indices(time_series.index)
        img_idxs = []
        spec_idxs_final = []
        del_ts_abs = []
        for idx in range(min(nearest_idxs),max(nearest_idxs) + 1):
            print "Current tseries index %s" %idx
            matches =  where(nearest_idxs == idx)[0]
            if len(matches) > 0:
                del_ts_temp = del_ts[matches]
                spec_idxs_final.append(idx)
                del_ts_abs.append(min(del_ts_temp))
                img_idxs.append(matches[argmin(del_ts_temp)])
        
        series_new = time_series[spec_idxs_final]
        try:
            series_new.fit_errs = time_series.fit_errs[spec_idxs_final]
        except:
            pass
        stack_new = self.stack[img_idxs]
        texps_new = asarray(self.texps[img_idxs])
        start_acq_new = asarray(self.start_acq[img_idxs])
        stack_obj_new = ImgStack(stack_id=self.stack_id + "_merged_nearest",
                                 img_prep=self.img_prep, stack=stack_new,
                                 start_acq=start_acq_new, texps=texps_new)
        stack_obj_new.roi_abs = self.roi_abs
        stack_obj_new.add_data = series_new
        return (stack_obj_new, series_new)
            
    def _merge_tseries_cross_interpolation(self, time_series,
                                           itp_type="linear"):
        """Merge this stack with input data using interpolation
        
        :param Series time_series_data: pandas Series object containing time 
            series data (e.g. DOAS column densities)
        :param str itp_type: interpolation type (passed to 
            :class:`pandas.DataFrame` which does the interpolation, default is
            linear)
            
        """
        h, w = self.shape[1:]
        
        stack, time_stamps, _ = self.get_data()
        
        #first crop time series data based on start / stop time stamps
        time_series = self.crop_other_tseries(time_series)
        time_series.name =  None
        if not len(time_series) > 0:
            raise IndexError("Time merging failed, data does not overlap")
        
        #interpolate exposure times
        s0 = Series(self.texps, time_stamps)
        try:
            errs = Series(time_series.fit_errs, time_series.index)
            df0 = concat([s0, time_series, errs], axis=1).\
                                interpolate(itp_type).dropna()
        except:
            df0 = concat([s0, time_series], axis=1).\
                                interpolate(itp_type).dropna()
        new_num = len(df0[0])
        if not new_num >= self.num_of_imgs:
            raise ValueError("Unexpected error, length of merged data "
                             "array does not exceed length of inital image "
                             "stack...")
        #create new arrays for the merged stack
        new_stack = empty((new_num, h, w))
        new_acq_times = df0[0].index
        new_texps = df0[0].values
        
        
        for i in range(h):
            for j in range(w):
                print ("Stack interpolation active...: current img row (y):"
                                                           "%s (%s)" %(i, j))
                #get series from stack at current pixel
                series_stack = Series(stack[:, i, j], time_stamps)
                #create a dataframe
                df = concat([series_stack, df0[1]], axis=1).\
                    interpolate(itp_type).dropna()
                #throw all N/A values
                #df = df.dropna()
                new_stack[:, i, j] = df[0].values
        
        stack_obj = ImgStack(new_num, h, w, 
                             stack_id=self.stack_id+"_interpolated", 
                             img_prep=self.img_prep)
        stack_obj.roi_abs = self.roi_abs
        #print new_stack.shape, new_acq_times.shape, new_texps.shape
        stack_obj.set_stack_data(new_stack, new_acq_times, new_texps)
        
        new_series = df[1]
        try:
            new_series.fit_errs = df0[2].values
        except:
            print "Failed to access / process errors on time series data"
        return (stack_obj, new_series)
        
        
    def _merge_tseries_average(self, time_series):
        """Make new stack of averaged images based on input start / stop arrays
        
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
            - :class:`ImgStack`: new stack object with averaged images
            - :obj:`list`: list of bad indices (where no overlap was found)
        """
        try:
            if not time_series.has_start_stop_acqtamps():
                raise ValueError("No start / stop acquisition time stamps "
                                 "available in input data...")
            start_acq = asarray(time_series.start_acq)
            stop_acq = asarray(time_series.stop_acq)
        except:
            raise 
        
        stack, times, texps = self.get_data()
        h, w = stack.shape[1:]
        num = len(start_acq)
        
        #new_stack = empty((h, w, self.num_of_imgs))
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
#==============================================================================
#                 print ("Found %s images for spectrum #%s (of %s)" 
#                                                 %(sum(cond), k, num))
#==============================================================================
                im = stack[cond].mean(axis = 0)
                if counter == 0:
                    new_stack = im
                else:
                    new_stack = dstack((new_stack, im))
                new_acq_times.append(i + (f - i)/2)
                #img_avg_info.append(sum(cond))
                new_texps.append(texp)
                counter += 1
            else:
                bad_indices.append(k)
        new_stack = rollaxis(new_stack, 2)
        stack_obj = ImgStack(len(new_texps), h, w, 
                             stack_id=self.stack_id+"_avg", 
                             img_prep=self.img_prep)
        stack_obj.roi_abs = self.roi_abs
        stack_obj.set_stack_data(new_stack, asarray(new_acq_times), 
                                 asarray(new_texps))
        
        tseries = time_series.drop(time_series.index[bad_indices])
        try:
            errs = delete(time_series.fit_errs, bad_indices)
            tseries.fit_errs = errs
        except:
            pass
        return (stack_obj, tseries)
    
    """Helpers
    """
    def crop_other_tseries(self, time_series):
        """Crops other time series object based on start / stop time stamps"""
#==============================================================================
#         start = self.start - self.total_time_period_in_seconds() * tol_borders
#         stop = self.stop + self.total_time_period_in_seconds() * tol_borders
#==============================================================================
        cond = logical_and(time_series.index >= self.start,
                           time_series.index <= self.stop)
        new = time_series[cond]
        try:
            new.fit_errs = new.fit_errs[cond]
        except:
            pass
        return new
        
    def total_time_period_in_seconds(self):
        """Returns start time stamp of first image"""
        return (self.stop - self.start).total_seconds()
        
    def get_nearest_indices(self, tstamps_other):
        """Find indices of time stamps nearest to img acq. time stamps
        
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
            
    def get_nearest_img(self, time_stamp):
        """Returns stack image which is nearest to input timestamp
        
        Searches the nearest image(s) with respect to input datetime(s)
        
        :param (datetime, ndarray) time_stamps: the actual time stamp(s) (for 
            instance from another time series object)
        """
        raise NotImplementedError
        
    def has_data(self):
        """Returns bool"""
        return bool(sum(self._access_mask))
     
    def sum(self, *args, **kwargs):
        """Sum over all pixels of stack
        
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
        """Applies numpy.mean function to stack data
        
        :param *args: non keyword arguments passed to :func:`numpy.mean` 
            applied to stack data
        :param **kwargs: keyword arguments passed to :func:`numpy.mean` 
            applied to stack data
        """
        return self.stack.mean(*args, **kwargs)

    def std(self, *args, **kwargs):
        """Applies numpy.std function to stack data
        
        :param *args: non keyword arguments passed to :func:`numpy.std` 
            applied to stack data
        :param **kwargs: keyword arguments passed to :func:`numpy.std` 
            applied to stack data
        """
        return self.stack.std(*args, **kwargs)

    @property
    def shape(self):
        """Returns stack shape"""
        return self.stack.shape
    
    @property
    def ndim(self):
        """Return stack dimension"""
        return self.stack.ndim   
        
    """Plots / visualisation"""
    def show_img(self, index=0):
        """Show image at input index
        
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
        """Reduce the stack image size using gaussian pyramid 
             
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
        """Increasing the image size using gaussian pyramide 
        
        
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
        """Down / upscale image to a given pyramide level"""
        steps = final_state - self.img_prep["pyrlevel"]
        if steps > 0:
            return self.pyr_down(steps)
        elif steps < 0:
            return self.pyr_up(-steps)
    
    def duplicate(self):
        """Returns deepcopy of this object"""
        return deepcopy(self)
        
    def _format_check(self):
        """Checks if all relevant data arrays have the same length"""
        if not all([len(x) == self.num_of_imgs for x in [self.add_data,
                                                         self.texps, 
                                                         self._access_mask, 
                                                         self.start_acq]]):
            raise ValueError("Mismatch in array lengths of stack data, check"
                "add_data, texps, start_acq, _access_mask")
    
    def load_stack_fits(self, file_path):
        """Load stack object (fits)
        
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
        self.set_stack_data(hdu[0].data.byteswap().newbyteorder().\
                            astype(self.dtype))
        prep = Img().edit_log
        for key, val in hdu[0].header.iteritems():
            if key.lower() in prep.keys():
                self.img_prep[key.lower()] = val
        self.stack_id = hdu[0].header["stack_id"]
        try:
            times = hdu[1].data["start_acq"].byteswap().newbyteorder()
            self.start_acq = asarray([datetime.strptime(x, "%Y%m%d%H%M%S%f") 
                              for x in times])
        except:
            warn("Failed to import acquisition times")
        try:
            self.texps = asarray(hdu[1].data["texps"].byteswap().newbyteorder())
        except:
            warn("Failed to import exposure times")
        try:
            self._access_mask = asarray(hdu[1].data["_access_mask"].\
                                        byteswap().newbyteorder())
        except:
            warn("Failed to import data access mask")    
        try:
            self.add_data = asarray(hdu[1].data["add_data"].byteswap().\
                            newbyteorder())
        except:
            warn("Failed to import data additional data")
        self.roi_abs = hdu[2].data["roi_abs"].byteswap().\
                        newbyteorder()
        self._format_check()
        
    def save_as_fits(self, save_dir=None, save_name=None):
        """Save stack as FITS file"""
        self._format_check()
        save_dir = abspath(save_dir) #returns abspath of current wkdir if None
        if not isdir(save_dir): #save_dir is a file path
            save_name = basename(save_dir)
            save_dir = dirname(save_dir)
        if save_name is None:
            save_name = ("pyplis_imgstack_id_%s_%s_%s_%s.fts" 
                            %(self.stack_id,
                              self.start.strftime("%Y%m%d"),
                              self.start.strftime("%H%M"),
                              self.stop.strftime("%H%M")))
        else:
            save_name = save_name.split(".")[0] + ".fts"
        print "DIR: %s" %save_dir
        print "Name: %s" %save_name
        hdu = fits.PrimaryHDU()
        start_acq_str = [x.strftime("%Y%m%d%H%M%S%f") for x in self.start_acq]
        col1 = fits.Column(name="start_acq", format="25A", array=start_acq_str)
        col2 = fits.Column(name="texps", format="D", array=self.texps)
        col3 = fits.Column(name="_access_mask", format="L", 
                           array=self._access_mask)
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
            try:
                print ("Stack already exists at %s and will be overwritten" 
                       %path)
                remove(path)
            except:
                warn("Failed to delete existing file...")
        try:
            hdulist.writeto(path)
        except:
            warn("Failed to save stack to FITS File "
                 "(check previous warnings)")
        
    """Magic methods"""
    def __str__(self):
        raise NotImplementedError
        
    def __sub__(self, other):
        """Subtract data
        
        :param other: data to be subtracted object (e.g. offband stack)
        
        """
        new = self.duplicate()
        try:
            new.stack = self.stack - other.stack
            new.stack_id = "%s - %s" %(self.stack_id, other.stack_id)
        except:
            new.stack = self.stack - other
            new.stack_id = "%s - %s" %(self.stack_id, other)
        return new
        
def model_dark_image(img, dark, offset):
    """Model a dark image for input image based on dark and offset images
    
    Determine a modified dark image (D_mod) from the current dark and 
    offset images. The dark image is determined based on the image 
    exposure time of the image object to be corrected (t_exp,I). 
    D_mod represents dark and offset signal for this image object and 
    is then subtracted from the image data.

    Formula for modified dark image:

    .. math::

        D_{mod} = O + \\frac{(D - O)*t_{exp,I}}{(t_{exp, D}-t_{exp, O})} 
        
    :param Img img: the image for which dark and offset is modelled
    :param Img dark: dark image object (dark with long(est) exposure time)
    :param Img offset: offset image (dark with short(est) exposure time)
    :returns: - :class:`Img`, modelled dark image 
 
    """
    if not all([x.meta["texp"] > 0.0 for x in [img, dark, offset]]):
        raise ImgMetaError("Could not model dark image, invalid value for "
            "exposure time encountered for at least one of the input images")
    if any([x.modified for x in [img, dark, offset]]):
        warn("Images used for modelling dark image are modified")
# =============================================================================
#         
#     if any([x.modified for x in [dark, offset]]):
#         raise ImgModifiedError("Could not model dark image at least one of the "
#             "input dark / offset images was already modified")
#     if img.modified:
#         img = Img(img.meta["path"], import_method=img.import_method)
#             
# =============================================================================
    dark_img = (offset.img + (dark.img - offset.img) * img.meta["texp"]/
                             (dark.meta["texp"] - offset.meta["texp"]))
    
    return Img(dark_img, start_acq=img.meta["start_acq"], 
               texp=img.meta["texp"], **img.edit_log)
#==============================================================================
# import matplotlib.animation as animation
# 
# def animate_stack(img_stack):
#     
#     fig = figure() # make figure
#     
#     # make axesimage object
#     # the vmin and vmax here are very important to get the color map correct
#     im = imshow(sta, cmap=plt.get_cmap('jet'), vmin=0, vmax=255)
#     
#     # function to update figure
#     def updatefig(j):
#         # set the data in the axesimage object
#         im.set_array(imagelist[j])
#         # return the artists set
#         return im,
#     # kick off the animation
#     animation.FuncAnimation(fig, updatefig, frames=range(20), 
#                                   interval=50, blit=True)
#     plt.show()
#==============================================================================
