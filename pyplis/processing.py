# -*- coding: utf-8 -*-
"""
Basic and advanced processing objects
-------------------------------------
"""
from numpy import vstack, ogrid, empty, ones, asarray, ndim, round, hypot,\
    linspace, sum, dstack, float32, zeros, poly1d, polyfit, argmin, where,\
    logical_and, rollaxis, complex, angle, array, ndarray
    
from numpy.linalg import norm
from scipy.ndimage import map_coordinates 
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from warnings import warn
from json import loads, dumps
from copy import deepcopy
from datetime import datetime, timedelta
from matplotlib.pyplot import subplot, subplots, tight_layout, draw
from matplotlib.dates import date2num
from pandas import Series, concat, DatetimeIndex
from cv2 import cvtColor, COLOR_BGR2GRAY, pyrDown, pyrUp
from os import getcwd, remove
from os.path import join, exists
from astropy.io import fits
    
from .image import Img
from .setupclasses import Camera
from .exceptions import ImgMetaError, ImgModifiedError
from .helpers import map_coordinates_sub_img, same_roi, map_roi

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
    
    def __init__(self, data, start_acq, std = None, texps = None,
                         roi_abs = None, img_prep = {}, **kwargs):
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
        
    def get_data_normalised(self, texp = None):
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
            
            return PixelMeanTimeSeries(self.values * facs, self.index,\
                        self.std, ts, self.roi_abs, self.img_prep)
            
        except Exception as e:
            print ("Failed to normalise data bases on exposure times:\n%s\n\n"
                        %repr(e))
                      
    
    def fit_polynomial(self, order = 2):
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
    
    def includes_timestamp(self, time_stamp, ext_border_secs = 0.0):
        """Checks if input time stamp is included in this dataset
        
        :param datetime time_stamp: the time stamp to be checked
        """
        i = self.start - timedelta(ext_border_secs / 86400.0)
        f = self.stop + timedelta(ext_border_secs / 86400.0)
        if i <= time_stamp <= f:
            return True
        return False
        
    def get_poly_vals(self, time_stamps, ext_border_secs = 0.0):
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
        
    def estimate_noise_amplitude(self, sigma_gauss = 1, median_size = 3,\
                                                                    plot = 0):
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
    
    def plot(self, include_tit = True, **kwargs):
        """Plot"""
        try:
            if not "style" in kwargs:
                kwargs["style"] = "--x"    
            ax = super(PixelMeanTimeSeries, self).plot(**kwargs)
            if include_tit:
                ax.set_title("Mean value (%s), roi_abs: %s" %(self.name,\
                        self.roi_abs))
            ax.grid()
            
            return ax
        except Exception as e:
            print repr(e)
            fig, ax = subplots(1,1)
            ax.text(.1,.1, "Plot failed...")
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
    
    Main purpose is data extraction along the line, which is done using 
    spline interpolation.    
    """
    def __init__(self, x0=0, y0=0, x1=1, y1=1,  normal_orientation="right",
                 roi_abs=[0, 0, 9999, 9999], pyrlevel=0, line_id = "",
                 color="lime", linestyle="-"):
        """Initiation of line
        
        :param int x0: start x coordinate
        :param int y0: start y coordinate
        :param int x1: stop x coordinate
        :param int y1: stop y coordinate
        :param str normal_orientation: orientation of normal vector, choose 
            from left or right (left means in negative x direction for a 
            vertical line)
        :param list roi_abs: region of interest in image for which start / stop
            coordinates are defined (is used to convert to other image shape
            settings)
        :param int pyrlevel: pyramid level of image for which start /stop
            coordinates are defined
        :param str line_id: string for identification (optional)
        
        .. note::
        
            The input coordinates correspond to relative image coordinates 
            with respect to the input ROI (``roi_abs``) and pyramid level
            (``pyrlevel``)
            
        """
        self.line_id = line_id # string ID of line
        self.color = color
        self.linestyle = linestyle
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
            
        self.x0 = x0 #start x coordinate
        self.y0 = y0 #start y coordinate
        self.x1 = x1 #stop x coordinate
        self.y1 = y1 #stop y coordinate
        
        self._roi_abs = roi_abs
        self._pyrlevel = pyrlevel
        
        self.profile_coords = None
        
        self._dir_idx = {"left"   :   0,
                         "right"  :   1}
                         
        self.check_coordinates()
        self.normal_orientation = normal_orientation
                                       
        self.prepare_profile_coordinates()
            
    @property
    def start(self):
        """Returns start coordinates ``[x0, y0]``"""
        return [self.x0, self.y0]
        
    @property
    def stop(self):
        """Returns stop coordinates ``[x1, y1]``"""
        return [self.x1, self.y1]
    
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
    def roi_abs(self):
        """Returns current ROI (in absolute detector coordinates)"""
        return self._roi_abs
    
    @roi_abs.setter
    def roi_abs(self):
        """Raises AttributeError"""
        raise AttributeError("This attribute is not supposed to be changed, "
            "please use method convert() to create a new LineOnImage object "
            "corresponding to other image shape settings")
            
    @property
    def pyrlevel(self):
        """Returns current pyramid level"""
        return self._pyrlevel
        
    @pyrlevel.setter
    def pyrlevel(self):
        """Raises AttributeError"""
        raise AttributeError("This attribute is not supposed to be changed, "
            "please use method convert() to create a new LineOnImage object "
            "corresponding to other image shape settings")
    
    def dist_other(self, other):
        """Determines the distance to another line
        
        :param LineOnImage other: the line to which the distance is retrieved
            
        .. note::
        
            The offset is applied in relative coordinates, i.e. it does not
            consider the pyramide level or ROI
            
        """
        dx0, dy0 = other.x0 - self.x0, other.y0 - self.y0
        dx1, dy1 = other.x1 - self.x1, other.y1 - self.y1
        if dx1 != dx0 or dy1 != dy0:
            print "dx0, dx1: %s, %s"%(dx0, dx1)
            print "dy0, dy1: %s, %s"%(dy0, dy1)
            raise ValueError("Lines are not parallel...")
        return norm([dx0, dy0]), dx0, dy0
        
    def offset(self, pixel_num = 20, line_id = None):
        """Returns a shifted line at given distance along normal orientation
        
        .. note::
                
            1. The offset is applied in relative coordinates, i.e. it does not
                consider the pyramide level or ROI
            
            2. The determined required displacement (dx, dy) is converted into
                integers                
            
        """
        if line_id is None:
            line_id = self.line_id + "_shifted_%dpix" %pixel_num
        dx, dy = self.normal_vector * pixel_num
        x0, x1 = self.x0 + int(dx), self.x1 + int(dx)
        y0, y1 = self.y0 + int(dy), self.y1 + int(dy)
        return LineOnImage(x0, y0, x1, y1,
                           normal_orientation=self.normal_orientation,
                           line_id=line_id)
        
        
    def convert(self, pyrlevel = 0, roi_abs = [0, 0, 9999, 9999]):
        """Convert to other image preparation settings"""
        if pyrlevel == self.pyrlevel and same_roi(self.roi_abs, roi_abs):
            print "Same shape settings, returning current line object"""
            return self

        (x0, x1), (y0, y1) = map_coordinates_sub_img([self.x0, self.x1],
                                                     [self.y0, self.y1],
                                                     roi_abs=self._roi_abs, 
                                                     pyrlevel=self._pyrlevel,
                                                     inverse=True)
                                                     
        (x0, x1), (y0, y1) = map_coordinates_sub_img([x0, x1], [y0, y1],
                                                     roi_abs=roi_abs,
                                                     pyrlevel=pyrlevel,
                                                     inverse=False)
        
        return LineOnImage(x0, y0, x1, y1, roi_abs=roi_abs, pyrlevel=pyrlevel,
                           normal_orientation=self.normal_orientation,
                           line_id=self.line_id)
            
    def check_coordinates(self):
        """Check if coordinates are in the right order and exchange start/stop
        points if necessary
        """
        if any([x < 0 for x in self.start]):
            raise ValueError("Invalid value encountered, start coordinates of "
                "line must be bigger than 0")
        if self.start[0] > self.stop[0]:
            print "X coordinate of Start point is larger than X of stop point"
            print "Start and Stop will be exchanged"
            self.start, self.stop = self.stop, self.start
     
    def in_image(self, img_array):
        """Check if this line is within the coordinates of an image array"""
        if not all(self.point_in_image(p, img_array)\
                                    for p in [self.start, self.stop]):
            return 0
        return 1
        
    def point_in_image(self, x, y, img_array):
        """
        Check, if input x and y coordinates are within an image array. It is
        assumed that y direction is first dimension of image array-
        
        :param int x: x coordinate of point
        :param int y: y coordinate of point
        
        :return bool val: True or false
        
        """
        h, w = img_array.shape[:2]
        if not 0 < x < w:
            print "x coordinate out of image"
            return 0
        if not 0 < y < h:
            print "y coordinate out of image"
            return 0
        return 1
    
    def get_roi_abs_coords(self, img_array, add_left = 80, add_right = 80,\
                                        add_bottom = 80, add_top = 80):
        """Get a rectangular ROI covering this line 
                
        :param int add_left: expand range to left of line (in pix)
        :param int add_right: expand range to right of line (in pix)
        :param int add_bottom: expand range to bottom of line (in pix)
        :param int add_top: expand range to top of line (in pix)
        
        :returns list: rectangle specifying the ROI ``[x0,y0,x1,y1]``
        
        """
        
        x0, x1 = self.start[0] - add_left, self.stop[0] + add_right
        #y start must not be smaller than y stop
        y_arr = [self.start[1], self.stop[1]]
        y_min, y_max = min(y_arr), max(y_arr)
        y0, y1 = y_min - add_top, y_max + add_bottom
        roi = self.check_roi_borders([x0, y0, x1, y1], img_array)
        roi_abs = map_roi(roi, pyrlevel_rel=-1 * self.pyrlevel)
        return roi_abs
    
    def check_roi_borders(self, roi, img_array):
        """ Check if all points of ROI are within image borders
        
        :param list roi: list specifying ROI rectangle: ``[x0,y0,x1,y1]``
        :param ndarray img_array: the image to which the ROI is applied
        :return: roi within image coordinates (unchanged, if input is ok, else
            image borders)
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
            
    def prepare_profile_coordinates(self):
        """Prepare the evaluation coordinates
        
        The number analysis points on this object correponds to the physical 
        length of this line in pixel coordinates. 
        """
        length = self.length()
        x0, y0 = self.start     
        x1, y1 = self.stop

        x = linspace(x0, x1, length)
        y = linspace(y0, y1, length)
        self.profile_coords = vstack((y, x))
        
    def length(self):
        """Determine the length in pixel coordinates"""
        return int(round(hypot(*self._delx_dely())))
        
    def get_line_profile(self, array):
        """Retrieve the line profile on input input grid
        
        :param ndarray array: the image array (color images are converted into
                            gray scale using :func:`cv2.cvtColor`) 
        :return
        """
        try:
            array = array.img #if input is Img object
        except:
            pass
        if ndim(array) != 2:
            if ndim(array) != 3:
                print ("Error retrieving line profile, invalid dimension of input "
                " array: " + str(ndim(array)))
                return
            if array.shape[2] != 3:
                print ("Error retrieving line profile, invalid dimension of input "
                " array: " + str(ndim(array)))
                return
            "Input in BGR, conversion into gray image"
            array = cvtColor(array, COLOR_BGR2GRAY)

        # Extract the values along the line, using interpolation
        zi = map_coordinates(array, self.profile_coords)
        return zi
        
    """Plotting / visualisation etc...
    """
    def plot_line_on_grid(self, img_arr = None, ax = None,\
                                include_normal = False, **kwargs):
        """Draw the line on the image
        
        :param ndarray img_arr: if specified, the array is plotted using 
            :func:`imshow` and onto that axes, the line is drawn
        :param ax: matplotlib axes object. Is created if unspecified. Leave 
            :param:`img_arr` empty if you want the line to be drawn onto an
            already existing image (plotted in ax)
        :param **kwargs: additional keyword arguments for plotting of line
            (please use following keys: marker for marker style, mec for marker 
            edge color, c for line color and ls for line style)
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
        if img_arr is not None:
            ax.imshow(img_arr, cmap = "gray")
        p = ax.plot([self.start[0],self.stop[0]], [self.start[1],self.stop[1]],\
             **kwargs)
        if img_arr is not None:
            ax.set_xlim([0,img_arr.shape[1]])
            ax.set_ylim([img_arr.shape[0],0])
        if include_normal:
            mag = self.norm * 0.03 #3 % of length of line
            n = self.normal_vector * mag
            xm, ym = self.center_pix
            epx, epy = n[0], n[1]
            c = p[0].get_color()
            ax.arrow(xm, ym, epx, epy, head_width=mag/2, head_length=mag,\
                                    fc = c, ec = c, label = "Scaled normal")
            
        #axis('image')
        if new_ax:
            ax.set_title("Line " + str(self.line_id))
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        draw()
        return ax
    
    def plot_line_profile(self, img_arr, ax=None):
        """Plots the line profile"""
        if ax is None:
            ax=subplot(111)
        p = self.get_line_profile(img_arr)
        ax.set_xlim([0,self.length()])
        ax.grid()
        ax.plot(p)
        ax.set_title("Profile")
        return ax
    
    def plot(self, img_arr):
        """Basically calls
        
            1. self.plot_line_on_grid() and
            #. self.plot_line_profile()
            
        and puts them next to each other in a subplot
        """
        fig, axes = subplots(1,2)
        self.plot_line_on_grid(img_arr,axes[0])
        self.plot_line_profile(img_arr,axes[1])
        tight_layout()
        
    def _delx_dely(self):
        """Length of x and y range covered by line"""
        return float(self.x1) - float(self.x0), float(self.y1) - float(self.y0)
    
    @property
    def norm(self):
        """Return length of line in pixels"""
        dx, dy = self._delx_dely()
        return norm([dx, dy])
    
    def normal_vecs(self):
        """Get both normal vectors"""
        dx, dy = self._delx_dely()
        v1, v2 = array([-dy, dx]), array([dy, -dx])
        return v1 / norm(v1), v2 / norm(v2)
            
    @property
    def normal_vector(self):
        """Get normal vector corresponding to current orientation setting"""
        return self.normal_vecs()[self._dir_idx[self.normal_orientation]]
        
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
        """Returns ID, coordinates and pyramide level in dictionary"""
        return {"class"                 :   "LineOnImage",
                "line_id"               :   self.line_id,
                "x0"                    :   self.x0,
                "y0"                    :   self.y0,
                "x1"                    :   self.x1, 
                "y1"                    :   self.y1,
                "normal_orientation"    :   self.normal_orientation,
                "pyrlevel"              :   self.pyrlevel, 
                "roi_abs"               :   self.roi_abs}
    
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
        """String representation
        """
        s=("LineOnImage " + str(self.line_id) + 
            "\n----------------------------------------\n")
        s=(s + "Start (X,Y): " + str(self.start) + 
                "\nStop (X,Y): " + str(self.stop) + "\n")
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
    def __init__(self, img_data = None, time_stamps = asarray([]), img_id =\
                    "", dtype = float32, profile_info_dict = {}, **meta_info):
        self.img_id = img_id
        self.time_stamps = asarray(time_stamps)
        self.profile_info = {}
        if isinstance(profile_info_dict, dict):
            self.profile_info = profile_info_dict
        #Initiate object as Img object
        super(ProfileTimeSeriesImg, self).__init__(input = img_data,\
                                            dtype = dtype, **meta_info)
                                                
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
              
    def save_as_fits(self, save_dir = None, save_name = None):
        """Save stack as FITS file"""
        self._format_check()
        try:
            save_name = save_name.split(".")[0]
        except:
            pass
        if save_dir is None:
            save_dir = getcwd()
        if save_name is None:
            save_name = "pyplis_profile_tseries_id_%s_%s_%s_%s.fts"\
                %(self.img_id, self.start.strftime("%Y%m%d"),\
                self.start.strftime("%H%M"), self.stop.strftime("%H%M"))
        else:
            save_name = save_name + ".fts"
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
            if key == "roi_abs":
                hdu.header["roi_abs"] = dumps(val)
            else:
                hdu.header[key] = val
    
        hdu.header.append()
        hdulist = fits.HDUList([hdu, arrays])
        path = join(save_dir, save_name)
        if exists(path):
            print "Stack already exists at %s and will be overwritten" %path
            remove(path)

        hdulist.writeto(path)
    
    def _profile_dict_keys(self, profile_type = "LineOnImage"):
        """Returns profile dictionary keys for input profile type"""
        d = {"LineOnImage"  :   LineOnImage().to_dict().keys()}
        return d[profile_type]
        
    def load_fits(self, file_path, profile_type = "LineOnImage"):
        """Load stack object (fits)
        
        :param str file_path: file path of fits image
        """
        
        if not exists(file_path):
            raise IOError("Img could not be loaded, path does not exist")
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
                if k == "roi_abs":
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
        
    .. todo::
    
        1. Include optical flow routine for emission rate retrieval
        
    """
    def __init__(self, height=0, width=0, img_num=0, dtype=float32,
                 stack_id="", img_prep=None, camera=None, **stack_data):
        """Specify input
        
        :param int height: height of images to be stacked
        :param int width: width of images to be stacked
        :param int num: number of images to be stacked
        :param dtype: accuracy of image data type (e.g. for an 8 bit image, 
            uint8 would be enough, makes the necessary space smaller,
            default: float32)
        :param str stack_id: string ID of this object ("")
        :param dict img_prep: additional information about the images (e.g.
            roi, gauss pyramid level, dark corrected?, blurred?)
        :param **stack_data:
        """
        self.stack_id = stack_id
        self.dtype = dtype
        self.current_index = 0
        
        try:
            self.stack = empty((img_num, height, width))
        except MemoryError:
            raise MemoryError("Could not initiate empty 3D numpy array "
                "(d, h, w): (%s, %s, %s)" %(img_num, height, width))
        self.start_acq = asarray([datetime(1900,1,1)] * img_num)
        self.texps = zeros(img_num, dtype = float32)
        self.add_data = zeros(img_num, dtype = float32)
        
        self._access_mask = zeros(img_num, dtype = bool)
        
        if img_prep is None:
            img_prep = {"pyrlevel"     :   0}
        self.img_prep = img_prep 
        
        
        self.roi_abs = [0, 0, 9999, 9999]
        
        self._cam = Camera()
        
        if stack_data.has_key("stack"):
            self.set_stack_data(**stack_data)
        
        if isinstance(camera, Camera):
            self.camera = camera
        
    @property
    def last_index(self):
        """Returns last index"""
        return self.num_of_imgs - 1
    
    @property
    def start(self):
        """Returns start time stamp of first image"""
        try:
            return self.start_acq[0]
        except:
            raise ValueError("Information about start acquisition time could"
                " not be retrieved")
                
    @property
    def stop(self):
        """Returns start time stamp of first image"""
        try:
            return self.start_acq[-1] + timedelta(self.texps[-1] / 86400.)
        except:
            raise ValueError("Information about stop acquisition time could"
                " not be retrieved")
        
    @property
    def time_stamps(self):
        """Compute time stamps for images from acq. times and exposure times"""
        try:
            dts = [timedelta(x /(2 * 86400.)) for x in self.texps]
            return self.start_acq + asarray(dts)
        except:
            raise ValueError("Failed to access information about acquisition "
                "time stamps and / or exposure times")
    
    @property
    def pyrlevel(self):
        """return current pyramide level (stored in ``self.img_prep``)"""
        return self.img_prep["pyrlevel"]
    
    @property
    def camera(self):
        """Get / set current camera object"""
        return self._cam
    
    @camera.setter
    def camera(self, value):
        """Set camera"""
        if isinstance(value, Camera):
            self._cam = value
        else:
            raise TypeError("Need Camera object...")
            
    def append_img(self, img_arr, start_acq = datetime(1900, 1, 1), texp = 0.0, 
                                                               add_data = 0.0):
        """Append at the end of the stack
        
        :param ndarray img_arr: image data (must have same dimension than
            ``self.stack.shape[:2]``)
        :param datetime start_acq (datetime(1900, 1, 1)): acquisition time of 
                                                                        image
        :param float texp: exposure time of image (in units of s)
        
        The image is inserted at the current index position ``self.current_index``
        which is increased by 1 afterwards.
        """
#==============================================================================
#         if self.current_index >= self.last_index:
#             print self.last_index
#             raise IndexError("Last stack index reached...")
#==============================================================================
        self.set_img(self.current_index, img_arr, start_acq, texp, add_data)
        self.current_index += 1
            
    def set_img(self, pos, img_arr, start_acq  = datetime(1900, 1, 1),\
                                                texp = 0.0, add_data = 0.0):
        """Place the imageArr in the stack
        :param int pos: Position of img in stack
        :param ndarray img_arr: image data (must have same dimension than
            ``self.stack.shape[:2]``)
        :param datetime start_acq (datetime(1900, 1, 1)): acquisition time of 
                                                                        image
        :param float texp: exposure time of image (in units of s)
        
        """
        self.stack[pos] = img_arr
        self.start_acq[pos] = start_acq
        self.texps[pos] = texp
        self.add_data[pos] = add_data
        self._access_mask[pos] = True
    
    def make_circular_access_mask(self, cx, cy, radius):
        """Create a circular mask for stack 
        :param int pos_x_abs: x position of centre
        :param int pos_y_abs: y position of centre
        :param int radius: radius
        """
        #cx, cy = self.img_prep.map_coordinates(pos_x_abs, pos_y_abs)
        h, w = self.stack.shape[1:]
        y, x = ogrid[:h, :w]
        m = (x - cx)**2 + (y - cy)**2 < radius**2
        return m
        
    @property
    def num_of_imgs(self):
        """Return current number of images in stack"""
        return self.stack.shape[0]
        
    def set_stack_data(self, stack, start_acq = None, texps = None):
        """Sets the current data based on input
        
        :param ndarray stack: the image stack data
        :param ndarray start_acq: array containing acquisition time stamps
        :param ndarray texps: array containing exposure times
        """
        num = stack.shape[0]
        self.stack = stack
        if start_acq is None:
            start_acq = asarray([datetime(1900, 1, 1)] * num)
        self.start_acq = start_acq
        if texps is None:
            texps = zeros(num, dtype = float32)
        self.texps = texps
        self._access_mask = ones(num, dtype = bool)
        
    def get_data(self):
        """Get stack data 
        
        :rtype: (ndarray, list, list)
        :returns: 
            - ndarray, stack data
            - ndarray, acq time stamps
            - ndarray, exposure times
        """
        m = self._access_mask
        return (self.stack[m], asarray(self.time_stamps)[m],\
                                                asarray(self.texps)[m])
        
    def apply_mask(self, mask):
        """Convolves the stack data with a input mask along time axis (2)
        
        :param ndarray mask: bool mask for image pixel access
        :returns:
            - ndarray, new stack 
            - list, acq times
            - list, exposure times
        """
        #mask_norm = boolMask.astype(float32)/sum(boolMask)
        d = self.get_data()
        data_conv = (d[0] * mask.astype(float32))#[:, :, newaxis])#, d[1], d[2])
        return (data_conv, d[1], d[2])
    
    def get_time_series(self, pos_x = None, pos_y = None, radius = 1,\
                                                            mask = None):
        """Get time series in a circular ROI
        
        Retrieve the time series at a given pixel position *in stack 
        coordinates* in a circular pixel neighbourhood.
        
        :param int pos_x: x position of center pixel on detector
        :param int pos_y: y position of center pixel on detector
        :param float radius: radius of pixel disk on detector (centered
            around pos_x, pos_y, default: 1)
        :param ndarray mask: boolean mask for image pixel access, 
            default is None, if the mask is specified and valid (i.e. same
            shape than images in stack) then the other three input parameters
            are ignored
        """
        d = self.get_data()
        try:
            data_mask, start_acq, texps = self.apply_mask(mask)
        except:
            if radius == 1:
                mask = zeros(self.shape[1:]).astype(bool)
                mask[pos_y, pos_x] = True
                return Series(d[0][self._access_mask, pos_y, pos_x], d[1]),\
                                                                        mask
            mask = self.make_circular_access_mask(pos_x, pos_y, radius)
            data_mask, start_acq, texps = self.apply_mask(mask)
        values = data_mask.sum((1, 2)) / float(sum(mask))
        return Series(values, start_acq), mask
    
    """Data merging functionality based on additional time series data"""
    def merge_with_time_series(self, time_series, method = "average", **kwargs):
        """High level wrapper for data merging"""
        if not isinstance(time_series, Series):
            raise TypeError("Could not merge stack data with input data: "
                "wrong type: %s" %type(time_series))
        
        if method == "average":
            try:
                return self._merge_tseries_average(time_series, **kwargs)
            except ValueError:
                print ("Failed to merge data using method average, trying "
                       "method nearest instead")
                method = "nearest"
        if method == "nearest":
            return self._merge_tseries_nearest(time_series, **kwargs)
        elif method == "interpolation":
            return self._merge_tseries_cross_interpolation(time_series,\
                                                                   **kwargs)
        else:
            raise TypeError("Unkown merge type: %s. Choose from "
                    "[nearest, average, interpolation]")
        
    def _merge_tseries_nearest(self, time_series):
        """Find nearest in time image for each time stamp in input series
        
        Find indices (and time differences) in input time series of nearest 
        data point for each image in this stack. Then, get rid of all indices
        showing double occurences using time delta information. 
        
        .. note::
            
            Hard coded for now, more elegant solution follows...
            
        .. todo::
            
            rewrite docstring
            
        """
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
        stack_new = self.stack[img_idxs]
        texps_new = self.texps[img_idxs]
        start_acq_new = self.start_acq[img_idxs]
        stack_obj_new = ImgStack(stack_id = self.stack_id + "_merged_nearest",\
            img_prep = self.img_prep, stack = stack_new, start_acq =\
                                        start_acq_new, texps = texps_new)
        stack_obj_new.roi_abs = self.roi_abs
        stack_obj_new.add_data = series_new
        return stack_obj_new, series_new
            
    def _merge_tseries_cross_interpolation(self, time_series,\
                                               itp_type = "linear"):
        """Merge this stack with input data using interpolation
        
        :param Series time_series_data: pandas Series object containing time 
            series data (e.g. DOAS column densities)
        :param str itp_type: interpolation type (passed to 
            :class:`pandas.DataFrame` which does the interpolation, default is
            linear)
            
        """
        h, w = self.shape[1:]
        stack = self.stack
        
        #first crop time series data based on start / stop time stamps
        time_series = self.crop_other_tseries(time_series)
        time_series.name =  None
        if not len(time_series) > 0:
            raise IndexError("Time merging failed, data does not overlap")
        
        time_stamps = self.time_stamps
        #interpolate exposure times
        s0 = Series(self.texps, time_stamps)
        df0 = concat([s0, time_series], axis =1).interpolate(itp_type).dropna()
        
        new_num = len(df0[0])
        if not new_num >= self.num_of_imgs:
            raise ValueError("Unexpected error, length of merged data array does"
                "not exceed length of inital image stack...")
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
                df = concat([series_stack, df0[1]], axis = 1).\
                    interpolate(itp_type).dropna()
                #throw all N/A values
                #df = df.dropna()
                new_stack[:, i, j] = df[0].values
        
        stack_obj = ImgStack(new_num, h, w, stack_id = self.stack_id +\
                                "_interpolated", img_prep = self.img_prep)
        stack_obj.roi_abs = self.roi_abs
        #print new_stack.shape, new_acq_times.shape, new_texps.shape
        stack_obj.set_stack_data(new_stack, new_acq_times, new_texps)
        return stack_obj, df[1]
        
        
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
        
        :param DoasResults time_series: DOAS results object including arrays
            for start / stop acquisition time stamps required for averaging
        :returns: tuple, containing
            - :class:`ImgStack`, new stack object with averaged images
            - list, containing bad indices, i.e. indices of all start / stop 
                intervals for which no images could be found
        
        
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
#==============================================================================
#             print ("Found %s images for input index %s (of %s)" 
#                                                 %(sum(cond), k, num))
#==============================================================================
            if sum(cond) > 0:
                print ("Found %s images for spectrum #%s (of %s)" 
                                                %(sum(cond), k, num))
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
        stack_obj = ImgStack(len(new_texps), h, w, stack_id =\
                        self.stack_id + "_avg", img_prep = self.img_prep)
        stack_obj.roi_abs = self.roi_abs
        stack_obj.set_stack_data(new_stack, new_acq_times, new_texps)
        time_series = time_series.drop(time_series.index[bad_indices])
        return stack_obj, time_series
    
    """Helpers
    """
    def crop_other_tseries(self, time_series):
        """Crops other time series object based on start / stop time stamps"""
#==============================================================================
#         start = self.start - self.total_time_period_in_seconds() * tol_borders
#         stop = self.stop + self.total_time_period_in_seconds() * tol_borders
#==============================================================================
        cond = logical_and(time_series.index >= self.start,\
                                 time_series.index <= self.stop)
        return time_series[cond]
        
    def total_time_period_in_seconds(self):
        """Returns start time stamp of first image"""
        return (self.stop - self.start).total_seconds()
        
    def get_nearest_indices(self, time_stamps):
        """Find indices of time stamps nearest to img acq. time stamps
        
        :param (datetime, ndarray, list) time_stamps: input time stamps
        """
        idx = []
        delt = []
        img_stamps = self.time_stamps
        for tstamp in img_stamps:
            diff = [x.total_seconds() for x in abs(time_stamps - tstamp)]
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
    def show_img(self, index = 0):
        """Show image at input index
        :param int index: index of image in stack        
        """
        stack, ts, _ = self.get_data()
        im = Img(stack[index], start_acq = ts[index], texp = self.texps[index])
        im.edit_log.update(self.img_prep)
        im.roi_abs = self.roi_abs
        return im.show()

    def pyr_down(self, steps = 0):
        """Reduce the stack image size using gaussian pyramid 
             
        :param int steps: steps down in the pyramide
        :return: 
            - ImgStack, new image stack object (downscaled)
        
        """
        
        if not steps:
            return
        #print "Reducing image size, pyrlevel %s" %steps
        h, w = Img(self.stack[0]).pyr_down(steps).shape
        prep = deepcopy(self.img_prep)        
        new_stack = ImgStack(height=h, width=w, img_num= self.num_of_imgs,\
            stack_id = self.stack_id, img_prep=prep)
        for i in range(self.shape[0]):
            im = self.stack[i]
            for k in range(steps):
                im = pyrDown(im)
            new_stack.append_img(img_arr = im, start_acq = self.start_acq[i],\
                texp = self.texps[i], add_data = self.add_data[i])
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
        #print "Reducing image size, pyrlevel %s" %steps
        h, w = Img(self.stack[0]).pyr_up(steps).shape
        prep = deepcopy(self.img_prep)        
        new_stack = ImgStack(height=h, width=w, img_num= self.num_of_imgs,\
            stack_id = self.stack_id, img_prep=prep)
        for i in range(self.shape[0]):
            im = self.stack[i]
            for k in range(steps):
                im = pyrUp(im)
            new_stack.append_img(img_arr = im, start_acq = self.start_acq[i],\
                texp = self.texps[i], add_data = self.add_data[i])
        new_stack._format_check()
        new_stack.img_prep["pyrlevel"] -= steps
        return new_stack
    
    def to_pyrlevel(self, final_state = 0):
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
        if not all([len(x) == self.num_of_imgs for x in [self.add_data,\
                            self.texps, self._access_mask, self.start_acq]]):
            raise ValueError("Mismatch in array lengths of stack data, check"
                "add_data, texps, start_acq, _access_mask")
    
    def load_stack_fits(self, file_path):
        """Load stack object (fits)
        
        :param str file_path: file path of stack
        """
        if not exists(file_path):
            raise IOError("ImgStack could not be loaded, path does not exist")
        hdu = fits.open(file_path)
        self.set_stack_data(hdu[0].data.astype(self.dtype))
        prep = Img().edit_log
        for key, val in hdu[0].header.iteritems():
            if key.lower() in prep.keys():
                self.img_prep[key.lower()] = val
        self.stack_id = hdu[0].header["stack_id"]
        try:
            self.start_acq = [datetime.strptime(x, "%Y%m%d%H%M%S%f") for x in\
                                                    hdu[1].data["start_acq"]]
        except:
            warn("Failed to import acquisition times")
        try:
            self.texps = hdu[1].data["texps"]
        except:
            warn("Failed to import exposure times")
        try:
            self._access_mask = hdu[1].data["_access_mask"]
        except:
            warn("Failed to import data access mask")    
        try:
            self.add_data = hdu[1].data["add_data"]
        except:
            warn("Failed to import data additional data")
        self.roi_abs = hdu[2].data["roi_abs"]
        self._format_check()
        
    def save_as_fits(self, save_dir = None, save_name = None):
        """Save stack as FITS file"""
        self._format_check()
        try:
            save_name = save_name.split(".")[0]
        except:
            pass
        if save_dir is None:
            save_dir = getcwd()
        if save_name is None:
            save_name = "pyplis_imgstack_id_%s_%s_%s_%s.fts" %(self.stack_id,\
                self.start.strftime("%Y%m%d"), self.start.strftime(\
                                    "%H%M"), self.stop.strftime("%H%M"))
        else:
            save_name = save_name + ".fts"
        hdu = fits.PrimaryHDU()
        start_acq_str = [x.strftime("%Y%m%d%H%M%S%f") for x in self.start_acq]
        col1 = fits.Column(name = "start_acq", format = "25A", array =\
            start_acq_str)
        col2 = fits.Column(name = "texps", format = "D", array =\
                                                        self.texps)
        col3 = fits.Column(name = "_access_mask", format = "L",\
                                            array = self._access_mask)
        col4 = fits.Column(name = "add_data", format = "D",\
                                            array = self.add_data)
        cols = fits.ColDefs([col1, col2, col3, col4])
        arrays = fits.BinTableHDU.from_columns(cols)
        
        col5 = fits.Column(name = "roi_abs", format = "I",\
                                            array = self.roi_abs)                                    
        
        roi_abs = fits.BinTableHDU.from_columns([col5])
        #==============================================================================
        # col1 = fits.Column(name = 'target', format = '20A', array=a1)
        # col2 = fits.Column(name = 'V_mag', format = 'E', array=a2)
        #==============================================================================
        hdu.data = self.stack
        hdu.header.update(self.img_prep)
        hdu.header["stack_id"] = self.stack_id
        hdu.header.append()
        hdulist = fits.HDUList([hdu, arrays, roi_abs])
        path = join(save_dir, save_name)
        if exists(path):
            print "Stack already exists at %s and will be overwritten" %path
            remove(path)

        hdulist.writeto(path)
        
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
    if any([x.modified for x in [dark, offset]]):
        raise ImgModifiedError("Could not model dark image at least one of the "
            "input dark / offset images was already modified")
    if img.modified:
#==============================================================================
#         print ("Input image is modified, try reloading raw data for dark model"
#             " retrieval")
#==============================================================================
        img = Img(img.meta["path"])
            
    dark_img = offset.img + (dark.img - offset.img) * img.meta["texp"]/\
                                (dark.meta["texp"] - offset.meta["texp"])
    
    return Img(dark_img, start_acq = img.meta["start_acq"],\
                                            texp = img.meta["texp"])
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
