# -*- coding: utf-8 -*-
"""
Module containing features related to plume velocity analysis

Todo
----

    1. Make class for signal cross correlation analysis (high level) based on
       image list (cf. methods in ex08)
       
"""
from time import time
from numpy import mgrid,vstack,int32,sqrt,arctan2,rad2deg, asarray, sin, cos,\
    logical_and, histogram, ceil, roll, argmax, arange, ndarray,\
    deg2rad, nan, inf, dot, mean
from numpy.linalg import norm
from traceback import format_exc
from copy import deepcopy
from warnings import warn
from datetime import datetime
from collections import OrderedDict as od
from matplotlib.pyplot import subplots, figure, GridSpec
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.stats.stats import pearsonr
from os.path import isdir, join, isfile
from os import getcwd
    
from pandas import Series, DataFrame

from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN,\
    cvtColor,COLOR_GRAY2BGR,line,circle,VideoCapture,COLOR_BGR2GRAY,\
    waitKey, imshow

from .helpers import bytescale, check_roi, map_roi, roi2rect, set_ax_lim_roi,\
    nth_moment
from .optimisation import MultiGaussFit
from .processing import LineOnImage
from .image import Img

def find_signal_correlation(first_data_vec, next_data_vec, 
                            time_stamps=None, reg_grid_tres=None, 
                            freq_unit="S", itp_method="linear", 
                            cut_border_idx=0, sigma_smooth=1, plot=False):
    """Determines cross correlation from two ICA time series
    
    Parameters
    ----------
    first_data_vec : array
        first data vector (i.e. left or before ``next_data_vec``)
    next_data_vec : array
        second data vector (i.e. behind ``first_data_vec``)
    time_stamps : array 
        array containing time stamps of the two data vectors. If default 
        (None), then the two vectors are assumed to be sampled on a regular 
        grid and the returned lag corresponds to the index shift with highest 
        correlation. If ``len(time_stamps) == len(first_data_vec)`` and if 
        entries are datetime objects, then the two input time series are 
        resampled and interpolated onto a regular grid, for resampling and 
        interpolation settings, see following 3 parameters.
    reg_grid_tres : int
        sampling resolution of resampled time series
        data in units specified by input parameter ``freq_unit``. If None, 
        then the resolution is determined automatically based on the mean time
        resolution of the data
    freq_unit : str
        pandas frequency unit (use S for seconds, L for ms)
    itp_method : str
        interpolation method, choose from ``["linear", "quadratic", "cubic"]``
    cut_border_idx : int
        number of indices to be removed from both ends of the input arrays 
        (excluded datapoints for cross correlation analysis)
    sigma_smooth : int
        specify width of gaussian blurring kernel applied to data before 
        correlation analysis (default=1)
    plot : bool
        if True, result is plotted
        
    Returns
    -------
    tuple
        5-element tuple containing
        
        - *float*: lag (in units of s or the index, see input specs)
        - *array*: retrieved correlation coefficients for all shifts 
        - *Series*: analysis signal 1. data vector 
        - *Series*: analysis signal 2. data vector 
        - *Series*: analysis signal 2. data vector shifted using ``lag``
        
    """
    if not all([isinstance(x, ndarray) for x in\
                    [first_data_vec, next_data_vec]]):
        raise IOError("Need numpy arrays as input")
    if not len(first_data_vec) == len(next_data_vec):
        raise IOError("Mismatch in lengths of input data vectors")
    lag_fac = 1 #factor to convert retrieved lag from indices to seconds  
    if time_stamps is not None and len(time_stamps) == len(first_data_vec)\
                    and all([isinstance(x, datetime) for x in time_stamps]):
        print "Input is time series data"
        if not itp_method in ["linear", "quadratic", "cubic"]:
            warn("Invalid interpolation method %s: setting default (linear)" 
                 %itp_method)
            itp_method = "linear"
            
        if reg_grid_tres is None:
            delts = asarray([delt.total_seconds() for delt in\
                            (time_stamps[1:] - time_stamps[:-1])])
             #time resolution for re gridded data
            reg_grid_tres = (ceil(delts.mean()) - 1) / 4.0
            if reg_grid_tres < 1: #mean delt is smaller than 4s
                freq_unit = "L" #L decodes to milliseconds
                reg_grid_tres = int(reg_grid_tres * 1000)
            else:
                freq_unit = "S"
                
        delt_str = "%d%s" %(reg_grid_tres, freq_unit)
        print "Delta t string for resampling: %s" %delt_str
        
        s1 = Series(first_data_vec, time_stamps)
        s2 = Series(next_data_vec, time_stamps)
        # this try except block was inserted due to bug when using code in 
        # exception statement with pandas > 0.19, it worked, though with 
        # pandas v0.16
        try:
            s1 = s1.resample(delt_str).agg(mean).interpolate(itp_method).dropna()
            s2 = s2.resample(delt_str).agg(mean).interpolate(itp_method).dropna()
        except:
            s1 = s1.resample(delt_str).interpolate(itp_method).dropna()
            s2 = s2.resample(delt_str).interpolate(itp_method).dropna()
        lag_fac = (s1.index[10] - s1.index[9]).total_seconds()
    else:
        s1 = Series(first_data_vec)
        s2 = Series(next_data_vec)
    if cut_border_idx > 0:
        s1 = s1[cut_border_idx:-cut_border_idx]
        s2 = s2[cut_border_idx:-cut_border_idx]
    s1_vec = gaussian_filter(s1, sigma_smooth) 
    s2_vec = gaussian_filter(s2, sigma_smooth) 
        
    coeffs = []
    max_coeff = -10
    max_coeff_signal = None
    print "Signal correlation analysis running..."
    for k in range(len(s1_vec)):
        shift_s1 = roll(s1_vec, k)
        coeffs.append(pearsonr(shift_s1, s2_vec)[0])
        if coeffs[-1] > max_coeff:
            max_coeff = coeffs[-1]
            max_coeff_signal = Series(shift_s1, s1.index)
    coeffs = asarray(coeffs)
    s1_ana = Series(s1_vec, s1.index)
    s2_ana = Series(s2_vec, s2.index)
    ax = None
    if plot:
        fig, ax = subplots(1, 3, figsize = (18,6))
        
        s1.plot(ax = ax[0], label="First line")
        s2.plot(ax = ax[0], label="Second line")
        ax[0].set_title("Original time series", fontsize = 10)
        ax[0].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
        ax[0].grid()
        
        max_coeff_signal.plot(ax = ax[1], label =\
                        "Data vector 1. line (best shift)")
        s2_ana.plot(ax = ax[1], label = "Data vector 2. line")
        
                        
        ax[1].set_title("Signal match", fontsize = 10)
        ax[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
        ax[1].grid()
        
        x = arange(0, len(coeffs), 1) * lag_fac
        ax[2].plot(x, coeffs, "-r")
        ax[2].set_xlabel(r"$\Delta$t [s]")
        ax[2].grid()
        #ax[1].set_xlabel("Shift")
        ax[2].set_ylabel("Correlation coeff")
        ax[2].set_title("Correlation signal", fontsize = 10)
    lag = argmax(coeffs) * lag_fac
    return lag, coeffs, s1_ana, s2_ana, max_coeff_signal, ax

class LocalPlumeProperties(object):
    """Class to store results about local properties of plume displacement
    
    This class represents statistical (local) plume (gas) displacement 
    information (e.g. retrieved using an optical flow algorithm). These 
    include the predominant local displacement direction (orientation of 
    displacement vectors) and the corresponding displacement length both 
    including uncertainties (e.g. retrieved from Gauss fits applied to 
    histogram distribution). 
    Further, the time difference between the two frames used to estimate the 
    displacement parameters is stored. This class is for instance used for
    plume displacement properties derived using 
    :func:`local_flow_params` from :class:`OptflowFarneback`
    which is based on a statistical analysis of histograms derived from 
    a dense optical flow algorithm.
    """
    def __init__(self, roi_id="", **kwargs):
        self.roi_id = roi_id
        self.color = "b"
        self._len_mu_norm = []
        self._len_sigma_norm = []
        self._dir_mu = []
        self._dir_sigma = []
        self._start_acq = []
        self._del_t = []
        self._significance = []
        
        for k, v in kwargs.iteritems():
            self[k] = v
            
    @property
    def start(self):
        """Acquisistion time of first image"""
        return self.start_acq[0]
        
    @property
    def stop(self):
        """Start acqusition time of last image"""
        return self.start_acq[-1]
        
    @property
    def len_mu(self):
        """Array containing displacement lengths (unit [pix/del_t])"""
        return asarray(self._len_mu_norm) * self.del_t
    
    @property
    def len_sigma(self):
        """Array with errors of displacement lengths (unit [pix/del_t])"""
        return asarray(self._len_sigma_norm) * self.del_t
    
    @property
    def len_mu_norm(self):
        """Array containing normalised displacement lengths (unit [pix/s])"""
        return asarray(self._len_mu_norm) 
    
    @property
    def len_sigma_norm(self):
        """Array with errors of normalised displ. lens (unit [pix/s])"""
        return asarray(self._len_sigma_norm)
        
    @property
    def dir_mu(self):
        """Return current displacement orientation vector"""
        return asarray(self._dir_mu)
    
    @property
    def dir_sigma(self):
        """Return current displacement orientation std vector"""
        return asarray(self._dir_sigma)
    
    @property
    def significance(self):
        """Significancy of data point
        
        This array is filled in :func:`get_and_append_from_farneback`, which
        calls :func:`local_flow_params` of 
        :class:`OptflowFarneback` object. The number corresponds to the 
        fraction of pixels used to determine the displacement parameters, 
        relative to the total number of pixels available in the corresponding
        ROI used. The latter can, for instance, be a rotated ROI around 
        a retrieval line (:class:`LineOnImage` class).
        """
        return asarray(self._significance)
        
    @property
    def del_t(self):
        """Return current del_t vector 
        
        Corresponds to the difference between frames for time series
        """
        return asarray(self._del_t)
    
    @property
    def start_acq(self):
        """Return current displacement length std vector
        
        Corresponds to the start acquisition times of the time series         
        """
        return asarray(self._start_acq)
    
    @property
    def displacement_vectors(self):
        """All displacement vectors (unit [pix / del_t])"""
        return (asarray([sin(deg2rad(self.dir_mu[:])),
                        -cos(deg2rad(self.dir_mu[:]))])\
                        * self.len_mu[:]).T
    
    def displacement_vector(self, idx=-1):
        """Get displacement vector for given index (
        
        The vector is returned in unit [pix / del_t].
        
        Parameters
        ----------
        idx : int
            index 
        
        Returns
        -------
        array
            2-element array containing displacement in x and y direction, i.e.
            ``(dx, dy)``
            
        """
        return asarray([sin(deg2rad(self.dir_mu[idx])),
                        -cos(deg2rad(self.dir_mu[idx]))])\
                        * self.len_mu[idx]
                        
    def dropna(self, **kwargs):
        """Drop all indices containing nans
        
        Remove all indices for which any of the data arrays 
        ``len_mu``, ``len_sigma``, ``dir_mu``, ``dir_sigma``, ``del_t`` 
        contains NaN values using the method :func:`dropna` of pandas 
        :class:`DataFrame` object.
        
        Parameters
        ----------
        **kwargs
            additional keyword arguments passed to :func:`dropna` of pandas
            :class:`DataFrame` object.
            
        Returns
        -------
        LocalPlumeProperties
            new object excluding nan values in any of the data arrays
        """
        p = LocalPlumeProperties(self.roi_id, color=self.color)
        df = self.to_pandas_dataframe()
        df = df.dropna(**kwargs)
        p.from_pandas_dataframe(df)
        return p
        
    def interpolate(self, **kwargs):
        """Interpolate missing
        
        Remove all indices for which any of the data arrays 
        ``len_mu``, ``len_sigma``, ``dir_mu``, ``dir_sigma``, ``del_t`` 
        contains NaN values using the method :func:`dropna` of pandas 
        :class:`DataFrame` object.
        
        Parameters
        ----------
        **kwargs
            additional keyword arguments passed to :func:`dropna` of pandas
            :class:`DataFrame` object.
            
        Returns
        -------
        LocalPlumeProperties
            new object excluding nan values in any of the data arrays
        """
        p = LocalPlumeProperties(self.roi_id, color=self.color)
        df = self.to_pandas_dataframe()
        df = df.interpolate(**kwargs)
        p.from_pandas_dataframe(df)
        return p
    
    def apply_median_filter(self, width=5):
        """Apply median filter to data
        
        The filter is only applied to :attr:`len_mu` and :attr:`dir_mu`, 
        and the corresponding uncertainty arrays :attr:`len_sigma` and 
        :attr:`dir_sigma`
        
        Note
        ----
        Creates and returns new :class:`LocalPlumeProperties` instance, the 
        data in this object remains unchanged
        
        Parameters
        ----------
        width : int
            width of 1D median filter
            
        Returns
        -------
        LocalPlumeProperties
            new data object
        """
        p = LocalPlumeProperties(self.roi_id, color=self.color)
        p.from_dict(self.to_dict())
        p._len_mu_norm = median_filter(self.len_mu_norm, width)
        p._len_sigma_norm = median_filter(self.len_sigma_norm, width)
        p._dir_mu = median_filter(self.dir_mu, width)
        p._dir_sigma = median_filter(self.dir_sigma, width)
        return p
        
    def apply_gauss_filter(self, width=5):
        """Apply Gaussian blurring filter to data
        
        The filter is only applied to :attr:`len_mu` and :attr:`dir_mu`, 
        and the corresponding uncertainty arrays :attr:`len_sigma` and 
        :attr:`dir_sigma`
        
        Note
        ----
        Creates and returns new :class:`LocalPlumeProperties` instance, the 
        data in this object remains unchanged
        
        Parameters
        ----------
        width : int
            width of Gaussian blurring kernel 
            
        Returns
        -------
        LocalPlumeProperties
            new data object
        """
        p = LocalPlumeProperties(self.roi_id, color=self.color)
        p.from_dict(self.to_dict())
        p._len_mu_norm = gaussian_filter(self.len_mu_norm, width)
        p._len_sigma_norm = gaussian_filter(self.len_sigma_norm, width)
        p._dir_mu = gaussian_filter(self.dir_mu, width)
        p._dir_sigma = gaussian_filter(self.dir_sigma, width)
        return p
        
    def get_and_append_from_farneback(self, optflow_farneback, **kwargs):
        """Retrieve main flow field parameters from Farneback engine
        
        Calls :func:`local_flow_params` from 
        :class:`OptflowFarneback` engine and appends the results to 
        the current data
        """
        res = optflow_farneback.local_flow_params(**kwargs)
        for key, val in res.iteritems():
            if self.__dict__.has_key(key):
                self.__dict__[key].append(val)
        
    def get_velocity(self, idx=-1, pix_dist_m=1.0, pix_dist_m_err=None, 
                     normal_vec=None):
        """Determine plume velocity from displacements
        
        Parameters
        ----------
        idx : int
            index of results for which velocity is determined
        pix_dist_m : float
            pixel to pixel distance in m (default is 1.0), e.g.
            determined using :class:`MeasGeometry` object
        pix_dist_m_err : :obj:`float`, optional
            uncertainty in pixel distance, if None (default), then 5% of the 
            actual pixel distance is assumed
        normal_vec : :obj:`tuple`, optional
            normal vector used for scalar product to retrieve effective 
            velocity (e.g. :attr:`normal_vector` of a :class:`LineOnImage`) 
            object. If None (default), the normal direction is assumed to be 
            aligned with the displacement direction, i.e. the absolute 
            magnitude of the velocity is retrieved
            
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`float`: magnitude of effective velocity
            - :obj:`float`: uncertainty of effective velocity
        """
        
        if pix_dist_m_err is None:
            pix_dist_m_err = pix_dist_m * 0.05
        vec = self.displacement_vector(idx)
        if normal_vec is None:
            normal_vec = vec / norm(vec)
        len_mu_eff = dot(normal_vec, vec)
        dt = self.del_t[idx]
        v =  len_mu_eff * pix_dist_m / dt
        verr = sqrt((pix_dist_m * self.len_sigma[idx] / dt)**2 +\
                    (pix_dist_m_err * len_mu_eff / dt)**2)
        return (v, verr)
        
    def get_orientation_tseries(self):   
        """Get time series (and uncertainties) of movement direction
        
        Returns
        -------
        tuple 
            3-element tuple containing
            
            - :obj:`Series`: time series of orientation angles
            - :obj:`Series`: time series of lower vals (using ``dir_sigma``)
            - :obj:`Series`: time series of upper vals (using ``dir_sigma``)
            
        """
        s = Series(self.dir_mu, self.start_acq)
        upper = Series(self.dir_mu + self.dir_sigma, self.start_acq)
        lower = Series(self.dir_mu - self.dir_sigma, self.start_acq)
        
        return (s, upper, lower)
        
    def get_magnitude_tseries(self, normalised=True):
        """Get time series (and uncertainties) of displacement lengths
        
                
        Note
        ----
        The time series are absolute magnitudes of the retrived displacement 
        lengths and are not considered relative to a certain normal direction.
        
        Parameters
        ----------
        normalised : bool
            if True, the lengths are normalised to a time difference of 1s
            
        Returns
        -------
        tuple 
            3-element tuple containing
            
            - :obj:`Series`: time series of displacement lengths
            - :obj:`Series`: time series of lower vals (using ``len_sigma``)
            - :obj:`Series`: time series of upper vals (using ``len_sigma``)
            
        """
        if normalised:
            l, err = self.len_mu_norm, self.len_sigma_norm
        else:
            l, err = self.len_mu, self.len_sigma
        s = Series(l, self.start_acq)
        upper = Series(l + err, self.start_acq)
        lower = Series(l - err, self.start_acq)
        
        return (s, upper, lower)
    
    #: VISUALISATION ETC.

    def plot_directions(self, ax=None, date_fmt=None, **kwargs):
        """Plot time series of displacement orientation
        
        Parameters
        ----------
        ax
            optional, matplotlib axes object
        date_fmt : str
            optional, x label datetime formatting string, passed to 
            :class:`DateFormatter` (e.g. "%H:%M")
        **kwargs
            additional keyword args passed to plot function of :class:`Series`
            object
            
        Returns 
        -------
        Axes
            matplotlib axes object
        """
        if ax is None:
            fig, ax = subplots(1,1)
        if not "color" in kwargs:
            kwargs["color"] = self.color
        if not "label" in kwargs:
            kwargs["label"] = self.roi_id
            
        s, upper, lower = self.get_orientation_tseries()
        s.index = s.index.to_pydatetime()
        s.plot(ax=ax, **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
        ax.fill_between(s.index, lower, upper, alpha=0.1, **kwargs)
        ax.set_ylabel(r"$\Theta\,[^{\circ}$]", fontsize=14)
        ax.grid()
        return ax
        
    def plot_magnitudes(self, normalised=True, ax=None, 
                        date_fmt=None, **kwargs):
        """Plot time series of displacement magnitudes
        
        Parameters
        ----------
        normalised : bool
            normalise magnitudes to time difference intervals of 1s
        ax
            optional, matplotlib axes object
        date_fmt : str
            optional, x label datetime formatting string, passed to 
            :class:`DateFormatter` (e.g. "%H:%M")
        **kwargs
            additional keyword args passed to plot function of :class:`Series`
            object
            
        Returns 
        -------
        Axes
            matplotlib axes object
        """
        if ax is None:
            fig, ax = subplots(1,1)
        if not "color" in kwargs:
            kwargs["color"] = self.color
        if not "label" in kwargs:
            kwargs["label"] = self.roi_id
            
        s, upper, lower = self.get_magnitude_tseries(normalised=normalised)
        if normalised:
            unit = "pix/s"
        else:
            unit = "pix"
        s.index = s.index.to_pydatetime()
        s.plot(ax=ax, **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
        ax.fill_between(s.index, lower, upper, alpha=0.1, **kwargs)
        ax.set_ylabel(r"Magnitude [%s]" %unit, fontsize=14)
        ax.grid()
        return ax
    
    def plot(self, date_fmt=None, **kwargs):
        """Plot showing detailed information about this time series"""
        fig = figure()
        gs = GridSpec(3, 1, height_ratios = [.4, .4, .2], hspace=0.05)
        ax2 = fig.add_subplot(gs[2]) #for significance plot
        ax0 = fig.add_subplot(gs[0])#, sharex=ax2) #for orientation plot
        ax1 = fig.add_subplot(gs[1])#, sharex=ax2) #for displ. lens
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        
        sign = Series(self.significance, self.start_acq)
        sign.index = sign.index.to_pydatetime()
        sign.plot(ax=ax2,**kwargs)
        ax2.set_ylabel("Significance")
        ax2.set_ylim([0, 1])
        ax2.set_yticks([0,.2,.4,.6,.8, 1])
        ax2.grid()
        try:
            if date_fmt is not None:
                ax2.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
        
        self.plot_directions(ax=ax0, date_fmt=date_fmt, **kwargs)
        self.plot_magnitudes(ax=ax1, date_fmt=date_fmt, **kwargs)
        ax0.set_xticklabels([])
        ax1.set_xticklabels([])
        return fig
        
    def plot_velocities(self, pix_dist_m=None, pix_dist_m_err=None, ax=None,
                        **kwargs):
        """Plot time series of velocity evolution 
        
        :param pix_dist_m: detector pixel distance in m, if unspecified, then
            velocities are plotted in units of pix/s
        :param pix_dist_m_err: uncertainty in pixel to pixel distance in m
        """
        raise NotImplementedError("Coming soon")
        velo_unit = "m/s"
        try:
            pix_dist_m = float(pix_dist_m)
        except:
            pix_dist_m = 1.0
            velo_unit = "pix/s"
        if not "color" in kwargs:
            kwargs["color"] = "b" 
            
        v, verr = self.get_velocity(pix_dist_m, pix_dist_m_err)
        if ax is None:
            fig, ax = subplots(1,1)
        velos = Series(v, self.start_acq)
        velos_upper = Series(v + verr, self.start_acq)
        velos_lower = Series(v - verr, self.start_acq)
        ax.plot(velos.index, velos, **kwargs)
        ax.fill_between(velos.index, velos_lower, velos_upper, alpha=0.1,
                        **kwargs)
        ax.set_ylabel("v [%s]" %velo_unit, fontsize=14)
        ax.grid()
        return ax
        
    #: I/O stuff
    def to_dict(self):
        """Write all data attributes into dictionary 
        
        Keys of the dictionary are the private class names
        
        Returns
        -------
        dict
            Dictionary containing results 
        """
        return dict(_len_mu_norm        =   self.len_mu_norm,
                    _len_sigma_norm     =   self.len_sigma_norm,
                    _dir_mu             =   self.dir_mu,
                    _dir_sigma          =   self.dir_sigma,
                    _del_t              =   self.del_t,
                    _start_acq          =   self.start_acq,
                    _significance       =   self._significance)
         
    def from_dict(self, d):
        """Read valid attributes from dictionary
        
        Parameters
        ----------
        d : dict
            dictionary containing data
            
        Returns
        -------
        LocalPlumeProperties
            this object
        """
        for k, v in d.iteritems():
            if self.__dict__.has_key(k):
                self.__dict__[k] = v
                
    def to_pandas_dataframe(self):
        """Converts object into pandas dataframe
        
        This can, for instance be used to store the data as csv (cf.
        :func:`from_pandas_dataframe`)        
        """
        d = self.to_dict()
        del d["_start_acq"]
        try:
            df = DataFrame(d, index=self.start_acq)
            return df
        except:
            warn("Failed to convert LocalPlumeProperties into pandas DataFrame")
    
    def from_pandas_dataframe(self, df):
        """Import results from pandas :class:`DataFrame` object
        
        Parameters
        ----------
        df : DataFrame
            pandas dataframe containing emisison rate results
        
        Returns
        -------
        LocalPlumeProperties
            this object
        """
        self._start_acq = df.index.to_pydatetime()
        for key in df.keys():
            if self.__dict__.has_key(key):
                self.__dict__[key] = df[key].values
        return self
                
    @property
    def default_save_name(self):
        """Returns default name for txt export"""
        try:
            d = self.start.strftime("%Y%m%d")
            i = self.start.strftime("%H%M")
            f = self.stop.strftime("%H%M")
        except:
            d, i, f = "nan", "nan", "nan"    
        return "plume_props_%s_%s_%s_%s.txt" %(self.roi_id, d, i, f)
        
    def save_txt(self, path=None):
        """Save this object as text file"""       
        try:
            if isdir(path): # raises exception in case path is not valid loc
                path = join(path, self.default_save_name)
            elif not isfile:
                raise Exception
        except:
            path = join(getcwd(), self.default_save_name)
            
        self.to_pandas_dataframe().to_csv(path)
        
    def load_txt(self, path):
        df = DataFrame.from_csv(path)
        return self.from_pandas_dataframe(df)
    
    def __setitem__(self, key, val):
        if self.__dict__.has_key(key):
            #print ("Updating attr. %s in LocalPlumeProperties, new val: %s"
            #    %(key, val))
            self.__dict__[key] = val
        
class FarnebackSettings(object):
    """Settings for optical flow Farneback calculations and visualisation
    
    .. todo::
    
        Finish docs
        
    This object contains settings for the opencv implementation of the 
    optical flow Farneback algorithm :func:`calcOpticalFlowFarneback`. 
    For a detailed description of the input parameters see `OpenCV docs 
    <http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_
    tracking.html#calcopticalflowfarneback>`__ (last access: 07.03.2017).
    
    Furthermore, it includes attributes for image preparation which are applied
    to the input images before :func:`calcOpticalFlowFarneback` is called.
    Currently, these include contrast changes specified by :attr:`i_min` and
    :attr:`i_max` which can be used to specify the range of intensities to be
    considered.
    
    In addition, post analysis settings of the flow field can be specified, 
    which are relevant, e.g. for a histogram analysis of the retrieved flow
    field:
    
        1. :attr:`roi_abs`: specifiy ROI for post analysis of flow field \
            (``abs`` indicates that the input is assumed to be in absolute \
            image coordinates and not in coordinates set based on a cropped \
            or size reduced image). Default corresponds to whole image.
            
        #. :attr:`min_length`: minimum length of optical flow vectors to be \
            considered for statistical analysis, default is 1 (pix)
            
        #. :attr:`hist_dir_sigma`: parameter for retrieval of mean flow \
            field parameters. It specifies the range of considered \
            orientation angles based on mu and sigma of the main peak of \
            flow field orientation histogram. All vectors falling into this \
            angular range are considered to determine the flow length \
            histogram used to estimate the average displacement length.
            
        #. :attr:`hist_dir_gnum_max`: maximum allowed number of gaussians for \
            multi gauss fit of orientation histogram (default = 5).
            
                    
    Parameters
    ----------
    **settings 
        valid keyword arguments for class attributes, e.g.::
        
            stp = FarnebackSettings(i_min=0, i_max=3500,
                                               iterations=8)
        
    """
    def __init__(self, **settings):
        self._contrast = od([("i_min"       ,   0),
                             ("i_max"       ,   1e30),
                             ("roi_rad"     ,   [0, 0, 9999, 9999]),
                             ("auto_update" ,   True)])
        
        self._flow_algo = od([("pyr_scale"  ,   0.5), 
                              ("levels"     ,   4),
                              ("winsize"    ,   20), 
                              ("iterations" ,   5), 
                              ("poly_n"     ,   5), 
                              ("poly_sigma" ,   1.1)])
                            
        self._analysis = od([("roi_abs"             ,   [0, 0, 9999, 9999]),
                             ("min_length"          ,   1.0),
                             ("min_count_frac"      ,   0.1),
                             ("hist_dir_sigma"      ,   1),
                             ("hist_dir_gnum_max"   ,   10),
                             ("hist_dir_binres"     ,   10)])
        
        self._display = od([("disp_skip"            ,   10),
                            ("disp_len_thresh"      ,   3)])
        
        
        self.update(**settings)
    
    def update(self, **settings):
        """Update current settings 
        
        Parameters
        ----------
        **settings 
            keyword args specifying new settings (only valid keys are
            considered, i.e. class attributes)
        """
        for k, v in settings.iteritems():
            self[k] = v # see __setitem__ method
            
    @property
    def i_min(self):
        """Lower intensity limit for image contrast preparation"""
        return self._contrast["i_min"]
            
    @i_min.setter
    def i_min(self, val):
        self._contrast["i_min"] = val
        
    @property
    def i_max(self):
        """Upper intensity limit for image contrast preparation"""
        return self._contrast["i_max"]
    
    @i_max.setter
    def i_max(self, val):
        self._contrast["i_max"] = val
    
    @property
    def roi_rad(self):
        """ROI used for measuring min / max intensities for contrast settings
        
        ROI (in absolute image coords) for updating the intensity range 
        ``i_min`` / ``i_max`` (only relevant if :attr:`auto_update` is True). 
        
        """
        return self._contrast["roi_rad"]
    
    @roi_rad.setter
    def roi_rad(self, val):
        if not check_roi(val):
            raise ValueError("Invalid ROI, need list [x0, y0, x1, y1], "
                "got %s" %val)
        self._contrast["roi_rad"] = val
        
    @property
    def auto_update(self):
        """Contrast is automatically updated based on min / max intensities
        
        If active, then :attr:`i_min` and :attr:`i_max` are updated
        automativally whenever new images are assigned to a 
        :class:`OptflowFarneback` using method :func:`set_images`. The
        update is performed based on min / max intensities of the images in
        the current ROI
        """
        return self._contrast["auto_update"]
    
    @auto_update.setter
    def auto_update(self, val):
        """Upper intensity limit for image contrast preparation"""
        if val in [0, 1]:
            self._contrast["auto_update"] = val
        
    @property
    def pyr_scale(self):
        """Farneback algo input: scale space parameter for pyramid levels
        
        pyplis default = 0.5        
        """
        return self._flow_algo["pyr_scale"]
    
    @pyr_scale.setter
    def pyr_scale(self, val):
        self._flow_algo["pyr_scale"] = val
        
    @property
    def levels(self):
        """Farneback algo input: number of pyramid levels
        
        pyplis default = 4        
        """
        return self._flow_algo["levels"]    
    
    @levels.setter
    def levels(self, val):
        self._flow_algo["levels"] = val
        
    @property
    def winsize(self):
        """Farneback algo input: width of averaging kernel
        
        The larger, the more stable the results are, but also more smoothed 
        
        pyplis default = 20
        """
        return self._flow_algo["winsize"]
    
    @winsize.setter
    def winsize(self, val):
        if val <= 0:
            raise ValueError("winsize must exceed 0")
        self._flow_algo["winsize"] = val
        
    @property
    def iterations(self):
        """Farneback algo input: number of iterations
        
        pyplis default = 5       
        """
        return self._flow_algo["iterations"]
    
    @iterations.setter
    def iterations(self, val):
        if val <= 0:
            raise ValueError("winsize must exceed 0")
        elif val < 4:
            warn("Small value for optical flow input parameter: iterations")
        elif val >10:
            warn("Large value for optical flow input parameter: iterations. "
                "This might significantly increase computation time")            
        self._flow_algo["iterations"] = val
        
    @property
    def poly_n(self):
        """Farneback algo input: size of pixel neighbourhood for poly exp
        
        default = 5
        """
        return self._flow_algo["poly_n"]
        
    @poly_n.setter
    def poly_n(self, val):
        self._flow_algo["poly_n"] = val

    @property
    def poly_sigma(self):
        """Farneback algo input: std of Gaussian to smooth poly derivatives
        
        pyplis default = 1.1
        """
        return self._flow_algo["poly_sigma"]
    
    @poly_sigma.setter
    def poly_sigma(self, val):
        self._flow_algo["poly_sigma"] = val
        
    @property
    def roi_abs(self):
        """Get ROI for analysis of flow field (in absolute image coords)"""
        return self._analysis["roi_abs"]
    
    @roi_abs.setter
    def roi_abs(self, val):
        if not check_roi(val):
            raise ValueError("Invalid ROI, need list [x0, y0, x1, y1], "
                "got %s" %val)
        self._analysis["roi_abs"] = val
        
    @property
    def min_length(self):
        """Get / set minimum flow vector length for post analysis"""
        return self._analysis["min_length"]
        
    @min_length.setter
    def min_length(self, val):
        if not val >= 1.0:
            print ("WARNING: Minimum length of optical flow vectors for "
                "analysis is smaller than 1 (pixel)")
        print "Updating param min_length: %.2f" %val
        self._analysis["min_length"] = val
    
    @property
    def min_count_frac(self):
        """Minimum fraction of significant vectors required for histo analysis"""
        return self._analysis["min_count_frac"]
        
    @min_count_frac.setter
    def min_count_frac(self, val):
        if not val <= 1.0:
            raise ValueError("Please use a fraction between 0 and 1")
        print "Updating param min_count_frac: %.2f" %val
        self._analysis["min_count_frac"] = val
    
    @property
    def hist_dir_sigma(self):
        """Sigma tolerance value for mean flow analysis"""
        return self._analysis["hist_dir_sigma"]
    
    @hist_dir_sigma.setter
    def hist_dir_sigma(self, val):
        if not 1 <= val < 4:
            raise ValueError("Value must be between 1 and 4")
        self._analysis["hist_dir_sigma"] = val
    
    @property
    def hist_dir_gnum_max(self):
        """Max number of gaussians for multigauss fit of orientation histo"""
        return self._analysis["hist_dir_gnum_max"]
    
    @hist_dir_gnum_max.setter
    def hist_dir_gnum_max(self, val):
        if not val > 0:
            raise ValueError("Value must be larger than 0")
        self._analysis["hist_dir_gnum_max"] = val
    
    @property
    def hist_dir_binres(self):
        """Angular resolution of orientation histo (bin width, in deg)"""
        return self._analysis["hist_dir_binres"]
    
    @hist_dir_binres.setter
    def hist_dir_binres(self, val):
        if not 1 <= val <= 180:
            raise ValueError("Please choose an angular resolution between "
                "1 and 180 degrees")
        self._analysis["hist_dir_binres"] = val
        
    @property
    def disp_skip(self):
        """Return current pixel skip value for displaying flow field"""
        return self._display["disp_skip"]
    
    @property
    def disp_len_thresh(self):
        """Return current pixel skip value for displaying flow field"""
        return self._display["disp_len_thresh"]    
    
    def duplicate(self):
        """Returns deepcopy of this object"""
        return deepcopy(self)
        
    def __str__(self):
        """String representation"""
        s="Image contrast settings (applied before flow calc):\n"
        for key, val in self._contrast.iteritems():
            s += "%s: %s\n" %(key, val)
        s += "\nOptical flow algo input (see OpenCV docs):\n"
        for key, val in self._flow_algo.iteritems():
            s += "%s: %s\n" %(key, val)
        s += "\nPost analysis settings:\n"
        for key, val in self._analysis.iteritems():
            s += "%s: %s\n" %(key, val)
        s += "\nDisplay settings:\n"
        for key, val in self._display.iteritems():
            s += "%s: %s\n" %(key, val)
        return s
    
    def __setitem__(self, key, value):
        """Set item method"""
        for k, v in self.__dict__.iteritems():
            try:
                if v.has_key(key):
                    v[key]=value
            except:
                pass
            
    def __getitem__(self, name):
        """Get item method"""
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        for k,v in self.__dict__.iteritems():
            try:
                if v.has_key(name):
                    return v[name]
            except:
                pass


    
class OptflowFarneback(object):
    """Implementation of Optical flow Farneback algorithm of OpenCV library
    
    Engine for autmatic optical flow calculation, for settings see
    :class:`FarnebackSettings`. The calculation of the flow field
    is performed for two consecutive images. 
    
    Includes features for histogram based post analysis of flow field which can
    be used to estimate flow vectors in low contrast image regions.
    
    Parameters
    ----------
    first_img : :obj:`Img`, optional
        first of two consecutive images
    next_img : :obj:`Img`, optional
        second of two consecutive images
        
    Attributes
    ----------
    images_input : dict
        Dictionary containing the current images used to determine flow field.
        The images can be updated using :func:`set_images`.
        Keys: ``this``, ``next``
    images_prep : dict
        Dictionary containing modified input images prepared for determining 
        the optical field using :func:`calcOpticalFlowFarneback` (e.g. 
        contrast changed, converted to 8 bit).
        Keys: ``this``, ``next``
    flow : array
        this attribute contains the flow field (i.e. raw output of 
        :func:`calcOpticalFlowFarneback`).
    settings : FarnebackSettings
        settings class including input specifications for flow calculation
        (i.e. input args for :func:`calcOpticalFlowFarneback`) and further, 
        settings for image preparation (before the flow field is calculated,
        cf. :attr:`images_prep`) as well as settings for post analysis of the
        optical flow field (e.g. for histogram analysis).
    """
    def __init__(self, first_img=None, next_img=None, **settings):
        #settings for determination of flow field
        self.settings = FarnebackSettings(**settings)

        self.images_input = {"this" : None,
                             "next" : None}
        #images used for optical flow
        self.images_prep = {"this" : None,
                            "next" : None}
        
        #the actual flow array (result from cv2 algo)
        self.flow = None
        
        #if you want, you can connect a TwoDragLinesHor object (e.g. inserted
        #in a histogram) to change the pre edit settings "i_min" and "i_max"
        #This will be done in both directions
        self._interactive_contrast_control = None
        
        if all([isinstance(x, Img) for x in [first_img, next_img]]):
            self.set_images(first_img, next_img)
            self.calc_flow()
            
    @property
    def auto_update_contrast(self):
        """Get / set mode for automatically update contrast range
        
        If True, the contrast parameters ``self.settings.i_min`` and 
        ``self.settings.i_max`` are updated when :func:`set_images`` is 
        called, based on the min / max intensity of the two images. The latter
        intensities are retrieved within the current ROI for the flow field 
        analysis (``self.roi_abs``)
        """
        return self.settings.auto_update
    
    @auto_update_contrast.setter
    def auto_update_contrast(self, val):
        self.settings.auto_update = val
        print ("Auto update contrast mode was updated in OptflowFarneback "
            "but not applied to current image objects, please call method "
            "set_images in order to apply the changes")
        return val
    
    def reset_flow(self):
        """Reset flow field"""
        self.flow = None
        
    @property
    def roi_abs(self):
        """Get / set current ROI (in absolute image coordinates)"""
        return self.settings.roi_abs
    
    @roi_abs.setter
    def roi_abs(self, val):
        self.settings.roi_abs = val
        if self.auto_update_contrast:
             self.update_contrast_range()
        
    @property
    def roi(self):
        """Get ROI converted to current image preparation settings"""
        try:
            return map_roi(self.roi_abs, self.images_input["this"].pyrlevel)
        except:
            raise ValueError("Error transforming ROI, check if images are set."
                "Error msg: %s" %format_exc())
    @roi.setter
    def roi(self):
        """Raises AttributeError"""
        raise AttributeError("Please use attribute roi_abs to change the "
            "current ROI")
    
    @property
    def pyrlevel(self):
        """Return pyramid level of current image"""
        im = self.images_input["this"]
        if not isinstance(im, Img):
            raise AttributeError("No image available")
        return im.edit_log["pyrlevel"]
        
    def set_mode_auto_update_contrast_range(self, value=True):
        """Activate auto update of image contrast range
        
        If this mode is active (the actual parameter is stored in 
        ``self._img_prep_modes["update_contrast"]``), then, whenever the 
        optical flow is calculated, the input contrast range is updated based
        on minimum / maxium intensity of the first input image within the 
        current ROI.
        
        :param bool value (True): new mode
        """
        self._img_prep_modes["update_contrast"] = value

    def check_contrast_range(self, img_data):
        """Check input contrast settings for optical flow calculation"""
        i_min, i_max = self.current_contrast_range()
        if i_min < img_data.min() and i_max < img_data.min() or\
                    i_min > img_data.max() and i_max > img_data.max():
            self.update_contrast_range(i_min, i_max)
            
    def current_contrast_range(self):
        """Get min / max intensity values for image preparation"""
        i_min = float(self.settings._contrast["i_min"])
        i_max = float(self.settings._contrast["i_max"])
        return i_min, i_max
    
    def update_contrast_range(self):
        """Update contrast range using min/max vals of current images in ROI"""
        img = self.images_input["this"]
        if self.settings.roi_rad == [0, 0, 9999, 9999]:
            self.settings.roi_rad = self.settings.roi_abs
        roi = map_roi(self.settings.roi_rad, img.edit_log["pyrlevel"])
        sub = img.img[roi[1]:roi[3], roi[0]:roi[2]]
        i_min, i_max = max([0, sub.min()]), sub.max()
        self.settings.i_min = i_min
        self.settings.i_max = i_max
        print ("Updated contrast range in opt flow: ROI=%s\ni_min=%s\ni_max%s\n" 
            %(roi, i_min, i_max))
    
    def set_images(self, this_img, next_img):
        """Update the current image objects 
        
        :param ndarray this_img: the current image
        :param ndarray next_img: the next image
        """
        self.flow = None
        self.images_input["this"] = this_img
        self.images_input["next"] = next_img
        if any([x.edit_log["crop"] for x in [this_img, next_img]]):
            warn("Input images for optical flow calculation are cropped")
        
        i_min, i_max = self.current_contrast_range() 
        if self.roi_abs == [0, 0, 9999, 9999]:
            self.roi_abs = this_img.roi_abs
        if i_max == 1e30 or self.auto_update_contrast:
            self.update_contrast_range()
         
        self.prep_images()
    
    def prep_images(self):
        """Prepare images for optical flow input"""
        i_min, i_max = self.current_contrast_range()
        self.images_prep["this"] = bytescale(self.images_input["this"].img,
                                             cmin=i_min, cmax=i_max)
        self.images_prep["next"] = bytescale(self.images_input["next"].img,
                                             cmin=i_min, cmax=i_max)
        
    
    def calc_flow(self, this_img=None, next_img=None):
        """Calculate the optical flow field
        
        Uses :func:`cv2.calcOpticalFlowFarneback` to calculate optical
        flow field between two images using the input settings specified in
        ``self.settings``.
        
        Parameters
        ----------
        this_img : Img
            the first of two successive images (if unspecified, the current 
            images in ``self.images_prep`` are used, else, they are updated)
        next_img : Img
            the second of two successive images (if unspecified, the current 
            images in ``self.images_prep`` are used, else, they are updated)
        
        Returns
        -------
        array 
            3D numpy array containing flow displacement field (is also
            assigned to :attr:`flow`)
        """
        if all([isinstance(x, Img) for x in [this_img, next_img]]):
            self.set_images(this_img, next_img)
            
        settings = self.settings._flow_algo
        print "Calculating optical flow"
        t0 = time()
        self.flow = calcOpticalFlowFarneback(self.images_prep["this"],
                                             self.images_prep["next"], 
                                             flags=OPTFLOW_FARNEBACK_GAUSSIAN,
                                             **settings)
        print "Elapsed time flow calculation: %.1f s" %(time() - t0)
        return self.flow 
        
    def get_flow_in_roi(self, roi_rel=None):
        """Get the flow field within in a ROI
        
        Parameters
        ----------
        roi_rel : list
            rectangular ROI aligned with image axis (``[x0, y0, x1, y1]``).
            
            .. note::
            
                The ROI is used as is, i.e. it needs to be defined for 
                current Gauss pyramid level.
            
        Returns
        -------
        array
            3D numpy array containing flow displacement field in ROI
            
        """
        if self.flow is None:
            raise ValueError("No flow field available..")
        try:
            x0, y0, x1, y1 = roi_rel
        except:
            x0, y0, x1, y1 = self.roi
            
        return self.flow[y0 : y1, x0 : x1, :]
    
    def prep_flow_for_analysis(self, mask=None):
        """Get flow field data from all pixels in a certain ROI
        
        This function provides access to the flow field in a certain region
        of interest. In the default case the currently set roi
        :attr:`roi` is used (which is a rectangle aligned with the image 
        x / y axis). Alternatively, a pixel access mask can be provided 
        (e.g. specifying pixels in a rotated rectangle) which is then be used.
        Returns 
        
        Parameters
        ----------
        mask : array
            boolean mask specifying all pixels used to retrieve displacement
            information (True pixels in mask)
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`array`, vector containing all x displacement lengths
            - :obj:`array`, vector containing all y displacement lenghts
            
        """
        fl = self.flow
        if fl is None:
            raise ValueError("Optical flow field not available")
        try:
            if mask.shape == fl.shape[:2]:
                return fl[:,:,0][mask], fl[:,:,1][mask]
            raise Exception
        except:
            fl = self.get_flow_in_roi()
        return fl[:, :, 0].flatten(), fl[:, :, 1].flatten()
        
    
#==============================================================================
#     def prepare_intensity_condition_mask(self, lower_val=0.0, upper_val=1e30):
#         """Apply intensity threshold to input image in ROI and make mask vector
#         
#         Parameters
#         ----------
#         lower_val : float
#             lower intensity value, default is 0.0
#         upper_val : float 
#             upper intensity value, default is 1e30
#         
#         Returns
#         -------
#         ndarray
#             flattened mask which can be used e.g. in 
#             :func:`flow_orientation_histo` as additional input param
#             
#         """
#         x0, y0, x1, y1 = self.roi
#         sub = self.images_input["this"].img[y0 : y1, x0 : x1].flatten()
#         return logical_and(sub > lower_val, sub < upper_val)
#==============================================================================
    
    def to_plume_speed(self, col_dist_img, row_dist_img=None):
        """Convert the current flow field to plume speed array
        
        Parameters
        ----------
        col_dist_img : Img
            image, where each pixel corresponds to horizontal pixel distance 
            in m
        row_dist_img
            optional, image where each pixel corresponds to vertical pixel 
            distance in m (if None, ``col_dist_img`` is also
            used for vertical pixel distances)
        
        """
        if row_dist_img is None:
            row_dist_img = col_dist_img
        if col_dist_img.edit_log["pyrlevel"] != self.pyrlevel:
            raise ValueError("Images have different pyramid levels")
        if not all([x.shape == self.flow.shape[:2] for x in [col_dist_img, 
                                                            row_dist_img]]):
            raise ValueError("Shape mismatch, check ROIs of input images")
        try:
            delt = self.del_t
        except:
            delt = 0
        if delt == 0:
            raise ValueError("Check image acquisition times...")
        dx = col_dist_img.img * self.flow[:,:,0] / delt
        dy = row_dist_img.img * self.flow[:,:,1] / delt
        return sqrt(dx**2 + dy**2)
        
    def get_flow_orientation_img(self, in_roi=False, roi_rel=None):
        """Returns flow angle image
        
        The pixel values correspond to the orientation angles of the vectors of
        the current flow field, where the values correspond to:
        
            - 0 -> upwards (**-y** direction)
            - 90 -> to the right (**+x** direction)
            - -90 -> to the left (**-x** direction)
            - -180, 180 -> down (**+y** direction)
            
        Parameters
        ----------
        in_roi : bool
            get the image for a certain ROI
        roi_rel : :obj:`list`, optional,
            the ROI supposed to be used if ``in_roi`` is True. If None 
            (default) then the current ROI is used (:attr:`roi`).
        
        Returns
        -------
        array
            2D numpy array corresponding to flow orientation image 
            
        """
        if self.flow is None:
            raise ValueError("No flow field available..")
                
        if in_roi:
            fl = self.get_flow_in_roi(roi_rel)
        else:
            fl = self.flow
        fx, fy = fl[:,:,0], fl[:,:,1]
        return rad2deg(arctan2(fx, -fy))
      
    def get_flow_vector_length_img(self, in_roi=False, roi=None):        
        """Returns flow vector length image
        
        The pixel values correspond to the magnitude of the vectors of the
        current flow field.
            
        Parameters
        ----------
        in_roi : bool
            get the image for a certain ROI
        roi : :obj:`list`, optional,
            the ROI supposed to be used if ``in_roi`` is True. If None 
            (default) then the current ROI is used (:attr:`roi`).
        
        Returns
        -------
        array
            2D numpy array corresponding to flow orientation image 
        """
        if self.flow is None:
            raise ValueError("No flow field available..")        
        if in_roi:
            fl = self.get_flow_in_roi(roi)
        else:
            fl = self.flow
        fx, fy = fl[:,:,0], fl[:,:,1]
        return sqrt(fx ** 2 + fy ** 2)
      
    def all_len_angle_vecs_roi(self, mask=None):
        """Get lengths and angles for all pixels in a ROI
        
        Parameters
        ----------
        mask : :obj:`array`, optional
            boolean mask specifying all pixels supposed to be used for 
            data access, defaults to None, in which case the current ROI is
            used (i.e. :attr:`roi`)
            
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`array`, vector with all displacement lengths in ROI / mask
            - :obj:`array`, vector with all displacement angles in ROI / mask
            
        """
        fx, fy = self.prep_flow_for_analysis(mask)
        angles = rad2deg(arctan2(fx, -fy))
        lens = sqrt(fx**2 + fy**2)
        return lens, angles        
    
    def _prep_histo_data(self, count, bins):
        """Check if histo data (count, bins) arrays have same length
        
        If not, shift bins to center of counts
        
        Parameters
        ----------
        count : array
            array containing histogram counts
        bins : array
            array containing bins corresponding to counts
        
        Returns
        -------
        tuple
            2-element tuple containing
    
            - count
            - bins (this was changed if input has length mismatch)
        """
        if len(bins) == len(count):
            return count, bins
        elif len(bins) == len(count) + 1:
            bins = asarray([0.5 * (bins[i] + bins[i + 1]) for\
                                                i in xrange(len(bins) - 1)])
            return count, bins
        else:
            raise ValueError("Invalid input for histogram data")
        
    def flow_orientation_histo(self, pix_mask=None, bin_res_degrees=None,
                               min_length=1.0, **kwargs):
        """Get histogram of orientation distribution of current flow field
        
        Parameters
        ----------
        pix_mask : :obj:`array`, optional
            boolean mask specifying image pixels supposed to be considered for
            the analysis. Is passed to :func:`all_len_angle_vecs_roi`, i.e. if
            this mask is unspecified the histogram data is retrieved using
            the current ROI (:attr:`roi`) for specifying the considered image 
            region. 
            
            .. note::
            
                This is ignored if two arrays containing lengths and angles are
                provided using ``**kwargs`` (for details see below)
                
        bin_res_degrees : int
            bin width of histogram (is rounded to nearest integer if not 
            devisor of 360), if unspecified use :attr:`hist_dir_binres` of 
            settings class
        min_length : float
            minimum length of vectors in order to be considered for histogram, 
            defaults to 1.0
        **kwargs : 
            additional key word args that can be used to pass lens and angles
            arrays (see e.g. :func:`local_flow_params`). Use keywords
            ``lens`` and ``angles`` to pass this information. 
            
        Returns
        -------
        tuple
            3-element tuple containing
            
            - :obj:`array`: histogram counts
            - :obj:`array`: histogram bins
            - :obj:`array`: all angles used to determine the histogram
        """
        try:
            lens = kwargs["lens"]
            angles = kwargs["angles"]
        except:
            lens, angles = self.all_len_angle_vecs_roi(pix_mask)
        if bin_res_degrees is None:
            bin_res_degrees = self.settings.hist_dir_binres
        cond = lens > min_length
        if sum(cond) == 0:
            raise ValueError("No data left for determining orientation histo")
        angs = angles[cond.astype(bool)]
        num_bins = int(round(360 / float(bin_res_degrees)))
        count, bins = histogram(angs, num_bins, range=(-180, 180))
        return count, bins, angs
    
    def flow_length_histo(self, pix_mask=None, bin_res_pix=1, 
                          min_length=1.0, **kwargs):
        """Get histogram of displacement length distribution of flow field
        
        Parameters
        ----------
        pix_mask : :obj:`array`, optional
            boolean mask specifying image pixels supposed to be considered for
            the analysis. Is passed to :func:`all_len_angle_vecs_roi`, i.e. if
            this mask is unspecified the histogram data is retrieved using
            the current ROI (:attr:`roi`) for specifying the considered image 
            region. 
            
            .. note::
            
                This is ignored if two arrays containing lengths and angles are
                provided using ``**kwargs`` (for details see below)
        
        bin_res_pix : int
            bin width in units of pixels, defaults to 2
        min_length : float
            minimum length of vectors in order to be considered for histogram, 
            defaults to 1.0
        **kwargs : 
            additional key word args that can be used to pass lens and angles
            arrays (see e.g. :func:`local_flow_params`). Use keyword
            ``lens`` to pass this information. 
            
        Returns
        -------
        tuple
            3-element tuple containing
            
            - :obj:`array`: histogram counts
            - :obj:`array`: histogram bins
            - :obj:`array`: all lengths used to determine the histogram
            
        """
        try:
            lens = kwargs["lens"]
        except:
            lens, _ = self.all_len_angle_vecs_roi(pix_mask)    
        cond = lens > min_length
        
        lens = lens[cond.astype(bool)]
        if not len(lens) > 0:
            raise ValueError("No data available...")
        upper = int(ceil(lens.max())) + 1
        upper = upper + upper%2
        count, bins = histogram(lens, bins=int(upper/float(bin_res_pix)), 
                                range=(0, upper))
    
        return count, bins, lens
      
    def fit_multigauss_to_histo(self, count, bins, noise_amp=None,
                                max_num_gaussians=None):
        """Fit multi gauss distribution to histogram
        
        Parameters
        ----------
        count : array
            array containing histogram counts
        bins : array
            array containing bins corresponding to counts
        noise_amp : float
            noise amplitude of the histogram data (you don't want to fit all
            the noise peaks). If None, then it is estimated automatically 
            within :class:`MultiGaussFit`.
        max_num_gaussians : int
            Maximum allowed number of Gaussians for :class:`MultiGaussFit`, 
            if None, then default of :class:`MultiGaussFit` is used
            
        Returns
        -------
        tuple 
            2-element tuple containing
            
            - *MultiGaussFit*: fit object
            - *bool*: success True / False
        """
        ok = True
        c, x = self._prep_histo_data(count, bins)
        fit = MultiGaussFit(c, x, noise_amp=noise_amp,
                            max_num_gaussians=max_num_gaussians,
                            do_fit=False) #make sure the object is initiated
        ok = False
        try:
            if fit.run_optimisation():
                ok = True
        except:
            pass
        return fit, ok
        
    def fit_orientation_histo(self, count, bins, noise_amp=None, 
                              max_num_gaussians=None, **kwargs):
        """Fit multi gauss distribution to flow orientation histogram
        
        Parameters
        ----------
        count : array
            histogram counts (see :func:`flow_orientation_histo`)
        bins : array
            histogram bins (see :func:`flow_orientation_histo`)
        noise_amp : :obj:`float`, optional
            minimum amplitude required for peaks in histogram in order to be 
            considered for multi gauss fit, if None (default) use 5% of max
            count
        max_num_gaussians : :obj:`int`, optional
            maximum number of Gaussians fitted to the distributions, if None
            (default) then use ``self.settings.hist_dir_gnum_max``
            
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`MultiGaussFit`, the fit object
            - bool, fit success
        """
        if max_num_gaussians is None:
            max_num_gaussians = self.settings.hist_dir_gnum_max
        if noise_amp is None:
            #set minimum amplitude for multi gauss fit 5% of max amp
            noise_amp = max(count) * 0.05 
        
        fit, ok = self.fit_multigauss_to_histo(count, bins, 
                                               noise_amp=noise_amp,
                                               max_num_gaussians=
                                               max_num_gaussians)
        return fit, ok
    
    def analyse_length_histo(self, count, bins):
        """Get mean and sigma of length histogram using 1. and 2nd moment
        
        Parameters
        ----------
        count : array
            array with counts per bin
        bins : array
            array containing bins
            
        Returns
        -------
        tuple
            2-element tuple, containing
            
            - :obj:`float`: expectation value mu
            - :obj:`float`: corresponding standard deviation
        """
        c, x = self._prep_histo_data(count, bins)

        mu = nth_moment(x, c, 0, 1)
        sigma = sqrt(nth_moment(x, c, mu, 2))
        print("Avg. displ. length: %.1f +/- %.1f" %(mu, sigma))
        return (mu, sigma)
        
    def fit_length_histo(self, count, bins, noise_amp=None,
                         max_num_gaussians=4, **kwargs):
        """Apply multi gauss fit to length distribution histogram
        
        Parameters
        ----------
        count : array
            histogram counts (see :func:`flow_orientation_histo`)
        bins : array
            histogram bins (see :func:`flow_orientation_histo`)
        noise_amp : :obj:`float`, optional
            minimum amplitude required for peaks in histogram in order to be 
            considered for multi gauss fit, if None (default) use 5% of max
            count
        max_num_gaussians : :obj:`int`, optional
            maximum number of Gaussians fitted to the distributions, if None
            (default) then use ``self.settings.hist_dir_gnum_max``
            
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`MultiGaussFit`, the fit object
            - bool, fit success
        """
        if noise_amp is None:
            noise_amp = max(count) * 0.05
            
        fit, ok = self.fit_multigauss_to_histo(count, bins, 
                                               noise_amp=noise_amp,
                                               max_num_gaussians=
                                               max_num_gaussians)
        return fit, ok
    
    def get_main_flow_field_params(self, **kwargs):
        """Old name of :func:`local_flow_params`"""
        return self.local_flow_params(**kwargs)
        
    def local_flow_params(self, line=None, pix_mask=None, noise_amp=None, 
                          min_count_frac=None, min_length=None):
        """Histogram based statistical analysis of flow field in current ROI
        
        This function analyses histograms of the current flow field within
        a  ROI in order to find the predominant movement direction (within the
        ROI) and the corresponding predominant displacement length.
        
        Parameters
        ----------
        line : :obj:`LineOnImage`, optional
            if provided, then the ROI corresponding to the line orientation 
            is used (see :func:`get_rotated_roi_mask` in :class:`LineOnImage` 
            objects). If unspecified the current roi (:attr:`roi`) is used.
        pix_mask : :obj:`array`, optional
            boolean mask specifying image pixels supposed to be considered for
            the analysis, e.g. only plume pixels (determined applying a 
            tau threshold to a tau image).
        noise_amp : :obj:`float`, optional
            this number specifies the minimum amplitude for individual peaks
            in the histograms (for multiple Gaussian fit). If unspecified here
            it will be set automatically in the corresponding methods
            :func:`fit_length_histo` and :func:`fit_orientation_histo`.
        min_count_frac : :obj:`float`, optional
            determins the minimum required number of significant vectors in 
            current ROI for histogram analysis (i.e. if ROI is NxM pixels and
            ``min_count_frac=0.1``, then at least (MxN)*0.1 pixels need to 
            remain after applying ``cond_mask_flat`` and exclusion of vectors
            shorter than current minimum length ``self.settings.min_length``)
        min_length : :obj:`float`, optional
            minimum length of vectors required in order to be considered for 
            historgram analysis
            
        Returns
        -------
        dict
            dictionary containing results of the analysis
                
        """
        del_t = self.del_t
        res = od([("_len_mu_norm"   ,   nan), #normalised displ. len [s-1]
                  ("_len_sigma_norm",   inf), #error norm. displ. len [s-1]
                  ("_dir_mu"        ,   nan), #predominant displ. dir. [deg]
                  ("_dir_sigma"     ,   inf), #error pred. displ. dir. [deg]
                  ("_del_t"         ,   del_t), #time diff 'this' -> 'next'
                  ("_start_acq"     ,   self.current_time), #time stamp 'this'
                  ("_significance"  ,   0.0), #fraction of usable pixels in ROI
                  ("_add_gauss_dir" ,   []), 
                  ("pix_mask"       ,   None),
                  ("fit_dir"        ,   None)])
        # EVALUATE INPUT AND INIT PARAMETERS
        
        # get current minimum length required to be included into statistics
        if min_length is None:
            min_length = self.settings.min_length
        
        # minimum fraction of significant vectors required (relative to the 
        # total number of vectors in ROI)                    
        if min_count_frac is None:
            min_count_frac = self.settings.min_count_frac
    
        #init pixel access mask 
        mask = pix_mask
        if isinstance(line, LineOnImage):
            print "Using rotated ROI mask for pixel access"""
            m = line.get_rotated_roi_mask(self.flow.shape[:2])
            if mask is None:
                mask = m
            else:
                mask = (mask * m).astype(bool)

        res["pix_mask"] = mask
        
        # vectors containing lengths and angles of flow field in ROI (if None
        # of the two input masks are specified, then the current ROI is used)
        lens, angles = self.all_len_angle_vecs_roi(mask)
        
        # get histogram of data exceeding minimum length
        try:
            (count, 
             bins, 
             angs) = self.flow_orientation_histo(lens=lens,
                                                 angles=angles,
                                                 min_length=min_length)  
        except:
            warn("Retrieval of flow orientation histogram failed")
            return res
        
        # Check if enough vectors are left to go on with the analysis
        frac = len(angs) / float(len(angles))
        if frac < min_count_frac:
            warn("Aborted retrieval of main flow field paramaters"
                "only %d %% of the vectors in current ROI are longer than "
                "minimum required length %.1f"
                %(frac*100, min_length))
            return res
        # Now try to apply multi gauss fit to histogram distribution
        fit, ok = self.fit_orientation_histo(count, bins, noise_amp)
        res["fit_dir"] = fit 
        if not fit.has_results():
            warn("Could not retrieve main flow field parameters, multi gauss "
                 "fit of orientation histogram failed")
            raise Exception
            return res
        sigma_tol = self.settings.hist_dir_sigma
        #analyse the fit result (i.e. find main gauss peak and potential other
        #significant peaks)
        (dir_mu, 
         dir_sigma, 
         tot_num, 
         add_gaussians) = fit.analyse_fit_result(sigma_tol + 1)
         
        res["_dir_mu"] = dir_mu
        res["_dir_sigma"] = dir_sigma
        res["_add_gauss_dir"] = add_gaussians
        
        print("Predominant movement direction: %.1f +/- %.1f" %(dir_mu,
                                                                dir_sigma))
                                                                
        for g in add_gaussians:
            sign = int(fit.integrate_gauss(*g) * 100 / tot_num)
            if sign > 20: #other peak exceeds 20% of main peak
                warn("Optical flow histogram analysis:\n"
                     "Detected additional gaussian in orientation histogram:\n"
                     "%sSignificany: %s %%\n" %(fit.gauss_str(g), sign))
        
        #limit range of reasonable orientation angles...
        dir_low = dir_mu - dir_sigma * sigma_tol
        dir_high = dir_mu + dir_sigma * sigma_tol
        
        #... and make a mask from it including min length condition
        cond = logical_and(angles > dir_low, 
                           angles < dir_high) * (lens > min_length)

        # Check if enough vectors are left to go on with the analysis
        frac = sum(cond) / float(len(lens))
        if frac < min_count_frac:
            warn("Aborted retrieval of main flow field parameters"
                "only %d %% of the vectors in current ROI remain after "
                "limiting angular range from fit result of orientation histo"
                %(frac*100))
            return res
        res["_significance"] = frac
        lens = lens[cond]
        
        count, bins, _ = self.flow_length_histo(lens=lens)
        len_mu, len_sigma = self.analyse_length_histo(count, bins)
        
        res["_len_mu_norm"] = len_mu / del_t #normalise to 1s ival
        res["_len_sigma_norm"] = len_sigma / del_t #normalise to 1s ival
        return res
    
    def apply_median_filter(self, shape=(3,3)):
        """Apply a median filter to flow field, i.e. to both flow images (dx, dy
        stored in self.flow) individually
        
        :param tuple shape (3,3): size of the filter
        """
        
        self.flow[:,:,0] = median_filter(self.flow[:,:,0], shape)
        self.flow[:,:,1] = median_filter(self.flow[:,:,1], shape)
    
    @property
    def del_t(self):
        """Return time difference in s between both images"""
        t0, t1 = self.get_img_acq_times()
        return (t1 - t0).total_seconds()
    
    @property
    def current_time(self):
        """Return acquisition time of current image"""
        try:
            return self.images_input["this"].meta["start_acq"]
        except:
            warn("Image acq. time cannot be accessed in OptflowFarneback")
            return datetime(1900, 1, 1)
        
    def get_img_acq_times(self):
        """Return acquisition times of current input images
        
        :return:
            - datetime, acquisition time of first image
            - datetime, acquisition time of next image
            
        """
        try:
            t0 = self.images_input["this"].meta["start_acq"]
            t1 = self.images_input["next"].meta["start_acq"]
        except: 
            warn("Image acquisition times cannot be accessed in" 
                        " OptflowFarneback")
            t0 = datetime(1900, 1, 1)
            t1 = datetime(1900, 1, 1, 0, 0, 1)
        return t0, t1
    """
    Plotting / visualisation etc...
    """  
    def plot_orientation_histo(self, pix_mask=None, min_length=None,
                               apply_fit=True, ax=None, 
                               tit="Orientation histo", color="b",
                               **fit_settings):
        """Plot flow orientation histogram
        
        Plots a histogram of the orientation angles of the flow vectors w
        within a certain ROI. By default, vectors shorter then 
        ``self.settings.min_length`` are excluded from the histogram, if you
        want a histogram including the short vectors, provide input parameter
        ``min_length=0.0``.
        
        Param
        
        """
        if ax is None:
            fig, ax = subplots(1,1)
        if min_length is None:
            min_length = self.settings.min_length
        lens, angles = self.all_len_angle_vecs_roi(pix_mask)
        try:
            (count, 
             bins, 
             angs) = self.flow_orientation_histo(lens=lens,
                                                 angles=angles,
                                                 min_length=min_length) 
        except:
            warn("Failed to retrieve orientation histogram: probably no vectors "
                "left for retrieval of histogram. Current time: %s "
                %self.current_time)
            return (ax, None, None)
        w = bins[1] - bins[0]
        ax.bar(bins[:-1], count, width=w, color=color, ec="none", alpha=0.3,
               label="Histogram")
        
        mu, sigma = 0, 180
        if apply_fit:
            fit, ok = self.fit_orientation_histo(count, bins, **fit_settings)
            if fit.has_results():
                mu, sigma,_,_ = fit.analyse_fit_result()
                fit.plot_multi_gaussian(ax=ax, label="Multi-Gauss fit",
                                        color=color)
                tit += (r": $\mu (+/-\sigma$) = %.1f (+/- %.1f)" 
                    %(mu, sigma))
                ax.plot([mu, mu], [0, count.max()*1.05], color=color, ls="-")
                ax.plot([mu-sigma, mu-sigma], [0, count.max()*1.05], 
                        color=color, ls="--")
                ax.plot([mu+sigma, mu+sigma], [0, count.max()*1.05], 
                        color=color, ls="--")
            else:
                tit += ": Fit failed..."
        ax.set_title(tit, fontsize=11)      
        ax.set_xlim([-180, 180])    
        ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
        ax.grid()
        return ax, mu, sigma
    
    def plot_length_histo(self, pix_mask=None, dir_low=-180, dir_high=180, 
                          min_length=None, apply_fit=True, ax=None, 
                          tit="Length histo", color="b", **fit_settings):
        if ax is None:
            fig, ax = subplots(1,1)
        if min_length is None:
            min_length = self.settings.min_length
        lens, angles = self.all_len_angle_vecs_roi(pix_mask)
        #... and make a mask from it including min length condition
        cond = logical_and(angles > dir_low, angles < dir_high)
        lens, angles = lens[cond], angles[cond]
        try:
            (count, 
             bins, 
             lens) = self.flow_length_histo(lens=lens, angles=angles,
                                            min_length=min_length) 
        except:
            warn("Failed to retrieve length histogram: probably no vectors "
                "left for retrieval of histogram. Current time: %s "
                %self.current_time)
        w = bins[1] - bins[0]
        ax.bar(bins[:-1], count, width=w, color=color, ec="none", alpha=0.3,
               label="Histogram")
        
        mu, sigma = self.analyse_length_histo(count, bins)
        
        tit += (r": $\mu (+/-\sigma$) = %.1f (+/- %.1f)" %(mu, sigma))
        if apply_fit:
            fit, ok = self.fit_length_histo(count, bins, **fit_settings)
            if fit.has_results():
                fit.plot_multi_gaussian(ax=ax, label="Multi-Gauss fit",
                                        color=color)
        ax.set_title(tit, fontsize=11)      
        ax.set_xlim([0, int(bins.max()) + 1])
        ax.plot([mu, mu], [0, count.max()*1.05], color=color, ls="-")
        ax.plot([mu-sigma, mu-sigma], [0, count.max()*1.05], 
                color=color, ls="--")
        ax.plot([mu+sigma, mu+sigma], [0, count.max()*1.05], 
                color=color, ls="--")
        ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
        ax.grid()
        return ax
        
    def plot_flow_histograms(self, line=None, pix_mask=None, apply_fits=True, 
                             **kwargs):
        """Plot detailed information about optical flow histograms
        
        Parameters
        ----------
        line : LineOnImage
            retrieval line for which historgrams plotted, if None (default), 
            then the curr
        """
        if self.flow is None:
            raise ValueError("No flow field available..")
        roi_temp = self.roi_abs
        
        if isinstance(line, LineOnImage):
            self.roi_abs = line.line_frame_abs
    
        aspect = self.images_input["this"].xy_aspect
        #set up figure and axes
        fig = figure(figsize=(16,8))

        #three strangely named axes for top row 
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        
        ax4 = fig.add_subplot(2,3,4)        
        ax5 = fig.add_subplot(2,3,5)
        ax6 = fig.add_subplot(2,3,6)
        
        #draw the optical flow image
        self.draw_flow(0, add_cbar=True, ax=ax1)
        self.draw_flow(1, add_cbar=True, ax=ax4)
        
        #load and draw the length and angle image
        angle_im = self.get_flow_orientation_img()
        len_im = self.get_flow_vector_length_img()
        angle_im_disp = ax2.imshow(angle_im, interpolation='nearest',
                                   vmin=-180, vmax=180, cmap="RdBu")
        ax2.set_title("Displacement orientation", fontsize=11)        
        fig.colorbar(angle_im_disp, ax=ax2)
        
        len_im_disp = ax5.imshow(len_im, interpolation='nearest',  
                                 cmap="Blues")
        fig.colorbar(len_im_disp, ax=ax5)
        ax5.set_title("Displacement lengths", fontsize=11)        
        
        set_ax_lim_roi(self.roi, ax2, xy_aspect=aspect)
        set_ax_lim_roi(self.roi, ax5, xy_aspect=aspect)
        set_ax_lim_roi(self.roi, ax4, xy_aspect=aspect)
        mask = pix_mask
        c="g"
        if isinstance(line, LineOnImage):
            print "Using rotated ROI mask for pixel access"""
            m = line.get_rotated_roi_mask(self.flow.shape[:2])
            if mask is None:
                mask = m
            else:
                mask = (mask * m).astype(bool)
                
            line.plot_line_on_grid(ax=ax1, include_roi_rot=1)
            line.plot_line_on_grid(ax=ax2, include_roi_rot=1)
            line.plot_line_on_grid(ax=ax4, include_roi_rot=1)
            line.plot_line_on_grid(ax=ax5, include_roi_rot=1)
            c=line.color
            
        _, mu, sigma = self.plot_orientation_histo(pix_mask=mask, 
                                                   apply_fit=True, ax=ax3, 
                                                   color=c)
        low, high = mu-sigma, mu+sigma
        self.plot_length_histo(pix_mask=mask, apply_fit=False, ax=ax6, 
                               dir_low=low, dir_high=high, color=c)
        
        fig.tight_layout()
        self.roi_abs = roi_temp
        return fig
        
    def calc_flow_lines(self, in_roi=True, roi=None):
        """Determine line objects for visualisation of current flow field
        
        Parameters
        ----------
        in_roi : bool
            if True (default), then the lines are calculated for pixels
            within ROI (either specified by 2. input param and else
            :attr:`roi_abs` is used).
        roi : list
            Region of interest supposed to be displayed
            
        Returns
        -------
        tuple
            the line coordinates
            
        """
        settings = self.settings
        step, len_thresh = settings.disp_skip, settings.disp_len_thresh
        #get the shape of the rectangle in which the flow was determined
        if in_roi:
            flow = self.get_flow_in_roi(roi)
        else:
            flow = self.flow
        h, w = flow.shape[:2]
        #create and flatten a meshgrid 
        y, x = mgrid[step / 2: h : step, step / 2: w : step].reshape(2, -1)
        fx, fy = flow[y, x].T
        
        if len_thresh > 0:
            #use only those flow vectors longer than the defined threshold
            cond = sqrt(fx**2 + fy**2) > len_thresh
            x, y, fx, fy = x[cond], y[cond], fx[cond], fy[cond]
        # create line endpoints
        lines = int32(vstack([x, y,x + fx, y + fy]).T.reshape(-1,2,2))
        return lines
    
    def plot(self, **kwargs):
        """Draw current flow field onto image
        
        Wrapper for :func:`draw_flow`
        
        :param **kwargs: key word args (see :func:`draw_flow`)
        """
        return self.draw_flow(**kwargs)
    
    def draw_flow(self, in_roi=False, roi_abs=None, add_cbar=False, ax=None):
        """Draw the current optical flow field
        
        Parameters
        ----------
        in_roi : bool
            if True, the flow field is plotted in a cropped image area 
            else, the whole image is drawn 
        roi_abs : :obj:`list`, optional
            region of interest for which the flow field is drawn (in absolute
            image coordinates, i.e. is converted to current pyrlevel). If None, 
            then the :attr:`roi_abs` is used.
        add_cbar : bool
            if True, a colorbar is added to the plot (note that the images
            are converted into 8 bit before the flow is calculated, therefore 
            the intensity range of the displayed image is between 0 and 256).
        ax : Axes
            matplotlib axes object
            
        Returns
        -------
        Axes
            the plot axes
        """
        if self.flow is None:
            print "Could not draw flow, no flow available"
            return
        if ax is None:
            fig, ax = subplots(1,1)
        else:
            fig = ax.figure
        
        i_min, i_max = self.current_contrast_range()
    
        img = self.images_input["this"]
        if not check_roi(roi_abs):
            roi_abs = self.roi_abs
        
        roi_rel = map_roi(roi_abs, self.pyrlevel)
        disp = bytescale(img.img, cmin=i_min, cmax=i_max)
        if img.is_tau: #invert intensities
            disp = (255 - disp)
        
        if add_cbar:
            disp_temp = ax.imshow(disp, cmap="gray")
            fig.colorbar(disp_temp, ax=ax)

        disp = cvtColor(disp, COLOR_GRAY2BGR) 
       
        lines = self.calc_flow_lines(in_roi, roi_rel)
        tit = r"1. img"
        x0, y0, w, h = roi2rect(roi_rel)
        if not in_roi:
            ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec="c"))
            x0, y0 = 0, 0
        else:
            tit += " (in ROI)"
        print "Drawing optical flow field into plot..."
        for (x1, y1), (x2, y2) in lines:
            line(disp, (x0 + x1, y0 + y1), (x0 + x2, y0 + y2),(0, 255, 255), 1)
            circle(disp, (x0 + x2, y0 + y2), 1, (255, 0, 0), -1)
        ax.imshow(disp)
        if in_roi:
            set_ax_lim_roi(roi_rel, ax)
            #img = img.crop(roi_abs=roi_abs, new_img=True)
#==============================================================================
#         for (x1, y1), (x2, y2) in lines:
#             ax.add_artist(Line2D([x0 + x1, x0 + x2], [y0 + y1, y0 + y2],
#                                 color="c"))
#             ax.add_patch(Circle((x0 + x2, y0 + y2), 1, ec="r", fc="r"))
#         #ax.imshow(disp)
#==============================================================================
        
        try:
            tit += (r": %s \n $\Delta$t (next) = %.2f s" %(\
                self.get_img_acq_times()[0].strftime("%H:%M:%S"), self.del_t))
            tit = tit.decode("string_escape")
        except:
            pass
        
        ax.set_title(tit, fontsize=12)
        return ax    
        
    def live_example(self):
        """Show live example using webcam"""
        cap = VideoCapture(0)
        ret,im = cap.read()
        gray = cvtColor(im,COLOR_BGR2GRAY)
        self.images_prep["this"] = gray
        while True:
            # get grayscale image
            ret, im = cap.read()
            self.images_prep["next"] = cvtColor(im,COLOR_BGR2GRAY)
            
            # compute flow
            flow = self.calc_flow()
            self.images_prep["this"] = self.images_prep["next"]
        
            # plot the flow vectors
            vis = cvtColor(self.images_prep["this"], COLOR_GRAY2BGR)
            lines = self.calc_flow_lines(False)
            for (x1,y1),(x2,y2) in lines:
                line(vis,(x1,y1),(x2,y2),(0,255,255),1)
                circle(vis,(x2,y2),1,(255,0,0), -1)
            imshow("Optical flow live view", vis)
            if waitKey(10) == 27:
                self.flow = flow
                break
            
    """
    Connections etc.
    """
    def connect_histo(self,canvasWidget):
        self._interactive_contrast_control = canvasWidget
        
    
    """
    Magic methods (overloading)
    """
    def __call__(self, item=None):
        if item is None:
            print "Returning current optical flow field, settings: "
            print self.settings
            return self.flow
        for key, val in self.__dict__.iteritems():
            try:
                if val.has_key(item):
                    return val[item]
            except:
                pass 

### OLD CLASS NAMES    
class OpticalFlowFarnebackSettings(FarnebackSettings):
    """Old name of :class:`FarnebackSettings`"""
    def __init__(self, *args, **kwargs):
        super(OpticalFlowFarnebackSettings, self).__init__(*args, **kwargs)
        warn("You are using an old name (OpticalFlowFarnebackSettings) for "
            "class FarnebackSettings")

class OpticalFlowFarneback(OptflowFarneback):
    """Old name of :class:`OptflowFarneback`"""
    def __init__(self, *args, **kwargs):
        super(OpticalFlowFarneback, self).__init__(*args, **kwargs)
        warn("You are using an old name OpticalFlowFarneback for class"
            "OptflowFarneback") 
