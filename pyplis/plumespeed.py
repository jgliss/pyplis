# -*- coding: utf-8 -*-
"""
Module containing features related to plume velocity analysis
"""
from time import time
from numpy import mgrid,vstack,int32,sqrt,arctan2,rad2deg, asarray, sin, cos,\
    logical_and, histogram, ceil, ones, roll, argmax, arange, ndarray,\
    deg2rad, nan, inf, dot, mean, e
from numpy.linalg import norm
from traceback import format_exc
from warnings import warn
from datetime import datetime
from collections import OrderedDict as od
from matplotlib.pyplot import subplots, figure, Figure, Circle, Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.stats.stats import pearsonr
    
from pandas import Series

from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN,\
    cvtColor,COLOR_GRAY2BGR,line,circle,VideoCapture,COLOR_BGR2GRAY,\
    waitKey, imshow

from .helpers import bytescale, check_roi, map_roi, roi2rect
from .optimisation import MultiGaussFit
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
    
    This class is used to store statistical local plume displacment 
    information, i.e. the predominant local displacement direction and the 
    corresponding displacment length both with a given uncertainty. 
    Further, the time difference between the two frames used to estimate the 
    displacement parameters is stored. This class is for instance used for
    plume displacement properties derived using 
    :func:`get_main_flow_field_params` from :class:`OpticalFlowFarneback`
    which is based on a statistical analysis of histograms derived from 
    a dense optical flow algorithm.
    """
    def __init__(self, **kwargs):
        self._len_mu = []
        self._len_sigma = []
        self._dir_mu = []
        self._dir_sigma = []
        self._start_acq = []
        self._del_t = []
        self._add_gauss_len = []
        self._add_gauss_dir = []
    
    @property
    def len_mu(self):
        """Return current displacement length vector"""
        return asarray(self._len_mu)
    
    @property
    def len_sigma(self):
        """Return current displacement length std vector"""
        return asarray(self._len_sigma)
    
    @property
    def dir_mu(self):
        """Return current displacement orientation vector"""
        return asarray(self._dir_mu)
    
    @property
    def dir_sigma(self):
        """Return current displacement orientation std vector"""
        return asarray(self._dir_sigma)
        
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
        
    def get_and_append_from_farneback(self, optflow_farneback, **kwargs):
        """Retrieve main flow field parameters from Farneback engine
        
        Calls :func:`get_main_flow_field_params` from 
        :class:`OpticalFlowFarneback` engine and appends the results to 
        the current data
        """
        res = optflow_farneback.get_main_flow_field_params(**kwargs)
        for key, val in res.iteritems():
            if self.__dict__.has_key(key):
                self.__dict__[key].append(val)
    
    def displacement_vector(self, index = -1):
        """Returns displacement vector for one of the data points in the series
        
        :param int index: index of data point (default is -1, which means the
            last index)
        """
        return asarray([sin(deg2rad(self.dir_mu[index])),
                        -cos(deg2rad(self.dir_mu[index]))])\
                        * self.len_mu[index]
                        
    def get_velocity(self, idx, pix_dist_m, pix_dist_m_err=None, 
                     normal_vec=None):
        """Determine plume velocity from displacements
        
        :param pix_dist_m: in plume distance between pixels (comes e.g.
            from :class:`MeasGeometry` object)
        :param pix_dist_m_err: uncertainty in pixel distance (if None, then
            a default uncertainty of 10% is assumed)
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
        return v, verr
        
    def plot_velocities(self, pix_dist_m=None, pix_dist_m_err=None, ax=None,
                        **kwargs):
        """Plot time series of velocity evolution 
        
        :param pix_dist_m: detector pixel distance in m, if unspecified, then
            velocities are plotted in units of pix/s
        :param pix_dist_m_err: uncertainty in pixel to pixel distance in m
        """
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
        
    def plot_directions(self, ax=None, **kwargs):
        """Plot time series of displacement orientation"""
        if ax is None:
            fig, ax = subplots(1,1)
        if not "color" in kwargs:
            kwargs["color"] = "b"
        angle = Series(self.dir_mu, self.start_acq)
        angle_upper = Series(self.dir_mu + self.dir_sigma, self.start_acq)
        angle_lower = Series(self.dir_mu - self.dir_sigma, self.start_acq)
        ax.plot(angle.index, angle, **kwargs)
        ax.fill_between(angle.index, angle_lower, angle_upper, alpha=0.1,
                        **kwargs)
        ax.set_ylabel(r"$\Theta$ [degrees]", fontsize=14)
        #ax.grid()
        return ax
    
class OpticalFlowFarnebackSettings(object):
    """Settings for optical flow Farneback calculations and visualisation
    
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
            
        #. :attr:`hist_len_how`: method to estimate the average displacement \
            length from the flow length histogram (see \
            :func:`flow_length_histo`) when performing \
            :func:`get_main_flow_field_params` analysis. Choose from:
            
                - *argmax*: the mean displacement length corresponds to \
                    histogram bin with largest count. This method is faster \
                    and more robust compared to method 
                - *multigauss*: apply :class:`MultiGaussFit` to length \
                    histogram and set mean displacement length based on \
                    x-position of main peak.
                    
    Parameters
    ----------
    **settings 
        valid keyword arguments for class attributes, e.g.::
        
            stp = OpticalFlowFarnebackSettings(i_min=0, i_max=3500,
                                               iterations=8)
        
    """
    def __init__(self, **settings):
        self._contrast = od([("i_min"       ,   0),
                             ("i_max"       ,   1e30),
                             ("auto_update" ,   True)])
        
        self._flow_algo = od([("pyr_scale"  ,   0.5), 
                              ("levels"     ,   4),
                              ("winsize"    ,   20), 
                              ("iterations" ,   5), 
                              ("poly_n"     ,   5), 
                              ("poly_sigma" ,   1.1)])
                            
        self._analysis = od([("roi_abs"             ,   [0, 0, 9999, 9999]),
                             ("min_length"          ,   1.0),
                             ("hist_dir_sigma"      ,   3),
                             ("hist_dir_gnum_max"   ,   5),
                             ("hist_len_how"        ,   "multigauss"),
                             ("hist_len_gnum_max"   ,   3)]) #only applies is hist_len_how=="multigauss"
        
        self._display = od([("disp_skip"            ,   10),
                            ("disp_len_thresh"      ,   3)])
        
        
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
    def auto_update(self):
        """Contrast is automatically updated based on min / max intensities
        
        If active, then :attr:`i_min` and :attr:`i_max` are updated
        automativally whenever new images are assigned to a 
        :class:`OpticalFlowFarneback` using method :func:`set_images`. The
        update is performed based on min / max intensities of the images in
        the current ROI
        """
        return self._contrast["auto_update"]
    
    @auto_update.setter
    def auto_update(self):
        """Upper intensity limit for image contrast preparation"""
        return self._contrast["auto_update"]
        
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
    def hist_dir_sigma(self):
        """Sigma tolerance value for mean flow analysis"""
        return self._analysis["hist_dir_sigma"]
    
    @hist_dir_sigma.setter
    def hist_dir_sigma(self, val):
        if not 1 <= val <= 4:
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
    def hist_len_gnum_max(self):
        """Max number of gaussians for multigauss fit of length histo"""
        return self._analysis["hist_len_gnum_max"]
    
    @hist_len_gnum_max.setter
    def hist_len_gnum_max(self, val):
        if not val > 0:
            raise ValueError("Value must be larger than 0")
        self._analysis["hist_len_gnum_max"] = val
        
    @property
    def hist_len_how(self):
        """Method to estimate the mean displacement length from histo"""
        return self._analysis["hist_len_how"]
    
    @hist_len_how.setter
    def hist_len_how(self, val):
        if not val in ["argmax", "multigauss"]:
            raise ValueError("Invalid input: choose from argmax or multigauss")
        self._analysis["hist_len_how"] = val
      
    @property
    def disp_skip(self):
        """Return current pixel skip value for displaying flow field"""
        return self._display["disp_skip"]
    
    @property
    def disp_len_thresh(self):
        """Return current pixel skip value for displaying flow field"""
        return self._display["disp_len_thresh"]    
        
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


    
class OpticalFlowFarneback(object):
    """Implementation of Optical flow Farneback algorithm of OpenCV library
    
    Engine for autmatic optical flow calculation, for settings see
    :class:`OpticalFlowFarnebackSettings`. The calculation of the flow field
    is performed for two consecut
    Advanced post processing analysis of flow field in order to automatically
    identify and distinguish reliable output from unreliable (the latter is 
    mainly in low contrast regions).      
    
    .. note::
    
        Image handling withhin this object is kept on low level base, i.e. on 
        numpy arrays. Input :class:`pyplis.Image.Img` also works on input but
        this class does not provide any funcitonality based on the functionality
        of :class:`pyplis.Image.Img` objects (i.e. access of meta data, etc).
        As a result, this engine cannot perform any wind speed estimates (it
        would need to know about image acquisition times for that and further
        about the measurement geometry) but only provides functionality to 
        calculate and analyse optical flow fields in detector pixel 
        coordinates.
    """
    def __init__(self, first_img=None, next_img=None, name="", **settings):        
        """Initialise the Optical flow environment"""
        self.name = name
    
        #settings for determination of flow field
        self.settings = OpticalFlowFarnebackSettings(**settings)

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
        print ("Auto update contrast mode was updated in OpticalFlowFarneback "
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
        
    def set_mode_auto_update_contrast_range(self, value = True):
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
        roi = map_roi(self.roi_abs, img.edit_log["pyrlevel"])
        sub = img.img[roi[1]:roi[3], roi[0]:roi[2]]
        i_min, i_max = max([0, sub.min()]), sub.max()
        self.settings.i_min = i_min
        self.settings.i_max = i_max
        print ("Updated contrast range in opt flow, i_min = %s, i_max = %s" 
                                                            %(i_min, i_max))
    
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
        
    
    def calc_flow(self, this_img = None, next_img = None):
        """Calculate the optical flow field
        
        Uses :func:`cv2.calcOpticalFlowFarneback` to calculate optical
        flow field between two images using the input settings specified in
        ``self.settings``.
        
        :param ndarray this_img (None): the first of two successive images (if 
            unspecified, the current images in ``self.images_prep`` are used, 
            else, they are updated)
        :param ndarray next_img (None): the second of two successive images (if 
            unspecified, the current images in ``self.images_prep`` are used, 
            else, they are updated)
            
        """
        if all([isinstance(x, Img) for x in [this_img, next_img]]):
            self.set_images(this_img, next_img)
            
        settings = self.settings._flow_algo
        print "Calculating optical flow"
        t0 = time()
        self.flow = calcOpticalFlowFarneback(self.images_prep["this"],\
                self.images_prep["next"], flags = OPTFLOW_FARNEBACK_GAUSSIAN,\
                **settings)
        print "Elapsed time flow calculation: %s" %(time() - t0)
        return self.flow 
        
    def get_flow_in_roi(self):
        """Get the flow field in the current ROI"""
        if self.flow is None:
            raise ValueError("No flow field available..")
        x0, y0, x1, y1 = self.roi
        return self.flow[y0 : y1, x0 : x1, :]
    
    def _prep_flow_for_analysis(self):
        """Flatten the flow fields for analysis
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - ``ndarray``, vector containing all x displacement lengths
            - ``ndarray``, vector containing all y displacement lenghts
            
        """
        fl = self.get_flow_in_roi()
        return fl[:,:,0].flatten(), fl[:,:,1].flatten()
    
    def prepare_intensity_condition_mask(self, lower_val=0.0, upper_val=1e30):
        """Apply intensity threshold to input image in ROI and make mask vector
        
        Parameters
        ----------
        lower_val : float
            lower intensity value, default is 0.0
        upper_val : float 
            upper intensity value, default is 1e30
        
        Returns
        -------
        ndarray
            flattened mask which can be used e.g. in 
            :func:`flow_orientation_histo` as additional input param
            
        """
        x0, y0, x1, y1 = self.roi
        sub = self.images_input["this"].img[y0 : y1, x0 : x1].flatten()
        return logical_and(sub > lower_val, sub < upper_val)
    
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
        
    def get_flow_orientation_img(self, in_roi = False):
        """Returns image corresponding to flow orientation values in each pixel"""
        if self.flow is None:
            raise ValueError("No flow field available..")
                
        if in_roi:
            fl = self.get_flow_in_roi()
        else:
            fl = self.flow
        fx, fy = fl[:,:,0], fl[:,:,1]
        return rad2deg(arctan2(fx, -fy))
      
    def get_flow_vector_length_img(self, in_roi=False):        
        """Returns image corresponding to displacement length in each pixel"""
        if self.flow is None:
            raise ValueError("No flow field available..")        
        if in_roi:
            fl = self.get_flow_in_roi()
        else:
            fl = self.flow
        fx, fy = fl[:,:,0], fl[:,:,1]
        return sqrt(fx ** 2 + fy ** 2)
      
    @property
    def flow_len_and_angle_vectors(self):
        """Returns vector containing all flow lengths
        
        :return:
            - ndarray, all displacement lengths of current flow field 
                (flattened)
            -ndarray, all orientation angles of current flow field (flattened)
            
        """
        fx, fy = self._prep_flow_for_analysis()
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
            
    def _estimate_mean_len_argmax(self, count, bins):
        """Estimate the mean displacement length and error for mode argmax
        
        See :class:`OpticalFlowFarnebackSettings` and 
        :func:`get_main_flow_field_params` for details
        
        Parameters
        ----------
        count : array
            array containing histogram counts
        bins : array
            array containing bins corresponding to counts
            
        """
        count, bins = self._prep_histo_data(count, bins)
        
        idx = argmax(count)
        #amplitude value at max
        amp = count[idx]
        pos = bins[idx]
        #print "Estimating peak width at peak, index: " + str(idx)
        #print "x,y:" + str(self.index[idx]) + ", " + str(amp)
        max_ind = len(bins) - 1  
        try:
            ind = next(val[0] for val in enumerate(count[idx:max_ind])\
                                                if val[1] < amp/e)
            #print "Width (index units): " + str(abs(ind))
            
            return pos, abs(bins[ind] - pos)
        except:
            #print "Trying to the left"#format_exc()
            try:
                inv = count[::-1]
                idx = len(inv) - 1 - idx
                ind = next(val[0] for val in enumerate(inv[idx:max_ind])\
                                                if val[1] < amp/2)
                #print "Width (index units): " + str(abs(ind))
                return pos, abs(bins[ind] - pos)
            except:
                pass
        warn("Failed to retrieve uncertainty in displacement length histo "
            "using method argmax, setting error to 100 maximum detected length")
        return pos, max(bins)
        
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
        try:
            fit.auto_fit()
        except:
            ok = False
        return fit, ok
        
    def flow_orientation_histo(self, bin_res_degrees=6, multi_gauss_fit=True,
                               exclude_short_vecs=True, cond_mask_flat=None,
                               noise_amp=None, max_num_gaussians=None,
                               **kwargs):
        """Get histogram of orientation distribution of current flow field
        
        :param int bin_res_degrees (6): bin width of histogram (is rounded to
            nearest integer if not devisor of 360)
        :param bool multi_gauss_fit (True): apply multi gauss fit to histo
        :param bool exclude_short_vecs: don't include flow vectors which are 
            shorter than ``self.settings.min_length``
        :param ndarray cond_mask_flat: additional conditional boolean vector
            applied to flattened orientation array (for instance all pixels
            in original imaged that exceed a certain tau value, see also
            :func:`prepare_intensity_condition_mask`)
        :param **kwargs: can be used to pass lens and angles arrays (see e.g.
            :func:`get_main_flow_field_params`)
        """
        try:
            lens = kwargs["lens"]
            angles = kwargs["angles"]
        except:
            lens, angles = self.flow_len_and_angle_vectors
        cond = ones(len(lens))
        if cond_mask_flat is not None:
            cond = cond * cond_mask_flat
        if exclude_short_vecs:
            cond = cond * (lens > self.settings.min_length)
        if max_num_gaussians is None:
            max_num_gaussians = self.settings.hist_dir_gnum_max
        angs = angles[cond.astype(bool)]
        
        num_bins = int(round(360 / bin_res_degrees))
        count, bins = histogram(angs, num_bins)
        if noise_amp is None:
            noise_amp = max(count) * 0.05 #set minimum amplitude for multi gauss fit 5% of max amp
        fit = None
        if multi_gauss_fit:
            try:
                fit, ok = self.fit_multigauss_to_histo(count, bins, 
                                                       noise_amp=noise_amp,
                                                       max_num_gaussians=
                                                       max_num_gaussians)
                if not ok:
                    raise Exception
            except:
                warn("MultiGaussFit failed in orientation histogram of optical"
                    "flow field at %s" %self.current_time)
        return count, bins, angs, fit
    
    def flow_length_histo(self, multi_gauss_fit=True, exclude_short_vecs=True,
                          cond_mask_flat=None, noise_amp=None,
                          max_num_gaussians=None, **kwargs):
        """Get histogram of displacement length distribution of flow field
        
        Parameters
        ----------
        multi_gauss_fit : bool
            apply multi gauss fit to histo
        exclude_short_vecs : bool
            don't include flow vectors which are shorter than 
            ``self.settings.min_length``
        cond_mask_flat : array
            conditional boolean vector applied to flattened array of 
            displacement lengths within current ROI (for instance all pixels
            in original image that exceed a certain tau value, see also
            :func:`prepare_intensity_condition_mask`)
        **kwargs
            can be used to pass lens and angles arrays (see e.g.
            :func:`get_main_flow_field_params`)
            
        Returns
        -------
        tuple
            
        """
        try:
            lens = kwargs["lens"]
            angles = kwargs["angles"]
        except:
            lens, angles = self.flow_len_and_angle_vectors    
        cond = ones(len(lens))
        if cond_mask_flat is not None:
            cond = cond * cond_mask_flat
        if exclude_short_vecs:
            cond = cond * (lens > self.settings.min_length)
        if max_num_gaussians is None:
            max_num_gaussians = self.settings.hist_len_gnum_max
        lens = lens[cond.astype(bool)]
        count, bins = histogram(lens, int(ceil(lens.max())))
        fit = None
        if noise_amp is None:
            noise_amp = max(count) * 0.05
        if multi_gauss_fit:
            try:
                fit, ok = self.fit_multigauss_to_histo(count, bins, 
                                                       noise_amp=noise_amp,
                                                       max_num_gaussians=
                                                       max_num_gaussians)
                if not ok:
                    raise Exception
            except:
                warn("MultiGaussFit failed in displacement length histogram "
                    "of optical flow field at %s" %self.current_time)
        return count, bins, lens, fit
    
    
    def get_main_flow_field_params(self, cond_mask_flat=None, noise_amp=None):
        """Histogram based statistical analysis of flow field in current ROI
        
        This function analyses histograms of the current flow field within the 
        current ROI (see :func:`roi`) in order to find the predominant 
        movement direction and the corresponding predominant displacement 
        length.
        
        Steps::
        
            1. Get main flow direction by fitting and analysing multi gauss
            (see :class:`MultiGaussFit`) flow orientation histogram
            using :func:`flow_orientation_histo`. The analysis yields mean 
            direction plus standard deviation
            
            .. note::
            
                Not finished yet ...
                
        Parameters
        ----------
        cond_mask_flat 
            optional, flattened mask specifying pixels used for statistical
            analysis (e.g. determined from an optical density image using a 
            tau threshold, i.e. only pixels exceeding a certain tau value.
            .. note::
                this mask must correspond to the sub image area specified by 
                :attr:`roi_abs`.

        
            
                
        """
        res = od([("_len_mu"        ,   nan), 
                  ("_len_sigma"     ,   inf),
                  ("_dir_mu"        ,   nan), 
                  ("_dir_sigma"     ,   inf),
                  ("_del_t"         ,   self.del_t), 
                  ("_start_acq"     ,   self.current_time),
                  ("_add_gauss_dir" ,   []),
                  ("_add_gauss_len" ,   []),
                  ("cond"           ,   cond_mask_flat),
                  ("fit_dir"        ,   None),
                  ("fit_len"        ,   None)])
               
        #vectors containing lengths and angles of flow field in ROI
        lens, angles = self.flow_len_and_angle_vectors
        
        #fit the orientation distribution histogram (excluding vectors shorter
        #than self.settings.min_length)
        _, _, _, fit = self.flow_orientation_histo(cond_mask_flat=
            cond_mask_flat, lens=lens, angles=angles, noise_amp=noise_amp)
        #in case the routine fails and you want to check the fit
        res["fit_dir"] = fit 
        
        if fit is None or not fit.has_results():
            warn("Could not retrieve main flow field parameters.. probably "
            "due to failure of multi gaussian fit to angular distribution "
            "histogram")
            return res
        #analyse the fit result (i.e. find main gauss peak and potential other
        #significant peaks)
        dir_mu, dir_sigma, tot_num, add_gaussians = fit.analyse_fit_result()
        res["_dir_mu"] = dir_mu
        res["_dir_sigma"] = dir_sigma
        res["_add_gauss_dir"] = add_gaussians
        
        print("Predominant movement direction: %.1f +/- %.1f" %(dir_mu,
                                                                dir_sigma))
        for g in add_gaussians:
            sign = int(fit.integrate_gauss(*g) * 100 / tot_num)
            if sign > 20: #other peak exceeds 20% of main peak
                warn("Optical flow hisogram analysis:\n"
                     "Detected additional gaussian in orientation histogram:\n"
                     "%sSignificany: %s %%\n" %(fit.gauss_str(g), sign))
        
        #limit range of reasonable orientation angles...
        dir_low = dir_mu - dir_sigma * self.settings.hist_dir_sigma
        dir_high = dir_mu + dir_sigma * self.settings.hist_dir_sigma
        
        #... and make a mask from it
        cond = logical_and(angles > dir_low, angles < dir_high)
        
        # now exclude short vectors from statistics (add condition to mask)
        cond = cond * (lens > self.settings.min_length)
        
        # and if a specified mask was provided at input, take that into account
        # too
        if cond_mask_flat is not None:
            cond = cond * cond_mask_flat
        
        res["cond"] = cond
        if sum(cond) < 0.01 * len(lens):
            warn("Total number of vectors fulfilling criteria is less "
                "insignificant (less than 1%), aborting retrieval of main"
                "flow fiel parameters")
        
            return res
        do_fit = False
        if self.settings.hist_len_how == "multigauss":
            do_fit = True
        fit_res = self.flow_length_histo(multi_gauss_fit=do_fit,
                                         cond_mask_flat=cond,
                                         lens=lens, 
                                         angles=angles, 
                                         noise_amp=noise_amp)
                                         
        count, bins, lens, fit2 = fit_res
        res["fit_len"] = fit2
        if fit2 is None or not fit2.has_results():
            
            len_mu, len_sigma = self._estimate_mean_len_argmax(count, bins)
            print("Estimating mean displacement length using method argmax")
            print("Retrieved value: %.1f +/- %.1f" %(len_mu, len_sigma))
            res["_len_mu"] = len_mu
            res["_len_sigma"] = len_sigma
            return res
        print("Estimating mean displacement length using method multigauss")
        len_mu, len_sigma, tot_num, add_gaussians = fit2.analyse_fit_result()
        print("Retrieved value: %.1f +/- %.1f" %(len_mu, len_sigma))
        for g in add_gaussians:
            sign = int(fit.integrate_gauss(*g) * 100 / tot_num)
            if sign > 20: #other peak exceeds 20% of main peak
                warn("Optical flow hisogram analysis:\n"
                     "Detected additional gaussian in length histogram:\n"
                     "%sSignificany: %s %%\n" %(fit2.gauss_str(g), sign))
        res["_len_mu"] = len_mu
        res["_len_sigma"] = len_sigma
        res["_add_gauss_len"] = add_gaussians
        return res
        
    def apply_median_filter(self, shape = (3,3)):
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
            warn("Image acquisition time cannot be accessed in" 
                        " OpticalFlowFarneback")
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
                        " OpticalFlowFarneback")
            t0 = datetime(1900, 1, 1)
            t1 = datetime(1900, 1, 1)
        return t0, t1
    """
    Plotting / visualisation etc...
    """        
    def plot_flow_histograms(self, multi_gauss_fit=1, exclude_short_vecs=True,
                             cond_mask_flat=None, for_app=0):
        """Plot histograms of flow field within roi
        
        Analyses the flow field in current ROI and plots 6 subfigures 
        containing:
        
            1. top left: flow field in the whole image, and
            #. bottom left: in the current ROI
            #. top middle: a color coded image showing the orientation of the \
                retrieved vectors in degrees:
                
                    - 0 is top direction
                    - 0 - 180: right orientation
                    - -180 - 0: left orientation
            
            #. bottom middle: a color coded image showing the retrieved \
                displacement lengths
            #. top right: the orientation histogram determined based on input \
                specifications (i.e. with multi gauss fit, considering short 
                vectors)
            #. bottom right: the length histogram determined based on input \
                specifications (i.e. with multi gauss fit, considering short 
                vectors
        
        Parameters
        ----------
        multi_gauss_fit : bool
            If True, try to fit multi gauss fit (:class:`MultiGaussFit`) to 
            both histograms and include results 
        exclude_short_vecs : bool
            If True, exclude vectors shorter than ``self.settings.min_length``
            in histograms
        cond_mask_flat : array
            optional, additional boolean array specifying pixels to be 
            considered for retrieval of histograms
                
        Returns
        -------
        figure
            matplotlib figure 
            
            
        """
        if self.flow is None:
            raise ValueError("No flow field available..")
        #set up figure and axes
        if not for_app:
            fig = figure(figsize=(16,8))
        else:
            fig = Figure(figsize=(16,8))
        #three strangely named axes for top row 
        ax1 = fig.add_subplot(2,3,1)
        ax4 = fig.add_subplot(2,3,2)
        ax2 = fig.add_subplot(2,3,3)
        
        ax11 = fig.add_subplot(2,3,4)        
        ax5 = fig.add_subplot(2,3,5)
        ax3 = fig.add_subplot(2,3,6)
        
        #draw the optical flow image
        self.draw_flow(0, add_cbar=True, ax=ax1)
        self.draw_flow(1, add_cbar=True, ax=ax11)
        
        #load and draw the length and angle image
        angle_im = self.get_flow_orientation_img(True)#rad2deg(arctan2(fx,-fy))
        len_im = self.get_flow_vector_length_img(True)#sqrt(fx**2+fy**2)
        angle_im_disp = ax4.imshow(angle_im, interpolation='nearest',
                                   vmin=-180, vmax=180, cmap="RdBu")
        ax4.set_title("Displacement orientation", fontsize=11)        
        fig.colorbar(angle_im_disp, ax = ax4)
        
        len_im_disp = ax5.imshow(len_im, interpolation='nearest',  cmap="Blues")
        fig.colorbar(len_im_disp, ax = ax5)
        ax5.set_title("Displacement lengths", fontsize=11)        
        
        #prepare the histograms
        n1, bins1, angles1, fit1 = self.flow_orientation_histo(\
            multi_gauss_fit = multi_gauss_fit, exclude_short_vecs =\
                    exclude_short_vecs, cond_mask_flat = cond_mask_flat)
        tit = "Flow angle histo"
        if multi_gauss_fit and fit1.has_results():
            mu, sigma,_,_ = fit1.analyse_fit_result()
        
            info_str=" mu (sigma) = %.1f (+/- %.1f)" %(mu, sigma)
            tit += info_str
        if exclude_short_vecs:
            thresh = self.settings.min_length
            ax3.axvline(thresh, linestyle="--", color="r")
            tit += "\nOnly vectors longer than %d" %thresh
            
        #n, bins,angles,m=self._ang_dist_histo(fx,fy,gaussFit=1)
        w = bins1[1] - bins1[0]
        ax2.bar(bins1[:-1], n1, width = w, label = "Histo")
        ax2.set_title(tit, fontsize=11)
        if multi_gauss_fit and fit1.has_results():
            fit1.plot_multi_gaussian(ax=ax2, label="Multi-gauss fit")
        ax2.set_xlim([-180, 180])    
        ax2.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)                
        #now the length histogram
        n2, bins2, lens2, fit2 = self.flow_length_histo(\
            multi_gauss_fit = multi_gauss_fit, exclude_short_vecs =\
                    exclude_short_vecs, cond_mask_flat = cond_mask_flat)
                    
        tit="Flow length histo"
        if multi_gauss_fit and fit2.has_results():
            mu, sigma,_,_ = fit2.analyse_fit_result()
        
            info_str=" mu (sigma) = %.1f (+/- %.1f)" %(mu, sigma)
            tit += info_str
    
        w = bins2[1] - bins2[0]
        ax3.bar(bins2[:-1], n2, width = w, label = "Histo")
        ax3.set_title(tit, fontsize=11)
        if multi_gauss_fit and fit2.has_results():
            fit2.plot_multi_gaussian(ax=ax3, label="Multi-gauss fit")   
        ax3.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10) 
        

        ax3.set_ylim([0, n2.max()*1.1])
        ax3.set_xlim([0, bins2[-1]])
        
        return fig
        
    def calc_flow_lines(self, in_roi = True):
        """Determine line objects for visualisation of current flow field
        
        .. note::

            The flow lines are calculated only in image area specified
            by current ROI (``self.roi``).
            
        """
        settings = self.settings
        step, len_thresh = settings.disp_skip, settings.disp_len_thresh
        #get the shape of the rectangle in which the flow was determined
        if in_roi:
            flow = self.get_flow_in_roi()
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

    def show_flow_field(self, rect=None):
        """Plot the actual flow field (dx,dy images) 
        
         :param list rect: sub-img rectangle specifying ROI for flow field \
            analysis (in absolute image coordinates)
        """
        raise NotImplementedError
#==============================================================================
#         if rect is None:
#             x0,y0=0,0
#             h,w = self.flow.shape[:2]
#         else:
#             x0,y0=self.settings.imgShapePrep.map_coordinates(rect[0][0],rect[0][1])
#             h,w=self.settings.imgShapePrep.get_subimg_shape()
#==============================================================================
    
    def plot(self, **kwargs):
        """Draw current flow field onto image
        
        Wrapper for :func:`draw_flow`
        
        :param **kwargs: key word args (see :func:`draw_flow`)
        """
        return self.draw_flow(**kwargs)
    
    def draw_flow_old(self, in_roi=False, add_cbar=False, ax=None):
        """Draw the current optical flow field
        
        :param bool in_roi: if True, the flow field is plotted in a
            cropped image area (using current ROI), else, the whole image is 
            drawn and the flow field is plotted within the ROI which is 
            indicated with a rectangle
        :param ax (None): matplotlib axes object
        """
        if ax is None:
            fig, ax = subplots(1,1)
        else:
            fig = ax.figure
        
        x0, y0 = 0, 0
        i_min, i_max = self.current_contrast_range()
    
        img = self.images_input["this"]#.bytescale(i_min, i_max)
        if in_roi:
            img = img.crop(roi_abs = self.roi_abs, new_img = True)
        
        #ugly (fast solution)

        img = bytescale(img.img, cmin = i_min, cmax = i_max)
        dsp = ax.imshow(img, cmap = "gray_r")
        if add_cbar:    
            fig.colorbar(dsp, ax = ax)
            

        disp = cvtColor(img, COLOR_GRAY2BGR) 
        
        if self.flow is None:
            print "Could not draw flow, no flow available"
            return
        lines = self.calc_flow_lines()
        tit = r"1. img"
        if not in_roi:
            x0, y0, w, h = roi2rect(self.roi)
            ax.add_patch(Rectangle((x0, y0), w, h, fc = "none", ec = "c"))
        else:
            tit += " (in ROI)"
        for (x1, y1), (x2, y2) in lines:
            line(disp, (x0+ x1, y0 + y1),\
                        (x0 + x2,y0 + y2),(0, 255, 255), 1)
            circle(disp, (x0 + x2, y0 + y2), 1, (255, 0, 0), -1)
        #ax.imshow(disp)
        
        try:
            tit += (r": %s \n $\Delta$t (next) = %.2f s" %(\
                self.get_img_acq_times()[0].strftime("%H:%M:%S"), self.del_t))
            tit = tit.decode("string_escape")
        except:
            pass
        
        ax.set_title(tit, fontsize = 10)
        return ax, disp
    
    def draw_flow(self, in_roi=False, add_cbar=False, ax=None):
        """Draw the current optical flow field
        
        :param bool in_roi: if True, the flow field is plotted in a
            cropped image area (using current ROI), else, the whole image is 
            drawn and the flow field is plotted within the ROI which is 
            indicated with a rectangle
        :param ax (None): matplotlib axes object
        """
        if ax is None:
            fig, ax = subplots(1,1)
        else:
            fig = ax.figure
        
        x0, y0 = 0, 0
        i_min, i_max = self.current_contrast_range()
    
        img = self.images_input["this"]#.bytescale(i_min, i_max)
        if in_roi:
            img = img.crop(roi_abs = self.roi_abs, new_img = True)
        
        #ugly (fast solution)

        img = bytescale(img.img, cmin = i_min, cmax = i_max)
        dsp = ax.imshow(img, cmap = "gray_r")
        if add_cbar:    
            fig.colorbar(dsp, ax = ax)
        
        if self.flow is None:
            print "Could not draw flow, no flow available"
            return
        lines = self.calc_flow_lines()
        tit = r"1. img"
        if not in_roi:
            x0, y0, w, h = roi2rect(self.roi)
            ax.add_patch(Rectangle((x0, y0), w, h, fc = "none", ec = "c"))
        else:
            tit += " (in ROI)"
        print "Drawing optical flow field into plot..."
        for (x1, y1), (x2, y2) in lines:
            ax.add_artist(Line2D([x0+ x1, x0 + x2], [y0 + y1, y0 + y2],
                                color="c"))
            ax.add_patch(Circle((x0 + x2, y0 + y2), 1, ec="r", fc="r"))
        #ax.imshow(disp)
        
        try:
            tit += (r": %s \n $\Delta$t (next) = %.2f s" %(\
                self.get_img_acq_times()[0].strftime("%H:%M:%S"), self.del_t))
            tit = tit.decode("string_escape")
        except:
            pass
        
        ax.set_title(tit, fontsize = 10)
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
            
