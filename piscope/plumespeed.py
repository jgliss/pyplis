# -*- coding: utf-8 -*-
"""
Classes for plume speed retrievals
----------------------------------
"""
from time import time
from numpy import mgrid,vstack,int32,sqrt,arctan2,rad2deg, asarray, sin, cos,\
    logical_and, histogram, ceil, ones, roll, argmax, arange, ndarray,\
    deg2rad, nan, inf, dot
from numpy.linalg import norm
from traceback import format_exc
from warnings import warn
from datetime import datetime
from collections import OrderedDict as od
from matplotlib.pyplot import subplots, figure, Figure, Circle, Line2D
from matplotlib.cm import RdBu
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.stats.stats import pearsonr
    
from pandas import Series

from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN,\
    cvtColor,COLOR_GRAY2BGR,line,circle,VideoCapture,COLOR_BGR2GRAY,\
    waitKey, imshow

from .helpers import bytescale, check_roi, map_roi, roi2rect, shifted_color_map
from .optimisation import MultiGaussFit
from .image import Img

def find_signal_correlation(first_data_vec, next_data_vec,\
        time_stamps = None, reg_grid_tres = None, freq_unit = "S",\
            itp_method = "linear", cut_border_idx = 0, sigma_smooth = 1,\
                                                                plot = False):
    """Determines cross correlation from two ICA time series
    
    :param ndarray first_data_vec: first data vector (i.e. left or before 
        ``next_data_vec``)
    :param ndarray next_data_vec: second data vector (i.e. behind
        ``first_data_vec``)
    :param ndarray time_stamps: array containing time stamps of the two data
        vectors. If default (None), then the two vectors are assumed to be
        sampled on a regular grid and the returned lag corresponds to the 
        index shift with highest correlation. If 
        ``len(time_stamps) == len(first_data_vec)`` and if entries are 
        datetime objects, then the two input time series are resampled and 
        interpolated onto a regular grid, for resampling and interpolation 
        settings, see following 3 parameters.
    :param int reg_grid_tres: sampling resolution of resampled time series
        data in units specified by input parameter ``freq_unit``. If None, 
        then the resolution is determined from the mean time difference 
        between consecutive points in ``time_stamps``,
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
            print ("Invalid interpolation method %s: setting default (linear)" 
                                                                %itp_method)
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
        s1 = Series(first_data_vec, time_stamps).resample(delt_str).\
                                        interpolate(itp_method).dropna()
        s2 = Series(next_data_vec, time_stamps).resample(delt_str).\
                                        interpolate(itp_method).dropna()
                                        
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
                        
    def get_velocity(self, idx, pix_dist_m, pix_dist_m_err = None, normal_vec=None):
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
        ax.set_ylabel("v [%s]" %velo_unit)
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
        ax.set_ylabel(r"$\Theta$ [$^\circ$]")
        #ax.grid()
        return ax
    
class OpticalFlowFarnebackSettings(object):
    """Settings for optical flow Farneback calculations and visualisation"""
    def __init__(self, **settings):
        """Initiation of settings object"""
        self._contrast = od([("i_min"  ,   0),
                             ("i_max"  ,   1e30)])
        
        self._flow_algo = od([("pyr_scale"  ,   0.5), 
                              ("levels"     ,   4),
                              ("winsize"    ,   16), 
                              ("iterations" ,   6), 
                              ("poly_n"     ,   5), 
                              ("poly_sigma" ,   1.1)])
                            
        self._analysis = od([("roi_abs"         ,   [0, 0, 9999, 9999]),
                             ("min_length"      ,   1.0),
                             ("sigma_tol_mean_dir", 3)])
        
        self._display = od([("disp_skip"            ,   10),
                            ("disp_len_thresh"      ,   3)])
        
        for k, v in settings.iteritems():
            self[k] = v # see __setitem__ method
            
    @property
    def i_min(self):
        """Get lower intensity limit for image contrast preparation"""
        return self._contrast["i_min"]
            
    @property
    def i_max(self):
        """Get upper intensity limit for image contrast preparation"""
        return self._contrast["i_max"]
    
    @property
    def pyr_scale(self):
        """Get param for optical flow algo input"""
        return self._flow_algo["pyr_scale"]
    
    @property
    def levels(self):
        """Get param for optical flow algo input"""
        return self._flow_algo["levels"]    
        
    @property
    def winsize(self):
        """Get param for optical flow algo input"""
        return self._flow_algo["winsize"]
        
    @property
    def iterations(self):
        """Get param for optical flow algo input"""
        return self._flow_algo["iterations"]
        
    @property
    def poly_n(self):
        """Get param for optical flow algo input"""
        return self._flow_algo["poly_n"]

    @property
    def poly_sigma(self):
        """Get param for optical flow algo input"""
        return self._flow_algo["poly_sigma"]
        
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
    def sigma_tol_mean_dir(self):
        """Get / set sigma tolerance level for mean flow analysis"""
        return self._analysis["sigma_tol_mean_dir"]
    
    @sigma_tol_mean_dir.setter
    def sigma_tol_mean_dir(self, val):
        if not 1 <= val <= 4:
            raise ValueError("Value must be between 1 and 4")
        self._analysis["sigma_tol_mean_dir"] = val
        
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
        s += "\nROI (for post analysis, e.g. histograms): %s\n" %self.roi_abs
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
        numpy arrays. Input :class:`piscope.Image.Img` also works on input but
        this class does not provide any funcitonality based on the functionality
        of :class:`piscope.Image.Img` objects (i.e. access of meta data, etc).
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
        
        self._img_prep_modes = {"auto_update_contrast"   :   False}
        
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
        return self._img_prep_modes["auto_update_contrast"]
    
    @auto_update_contrast.setter
    def auto_update_contrast(self, val):
        self._img_prep_modes["auto_update_contrast"] = val
        print ("Auto update contrast mode was updated in OpticalFlowFarneback "
            "but not applied to current image objects, please call method "
            "set_images in order to apply the changes")
        return val
    
    @property
    def roi_abs(self):
        """Get / set current ROI (in absolute image coordinates)"""
        return self.settings.roi_abs
    
    @roi_abs.setter
    def roi_abs(self, val):
        self.settings.roi_abs = val
        
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

        
    def current_contrast_range(self):
        """Get min / max intensity values for image preparation"""
        i_min = float(self.settings._contrast["i_min"])
        i_max = float(self.settings._contrast["i_max"])
        return i_min, i_max
    
    def update_contrast_range(self, i_min, i_max):
        """Updates the actual contrast range for opt flow input images"""
        self.settings._contrast["i_min"] = i_min
        self.settings._contrast["i_max"] = i_max
        print ("Updated contrast range in opt flow, i_min = %s, i_max = %s" 
                                                            %(i_min, i_max))
    def check_contrast_range(self, img_data):
        """Check input contrast settings for optical flow calculation"""
        i_min, i_max = self.current_contrast_range()
        if i_min < img_data.min() and i_max < img_data.min() or\
                    i_min > img_data.max() and i_max > img_data.max():
            self.update_contrast_range(i_min, i_max)
    
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
            roi = map_roi(self.roi_abs, this_img.edit_log["pyrlevel"])
            sub = this_img.img[roi[1]:roi[3], roi[0]:roi[2]]
            i_min, i_max = max([0, sub.min()]), sub.max()
            self.update_contrast_range(i_min, i_max)
         
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
        x0, y0, x1, y1 = self.roi
        return self.flow[y0 : y1, x0 : x1, :]
    
    def _prep_flow_for_analysis(self):
        """Flatten the flow fields for analysis
        
        :return:
            - ndarray, vector containing all x displacement lengths
            - ndarray, vector containing all y displacement lenghts
            
        """
        fl = self.get_flow_in_roi()
        return fl[:,:,0].flatten(), fl[:,:,1].flatten()
    
    def prepare_intensity_condition_mask(self, lower_val = 0.0,\
                                                    upper_val = 1e30):
        """Apply intensity threshold to input image in ROI and make mask vector
        
        :param float lower_val: lower intensity value, default is 0.9
        :param float upper_val: upper intensity value, default is 9999
        :return:
            - ndarray, flattened mask which can be used e.g. in 
                :func:`flow_orientation_histo` as additional input param     
        """
        x0, y0, x1, y1 = self.roi
        sub = self.images_input["this"].img[y0 : y1, x0 : x1].flatten()
        return logical_and(sub > lower_val, sub < upper_val)
    
    def to_plume_speed(self, col_dist_img, row_dist_img = None):
        """Convert the current flow field to plume speed array
        
        :param Img col_dist_img: image, where each pixel corresponds to 
            horizontal pixel distance in m
        :param row_dist_img: image, where each pixel corresponds to 
            vertical pixel distance in m (if None, ``col_dist_img`` is also
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
        if in_roi:
            fl = self.get_flow_in_roi()
        else:
            fl = self.flow
        fx, fy = fl[:,:,0], fl[:,:,1]
        return rad2deg(arctan2(fx, -fy))
      
    def get_flow_vector_length_img(self, in_roi = False):        
        """Returns image corresponding to displacement length in each pixel"""
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
    
    def flow_orientation_histo(self, bin_res_degrees=6, multi_gauss_fit=True,
                               exclude_short_vecs=True, cond_mask_flat=None,
                               noise_amp=None, max_num_gaussians=5,**kwargs):
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
        angs = angles[cond.astype(bool)]
        
        num_bins = int(round(360 / bin_res_degrees))
        count, bins = histogram(angs, num_bins)
        if noise_amp is None:
            noise_amp = max(count) * 0.05 #set minimum amplitude for multi gauss fit 5% of max amp
        fit = None
        if multi_gauss_fit:
            try:
                x = asarray([0.5 * (bins[i] + bins[i + 1]) for\
                                                i in xrange(len(bins) - 1)])
                fit = MultiGaussFit(count, x, noise_amp = noise_amp,
                                    max_num_gaussians=max_num_gaussians,
                                    do_fit=False)
                fit.auto_fit()
            except:
                warn("MultiGaussFit failed in orientation histogram of optical"
                    "flow field at %s" %self.current_time)
        return count, bins, angs, fit
    
    def flow_length_histo(self, multi_gauss_fit=True, exclude_short_vecs=True,
                          cond_mask_flat=None, noise_amp=None,
                          max_num_gaussians=5, **kwargs):
        """Get histogram of displacement length distribution of flow field
        
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
        lens = lens[cond.astype(bool)]
        count, bins = histogram(lens, int(ceil(lens.max())))
        fit = None
        if noise_amp is None:
            noise_amp = max(count) * 0.05
        if multi_gauss_fit:
            try:
                x = asarray([0.5 * (bins[i] + bins[i + 1]) for\
                                                i in xrange(len(bins) - 1)])
                fit = MultiGaussFit(count, x, noise_amp=noise_amp,
                                    max_num_gaussians=max_num_gaussians,
                                    do_fit=False) #make sure the object is initiated
                fit.auto_fit()
            except:
                warn("MultiGaussFit failed in displacement length histogram "
                    "of optical flow field at %s" %self.current_time)
        return count, bins, lens, fit
    
    def get_main_flow_field_params(self, cond_mask_flat = None, noise_amp=None,
                                   max_num_gaussians=5):
        """Historgam based statistical analysis of flow field in current ROI
        
        This function analyses histograms of the current flow field within the 
        current ROI (see :func:`roi`) in order to find the predominant 
        movement direction and the corresponding predominant displacement 
        length.
        
        Steps::
        
            1. Get main flow direction by fitting and analysing multi gauss
            (see :class:`MultiGaussFit`) flow orientation histogram
            using :func:`flow_orientation_histo`. The analysis yields mean 
            direction plus standard deviation
            #. 
        """
        res = {"_len_mu"            :   nan, 
               "_len_sigma"         :   inf, 
               "_dir_mu"            :   nan, 
               "_dir_sigma"         :   inf,
               "_del_t"             :   self.del_t, 
               "_start_acq"         :   self.current_time,
               "_add_gauss_dir"     :   [],
               "_add_gauss_len"     :   [],
               "cond"               :   cond_mask_flat}
        #vectors containing lengths and angles of flow field in ROI
        lens, angles = self.flow_len_and_angle_vectors
        
        #fit the orientation distribution histogram (excluding vectors shorter
        #than self.settings.min_length)
        _, _, _, fit = self.flow_orientation_histo(cond_mask_flat=
            cond_mask_flat, lens=lens, angles=angles, noise_amp=noise_amp,
            max_num_gaussians=max_num_gaussians)
            
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
        dir_low = dir_mu - dir_sigma * self.settings.sigma_tol_mean_dir
        dir_high = dir_mu + dir_sigma * self.settings.sigma_tol_mean_dir
        
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
        count, bins, lens, fit2 = self.flow_length_histo(cond_mask_flat = cond,
                            lens=lens, angles=angles, noise_amp=noise_amp,
                            max_num_gaussians=max_num_gaussians)
        
        if fit2 is None or not fit2.has_results():
            warn("Could not retrieve main flow field parameters..probably "
            "due to failure of multi gaussian fit to vector length "
            "histogram")
            return res

        len_mu, len_sigma, tot_num, add_gaussians = fit2.analyse_fit_result()
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
    def plot_flow_histograms(self, multi_gauss_fit = 1,\
                exclude_short_vecs = True, cond_mask_flat = None, for_app = 0):
        """Plot histograms of flow field within roi"""
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
            
