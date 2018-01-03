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
"""Module containing optimisation routines"""
from numpy import abs, linspace, random, asarray, ndarray, where, diff,\
    insert, argmax, average, gradient, arange, nanmean, full, inf, sqrt, pi,\
    mod, mgrid, ndim, ones_like,ogrid, finfo, remainder, e, sum, uint8, int,\
    histogram, nan, isnan
    
from warnings import catch_warnings, simplefilter, warn
from matplotlib.pyplot import subplots

from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter

from scipy.ndimage.filters import gaussian_filter1d
from scipy.integrate import quad
from scipy.optimize import curve_fit, least_squares

from cv2 import pyrUp, pyrDown
from copy import deepcopy
#from scipy.signal import find_peaks_cwt
#from peakutils import indexes
from traceback import format_exc

try:
    from .model_functions import supergauss_2d, supergauss_2d_tilt,\
        multi_gaussian_no_offset, gaussian_no_offset, gaussian,\
        multi_gaussian_same_offset, dilutioncorr_model
    from .helpers import mesh_from_img
except:
    from pyplis.model_functions import supergauss_2d, supergauss_2d_tilt,\
        multi_gaussian_no_offset, gaussian_no_offset, gaussian,\
        multi_gaussian_same_offset
    from pyplis.helpers import mesh_from_img

GAUSS_2D_PARAM_INFO = ["amplitude", "mu_x", "mu_y", "sigma", "asymmetry",\
    "shape", "offset", "tilt_theta"]
 

def dilution_corr_fit(rads, dists, rad_ambient, i0_guess=None,
                      i0_min=0.0, i0_max=None, ext_guess=1e-4, ext_min=0.0,
                      ext_max = 1e-3):
    """Performs least square fit of data
    
    :param ndarray rads: vector containing measured radiances
    :param ndarray dists: vector containing corresponding dictances
    :param float rad_ambient: ambient intensity
    :param i0_guess: guess value for initial intensity of topographic features,
        i.e. the reflected radiation before entering scattering medium 
        (if None, then it is set 5% of the ambient intensity ``rad_ambient``)
    :param float i0_min: minimum initial intensity of topographic features
    :param float i0_max: maximum initial intensity of topographic features
    :param float ext_guess: guess value for atm. extinction coefficient
    :param float ext_min: minimum value for atm. extinction coefficient
    :param float ext_max: maximum value for atm. extinction coefficient
    """
    if i0_guess is None:
        print "No input for i0 guess, assuming albedo of 5%"
        i0_guess = rad_ambient * 0.05
    if i0_max is None:
        print "No input for i0 max, assuming maximum albedo of 50%"
        i0_max = rad_ambient * 0.5
    guess = [i0_guess, ext_guess]
    lower = [i0_min, ext_min]
    upper = [i0_max, ext_max]
    bounds = (lower, upper)
    print lower
    print upper
    errfun = lambda p, x, y: (dilutioncorr_model(x, rad_ambient, *p) - y)**2
    
    return least_squares(errfun, guess, args = (dists, rads), bounds=bounds)

def gauss_fit_2d(img_arr, cx, cy, g2d_asym=True, g2d_super_gauss=True, 
                 g2d_crop=True, g2d_tilt=False, **kwargs):
    """Apply 2D gauss fit to input image at its maximum pixel coordinate
    
    Parameters
    ----------
    corr_img : array
        correlation image
    cx : float 
        x-position of peak in image (used for initial guess)
    cy : float
        y-position of peak in image (used for initial guess)
    g2d_asym : bool
        allow for assymetric shape (sigmax != sigmay), True
    g2d_super_gauss : bool
        allow for supergauss fit, True
    g2d_crop : bool
        if True, set outside (1/e amplitude) datapoints = 0, True
    g2d_tilt : bool
        allow gauss to be tilted with respect to x/y axis
        
    Returns
    -------
    tuple
        3-element tuple containing
        
        - array (popt): optimised multi-gauss parameters
        - 2d array (pcov): estimated covariance of popt
        - 2d array: correlation image
    """
    xgrid, ygrid = mesh_from_img(img_arr)
    amp = img_arr[cy, cx]
    # constrain fit, if requested
    print "2D Gauss fit"
    if g2d_asym:
        print "g2d_asym active"
        asym_lb = -inf
        asym_ub =  inf
    else:
        asym_lb = 1 - finfo(float).eps
        asym_ub = 1 + finfo(float).eps
    if g2d_super_gauss:
        print "g2d_super_gauss active"
        shape_lb = -inf
        shape_ub =  inf
    else:
        shape_lb = 1 - finfo(float).eps
        shape_ub = 1 + finfo(float).eps
    if g2d_tilt and not g2d_asym:
        raise ValueError("With tilt and without asymmetry makes no sense")
    if g2d_tilt:
        print "g2d_tilt active"
        guess = [amp, cx, cy, 20, 1, 1, 0, 0]
        lb = [-inf, -inf, -inf, -inf, asym_lb, shape_lb, -inf, -inf]
        ub = [ inf,  inf,  inf,  inf, asym_ub, shape_ub,  inf,  inf]
        if any(lb >= ub):
            print "Bound-Problem"
        popt, pcov = curve_fit(supergauss_2d_tilt, (xgrid, ygrid),\
                    img_arr.ravel(), p0 = guess, bounds = (lb, ub))
        popt[-1] = remainder(popt[-1], pi * 2)
        if all(guess == popt):
            raise Exception("FOV gauss fit failed, popt == guess")
        result_img = supergauss_2d_tilt((xgrid, ygrid), *popt)
    else:
        guess = [amp, cx, cy, 20, 1, 1, 0]
        lb = [-inf, -inf, -inf, -inf, asym_lb, shape_lb, -inf]
        ub = [ inf,  inf,  inf,  inf, asym_ub, shape_ub,  inf]
        popt, pcov = curve_fit(supergauss_2d, (xgrid, ygrid),\
                               img_arr.ravel(), p0=guess, bounds=(lb,ub))
        
        if all(guess == popt):
            raise Exception("FOV gauss fit failed, popt == guess")
        result_img = supergauss_2d((xgrid, ygrid), *popt)
    # eventually crop FOV distribution (makes it more robust against outliers (eg. mountan ridge))
    if g2d_crop:
        print "g2d_crop active"
        # set outside (1/e amplitude) datapoints = 0
        result_img[result_img < popt[0] / e] = 0
    # reshape fov_mask as matrix instead of vector required for fitting
    result_img = result_img.reshape(img_arr.shape)
    # normalise
    return (popt, pcov, result_img)

def gauss_fit(data, idx=None, has_offset=False, plot=False):
    """Fits Gaussian function to data
    
    Parameters
    ----------
    data : array
        data array
    idx : :obj:`array`, optional
        indices of data (e.g. angles)
    has_offset : bool
        if True, the fitted Gauss is allowed to have a constant offset
        
    Returns
    -------
    array
        optimised parameters of gauss
    """
    data = asarray(data)
    if idx is None:
        idx = arange(len(data))
    if has_offset:
        model = gaussian    
    else:
        model = gaussian_no_offset
    err_fun = lambda p, x, y: (model(x, *p) - y)
    guess = [data.max(), idx[argmax(data)], data.std()]
    res = least_squares(err_fun, guess, args=(idx, data))
    opt = res.x
    if plot:
        fig, ax = subplots(1,1)
        ax.plot(idx, data, "--xg", label="data")
        x = linspace(idx.min(), idx.max(), 100)
        d = model(x, *opt)
        ax.plot(x, d, "-r", label="Fit result")
        ax.legend(loc='best', fancybox=True, framealpha=0.5)
        
    return opt
    
def get_histo_data(data, **kwargs):
    """Determine histogram of data and set bin array to center of bins"""
    c, b = histogram(data, **kwargs)
    b = asarray([0.5 * (b[i] + b[i + 1]) for i in xrange(len(b) - 1)])
    return (c, b)
    
    
class MultiGaussFit(object):
    """Fitting environment for fitting an arbitrary (i.e. unknown) amount of
    Gaussians to noisy 1D (x,y) data. It was initally desinged and developed
    for histogram data and aims to find a solution based on a minimum of 
    required superimposed Gaussians to describe the distribution. Therefore,
    the fit is performed in a controlled way (i.e. allowed Gaussians are
    required to be within certain parameter bounds, details below) starting 
    with a noise analysis (if noise level is not provided on class 
    initialisation).
    Based on the noise level, and the x-range of the data, the boundaries for
    accepted gauss parameters are set. These are::
    
        self.gauss_bounds["amp"][0] = 2*self.noise_amp
        self.gauss_bounds["amp"][1] = (self.y_range - self.offset) * 1.5
        self.gauss_bounds["mu"][0] = self.index[0]
        self.gauss_bounds["mu"][1] = self.index[-1]
        self.gauss_bounds["sigma"][0] = self.x_resolution/2.
        self.gauss_bounds["sigma"][1] = self.x_range/2.
        
    i.e. the amplitude of each of the superimposed Gaussians must be positive
    and larger then 2 times the noise amplitude. The max allowed amplitude
    is set 1.5 times the min / max difference of the data. The mean of each 
    Gaussian must be within the index range of the data and the standard
    deviation must at least be half the x resolution (the smallest allowed peak
    must be at least have a of FWHM = 1 index) and the max FHWM must not
    exceed the covered x-range. The fit boundaries can also be set manually
    using :func:`set_gauss_bounds` but this might have a strong impact on the
    quality of the result.
    
    Parameters
    ----------    
    data : array
        data array 
    index : :obj:`array`, otional 
        x-coordinates
    noise_amp : :obj:`float`, optional, 
        amplitude of noise in the signal. Defines the minimum required 
        amplitude for fitted Gaussians (you don't want to fit all the noise 
        peaks). If None, it will be estimated automatically on data import 
        using :func:`estimate_noise_amp`
    noise_amp_thresh_fac : float
        factor multiplied with :attr:`noise_amp` in order to determine the 
        minimum amplitude threshold required for detecting additional peaks
        in residual (see :func:`find_additional_peaks`)
    sigma_smooth : int
        width of Gaussian kernel to determine smoothed analysis signal (is 
        used to determine data baseline offset)
    sigma_tol_overlaps : int
        sigma range considered to find overlapping Gauss functions (after 
        fit was applied). This is, for instance used in 
        :func:`analyse_fit_result` in order to find the main peak parameters
    max_num_gaussians : int 
        max number of superimposed, defaults to 20 Gaussians for data 
    max_iter : int
        max number of iterations for optimisation, if None (default), use 
        ``max_num_gaussians + 1``
    auto_bounds : bool
        if True, bounds will be set automatically from data ranges whenever 
        data is updated, defaults to True
    do_fit : bool
        if True and input data available & ok, then :func:`run_optimisation` 
        will be called on initialisation, defaults to True
    """
    def __init__(self, data=None , index=None, noise_amp=None, 
                 noise_amp_thresh_fac=2.0, sigma_smooth=3,
                 sigma_tol_overlaps=3, max_num_gaussians=20, max_iter=None, 
                 auto_bounds=True, do_fit=True):
        #data
        self.index = []
        self.data = []
        
        # init relevant parameters
        self.offset = 0.0
        self.noise_amp = noise_amp
        self.noise_amp_thresh_fac = noise_amp_thresh_fac
        self.sigma_smooth =sigma_smooth
        self.sigma_tol_overlaps = sigma_tol_overlaps
        
        self.max_num_gaussians = max_num_gaussians
        if max_iter is None:
            max_iter = max_num_gaussians + 1
        self.max_iter = max_iter
          
        self.auto_bounds = auto_bounds
        self.gauss_bounds = {"amp"  :   [0, inf],
                             "mu"   :   [-inf, inf],
                             "sigma":   [-inf, inf]}
        
        #Fitting related stuff
        self._fit_result = None #the actual output from the minimisation
        
        self.params = [] #this is where the fit parameters are stored in
        
        #function to be minimised
        self.err_fun = lambda p, x, y:\
                    (multi_gaussian_no_offset(x, *p) - y)#**2
        
        #will be filled with optimisation results
        self.opt_log = {"chis"     : [],
                        "residuals": []}
                                
        self.set_data(data, index)
        if do_fit and self.has_data:
            self.run_optimisation()
    
    ### @property decorators
    @property
    def residual(self):
        """Get and return residual"""
        return self.get_residual()
    
    @property
    def data_grad(self):
        """Gradient of analysis signal"""
        return self.first_derivative(self.data)
        
    @property
    def data_grad_smooth(self):
        """Smoothed gradient of analysis signal"""        
        return self.apply_binomial_filter(self.data_grad, 
                                          sigma=self.sigma_smooth)
        
    @property
    def data_smooth(self):
        """Data smoothed using Gaussian kernel of width ``self.sigma_smooth``"""
        return self.apply_binomial_filter(self.data, sigma=self.sigma_smooth)
    
    @property
    def min_amp(self):
        """Minimum required amplitude to idenitify significant peaks"""
        return self.noise_amp * self.noise_amp_thresh_fac
        
    ### Initialisation, data preparation, I/O, etc...
    def init_results(self):
        """Initiate all result parameters"""
        self._peak_indices = []
        self.params = []
        self.offset = 0.0
    
    def set_data(self, data, index=None):
        """Set x and y data
        
        Parameters
        ----------
        data : array
            data array which is fitted
        index : :array
            optional, x index array of data, if None, the array index of data
            is used
        """
        if isinstance(data, ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = asarray(data)
        else:
            self.data = asarray([])  
        if isinstance(index, ndarray):
            self.index = index
        elif isinstance(index, list):
            self.index = asarray(index)
        else:
            self.index = arange(len(self.data))
                
        self.init_results()
        self.init_data()
        
    def init_data(self):
        """Initiate the input data and set constraints for Gaussians"""
        if self.has_data:
            # helper for optimisation (analysis signal)
            if self.noise_amp is None:
                self.estimate_noise_amp()
            self.offset = self.data_smooth.min()
            self.y = self.data - self.offset
            self.init_gauss_bounds_auto()
    
    def set_gauss_bounds(self, amp_range=[0, inf], mu_range=[-inf, inf],
                         sigma_range=[-inf, inf]):
        """Manually set boundaries for gauss parameters
        
        Parameters
        ----------
        amp_range : array
            accepted amplitude range, defaults to ``[0, inf]``
        mu_range : array
            accepted mu range, defaults to ``[-inf, inf]``
        sigma_range : array 
            accepted range of standard deveiations, defaults to ``[-inf, inf]``
        """
        self.gauss_bounds["amp"] = amp_range
        self.gauss_bounds["mu"] = mu_range
        self.gauss_bounds["sigma"] = sigma_range
        
    def init_gauss_bounds_auto(self):
        """Set parameter bounds for individual Gaussians"""
        if not self.has_data:
            #print "Could not init gauss bounds, no data available..."
            return 0
        if not self.auto_bounds:
            #print "Automatic update of boundaries is deactivated..."
            return 1
        self.gauss_bounds["amp"][0] = self.min_amp
        self.gauss_bounds["amp"][1] = (self.y_range - self.offset) * 1.5
        self.gauss_bounds["mu"][0] = self.index[0]
        self.gauss_bounds["mu"][1] = self.index[-1]
        self.gauss_bounds["sigma"][0] = self.x_resolution / 2.
        self.gauss_bounds["sigma"][1] = self.x_range / 2.
        return 1
                
    ### Fit preparations, peak search, etc
    def estimate_main_peak_params(self):
        """Get rough estimate and position of main peak"""
        data = self.y
        ind = argmax(data)
        amp = data[ind] 
        if not amp > self.min_amp:
            raise IndexError("No significant peak could be found in data")
        w = self.estimate_peak_width(data, ind)
        guess = [amp, self.index[ind], w * self.x_resolution]
        params, bds = self.prepare_fit_boundaries(guess)
        return least_squares(self.err_fun, params, args=(self.index, self.y),
                             bounds=bds).x
    
    ### index based processing (functions that get slow if data is large)
    def find_additional_peaks(self):
        """Search for significant peaks in the current residual
        
        Returns
        -------
        list
            list containing additional peak parameters (for optimisation), or 
            empty list if no additional peaks can be found
        """
        dat = self.residual
        add_params = []
        num = self.num_of_gaussians #current number of fitted gaussians
        for k in range(self.max_num_gaussians - num):
            if not dat.max() > self.min_amp:
                #print "Residual peak search finished..."
                return add_params
            else: #estimate peak and cut out sigma 3 sigma range
                ind = argmax(dat)
                w = self.estimate_peak_width(dat, ind)
                add_params.append(dat[ind])
                add_params.append(self.index[ind])
                add_params.append(w * self.x_resolution)
                cut_low = ind - 3 * w
                if cut_low < 0:
                    cut_low = 0
                cut_high = ind + 3 * w
                dat[cut_low : cut_high] = 0
        warn("Peak search in residual aborted: reached maxmimum number of "
            "allowed Gaussians: %d" %self.max_num_gaussians)
        return add_params
    
    def estimate_peak_width(self, dat, idx):
        """"Estimate width of a peak at given index 
        
        The peak width is estimated by finding the closest
        datapoint smaller than 0.5 the amplitude of data at index position
        
        Parameters
        ----------
        dat : array
            data (with baseline zero)
        idx : int
            index position of peak
        
        Returns
        -------
        int
            Estimated peak width in index units
            
        """
        amp = dat[idx]
        max_ind = len(self.index) - 1  
        lr_arr = [nan, nan]
        try: #
            ind = (next(val[0] for val in 
                    enumerate(dat[idx:max_ind]) if val[1] < amp / 2))
            #set estimate in right direction
            lr_arr[1] = ind
        except:
            pass
        try:
            inv = dat[::-1]
            idx = len(inv) - 1 - idx
            ind = next(val[0] for val in enumerate(inv[idx:max_ind])\
                                            if val[1] < amp/2)
            # set estimate in left direction
            lr_arr[0] = ind
        except:
            pass
        w = nanmean(lr_arr)
        if isnan(w):
            w = 3
            warn("Width of detected peak at index %d could not be estimated"
                "assuming 3 indices")
        return int(w)
            
    ### Fitting, fitting preparations, etc.        
    def prepare_fit_boundaries(self, guess):
        """Prepare the boundaries tuple
        
        For details see `used least squares solver <http://docs.scipy.org/doc/
        scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html>`_)
        
        Prepare fit boundary tuple for a multi-gauss parameter array supposed
        to be optimised (e.g. for two Gaussians  this could look like
        ``params=[300, 10, 2, 150, 15, 1]`` using the boundaries specified in 
        ``self.gauss_bounds``.
        
        Note
        ----
        
        If any of the parameters in ``params`` is out of the acceptance
        borders specified in ``self.gauss_bounds``, the corresponding 
        parameters will be updated to the corresponding boundary value.
        
        Parameters
        ----------
        params : list
            list of gauss parameters (e.g. ``self.params``)
 
        Returns
        -------
        tuple 
            2-element tuple, containing
            
            - :obj:`list`: new parameter list (only those matching boundaries)
            - :obj:`tuple`: corresponding lower and upper boundaries
            
        """
        if not mod(len(guess), 3) == 0:
            #print "Error: length of gauss param list must be divisable by three.."
            return []
        sub = [guess[x : x + 3] for x in range(0, len(guess), 3)]
        params_new = []
        for peak in sub:
            params_new.extend(self.check_peak_bounds(peak))
        lower, upper = [], []
        l, u = self._prep_bounds_single_gauss()
        for k in range(len(params_new) / 3):
            lower.extend(l)
            upper.extend(u)
        bds = (lower, upper)
        return params_new, bds
        
    def do_fit(self, x, y, guess):
        """Perform a least squares minimisation
        
        Perform least squares optimisiation for initial set of parameters and
        input data (includes determination of fit boundary array for all 
        initial peak guesses of input array).
        
        Parameters
        ----------
        x : array
            x-argument for model function (index of data)
        y : array
            y-argument for input function (data)
        guess : list
            initial guess of fit parameters
        
        Returns
        -------
        bool
            True, if optimisation was successful, False if not
        """
        try:
            params, bds = self.prepare_fit_boundaries(guess)    
            #print "Fitting data..."
            self._fit_result = res = least_squares(self.err_fun, params,
                                                   args=(x, y), bounds=bds)
            #params,ok=optimize.leastsq(self.err_fun, *guess, args=(x, y))
            if not res.success:
                #print "Fit failed"
                return False
            self.params = res.x
            return True
        except:
            #print "Fit failed with exception: %s" %repr(e)
            return False
    
    def opt_iter(self, add_params=[]):
        """Search additional peaks in residual and apply fit 
        
        Extends current optimisation parameters by additional peaks (either
        provided on input or automatically searched in residual) and performs
        multi gauss fit.
        
        Parameter
        ---------
        add_params : list
            list containing additional gauss parameters which are supposed to
            be added to ``self.params`` before the fit is applied
        Returns
        -------
        bool
        
            - False: Optimisation failed
            - True: Optimisation was successful
            
        """
        guess = list(self.params)
        guess.extend(add_params)
        if not self.do_fit(self.index, self.y, guess):
            return False
        return True
        
    def run_optimisation(self):
        """Run optimisation"""
        res = self.get_residual()
        # init optimisation info arrays
        params_log = [self.params]
        res_mus = [res.mean()]
        res_stds = [res.std()]
        for k in range(self.max_iter):
            if self.num_of_gaussians >= self.max_num_gaussians:
                #print ("Max num of gaussians (%d) reached "
                   # "abort optimisation" %self.max_num_gaussians)
                self._write_opt_log(params_log, res_mus, res_stds)
                warn ("MultiGaussFit reached aborted: maximum number of "
                      "Gaussians reached")
                return False
            add_params = self.find_additional_peaks()
            if len(add_params) == 0:
                # Optimisation was successful
                self._write_opt_log(params_log, res_mus, res_stds)
                return True
            
            # perform fit based on current parameters
            if not self.opt_iter(add_params):
                #print ("Optimisation failed,  aborted at iter %d" %k)
                self._write_opt_log(params_log, res_mus, res_stds)
                warn("Optimisation failed in MultiGaussFit")
                return False
            
            res = self.residual
            # append relevant information to optimisation log
            params_log.append(self.params)
            res_mus.append(res.mean())
            res_stds.append(res.std())   
        #print "Optimisation aborted, maximum number of iterations reached..."
        self._write_opt_log(params_log, res_mus, res_stds)
        warn("MultiGaussFit max iter reached..")
        return False
        
    def _write_opt_log(self, params, mus, stds):
        """Log optimisation params, and corresponding residual info"""
        self.opt_log["params"] = asarray(params)
        self.opt_log["mu"] = asarray(mus)
        self.opt_log["std"] = asarray(stds)
    
    ### Quality checks, etc..
    def result_ok(self):
        """Compares current peak to peak residual (ppr) with noise amplitude
        :returns bool: 1 if ``2*self.noise_amp > ppr``, else 0
        """
        if len(self.find_additional_peaks()) == 0:
            return True
        return False
        
    ### Post analysis methods for fitted Gaussian mixture model
    def find_overlaps(self, sigma_tol=None):
        """ Find overlapping Gaussians for current optimisation params
        
        Loops over all current Gaussians (``self.gaussians``) and for each of
        them, finds all which fall into :math:`3\\sigma` range.
        
        Parameters
        ----------
        sigma_tol : :obj:`float`, optional
            sigma tolerance level for finding overlapping Gaussians, if None, 
            use :attr:`sigma_tol_overlaps`.
            
        Returns
        -------
        tuple
            2-element tuple containing
            
            - list, contains all Gaussians overlapping with Gaussian (within \
                sigma tolerance range defined by ``sigma_tol``) at \
                index *k* in list returned by :func:`gaussians`.
            - list, integral values of each of the overlapping sub regions
        """
        info = []
        int_vals = [] # integral values of overlapping gaussians
        all_gaussians = self.gaussians()
        
        for gauss in all_gaussians:
            mu, sigma = gauss[1], gauss[2]
            gs = self.get_all_gaussians_within_sigma_range(mu, sigma, 
                                                           sigma_tol)
            info.append(gs)
            int_val = 0  
            for g in gs:
                int_val += self.integrate_gauss(*g)
            int_vals.append(int_val)
            
        return info, int_vals
    
    def analyse_fit_result(self, sigma_tol_overlaps=None):
        """Analyse result of optimisation
        
        Find main peak (can be overlap of single Gaussians) and potential other
        peaks.
        
        Parameters
        ----------
        sigma_tol : :obj:`float`, optional
            sigma tolerance level for finding overlapping Gaussians, if None, 
            use :attr:`sigma_tol_overlaps`.
            
        Returns
        -------
        tuple
            4-element tuple containing
            
            - :obj:`float`: center position (:math:`\\mu`) of predominant peak
            - :obj:`float`: corresponding standard deviation 
            - :obj:`float`: integral value of predominant peak
            - :obj:`list`: additional Gaussians (from fit result) that are \
                not lying within specified tolerance interval of main peak
        """
        if sigma_tol_overlaps is None:
            sigma_tol_overlaps = self.sigma_tol_overlaps
    
        info, ints = self.find_overlaps(sigma_tol_overlaps)
        #the peak index with largest integral value for integrated superposition
        #of all gaussians which are within 3sigma of this peak
        ind = argmax(ints) 
        #list of all gaussians contributing to max integral val
        gs = info[ind] 
        #Gaussians belonging to main peak
        params_mp = []
        for g in gs:
            params_mp.extend(g)
    
        x = self.index
        params_mp_norm = self.normalise_params(params_mp)
        mp_norm  = multi_gaussian_no_offset(x, *params_mp_norm)
        mu = self.det_moment(x, mp_norm, 0, 1)
        sigma = sqrt(self.det_moment(x, mp_norm, mu, 2))
        add_g = self.get_all_gaussians_out_of_sigma_range(mu,
                                                          sigma,
                                                          sigma_tol_overlaps)
          
#==============================================================================
#         print "Retrieved main peak parameters: %.3f +/- %.3f" %(mu, sigma)
#         print "Gauss overlap tol.: %d" %sigma_tol_overlaps
#         print "No. of additional Gaussians (excluded from stats): %d" %len(add_g)
#         for g in add_g:
#             print g
#==============================================================================
        return (mu, sigma, ints[ind], add_g)
        
    def analyse_fit_result_old(self, sigma_tol=None):
        """Analyse result of optimisation
        
        Find main peak (can be overlap of single Gaussians) and potential other
        peaks.
        
        Parameters
        ----------
        sigma_tol : :obj:`float`, optional
            sigma tolerance level for finding overlapping Gaussians, if None, 
            use :attr:`sigma_tol_overlaps`.
            
        Returns
        -------
        tuple
            4-element tuple containing
            
            - :obj:`float`: center position (:math:`\\mu`) of predominant peak
            - :obj:`float`: corresponding standard deviation 
            - :obj:`float`: integral value of predominant peak
            - :obj:`list`: additional Gaussians (from fit result) that are \
                not lying within specified tolerance interval of main peak
            
        """
        #get index of peak position
        mu0 = self.index[argmax(self.data)]
        
        info, ints = self.find_overlaps(sigma_tol)
        #the peak index with largest integral value for integrated superposition
        #of all gaussians which are within 3sigma of this peak
        ind = argmax(ints) 
        gs = info[ind] #list of all gaussians contributing to max integral val
        max_int = ints[ind] #value of integrated superposition
        #mu = self.gaussians()[ind][1] #mu of main peak
        #if not low < mu < high:
            #print("Main peak of multi gauss retrieval does not "
           #     "match with main peak estimate from single gauss fit")
        sigmas = []
        weights = []
        mus = []
        del_mus = []
        for g in gs:
            del_mus.append(abs(g[1] - mu0))
            mus.append(g[1])
            weights.append(self.integrate_gauss(*g) / max_int)
            sigmas.append(g[2])
        weights = asarray(weights)
        mean_mu = average(asarray(mus), weights=weights)
        mean_del_mu = average(asarray(del_mus), weights=weights)
        mean_sigma = average(asarray(sigmas), weights=weights) + mean_del_mu
        add_gaussians = self.get_all_gaussians_out_of_sigma_range(mean_mu,
                                                                  mean_sigma,
                                                                  sigma_tol)
                                                            
        return mean_mu, mean_sigma, max_int, add_gaussians
        
    ### Helpers / post analysis
    def normalise_params(self, params=None):
        """Get normalised distribution of Gaussians"""
        if params is None:
            params = self.params
        ints = self.integrate_all_gaussians(params)
        weights = ints / ints.sum()
        norm = []
        gs = self._params_to_sublist(params)
        #print "NUM of cons. peaks for normalisation: (%d | %d)" %(len(gs), len(self.gaussians()))
        for k in range(len(gs)):
            g = gs[k]
            mu, sigma = g[1], g[2]
            norm.extend([weights[k]/(sigma*sqrt(2*pi)), mu, sigma])
        return norm
        
    def gaussians(self):
        """Get list containing fitted parameters for each Gaussian"""
        return self._params_to_sublist(self.params)
        
    def _params_to_sublist(self, params):
        """Convert fit paramer list to list containing each Gaussian sep"""
        return [params[i:i + 3] for i in range(0, len(params), 3)]
        
    def integrate_gauss(self, amp, mu, sigma, start=-inf, stop=inf):
        """Return integral value of one Gaussian
        
        Parameters
        ----------
        amp : float
            amplitude of Gaussian
        mu : float
            mean of Gaussian
        sigma : float
            standard deviation
        start : 
            left integral border, defaults to :math:`-\infty`            
        stop : 
            right integral border, defaults to :math:`\infty`            
        """
        if start == -inf and stop == inf:
            return amp * sigma * sqrt(2 * pi)
        g = [amp, mu, sigma]
        return quad(lambda x: gaussian_no_offset(x, *g), start, stop)[0]
    
    def integrate_all_gaussians(self, params=None):
        """Determines the integral values of all Gaussians in ``self.gaussians`` 
        :returns list: integral values for each Gaussian
        """
        vals = []
        if params is None:
            gaussians = self.gaussians()
        else:
            gaussians = self._params_to_sublist(params)
        for g in gaussians:
            vals.append(self.integrate_gauss(*g))
        return asarray(vals)
    
    def det_moment(self, index, data, center, n):
        """Determine n-th moment of distribution"""
        return sum((index - center)**n * data) / sum(data)
        
    ### Creation of test data
    def create_test_data_singlegauss(self, add_noise=True, noise_frac=0.05):
        """Make a test data set containing a single Gaussian (without offset)
        
        The parameters of the Gaussian are ``[300, 150, 20]``
        
        Parameters
        ----------        
        add_noise : bool
            add noise to test data
        noise_frac : float
            determines noise amplitude (fraction relative to max amplitude of
            Gaussian)
        """
        x = linspace(0, 400, 401)
        amp = 300
        params = [amp, 150, 20]
        y = multi_gaussian_same_offset(x, 15, *params)
        if add_noise:
            y = y + amp * noise_frac * random.normal(0, 1, size = len(x))
        self.set_data(y, x)
        
    def create_test_data_multigauss(self, add_noise=True, noise_frac=0.03):
        """Create test data set containing 5 overlapping Gaussians
        
        Parameters
        ----------        
        add_noise : bool
            add noise to test data
        noise_frac : float
            determines noise amplitude (fraction relative to max amplitude of
            Gaussian)
    
        """
        x = linspace(0,400,401)
        params = [150,30,8,200,110,3,300,150,20,75,370,40,300,250,1]
        y = multi_gaussian_same_offset(x, 45, *params)
        if add_noise:
            y = y + 300 * noise_frac * random.normal(0, 1, size=len(x))
        self.set_data(y, x)
    
    def create_test_data_multigauss2(self, add_noise=True, noise_frac=0.03):
        """Create test data set containing 5 overlapping Gaussians
        
        Parameters
        ----------        
        add_noise : bool
            add noise to test data
        noise_frac : float
            determines noise amplitude (fraction relative to max amplitude of
            Gaussian)
    
        """
        x = linspace(-180,180,361)
        params = [150,-110,25,300,-50,20,150,90,10]
        y = multi_gaussian_same_offset(x, 45, *params)
        if add_noise:
            y = y + 300 * noise_frac * random.normal(0, 1, size=len(x))
        self.set_data(y, x)
        
    def create_noise_dataset(self):
        """Make pure noise and set as current data"""
        x = linspace(0,400,401)
        y = 5 * random.normal(0, 1, size=len(x))
        self.set_data(y, x)
     
    def apply_binomial_filter(self, data=None, sigma=1):
        """Returns filtered data using 1D gauss filter
        
        Parameters
        ----------
        data : :obj:`array`, optional
            data to be smoothed, if None, use ``self.data``
        sigma : int
            width of smoothing kernel, defaults to 1
            
        Returns
        -------
        array
            smoothed data array
        
        """
        if data is None:
            data = self.data
        return gaussian_filter1d(data, sigma)
    
    def first_derivative(self, data=None):
        """Determines and returns first derivatieve of data
        
        The derivative is determined using the numpy method :func:`gradient`
        
        Parameters
        ----------
        data : :obj:`array`, optional
            data to be smoothed, if None, use ``self.data``
        
        Returns
        -------
        array
            array containing gradients
        """
        if data is None:
            data = self.data
        return gradient(data)        
        
    def set_noise_amp(self, ampl):
        """Set the current fit amplitude threshold
        
        :param float ampl: amplitude of noise level
        """
        self.noise_amp = ampl
    
    def estimate_noise_amp(self, sigma_gauss=3, sigma_tol=3,
                           cut_out_width=None):
        """Estimate the noise amplitude of the current data
        
        
        Parameters
        ----------
        sigma_gauss : int
            width of smoothing kernel applied to data in order to determine 
            analysis signal
        sigma_tol : float
            factor by which noise signal standard deviation is multiplied in 
            order to estimate noise amplitude
        cut_out_width : 
            specifyies the width of index neighbourhood around narrow peaks 
            which is to be disregarded for statistics of noise amplitude. Such 
            narrow peaks can remain in the analysis signal. If None, it is set 
            3 times the width of the smoothing kernel used to determine the 
            analysis signal.
            
        Returns
        -------
        tuple
            3-element tuple containing
            
            - :obj:`float`: the analysis signal
            - :obj:`array`: mask specifying indices used to determine the ampl.
            - :obj:`array`: inital index array
            
        """
        if cut_out_width is None:
            cut_out_width = sigma_gauss * 3
        mask = full(len(self.data), True, dtype = bool)
        width = int(self.x_resolution * cut_out_width)
        #Analysis signal
        signal = self.data - self.apply_binomial_filter(sigma=sigma_gauss)
        idxs = where(abs(signal) > sigma_tol * signal.std())[0]
        for idx in idxs:
             mask[idx - width : idx + width] = False
        try:
            self.noise_amp = sigma_tol * signal[mask].std()
        except:
            warn("Using conservative estimate for noise amplitude")
            self.noise_amp = sigma_tol * signal.std()
        return signal, mask, idxs
    
    def max(self):
        """Return max value and x position of current parameters (not of data)"""
        if self.has_results():
            vals = multi_gaussian_no_offset(self.index, *self.params) +\
                                                                self.offset
            return max(vals), self.index[argmax(vals)]
        return [None, None]
    
    @property   
    def num_of_gaussians(self):
        """Get the current number of Gaussians, which is the length 
        
        :returns: 
            - float, ``len(self.params) / 3`` 
        """
        return len(self.params) / 3
        
    @property
    def max_amp(self):
        """Get the max amplitude of the current fit results"""
        if self.has_results():
            return self.max()

    @property
    def y_range(self):
        """Range of y values """
        return float(self.data.max() - self.data.min())

    @property
    def x_range(self):
        """Range of x values"""
        return float(self.index.max() - self.index.min())
    
    @property
    def x_resolution(self):
        """Returns resolution of x data array"""
        return self.x_range / (len(self.index) - 1)
    
    def get_sub_intervals_bool_array(self, bool_arr):
        """Get all subintervals of the input bool array
        
        .. note:: 
        
            Currently not in use, but might be helpful at any later stage
            
        """
        ind = where(diff(bool_arr) == True)[0]
        if bool_arr[0] == True:
            ind = insert(ind, 0, 0)
        if bool_arr[-1] == True:
            ind = insert(ind, len(ind), len(bool_arr) - 1)
        #print "Found sub intervals: " + str(ind)
        return ind
        
    def get_residual(self, params=None, mask=None):
        """Get the current residual
        
        :param list params: Multi gauss parameters, if None, use 
                                                        ``self.params``
        :param logical mask: use only certain indices
        
        """
        if not self.has_results():
            #print "No fit results available"
            return self.data - self.offset 
        if params is None:
            params = self.params
        x, y = self.index, self.data
        if mask is not None and len(mask) == len(x):
            x = x[mask]
            y = y[mask]
        dat = y - self.offset
        return dat - multi_gaussian_no_offset(x, *params)
        
    def get_peak_to_peak_residual(self, params=None):
        """Return peak to peak difference of fit residual
        
        :param list params: mutligauss optimisation parameters, if default
            (None), use ``self.params``
        """
        if params is None:
            params = self.params
        res = self.get_residual(params)
        return res.max() - res.min()
    
    def cut_sigma_range(self, x, y, params, n_sigma=4):
        """Cut out a N x sigma environment around Gaussian from data
        
        :param array x: x-data array
        :param array y: y-data array
        :param list params: Gaussian fit parameters [ampl, mu, sigma]
        :param int n_sigma: N (e.g. 3 => 3* sigma environment will be cutted)
        """
        data = []
        mu, sigma = params[1], params[2]
        l, r = mu - n_sigma * sigma, mu + n_sigma * sigma
        x1, y1 = x[x<l], y[x<l]
        x2, y2 = x[x>r], y[x>r]
        if len(x1) > 0:
            data.append([x1, y1])
        if len(x2) > 0:
            data.append([x2, y2])
        #print "Mu: %s, sigma: %s" %(mu, sigma)
        #print "Cutting out range (left, right): %s - %s" %(l, r)
        return data
    
    def _prep_bounds_single_gauss(self):
        """Prepare fit boundary arrays (lower, higher)  
        
        Uses parameters specified in ``self.gauss_bounds``
        """
        bds = self.gauss_bounds
        low = [bds["amp"][0], bds["mu"][0], bds["sigma"][0]]
        high = [bds["amp"][1], bds["mu"][1], bds["sigma"][1]]
        return (low, high)
    
    def _in_bounds(self, params):
        """Checks if gauss params fulfill current boundary conditions
        
        :param params: parameters of a single gauss ``[amp, mu, sigma]``
        """
        bds = self.gauss_bounds
        if not bds["amp"][0] <= params[0] <= bds["amp"][1]:
            #print "Amplitude out of range, value: " + str(params[0]) 
            return 0
        if not bds["mu"][0] <= params[1] <= bds["mu"][1]:
            #print "Mu out of range, value: " + str(params[1]) 
            return 0
        if not bds["sigma"][0] <= params[2] <= bds["sigma"][1]:
            #print "Sigma out of range, value: " + str(params[2]) 
            return 0
        return 1
        
    def check_peak_bounds(self, params):
        """Checks if gauss params fulfill current boundary conditions
        
        :param params: parameters of a single gauss ``[amp, mu, sigma]``
        """
        bds = self.gauss_bounds
        if params[0] < bds["amp"][0]:
            params[0] = bds["amp"][0]
        elif params[0] > bds["amp"][1]:
            params[0] = bds["amp"][1]
            
        if params[1] < bds["mu"][0]:
            params[1] = bds["mu"][0]
        elif params[1] > bds["mu"][1]:
            params[1] = bds["mu"][1]
        
        if params[2] < bds["sigma"][0]:
            params[2] = bds["sigma"][0]
        elif params[2] > bds["sigma"][1]:
            params[2] = bds["sigma"][1]
        return params
        
    def _value_range_single_gauss(self, x, p):
        """Return max amplitude of min/max of Gaussian in x array
        
        :param x: x values
        :param p: gauss params
        """
        params = list(p)
        if len(params) == 3:
            params.append(0)
        vals = self.gaussian(x, *params)
        return abs(vals.max() - vals.min())     
        
    def get_all_gaussians_within_sigma_range(self, mu, sigma,
                                             sigma_tol=None):
        """Find all current Gaussians within sigma range of a Gaussian
        
        Parameters
        ----------
        mu : float 
            mean (x pos) of considered Gaussian
        sigma : float
            corresponding standard deviation
        sigma_tol : :obj:`float`, optional
            sigma tolerance factor for finding overlaps, if None, 
            use :attr:`sigma_tol_overlaps`
        
        Returns
        -------
        list
            list containing parameter lists ``[amp, mu, sigma]`` for all 
            Gaussians of the current fit result having their mu values within 
            the specified sigma interval of the input Gaussian
        """
        if sigma_tol is None:
            sigma_tol = self.sigma_tol_overlaps
        l, r = mu - sigma_tol * sigma, mu + sigma_tol * sigma
        gaussians = []
        gs = self.gaussians()
        for g in gs:
            if l < g[1] < r:# or l1 < g[1] < r1:
                gaussians.append(g)
        return gaussians
    
    def get_all_gaussians_out_of_sigma_range(self, mu, sigma, 
                                             sigma_tol=None):
        """Find all current Gaussians out of sigma range of a Gaussian
        
        Parameters
        ----------
        mu : float 
            mean (x pos) of considered Gaussian
        sigma : float
            corresponding standard deviation
        sigma_tol : :obj:`float`, optional
            sigma tolerance factor for finding overlaps, if None, 
            use :attr:`sigma_tol_overlaps`
        
        Returns
        -------
        list
            list containing parameter lists ``[amp, mu, sigma]`` for all 
            Gaussians of the current fit result having their mu values within 
            the specified sigma interval of the input Gaussian
        """
        if sigma_tol is None:
            sigma_tol = self.sigma_tol_overlaps
        l, r = mu - sigma_tol * sigma, mu + sigma_tol * sigma
        gaussians = []
        gs = self.gaussians()
        for g in gs:
            if g[1] < l or g[1] > r:# or l1 < g[1] < r1:
                gaussians.append(g)
        return gaussians
        
    """
    Plotting / Visualisation etc..
    """
    def plot_signal_details(self):
        """Plot signal and derivatives both in original and smoothed version
        
        Returns
        -------
        array
            axes of two subplots
        """
        if not self.has_data:
            print "No data available..."
            return 0 
        fig, ax = subplots(2,1)
        ax[0].plot(self.index, self.data, "--g", label="Signal ")
        ax[0].plot(self.index, self.data_smooth, "-r", label="Smoothed")
        ax[0].legend(loc='best', fancybox=True, framealpha=0.5)
        ax[0].set_title("Signal")
        ax[0].grid()
        ax[1].plot(self.index, self.data_grad, "--g", label="Gradient")
        ax[1].plot(self.index, self.data_grad_smooth, "-r", 
                   label="Smoothed (width 3)")
        ax[1].legend(loc='best', fancybox=True, framealpha=0.5)
        ax[1].set_title("Derivative")
        ax[1].grid()
        return ax
        
    def plot_data(self, ax=None, sub_min=False):
        """Plot the input data
        
        :param ax: matplotlib axes object (default = None)
        :param bool sub_min: if true, ``self.offset`` will be 
            subtracted from data, (default = False)
            
        """
        if not self.has_data:
            print "No data available..."
            return 0 
        if ax is None:
            fig, ax = subplots(1,1)
        y = self.data
        l_str = "Data"
        if sub_min:
            y = self.data - self.offset
            l_str += " (submin)"
            
        ax.plot(self.index, y," x", lw=2, c='b', label = l_str)
        return ax

    def plot_multi_gaussian(self,x=None, params=None ,ax=None, color="r", 
                            lw=2,**kwargs):
        """Plot multi gauss
        
        :param array x: x data array, if None, use ``self.index``
            (default = None)
        :param list params: multi gauss parameter list if None, use 
            ``self.params`` (default = None)
        :param axes ax: matplotlib axes object (default = None)
        :param **kwargs: additional keyword args passed to matplotlib  plot 
            method
        
        """
        if ax is None:
            fig, ax = subplots(1,1)
        if x is None:
            x = self.index
        if params is None:
            params = self.params
        model = multi_gaussian_no_offset(x, *params) + self.offset
        ax.plot(x, model, color=color, lw=lw, **kwargs) 
        return ax
                     
    def plot_gaussian(self, x, params, ax = None, **kwargs):
        """Plot gaussian
        
        :param array x: x data array
        :param list params: single gauss parameter list
        :param axes ax: matplotlib axes object (default = None)
        :param **kwargs: additional keyword args passed to matplotlib  plot 
            method
        
        
        """
        if ax is None:
            fig, ax = subplots(1,1)
        params = list(params)            
        if len(params) == 3:
            params.append(0)
        dat = gaussian(x, *params) + self.offset
        ax.plot(x, dat, lw = 1, ls = "--", marker = " ", **kwargs)
        return ax
    
    def plot_result(self, add_single_gaussians=False, figsize=(16, 10)):
        """Plot the current fit result
        
        :param bool add_single_gaussians: if True, all individual Gaussians are 
            plotted
            
        """
        if not self.has_data:
            #print "Could not plot result, no data available.."
            return 0
        fig, axes = subplots(2,1, figsize=figsize)
        self.plot_data(sub_min = 0, ax = axes[0])
        x = linspace(self.index.min(), self.index.max(), len(self.index) * 3)
        if not self.has_results():
            #print "Only plotted data, no results available"
            return axes
        self.plot_multi_gaussian(x,self.params,ax=axes[0],\
                                        label = "Fit result", lw=2, c="b")        
        if add_single_gaussians:
            k = 1
            for g in self.gaussians(): 
                self.plot_gaussian(self.index, g, ax = axes[0],\
                    label = ("%d. Gaussian" %k))
                k += 1
                
        axes[0].legend(loc = 'best', fancybox = True, framealpha = 0.5)
        tit = r"Result"
        try:
            mu, sigma, _, _= self.analyse_fit_result()
            tit += r" main peak: $\mu (+/-\sigma$) = %.1f (+/- %.1f)" %(mu, sigma)
        except:
            pass
        
        axes[0].set_title(tit)
        
        res = self.get_residual(self.params)
        axes[1].plot(self.index, res)
        axes[1].set_title("Residual")
        fig.tight_layout()
        
        return axes
    
    """
    I/O stuff, prints etc...
    """    
    def print_gauss(self, ind):
        """print gauss string
        
        :param int ind: index of gauss in ``self.params``        
        """
        g = self.gaussians()[ind]
        print self.gauss_str(g)
        
    
    def gauss_str(self, g):
        """String representation of a Gaussian
        
        :param list g: gauss parameter list ``[amp, mu, sigma]``
                        
        """
        return "Amplitude: %.2f\nMu: %.2f\nSigma: %.2f\n" %(g[0], g[1], g[2])
            
    def info(self):
        """Print string representation"""
        print self.__str__()
    
    @property
    def has_data(self):
        """Returns True, if data available, else False"""
        if len(self.data) > 0:
            return True
        return False
    
    def has_results(self):
        """Check if fit results are available"""
        if self._fit_result is not None and sum(self.params) > 0:
            return 1
        #print "No multi gauss fit results available"
        return 0    
    """
    Magic methods
    """
    def __str__(self):
        """String representation"""
        gs = self.gaussians()
        s=("pyplis MultiGaussFit info\n--------------------------------\n\n"
            "All current Gaussians:\n\n")
        for k in range(len(gs)):
            g = gs[k]
            s+= "Gaussian #%d\n%s\n" %(k, self.gauss_str(g))
            
        s += ("Current peak to peak residual: %s\nNoise amplitude: %s" 
        %(self.get_peak_to_peak_residual(self.params), self.noise_amp))
        return s

class PolySurfaceFit(object):
    """Fit a 2D polynomial to data (e.g. a blue sky background image)
    
    This class can be used to fit 2D polynomials to image data. It includes 
    specifying pixels supposed to be used for the fit which have to be
    provided using a mask. The fit can be performed at arbitrary Gauss pyramid
    levels which can dramatically increase the performance.
    
    Note
    ----

    The fit result image can be accessed via the attribute ``model``
        
    Parameters
    ----------
    data_arr : array
        image data to be fitted (NxM matrix)
    mask : array
        mask specifying pixels considered for the fit (if None, then all 
        pixels of the image data are considered
    polyorder : int
        order of polynomial for fit (default=3)
    pyrlevel : int
        level of Gauss pyramid at which the fit is performed (relative to
        Gauss pyramid level of input data)
    do_fit : bool
        if True, and if input data is valid, then the fit is performed on 
        intialisation of the class
        
    """    
    def __init__(self, data_arr, mask=None, polyorder=3, pyrlevel=4, do_fit=1):
        self.data = None
        self.mask = None
        
        self.pyrlevel = pyrlevel
        self.polyorder = polyorder
        self.err_fun = models.Polynomial2D(degree=self.polyorder)
        self.fitter = LevMarLSQFitter()
        self.params = None
        self.model = None
        self.auto_update = 1
        if self.set_data(data_arr, mask) and do_fit:
            self.do_fit()
    
    def set_data(self, data_arr, mask=None):
        """Set the data array (must be dimension 2)
        
        Create ``self.mask`` for array shape which can be used to exclude 
        picxel areas from the image
        
        Parameters
        ----------
        data_arr: array
            image data (can also be :class:`Img`)
        mask : array
            boolean mask specifying pixels considered for fit, if None, all
            pixels are considered
            
        Returns
        -------
        bool
            True if data is valid, False if not
        """
        try:
            data_arr = data_arr.img
        except:
            pass
        try:
            mask = mask.img
        except:
            pass
        if not ndim(data_arr) == 2:
            warn("Could not set data, dimension mismatch...")
            return 0
        if mask is None or mask.shape != data_arr.shape:
            mask = ones_like(data_arr)
        self.data = data_arr
        self.mask = mask.astype(uint8)
        self.params = None #storage of fit results
        self.model = None
        return 1
    
    def activate_auto_update(self, val=1):
        """Activate or deactivate auto update mode. 
        
        If active, the fit is reapplied each time some input parameter is 
        changed
        
        Parameters
        ----------
        val : bool
            new value for :attr:`auto_update` 
            
        """
        self.auto_update = val
    
    def change_pyrlevel(self, newlevel):
        """Change the level in the Gaussian pyramide where the fit is applied
        """
        self.pyrlevel = newlevel
        if self.auto_update:
            self.do_fit()
            
    def change_polyorder(self, new_order):
        """Change the order of the polynomial which is fitted
        
        Sets new poly order and re-applies fit in case ``auto_update == True``
        
        Parameters
        ----------
        new_order : int
            new order of poly fit
        
        """
        self.polyorder = int(new_order)
        if self.auto_update:
            self.do_fit()
            
    def exclude_pix_range_rect(self, x0, x1, y0, y1):
        """Add a rectangular pixel area which will be excluded from the fit
        :param int x0: start x coordinate (original image resolution)
        :param int x1: stop x coordinate (original image resolution)
        :param int y0: start y coordinate (original image resolution)
        :param int y1: stop y coordinate (original image resolution)
        """
        self.mask[y0:y1,x0:x1]=0
        if self.auto_update:
            self.do_fit()
            
    def exclude_pix_range_circ(self, x0, y0, r):
        """Add a circular pixel area which will be excluded from the fit
        :param int x0: centre x coordinate (original image resolution)
        :param int y0: centre y coordinate (original image resolution)
        :param int r: radius in pixel coordinates (original image resolution)
        """
        m=self.mask
        h,w=m.shape
        y,x = ogrid[:h,:w]
        m1=(x-x0)**2+(y-y0)**2 > r**2
        self.mask=m*m1
        if self.auto_update:
            self.do_fit()
            
    def pyr_down(self, arr, steps):
        """Reduce the image size using Gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrDown` 
        """
        for i in range(steps):
            arr=pyrDown(arr)
        return arr
    
    def pyr_up(self, arr, steps):
        """Increasing the image size using Gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrUp` 
        """
        for i in range(steps):
            arr=pyrUp(arr)
        return arr
        
    def make_weights_mask(self):
        """Make a mask for pixel fit weights based on values in self.mask"""
        cond = self.mask < .99
        weights = deepcopy(self.mask)
        weights[cond] = 1E-30
        return weights
    
    def do_fit(self):
        """Apply the fit to the data and write results"""
        #print "Fitting 2D polynomial to data...(this might take a moment)"
        
        try:
            weights=self.make_weights_mask()
            weights=self.pyr_down(weights, self.pyrlevel)
            inp=self.pyr_down(self.data, self.pyrlevel)            
            h,w=inp.shape
            with catch_warnings():
                # Ignore model linearity warning from the fitter
                simplefilter('ignore')
                x,y=mgrid[:h,:w]
                self.params = self.fitter(self.err_fun,x,y,inp,weights=weights)
                self.model=self.pyr_up(self.params(x,y), self.pyrlevel)
            return self.model
        except:
            print format_exc()
            return 0   
    
    
        
    def get_residual(self):
        """Get the current residual image array"""
        return self.data - self.model  
        
    def plot_result(self):
        """Plot the fit result
        
        Plots the results, the original data and the residual in two 
        versions (2 different intensity ranges)
        """
        l,h=self.data.min(),self.data.max()
        fig, axes=subplots(2,2,figsize=(16,8))
        ax=axes[0,0]
        p0=ax.imshow(self.data,vmin=l, vmax=h)
        fig.colorbar(p0,ax=ax,shrink=0.8)
        ax.set_title("RAW input")
        
        ax=axes[0,1]
        p1=ax.imshow(self.model,vmin=l, vmax=h)
        fig.colorbar(p1,ax=ax,shrink=0.8)
        ax.set_title("Fit result")
        
        ax=axes[1,0]
        p2=ax.imshow(self.get_residual(),vmin=-h, vmax=h)
        fig.colorbar(p2,ax=ax,shrink=0.8)
        ax.set_title("Residual (scaled)")
        
        ax=axes[1,1]
        p3=ax.imshow(self.get_residual())#,vmin=l, vmax=h)
        fig.colorbar(p3,ax=ax,shrink=0.9)
        ax.set_title("Residual")

    
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.pyplot import rc_context
    rc_context({'font.size':'12'})
    
    TOL = 3
    plt.close("all")
    f = MultiGaussFit()
    f.create_test_data_multigauss(1, 0.02)
    #f.create_test_data_singlegauss(0)
    f.run_optimisation()
    
    axes=f.plot_result(True, figsize=(12,10))
    ax=axes[0]
    ax.set_ylim([0, 400])
    ax.set_title("")
    
    p_norm = f.normalise_params()
    
    x = f.index
    data_norm = multi_gaussian_no_offset(x, *p_norm)
    
    mu0 = f.det_moment(x, data_norm,0,1)
    sigma0 = np.sqrt(f.det_moment(x, data_norm, mu0, 2))
    
    ### COPY OF FUNC analyse_fit_result        
    info, ints = f.find_overlaps(TOL)
        #the peak index with largest integral value for integrated superposition
        #of all gaussians which are within 3sigma of this peak
    ind = argmax(ints) 
    gs = info[ind] #list of all gaussians contributing to max integral val
    max_int = ints[ind] #value of integrated superposition
        #mu = self.gaussians()[ind][1] #mu of main peak
        #if not low < mu < high:
            #print("Main peak of multi gauss retrieval does not "
           #     "match with main peak estimate from single gauss fit")
    main_peak = []
    for g in gs:
        main_peak.extend(g)
        
    mp  = multi_gaussian_no_offset(x, *main_peak)
    mu1 = f.det_moment(x, mp, 0, 1)
    sigma1 = np.sqrt(f.det_moment(x, mp, mu1, 2))
    
    mean_mu, mean_sigma, max_int, add_gaussians = f.analyse_fit_result_old(TOL)
    
    pos_add = [g[1] for g in add_gaussians]
    for g in f.gaussians():
        if g[1] in pos_add:
            axes[0].annotate("Additional\npeak", xy=(g[1], g[0] + f.offset), xytext=(g[1]-10, g[0] + 20 + f.offset), 
                arrowprops=dict(arrowstyle="->", color="k", connectionstyle=
                "arc,angleA=10,armA=20,rad=6", shrinkA=2, 
                shrinkB=2), color="k", fontsize=14)
    mu2, sigma2, _, _ = f.analyse_fit_result(TOL)
 
    print "Mu, sigma (moments ALL): %.2f, %.2f" %(mu0, sigma0)
    print "Mu, sigma (OLD METHOD): %.2f, %.2f" %(mean_mu, mean_sigma)
    print "Mu, sigma (moments MAIN PEAK): %.2f, %.2f" %(mu1, sigma1)
    print "Mu, sigma (moments MAIN PEAK normalised): %.2f, %.2f" %(mu2, sigma2)
    
    
