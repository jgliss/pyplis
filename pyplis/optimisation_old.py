# -*- coding: utf-8 -*-
"""Module containing optimisation and fitting algorithms"""
from numpy import abs, linspace, random, asarray, ndarray, where, diff,\
    insert, argmax, average, gradient, arange,argmin, full, inf, sqrt, pi,\
    mod, mgrid, ndim, ones_like,ogrid, finfo, remainder, e, sum, uint8
    
from warnings import catch_warnings, simplefilter, warn
from matplotlib.pyplot import subplots

from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter

from scipy.ndimage.filters import gaussian_filter1d, median_filter
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

def gauss_fit_2d(img_arr, cx, cy, g2d_asym = True,\
        g2d_super_gauss = True, g2d_crop = True, g2d_tilt = False, **kwargs):
    """Apply 2D gauss fit to input image at its maximum pixel coordinate
    
    :param corr_img: correlation image
    :param float cx: x position of peak in image (used for initial guess)
    :param float cy: y position of peak in image (used for initial guess)
    :param bool g2d_asym: allow for assymetric shape (sigmax != sigmay), True
    :param bool g2d_super_gauss: allow for supergauss fit, True
    :param bool g2d_crop: if True, set outside (1/e amplitude) datapoints = 0,
        True
    :param bool g2d_tilt: allow gauss to be tilted with respect to x/y axis
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

class MultiGaussFit(object):
    """Fitting environment for fitting an arbitrary (i.e. unknown) amount of
    gaussians to noisy 1D (x,y) data. It was initally desinged and developed
    for histogram data and aims to find a solution based on a minimum of 
    required superimposed gaussians to describe the distribution. Therefore,
    the fit is performed in a controlled way (i.e. allowed gaussians are
    required to be within certain parameter bounds, details below) starting 
    with a noise analysis (if noise level is not provided on class 
    initialisation).
    Based on the noise level, and the x-range of the data, the boundaries for
    accepted gauss parameters are set. These are::
    
        self.gauss_bounds["amp"][0] = 2*self.noise_amplitude
        self.gauss_bounds["amp"][1] = (self.y_range - self.offset) * 1.5
        self.gauss_bounds["mu"][0] = self.index[0]
        self.gauss_bounds["mu"][1] = self.index[-1]
        self.gauss_bounds["sigma"][0] = self.x_resolution/2.
        self.gauss_bounds["sigma"][1] = self.x_range/2.
        
    i.e. the amplitude of each of the superimposed gaussians must be positive
    and larger then 2 times the noise amplitude. The max allowed amplitude
    is set 1.5 times the min / max difference of the data. The mean of each 
    gaussian must be within the index range of the data and the standard
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
        amplitude for fitted gaussians (you don't want to fit all the noise 
        peaks). If None, it will be estimated automatically on data import 
        using :func:`estimate_noise_amplitude`
    smooth_sig : int
        width of Gaussian kernel to determine smoothed analysis signal
    sigma_tol_overlaps : int
        sigma range considered to find overlapping Gauss functions (after 
        fit was applied). This is, for instance used in 
        :func:`analyse_fit_result` in order to find the main peak parameters
    max_num_gaussians : int 
        max number of superimposed, defaults to 20
        gaussians for data 
    max_iter : int
        max number of iterations for optimisation, defaults to 10
    auto_bounds : bool
        if True, bounds will be set automatically from data ranges whenever 
        data is updated, defaults to True
    do_fit : bool
        if True and input data available & ok, then :func:`self.auto_fit` will 
        be performed on initialisation, defaults to True
    horizontal_baseline : bool
        data has horizontal baseline (if this  is set False, the fit strategy 
        is different... some more info follows)
    """
    def __init__(self, data=None , index=None, noise_amp=None, smooth_sig=3,
                 sigma_tol_overlaps=2, max_num_gaussians=20, max_iter=10, 
                 auto_bounds=True, horizontal_baseline=True, do_fit=True):
        #data
        self.index = []
        self.data = []
        self.data_smooth = []
        self.data_grad = []
        self.data_grad_smooth = []
        
        # init relevant parameters
        self.offset = 0.0
        self.noise_amplitude = noise_amp
        self.smooth_sig =smooth_sig
        self.sigma_tol_overlaps = sigma_tol_overlaps
        
        self.max_num_gaussians = max_num_gaussians
        self.max_iter = max_iter
    
        #bounds for gaussians (will be set for each gauss guess before fitting)
        self._horizontal_baseline = horizontal_baseline
        #make sure autobounds is off in case of none horizontal baseline
        if not horizontal_baseline: 
            auto_bounds = False
          
        self.auto_bounds = auto_bounds
        self.gauss_bounds = {"amp"  :   [0, inf],
                             "mu"   :   [-inf, inf],
                             "sigma":   [-inf, inf]}
        
        #Fitting related stuff
        self._fit_result = None #the actual output from the minimisation
        
        self.params = [0,0,0] #this is where the fit parameters are stored in
        
        #function to be minimised
        self.err_fun = lambda p, x, y:\
                    (multi_gaussian_no_offset(x, *p) - y)#**2
        
        #will be filled with optimisation results
        self.optimise_log = {"chis"     : [],
                             "residuals": []}

        self.plot_font_sizes = {"titles":  14,
                                "labels":  12,
                                "legends": 10}
                                
        self.set_data(data, index)
        if do_fit and self.has_data:
            self.auto_fit()
    
    """
    Initialisation, data preparation, I/O, etc...
    """
    def init_results(self):
        """Initiate all result parameters"""
        self._peak_indices = []
        self.params = [0, 0, 0]
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
        """Initiate the input data and set constraints for valid gaussians
        
        Main steps:
        
            1.  Estimate the data offset from minimum of smoothed data (using
                gaussian filter, width 3 to reduce noise)
            #.  Determine 1st derivative of data (stored in
                                            ``self.data_grad``)
            #.  Smooth data and gradient (using gaussian filter width 1)
            #.  Call :func:`self.estimate_noise_amplitude`
            #.  Call :func:`self.init_gauss_bounds_auto`
        """
        if self.has_data:
            sigma = self.smooth_sig
            self.offset = gaussian_filter1d(self.data, sigma).min()
            self.data_grad = self.first_derivative(self.data)
            self.data_smooth = self.apply_binomial_filter(self.data, 
                                                          sigma=sigma-1)
            self.data_grad_smooth = self.apply_binomial_filter(self.data_grad, 
                                                               sigma=sigma-1)
            if self.noise_amplitude is None:
                self.estimate_noise_amplitude()
            self.init_gauss_bounds_auto()
     
    def set_gauss_bounds(self, amp_range=[0, inf], mu_range=[-inf, inf],
                         sigma_range=[-inf, inf]):
        """Manually set boundaries for gauss parameters
        
        :param array amp_range ([0, nan]): accepted amplitude range
        :param array mu_range ([nan, nan]): accepted mu range
        :param array sigma_range ([nan, nan]): accepted range of gauss widths
        
        This routine was initially developed for fitting histogram distributions
        (e.g. gray values in an image).
        
        """
        self.gauss_bounds["amp"] = amp_range
        self.gauss_bounds["mu"] = mu_range
        self.gauss_bounds["sigma"] = sigma_range
        
    def init_gauss_bounds_auto(self):
        """Set parameter bounds for individual gaussians"""
        if not self.has_data:
            #print "Could not init gauss bounds, no data available..."
            return 0
        if not self.auto_bounds:
            #print "Automatic update of boundaries is deactivated..."
            return 1
        self.gauss_bounds["amp"][0] = 2 * self.noise_amplitude
        self.gauss_bounds["amp"][1] = (self.y_range - self.offset) * 1.5
        self.gauss_bounds["mu"][0] = self.index[0]
        self.gauss_bounds["mu"][1] = self.index[-1]
        self.gauss_bounds["sigma"][0] = self.x_resolution / 2.
        self.gauss_bounds["sigma"][1] = self.x_range / 2.
        return 1
                
    """
    Fit preparations, peak search, etc
    """
    def estimate_main_peak_params(self):
        """Get rough estimate and position of main peak"""
        data = self.data - self.offset
        ind = argmax(data)
        amp = data[ind] 
        if not amp > 1.5 * self.noise_amplitude:
            raise IndexError("No significant peak could be found in data")
        w = self.estimate_peak_width(data, ind)
        guess = [amp, self.index[ind], w * self.x_resolution]
        y = self.data - self.offset
        params, bds = self.prepare_fit_boundaries(guess)
        
        return least_squares(self.err_fun, params, args=(self.index, y),
                             bounds=bds).x

    def find_peak_positions_residual(self):
        """Search for significant peaks in the current residual
        
        :returns list: list containing additional peak parameters (for 
            optimisation)        
        """
        dat = self.get_residual()
        add_params = []
        num = self.num_of_gaussians #current number of fitted gaussians
        for k in range(self.max_num_gaussians - num):
            if not dat.max() > 2.0 * self.noise_amplitude:
                #print "Residual peak search finished..."
                return add_params
            else:
                ind = argmax(dat)
                w = self.estimate_peak_width(dat, ind)
                add_params.append(dat[ind])
                add_params.append(self.index[ind])
                add_params.append(w * self.x_resolution)
                cut_low = ind - 3*w
                if cut_low < 0:
                    cut_low = 0
                cut_high = ind + 3*w
                dat[cut_low:cut_high] = 0
        warn("Number of detected peaks exceeds allowed max "
            "number of superimposed gaussians in model")
        
    def add_peak_from_residual(self):
        """Search for significant peaks in the current residual
        
        :returns list: list containing additional peak parameters (for 
            optimisation)        
        """
        if self.num_of_gaussians == self.max_num_gaussians:
            raise ValueError("Number of detected peaks exceeds allowed max "
                "number of superimposed gaussians in model")
        dat = self.get_residual()
        
        add_params = []
        if dat.max() > 2.0 * self.noise_amplitude:
            ind = argmax(dat)
            w = self.estimate_peak_width(dat, ind)
            add_params.append(dat[ind])
            add_params.append(self.index[ind])
            add_params.append(w * self.x_resolution)
        return add_params
        
            
    def estimate_peak_width(self, dat, idx):
        """"Estimate width of a peak at given index by finding the closest
        datapoint smaller than 0.5 the amplitude of data at index position
        
        :param int dat: data (with baseline zero)
        :param int idx: index position of peak
        
        """
        amp = dat[idx]
        #print "Estimating peak width at peak, index: " + str(idx)
        #print "x,y:" + str(self.index[idx]) + ", " + str(amp)
        max_ind = len(self.index) - 1  
        try:
            ind = next(val[0] for val in enumerate(dat[idx:max_ind])\
                                                if val[1] < amp/2)
            #print "Width (index units): " + str(abs(ind))
            
            return ind
        except:
            #print "Trying to the left"#format_exc()
            try:
                inv = dat[::-1]
                idx = len(inv)-1-idx
                ind = next(val[0] for val in enumerate(inv[idx:max_ind])\
                                                if val[1] < amp/2)
                #print "Width (index units): " + str(abs(ind))
                return ind
            except:
                pass
        #print "Peak width could not be estimated, return 1"
        return int(len(dat))*0.05#2#int(.05*len(dat))
            
    
    """
    Fitting etc
    """         
    def prepare_fit_boundaries(self, params):
        """Prepare the boundaries tuple (for details see 
        `Minimisation tool <http://docs.scipy.org/doc/scipy-0.17.0/reference/
        generated/scipy.optimize.least_squares.html>`_)
        using the boundaries specified in ``self.gauss_bounds``.
        
        :param list params: list of gauss parameters (e.g. ``self.params``)
        
        .. note::
        
            If any of the parameters in ``paramList`` is out of the acceptance
            borders specified in ``self.gauss_bounds`` it will be disregarded in
            the fit
            
        """
        if not mod(len(params),3) == 0:
            #print "Error: length of gauss param list must be divisable by three.."
            return []
        sub = [params[x : x + 3] for x in range(0, len(params), 3)]
        new_list = []
        for pl in sub:
            if self._in_bounds(pl):
                new_list.extend(pl)
            #else:
                #print ("Gaussian in param list with params: " + str(pl) + "out "
                   # "of acceptance boundaries -> will be disregarded..:")
        lower, upper = [], []
        l, u = self._prep_bounds_single_gauss()
        for k in range(len(new_list) / 3):
            lower.extend(l)
            upper.extend(u)
        bds = (lower, upper)
        return new_list, bds
        
    def do_fit(self, x, y, *guess):
        """Perform a least squares minimisation
        
        :param array x: x argument for input function
        :param array y: y argument for input function
        :param list guess: initial guess of fit parameters
        """
        try:
            params, bds = self.prepare_fit_boundaries(*guess)
            #print "Fitting data..."
            self._fit_result = res = least_squares(self.err_fun, params,
                                                   args=(x, y), bounds=bds)
            #params,ok=optimize.leastsq(self.err_fun, *guess, args=(x, y))
            if not res.success:
                #print "Fit failed"
                return False
            self.params = res.x
            return True
        except Exception:
            #print "Fit failed with exception: %s" %repr(e)
            return False
    
    def result_ok(self):
        """Compares current peak to peak residual (ppr) with noise amplitude
        :returns bool: 1 if ``2*self.noise_amplitude > ppr``, else 0
        """
        if len(self.find_peak_positions_residual()) == 0:
            return True
        return False
    
    def auto_fit(self):
        """Automatic least square analysis"""
        idx_max = argmax(self.data)
        if any([idx_max == x for x in [0, (len(self.data) - 1)]]):
            raise ValueError("Could not perform MultiGaussFit: maximum of "
                "data is at first or last index of data")
        #print "Running multi gauss auto fit routine"
        guess = self.find_peak_positions_residual()
        #print "Initial peak search: found %s peaks" %(len(guess)/3.0)
                
        y = self.data - self.offset
        if not self.do_fit(self.index, y, guess):
            return 0
        #print "Fit applied (first iter), current settings:"
        #print self.info()
        add_params = self.find_peak_positions_residual()
        if add_params:
            #print ("Found %s more peaks after first iter, running "
               # "optimisation..." %(len(add_params)/3.0))
            if not self.run_optimisation():
                return 0
        #print "Auto fit successfully finished"""
        return 1

    def run_optimisation(self):
        """Run optimisation"""
        last_params = self.params
        residuals = [self.get_residual()]
        chis = []
        for k in range(self.max_iter):
            if self.num_of_gaussians >= self.max_num_gaussians:
                #print ("Max num of gaussians (%d) reached "
                   # "abort optimisation" %self.max_num_gaussians)
                self._write_opt_log(chis, residuals)
                warn ("MultiGaussFit reached aborted at maximum number of "
                    "allowed gaussians")
                return 0
            if not self.optimise_result():
                #print ("Optimisation failed,  aborted at iter %d" %k)
                self._write_opt_log(chis, residuals)
                warn("Optimisation failed in MultiGaussFit")
                return 0
            
            residuals.append(self.get_residual())
            chis.append(residuals[-1].std() / residuals[-2].std())
            #print "Current iter: %s, current chi: %.2f" %(k, chis[-1])
            if chis[-1] > 1:
                #print "Optimisation stopped, chi increased..."
                self.params = last_params
                self._write_opt_log(chis, residuals)
                return 1
            last_params = self.params
            if self.get_peak_to_peak_residual() < self.noise_amplitude or\
                                    0.9 < chis[-1] <= 1.0:
                #print "Optimisation successfully finished"
                self._write_opt_log(chis, residuals)
                return 1
        #print "Optimisation aborted, maximum number of iterations reached..."
        self._write_opt_log(chis, residuals)
        warn("MultiGaussFit max iter reached..")
        return 0
        
    def _write_opt_log(self, chis, residuals):
        """Write chi values and residuals from optimisation"""
        self.optimise_log["chis"] = chis
        self.optimise_log["residuals"] = residuals
        
    def optimise_result(self, add_params=None):
        """Optimise current result
        
        :param add_params: list containing additional peak params (guess) which
            are not yet included within the fit. If default (None), additional
            peaks are searched in the current fit residual
        
        Extends current optimisation parameters by additional peaks (either
        provided on input or automatically searched in residual) and performs
        multi gauss fit.
        """
        if add_params is None:
            add_params = self.find_peak_positions_residual()
        if not len(add_params) > 0:
            #print ("Optimisation aborted, no additional peaks could be "
                  #  "detected in current fit residual")
            return 1
        guess = list(self.params)
        guess.extend(add_params)
        y = self.data - self.offset
        if not self.do_fit(self.index, y, guess):
            return 0
        return 1

    """
    Fit analysis, results, post processing, etc..
    """
    def find_overlaps(self, sigma_tol=None):
        """ Find overlapping gaussians for current optimisation params
        
        Loops over all current gaussians (``self.gaussians``) and for each of
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

    def analyse_fit_result(self, sigma_tol=None):
        """Analyse result of optimisation
        
        Find main peak (can be overlap of single gaussians) and potential other
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
        
    """
    Helpers
    """
    def gaussians(self):
        """Split self.params (i.e. parameters of all gaussians) into sublists
        containing information of individual gaussians
        """
        return [self.params[i:i + 3] for i in range(0, len(self.params), 3)]
        
    def integrate_gauss(self, amp, mu, sigma, start=-inf, stop=inf):
        """Return integral value of one gaussian
        :param float amp: amplitude of gaussian
        :param float mu: mean of gaussian
        :param float sigma: standard deviation
        :param float start (-inf): left integral border
        :param float stop (inf): right integral border
        """
        if start == -inf and stop == inf:
            return amp * sigma * sqrt(2 * pi)
        g = [amp, mu, sigma]
        return quad(lambda x: gaussian_no_offset(x, *g), start, stop)[0]
    
    def integrate_all_gaussians(self):
        """Determines the integral values of all gaussians in ``self.gaussians`` 
        :returns list: integral values for each gaussian
        """
        vals = []
        for g in self.gaussians():
            vals.append(self.integrate_gauss(*g))
        return vals
    
    def create_test_data_singlegauss(self, add_noise=1):
        """Make a test data set with a single gaussian
        
        :param bool add_noise: add noise to test data
        """
        self.id = "test_data"
        x = linspace(0, 400, 401)
        params = [300, 150, 20]
        y = multi_gaussian_same_offset(x, 15, *params)
        if add_noise:
            y = y + 5*random.normal(0, 1, size = len(x))
        self.set_data(y, x)
        
    def create_test_data_multigauss(self, add_noise=1):
        """Make a test data set
        
        :param bool add_noise: add noise to test data
    
        """
        self.id="test_data"
        x = linspace(0,400,401)
        params = [150,30,8,200,110,3,300,150,20,75,370,40,300,250,1]
        y = multi_gaussian_same_offset(x, 45, *params)
        if add_noise:
            y = y + 5*random.normal(0, 1, size = len(x))
        self.set_data(y, x)
        
    def create_noise_dataset(self):
        """Make pure noise and set as current data"""
        self.id="noise"
        x = linspace(0,400,401)
        y = 5 * random.normal(0, 1, size=len(x))
        self.set_data(y, x)
     
    def apply_binomial_filter(self, data=None, sigma=1):
        """Returns filtered data using 1D gauss filter
        
        :param array data (None): data to be filtered, default None means that
            ``self.data`` is used
        :param int sigma (1): width of smoothing kernel
        
        """
        if data is None:
            data=self.data
        return gaussian_filter1d(data, sigma)
    
    def first_derivative(self, data=None):
        """Apply discrete gradient to data
        
        :param ndarray data (None): data array (if None, use ``self.data``)
        """
        if data is None:
            data = self.data
        return gradient(data)        
        
    def set_noise_amplitude(self, ampl):
        """Set an amplitude threshold for the peak search
        
        :param float ampl: amplitude of noise level
        """
        self.noise_amplitude = ampl
    
    def estimate_noise_amplitude(self, sigma_gauss = 1, std_fac = 3,\
                                                        cut_out_width = None):
        """Estimate the noise amplitude of the current data
        
        :param int sigma_gauss: width of smoothing kernel applied to data in 
            order to determine analysis signal
        :param float std_fac: factor by which noise signal standard deviation
            is multiplied in order to estimate noise amplitude
        :param cut_out_width: specifyies the width of index neighbourhood 
            around narrow peaks which is to be disregarded for statistics of
            noise amplitude. Such narrow peaks can remain in the analysis 
            signal. If None, it is set 3 times the width of the smoothing 
            kernel used to determine the analysis signal.
            
        """
        if cut_out_width is None:
            cut_out_width = sigma_gauss * 3
        mask = full(len(self.data), True, dtype = bool)
        width = int(self.x_resolution * cut_out_width)
        #Analysis signal
        signal = self.data - self.apply_binomial_filter(sigma = sigma_gauss)
        idxs = where(abs(signal) > std_fac * signal.std())[0]
        for idx in idxs:
             mask[idx - width : idx + width] = False
        self.noise_amplitude = std_fac * signal[mask].std()
        return signal, mask, idxs
        
    def estimate_noise_amplitude_alt(self, sigma_gauss = 1, medianwidth = 3,\
                                cut_out_width = 20, thresh = 0.75, max_iter = 10):
        """Estimate the amplitude of the noise in the data
        
        :param int sigma_gauss (3): width of gaussian smoothing kernel for high
            pass filter
        :param int medianwidth (3): width of median filter applied to low pass
            signal (for removal of narrow peaks)
        :param int cut_out_width (20): factor defining width of cutted out region
            around bad areas at each iteration
        :param float thresh (0.75): percentage threshold for std comparison of 
            target signals to fulfill "ok" condition 
        :param int max_iter (10): max number of iterations for optimisation
        
        Steps:

            1. Determine bool array ``B`` of ones (all data is considered 
                                                                    initially)
            #. Determine high pass filtered data array (hp) by subtracting 
                                                                    original
                data from smoothed data using gaussian kernel of input width
            #. Apply median filter to hp to reduce amplitudes of 
                remaining narrow and sharp peaks (e.g. salt & pepper peaks), 
                resulting in signal "hp_median"
            #. Run optimisation until max_iter is reached or "ok" condition is
                fulfilled. The ok condition is
                ``hp_median[B].std() / hp[B].std > thresh``
                and ``B`` is reduced by sharp peak areas at each iteration by 
                analysing target signal ``hp[B]`` for position of max 
                amplitude and cutting window around this region where the width 
                of this window is specified by ``self.x_resolution*cut_out_width``
            #. Finally, if "ok" condition is reached, it is checked whether 
                enough datapoints remain when applying ``B`` to ``hp`` and if 
                not, "ok" condition is set to False
            #. Finally, based on result (i.e. "ok" is True or False) the 
                noise amplitude is estimated from the standard deviation of the
                high pass filtered signal (hp) either considering all datapoints
                (i.e. if optimisation failed) or of all datapoints within ``B``
                
            
        """
        #make bool array of indices considered (initally all)
        b = full(len(self.data), True, dtype=bool)
        #width in indices of window cutted out around deviations at each 
        #optimisation iteration
        w = self.x_resolution * cut_out_width
        #Smooth by width 3 and pick every 3rd entry of difference signal
        d = self.data - self.apply_binomial_filter(sigma = sigma_gauss)
        dm = median_filter(d,medianwidth)
        #diff = dm - d
        ok = False
        for k in range(max_iter):
            #print "Current iteration: " + str(k)
            if dm[b].std() / d[b].std() < thresh: 
                #print "Significant difference found, optimising..."
                ind1 = argmax(d[b])
                ind2 = argmin(d[b])       
                if abs(d[b][ind1]) > abs(d[b][ind2]):
                    ind = ind1
                else: 
                    ind = ind2
                #print "Bad index at: " + str(ind)
                #print "Cut out index window: " + str(ind-w) + " - " + str(ind+w)
                b[ind-w:ind+w]=False
                #print "Current number of considered indices: " + str(sum(b))
            else:
                ok = True
                #print "Optimisation finished"
                break
        
        if float(sum(b)) / len(d) < 0.2:
            #print "Too little remaining data points, using conservative approach"
            ok = False
        if ok:
            self.noise_amplitude = 6*d[b].std()#d[b].max()-d[b].min()
        else:
            #print "Optimisation failed, use std of high pass filtered signal"
            self.noise_amplitude = 6*d.std()
        #print "Estimated noise amplitude: %s" %self.noise_amplitude
        return self.noise_amplitude
    
    def max(self):
        """Return max value and x position of current parameters (not of data)"""
        if self.has_results():
            vals = multi_gaussian_no_offset(self.index, *self.params) +\
                                                                self.offset
            return max(vals), self.index[argmax(vals)]
        return [None, None]
    
    @property   
    def num_of_gaussians(self):
        """Get the current number of gaussians, which is the length 
        
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
        
    def get_residual(self, params = None, mask = None):
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
        
    def get_peak_to_peak_residual(self, params = None):
        """Return peak to peak difference of fit residual
        
        :param list params: mutligauss optimisation parameters, if default
            (None), use ``self.params``
        """
        if params is None:
            params = self.params
        res = self.get_residual(params)
        return res.max() - res.min()
    
    def cut_sigma_range(self, x, y, params, n_sigma = 4):
        """Cut out a N x sigma environment around gaussian from data
        
        :param array x: x-data array
        :param array y: y-data array
        :param list params: gaussian fit parameters [ampl, mu, sigma]
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
        
    def _value_range_single_gauss(self, x, p):
        """Return max amplitude of min/max of gaussian in x array
        
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
        """Find all current gaussians within sigma range of a gaussian
        
        Parameters
        ----------
        mu : float 
            mean (x pos) of considered gaussian
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
        """Find all current gaussians out of sigma range of a gaussian
        
        Parameters
        ----------
        mu : float 
            mean (x pos) of considered gaussian
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
        ax[0].legend(loc='best', fancybox=True, framealpha=0.5,
                     fontsize=self.plot_font_sizes["legends"])
        ax[0].set_title("Signal", fontsize = self.plot_font_sizes["titles"])
        ax[0].grid()
        ax[1].plot(self.index, self.data_grad, "--g", label="Gradient")
        ax[1].plot(self.index, self.data_grad_smooth, "-r", 
                   label="Smoothed (width 3)")
        ax[1].legend(loc='best', fancybox=True, framealpha=0.5,
                     fontsize=self.plot_font_sizes["legends"])
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
            
        ax.plot(self.index, y," x", lw=2, c='g', label = l_str)
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
    
    def plot_result(self, add_single_gaussians=False):
        """Plot the current fit result
        
        :param bool add_single_gaussians: if True, all individual gaussians are 
            plotted
            
        """
        if not self.has_data:
            #print "Could not plot result, no data available.."
            return 0
        fig, axes = subplots(2,1)
        self.plot_data(sub_min = 0, ax = axes[0])
        x = linspace(self.index.min(), self.index.max(), len(self.index) * 3)
        if not self.has_results():
            #print "Only plotted data, no results available"
            return axes
        self.plot_multi_gaussian(x,self.params,ax=axes[0],\
                                        label = "Superposition")        
        if add_single_gaussians:
            k = 1
            for g in self.gaussians(): 
                self.plot_gaussian(self.index, g, ax = axes[0],\
                    label = ("%d. gaussian" %k))
                k += 1
                
        axes[0].legend(loc = 'best', fancybox = True, framealpha = 0.5,\
                            fontsize = self.plot_font_sizes["legends"])
        tit = r"Result"
        try:
            mu, sigma, _, _= self.analyse_fit_result()
            tit += r" main peak: $\mu (+/-\sigma$) = %.1f (+/- %.1f)" %(mu, sigma)
        except:
            pass
        
        axes[0].set_title(tit, fontsize =\
                            self.plot_font_sizes["titles"])
        
        res = self.get_residual(self.params)
        axes[1].plot(self.index, res)
        axes[1].set_title("Residual", fontsize =\
                            self.plot_font_sizes["titles"])
        
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
        """String representation of a gaussian
        
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
            "All current gaussians:\n\n")
        for k in range(len(gs)):
            g = gs[k]
            s+= "Gaussian #%d\n%s\n" %(k, self.gauss_str(g))
            
        s += ("Current peak to peak residual: %s\nNoise amplitude: %s" 
        %(self.get_peak_to_peak_residual(self.params), self.noise_amplitude))
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
        """Change the level in the gaussian pyramide where the fit is applied
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
            
    def exclude_pix_range_rect(self, x0,x1,y0,y1):
        """Add a rectangular pixel area which will be excluded from the fit
        :param int x0: start x coordinate (original image resolution)
        :param int x1: stop x coordinate (original image resolution)
        :param int y0: start y coordinate (original image resolution)
        :param int y1: stop y coordinate (original image resolution)
        """
        self.mask[y0:y1,x0:x1]=0
        if self.auto_update:
            self.do_fit()
            
    def exclude_pix_range_circ(self, x0,y0,r):
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
        """Reduce the image size using gaussian pyramide 
        
        :param int steps: steps down in the pyramide
        
        Algorithm used: :func:`cv2.pyrDown` 
        """
        for i in range(steps):
            arr=pyrDown(arr)
        return arr
    
    def pyr_up(self, arr, steps):
        """Increasing the image size using gaussian pyramide 
        
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
    
    @property
    def residual(self):
        """Get and return residual"""
        return self.get_residual()
        
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
    plt.close("all")
    f=MultiGaussFit()
    f.create_test_data_multigauss()
    f.auto_fit()
    axes=f.plot_result()
    f.plot_signal_details()
    
    curv = np.gradient(f.data_grad_smooth)
    
    cond1 = abs(f.data_grad_smooth) < 1
    cond2 = curv < -0.5
    cond = cond1 * cond2
    peaks = f.index[cond]
    print peaks
    
    mean_mu, mean_sigma, max_int, add_gaussians = f.analyse_fit_result(1)