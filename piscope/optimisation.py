# -*- coding: utf-8 -*-
from numpy import abs, linspace, random, asarray, ndarray, where, diff,\
    insert, argmax, average, gradient, arange,argmin, full, inf, sqrt, pi,\
    nan, mod, mgrid, ndim, ones_like,ogrid, finfo, remainder, e
    
from warnings import catch_warnings, simplefilter
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
        multi_gaussian_same_offset
    from .helpers import mesh_from_img
except:
    from piscope.model_functions import supergauss_2d, supergauss_2d_tilt,\
        multi_gaussian_no_offset, gaussian_no_offset, gaussian,\
        multi_gaussian_same_offset
    from piscope.helpers import mesh_from_img

GAUSS_2D_PARAM_INFO = ["amplitude", "mu_x", "mu_y", "sigma", "asymmetry",\
    "exp_super_gauss", "offset", "tilt_theta"]
    
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
    with a noise analysis (if noise level is not provided on class initialisation).
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
    
    """
    def __init__(self, data = None , index = None, noise_amp = None,\
            max_num_gaussians = 20, max_iter = 10, auto_bounds = True,\
                                do_fit = True, horizontal_baseline = True):
        """
        :param array data: data array 
        :param array index (None): x-index array (if None, then one is created)
        :param float noise_amp (None): amplitude of noise in the signal. Defines
            the minimum required amplitude for fitted gaussians (we don't want
            to fit all the noise peaks). It None, then it will be estimated
            automatically on data import using :func:`estimate_noise_amplitude`
        :param int max_num_gaussians (20): max number of superimposed 
            gaussians for data 
        :param int max_iter (10): max number of iterations for optimisation
            routine
        :param bool auto_bounds (True): if True, bounds will be set automatically from
            data ranges whenever data is updated
        :param bool do_fit (True): if True and input data available & ok, then
            :func:`self.auto_fit` will be performed on initialisation
        :param bool horizontal_baseline: data has horizontal baseline (if this 
            is set False, the fit strategy is different... some more info 
            follows)
            
        """
        #data
        self.index = []
        self.data = []
        self.data_smooth = []
        self.data_gradient = []
        self.data_gradient_smooth = []
        
        #pre evaluation results of data
        self.offset = 0.0
        self.noise_mask = None
        self.noise_amplitude = noise_amp
        
        self.max_num_gaussians = max_num_gaussians
        self.max_iter = max_iter
    
        #bounds for gaussians (will be set for each gauss guess before fitting)
        self._horizontal_baseline = horizontal_baseline
        if not horizontal_baseline: #make sure autobounds is off in case of none horizontal baseline
            auto_bounds = False
          
        self.auto_bounds = auto_bounds
        self.gauss_bounds = {"amp"  :   [0, inf],
                             "mu"   :   [-inf, inf],
                             "sigma":   [-inf, inf]}
        
        #Fitting related stuff
        self._fit_result = None #the actual output from the minimisation
        
        self.params = [0, 0, inf] #this is where the fit parameters are stored in
        
        #function to be minimised
        self.err_fun = lambda p, x, y:\
                    (multi_gaussian_no_offset(x, *p) - y)**2
        
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
        self.params = [0, 0, inf]
        self.offset = 0.0
    
    def set_data(self, data, index = None, amp_range = [0, nan],\
                        mu_range = [nan, nan], sigma_range = [nan, nan]):
        """Set x and y data
        
        :param array data: data array which is fitted
        :param array index (None): index array to data 
        :param list amp_range ([0, nan]): range of allowed amplitudes for 
            fitted gaussians
        :param list mu_range ([nan, nan]): range of allowed mean positions of
            fitted gaussians
        :param list sigma_range ([nan, nan]): range of allowed standard 
            deviations for fitted gaussians
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
            self.index = arange(0, len(self.data),1)
                
        self.init_results()
        self.init_data()
            
            
    def init_data(self):
        """Initiate the input data and set constraints for valid gaussians
        
        Main steps
        
            1.  Estimate the data offset from minimum of smoothed data (using
                gaussian filter, width 3 to reduce noise)
            #.  Determine 1st derivative of data (stored in
                                            ``self.data_gradient``)
            #.  Smooth data and gradient (using gaussian filter width 1)
            #.  Call :func:`self.estimate_noise_amplitude`
            #.  Call :func:`self.init_gauss_bounds_auto`
        """
        if self.has_data:
            self.offset = gaussian_filter1d(self.data, 3).min()
            self.data_gradient = self.first_derivative(self.data)
            self.data_smooth = self.apply_binomial_filter(self.data, sigma = 2)
            self.data_gradient_smooth = self.apply_binomial_filter(\
                                                self.data_gradient, sigma = 2)
            if self.noise_amplitude is None:
                self.estimate_noise_amplitude()
            self.init_gauss_bounds_auto()
     
    def set_gauss_bounds(self, amp_range = [0, inf], mu_range = [-inf, inf],\
                                                    sigma_range = [-inf, inf]):
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
            print "Could not init gauss bounds, no data available..."
            return 0
        if not self.auto_bounds:
            print "Automatic update of boundaries is deactivated..."
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
        data = self.data_smooth - self.offset
        ind = argmax(data)
        amp = data[ind] 
        if not amp > 1.5 * self.noise_amplitude:
            raise IndexError("No significant peak could be found in data")
        w = self.estimate_peak_width(data, ind)
        return [amp, self.index[ind], w * self.x_resolution]
        
#==============================================================================
#     def find_peak_positions_data(self):
#         """Perform peak search and write positions of peaks (indices) into 
#         variable ``self._peak_indices``
#         
#         .. note::
#         
#             Performs peak search algorighm using :func:`indexes` from
#             :mod:`peakutils` package.
#         """
#         if not self.has_data:
#             print "No data available"
#             return 0
#         self._peak_indices = []
#         indices = []
#         if self._horizontal_baseline:
#             print ("Automatic peak search...")
#             #ws=linspace(self.sigmaMin/res,self.sigmaMax/res,len(self.index)/res)
#             #indices=find_peaks_cwt(self.data,ws)
#             indices = indexes(self.data_smooth, min_dist = .05 * len(self.data),\
#                 thres = 3 * self.noise_amplitude / self.gauss_bounds["amp"][1])
#         
#         if not len(indices) > 0:
#             indices = [self.find_main_peak_position_data()]
#         print "Detected peaks: %s\n" %indices
#         for ind in indices:
#             if self.data_smooth[ind] - self.offset > 2 * self.noise_amplitude:
#                 self._peak_indices.append(ind)
#             
#         #find_peaks_cwt(gaussian_filter1d(self.data,15),ws)
#         if len(self._peak_indices) > 0:
#             return self._peak_indices
#         else:
#             return 0
#     
#==============================================================================
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
                print "Residual peak search finished..."
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
        raise ValueError("Number of detected peaks exceeds allowed max "
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
        print "Estimating peak width at peak, index: " + str(idx)
        print "x,y:" + str(self.index[idx]) + ", " + str(amp)
        maxInd = len(self.index) - 1  
        try:
            ind = next(val[0] for val in enumerate(dat[idx:maxInd])\
                                                if val[1] < amp/2)
            print "Width (index units): " + str(abs(ind))
            
            return ind
        except:
            print "Trying to the left"#format_exc()
            try:
                inv = dat[::-1]
                idx = len(inv)-1-idx
                ind = next(val[0] for val in enumerate(inv[idx:maxInd])\
                                                if val[1] < amp/2)
                print "Width (index units): " + str(abs(ind))
                return ind
            except:
                pass
        print "Peak width could not be estimated, return 1"
        return 1#int(.05*len(dat))
            
    
    """
    Fitting etc
    """         
#==============================================================================
#     def prepare_fit_from_peak_search(self):
#         """Initiate fit parameter list using results from :func:`find_peak_positions`
#         and the corresponding estimates of peak widths
#         """
#         if not self.find_peak_positions_data():# or not self.check_peaks():
#             print ("No peaks detected...")
#             return []
#         indices = self._peak_indices
#         params = []
#         dat = self.data - self.offset
#         for k in range(len(indices)):
#             ind = indices[k]
#             #estimate the amplitude of this peak
#             params.append(dat[ind])
#             params.append(self.index[ind])
#             params.append(self.estimate_peak_width(dat, ind)*self.x_resolution)
#         
#         return params            
#==============================================================================
    
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
            print "Error: length of gauss param list must be divisable by three.."
            return []
        sub = [params[x : x + 3] for x in range(0, len(params), 3)]
        new_list = []
        for pl in sub:
            if self._in_bounds(pl):
                new_list.extend(pl)
            else:
                print ("Gaussian in param list with params: " + str(pl) + "out "
                    "of acceptance boundaries -> will be disregarded..:")
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
            print "Fitting data..."
            self._fit_result = res = least_squares(self.err_fun, params,\
                                                    args=(x, y), bounds=bds)
            #params,ok=optimize.leastsq(self.err_fun, *guess, args=(x, y))
            if not res.success:
                print "Fit failed"
                return False
            self.params = res.x
            return True
        except Exception as e:
            print "Fit failed with exception: %s" %repr(e)
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
        print "Running multi gauss auto fit routine"
        guess = self.find_peak_positions_residual()
        print "Initial peak search: found %s peaks" %(len(guess)/3.0)
                
        y = self.data - self.offset
        if not self.do_fit(self.index, y, guess):
            return 0
        print "Fit applied (first iter), current settings:"
        print self.info()
        add_params = self.find_peak_positions_residual()
        if add_params:
            print ("Found %s more peaks after first iter, running "
                "optimisation..." %(len(add_params)/3.0))
            if not self.run_optimisation():
                return 0
        print "Auto fit successfully finished"""
        return 1
#==============================================================================
#         if not self.fit_multiple_gaussian():
#             #and if this does not work, try to fit a single gaussian (based
#             #on position of max count)
#             return 0
#         if self.has_results() and not self.result_ok():   
#             self.run_optimisation()    
#==============================================================================

    def run_optimisation(self):
        """Run optimisation"""
        last_params = self.params
        residuals = [self.get_residual()]
        chis = []
        for k in range(self.max_iter):
            if self.num_of_gaussians >= self.max_num_gaussians:
                print ("Max num of gaussians (%d) reached "
                    "abort optimisation" %self.max_num_gaussians)
                self._write_opt_log(chis, residuals)
                return 0
            if not self.optimise_result():
                print ("Optimisation failed,  aborted at iter %d" %k)
                self._write_opt_log(chis, residuals)
                return 0
            
            residuals.append(self.get_residual())
            chis.append(residuals[-1].std() / residuals[-2].std())
            print "Current iter: %s, current chi: %.2f" %(k, chis[-1])
            if chis[-1] > 1:
                print "Optimisation stopped, chi increased..."
                self.params = last_params
                self._write_opt_log(chis, residuals)
                return 1
            last_params = self.params
            if self.get_peak_to_peak_residual() < self.noise_amplitude or\
                                    0.9 < chis[-1] <= 1.0:
                print "Optimisation successfully finished"
                self._write_opt_log(chis, residuals)
                return 1
        print "Optimisation aborted, maximum number of iterations reached..."
        self._write_opt_log(chis, residuals)
        return 0
        
    def _write_opt_log(self, chis, residuals):
        """Write chi values and residuals from optimisation"""
        self.optimise_log["chis"] = chis
        self.optimise_log["residuals"] = residuals
        
    def optimise_result(self, add_params = None):
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
            print ("Optimisation aborted, no additional peaks could be "
                    "detected in current fit residual")
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
    def find_overlaps(self):
        """ Find overlapping gaussians for current optimisation params
        
        Loops over all current gaussians (``self.gaussians``) and for each of
        them, finds all which fall into :math:`3\\sigma` range.
        
        :return:
            - list, containing list with all gaussians for each gaussian
            - list, integrated values for each of the ovelapped results
        """
        info = []
        int_vals = [] # integral values of overlapping gaussians
        for k in range(len(self.gaussians())):
            gs = self._get_all_gaussians_within_3sigma(k)[0]
            info.append(gs)
            int_val = 0  
            for g in gs:
                int_val += self.integrate_gauss(*g)
            int_vals.append(int_val)
            
        return info, int_vals
            
    def analyse_fit_result(self):
        """Analyse result of optimisation
        
        Find main peak (can be overlap of single gaussians) and potential other
        peaks
        
        .. todo::
        
            This method might require some more thinking, there is potentially
            more useful information to be extracted
            
        """
        info, ints = self.find_overlaps()
        ind = argmax(ints)
        gs = info[ind]
        max_int = ints[ind]
        mu0 = self.gaussians()[ind][1]
        sigmas = []
        weights = []
        mus = []
        del_mus = []
        for g in gs:
            del_mus.append(abs(g[1] - mu0))
            mus.append(g[1])
            weights.append(self.integrate_gauss(*g) / max_int)
            sigmas.append(g[2])
        print "Mu array (used for averaging): " + str(mus)
        print "del_mu array: " + str(del_mus)
        print "Sigma array (used for averaging): " + str(sigmas)
        print "Weights (used for averaging): " + str(weights)
        mean_mu = average(asarray(mus), weights = asarray(weights))
        mean_sigma = average(asarray(sigmas), weights = asarray(weights))
        print "Mu, Sigma: " + str(mean_mu) + ", " + str(mean_sigma)
                
        return mean_mu, mean_sigma, mus, sigmas
        
    """
    Helpers
    """
    def gaussians(self):
        """Split self.params (i.e. parameters of all gaussians) into sublists
        containing information of individual gaussians
        """
        return [self.params[i:i + 3] for i in range(0, len(self.params), 3)]
        
    def integrate_gauss(self, amp, mu, sigma, start = -inf, stop = inf):
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
    
    def first_derivative(self, data = None):
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
        
    def estimate_noise_amplitude_old(self, sigma_gauss = 1, medianwidth = 3,\
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
            print "Current iteration: " + str(k)
            if dm[b].std() / d[b].std() < thresh: 
                print "Significant difference found, optimising..."
                ind1 = argmax(d[b])
                ind2 = argmin(d[b])       
                if abs(d[b][ind1]) > abs(d[b][ind2]):
                    ind = ind1
                else: 
                    ind = ind2
                print "Bad index at: " + str(ind)
                print "Cut out index window: " + str(ind-w) + " - " + str(ind+w)
                b[ind-w:ind+w]=False
                print "Current number of considered indices: " + str(sum(b))
            else:
                ok = True
                print "Optimisation finished"
                break
        
        if float(sum(b)) / len(d) < 0.2:
            print "Too little remaining data points, using conservative approach"
            ok = False
        if ok:
            self.noise_mask = b
            self.noise_amplitude = 6*d[b].std()#d[b].max()-d[b].min()
        else:
            print "Optimisation failed, use std of high pass filtered signal"
            self.noise_mask = full(len(self.data), True, dtype=bool)
            self.noise_amplitude = 6*d.std()
        print "Estimated noise amplitude: %s" %self.noise_amplitude
        return self.noise_amplitude
    
    def max(self):
        """Return max value and x position of current parameters (not of data)"""
        if self.has_results():
            vals = self.multi_gaussian_no_offset(self.index, *self.params) +\
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
        print "Found sub intervals: " + str(ind)
        return ind
        
    def get_residual(self, params = None, mask = None):
        """Get the current residual
        
        :param list params: Multi gauss parameters, if None, use 
                                                        ``self.params``
        :param logical mask: use only certain indices
        
        """
        if not self.has_results():
            print "No fit results available"
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
        print "Mu: %s, sigma: %s" %(mu, sigma)
        print "Cutting out range (left, right): %s - %s" %(l, r)
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
            print "Amplitude out of range, value: " + str(params[0]) 
            return 0
        if not bds["mu"][0] <= params[1] <= bds["mu"][1]:
            print "Mu out of range, value: " + str(params[1]) 
            return 0
        if not bds["sigma"][0] <= params[2] <= bds["sigma"][1]:
            print "Sigma out of range, value: " + str(params[2]) 
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

    def _get_all_gaussians_within_3sigma(self, index):
        """Find all current gaussians within 3 sigma of a specified gaussian
        
        :param int index: list index of considered gaussian
        
        """
        gs = self.gaussians()
        g = gs[index]
        l, r = g[1] - 3 * g[2], g[1] + 3 * g[2]
        gaussians = [g]
        rl = l
        rr = r
        for k in range(len(gs)):
            if not k == index:
                mu = gs[k][1]
                sig = gs[k][2]
                l1, r1 = mu - 3 * sig, mu + 3 * sig
                if l < mu < r:# or l1 < g[1] < r1:
                    gaussians.append(gs[k])
                    if l1 < rl:
                        rl = l1
                    if r1 > rr:
                        rr = r1
        return gaussians, rl, rr
    
    """
    Plotting / Visualisation etc..
    """
    def plot_signal_details(self):
        """Plot signal and derivatives both in original and smoothed version
        """
        if not self.has_data:
            print "No data available..."
            return 0 
        fig, ax = subplots(2,1)
        ax[0].plot(self.index, self.data, "--g", label="Signal " + self.id)
        ax[0].plot(self.index, self.apply_binomial_filter(sigma=3), "-r",\
                                                label="Smoothed (width 3)")
        ax[0].legend(loc='best', fancybox=True, framealpha=0.5,\
                            fontsize = self.plot_font_sizes["legend"])
        ax[0].set_title("Signal", fontsize = self.plot_font_sizes["titles"])
        ax[0].grid()
        ax[1].plot(self.index, self.first_derivative(), "--g", label="Grad signal")
        ax[1].plot(self.index, self.apply_binomial_filter(self.first_derivative(),\
                                    sigma=3), "-r", label="Smoothed (width 3)")
        ax[1].legend(loc='best', fancybox=True, framealpha=0.5,\
                                fontsize=self.plot_font_sizes.legend)
        ax[1].set_title("Derivative")
        ax[1].grid()
        
    def plot_data(self, ax = None, sub_min = False):
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

    def plot_multi_gaussian(self,x = None, params = None ,ax = None, **kwargs):
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
        ax.plot(x, model, lw = 3, c = 'r', ls = '-', **kwargs) 
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
    
    def plot_result(self, add_single_gaussians = False):
        """Plot the current fit result
        
        :param bool add_single_gaussians: if True, all individual gaussians are 
            plotted
            
        """
        if not self.has_data:
            print "Could not plot result, no data available.."
            return 0
        fig,axes = subplots(2,1)
        self.plot_data(sub_min = 0, ax = axes[0])
        x = linspace(self.index.min(), self.index.max(), len(self.index) * 3)
        if not self.has_results():
            print "Only plotted data, no results available"
            return axes
        self.plot_multi_gaussian(x,self.params,ax=axes[0],\
                                        label = "Superposition")        
        if add_single_gaussians:
            k = 1
            for g in self.gaussians(): 
                self.plot_gaussian(self.index, g, ax = axes[0],\
                    label = ("%d. gasussian" %k))
                k += 1
                
        axes[0].legend(loc = 'best', fancybox = True, framealpha = 0.5,\
                            fontsize = self.plot_font_sizes["legends"])
        axes[0].set_title("Fit result", fontsize =\
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
        return "Amplitude: %.2f\nMu: %.2f\n Sigma: %.2f\n" %(g[0], g[1], g[2])
            
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
        if self._fit_result is not None and len(self.params) >= 3:
            return 1
        print "No multi gauss fit results available"
        return 0    
    """
    Magic methods
    """
    def __str__(self):
        """String representation"""
        gs = self.gaussians()
        s=("piscope MultiGaussFit info\n--------------------------------\n\n" 
            "All current gaussians:\n\n")
        for k in range(len(gs)):
            g = gs[k]
            s+= "Gaussian #%d\n%s\n" %(k, self.gauss_str(g))
            
        s += ("Current peak to peak residual: %s\nNoise amplitude: %s" 
        %(self.get_peak_to_peak_residual(self.params), self.noise_amplitude))
        return s

class PolySurfaceFit(object):
    """Fit a 2D polynomial to data (e.g. a blue sky background image)"""    
    def __init__(self, im, mask=None, polyorder=3, pyrlevel = 4, do_fit=1):
        """Fit a 2D polynomial to 2D array"""
        self.data = None
        self.mask = None
        
        self.pyrlevel = pyrlevel
        self.polyorder = polyorder
        self.err_fun = models.Polynomial2D(degree = self.polyorder)
        self.fitter = LevMarLSQFitter()
        self.params = None
        self.model = None
        self.auto_update = 1
        if self.set_data(im, mask) and do_fit:
            self.do_fit()
    
    def set_data(self, arr, mask = None):
        """Set the data array (must be dimension 2)
        
        Create ``self.mask`` for array shape which can be used to exclude 
        picxel areas from the image
        
        :param array arr: data array
        :param array mask: boolean mask (must have same shape than arr) 
        """
        if not ndim(arr) == 2:
            print "Could not set data, dimesion mismatch..."
            return 0
        if mask is None or mask.shape != arr.shape:
            mask = ones_like(arr)
        self.data = arr
        self.mask = mask
        self.params = None #storage of fit results
        self.model = None
        return 1
    
    def activate_auto_update(self, Bool=1):
        """Activate or deactivate auto update mode. If active, the fit is 
        reapplied each time some input parameter is changed
        """
        self.auto_update=Bool
    
    def change_pyrlevel(self, newlevel):
        """Change the level in the gaussian pyramide where the fit is applied
        """
        self.pyrlevel = newlevel
        if self.auto_update:
            self.do_fit()
            
    def change_polyorder(self, order):
        """Change the order of the polynomial which is fitted
        """
        self.polyorder=order
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
        print "Fitting 2D polynomial to data...(this might take a moment)"
        
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
    plt.close("all")
    f=MultiGaussFit()
    f.create_test_data_multigauss()
    f.auto_fit()
    f.plot_result()
    print f