# -*- coding: utf-8 -*-
from numpy import abs, linspace, exp, random, asarray, ndarray,\
    where, diff, insert, argmax, average, gradient, arange,argmin,\
    full, inf, sqrt, pi, nan, mod, mgrid, ndim, ones_like,ogrid, sin, cos,\
    meshgrid, nanargmax, unravel_index
    
from warnings import catch_warnings, simplefilter
from matplotlib.pyplot import subplots

from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter

from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.integrate import quad
from scipy.optimize import curve_fit, least_squares

from cv2 import pyrUp, pyrDown
from copy import deepcopy
#from scipy.signal import find_peaks_cwt
#from peakutils import indexes
from traceback import format_exc

def gauss2D((x, y), amplitude, xm, ym, xsigma, ysigma, offset):
    """2D gaussian
    
    :param tuple (x,y): evaluation grid    
    :param float amplitude: amplitude of 2D gaussian
    :param float xm: x peak position
    :param float ym: y peak position
    :param float xsigma: x standard deviation
    :param float ysigma: y standard deviation
    :param float offset: baseline offset
    :return: evaluated valuesat input grid points 
    """
    
    U = ((x-xm)/xsigma)**2 + ((y-ym)/ysigma)**2
    g = offset + amplitude * exp(-U/2)
    return g.ravel()

def gauss2D_tilt((x, y), amplitude, xm, ym, xsigma, ysigma, offset, theta):
    """ Tilted 2D gaussian
    
    :param tuple (x,y): evaluation grid    
    :param float amplitude: amplitude of 2D gaussian
    :param float xm: x peak position
    :param float ym: y peak position
    :param float xsigma: x standard deviation
    :param float ysigma: y standard deviation
    :param float offset: baseline offset
    :param float theta: tilting angle
    :return: evaluated values at input grid points 
    """
    xprime = (x-xm)*cos(theta) - (y-ym)*sin(theta)
    yprime = (y-ym)*sin(theta) + (y-ym)*cos(theta)
    U = (xprime/xsigma)**2 + (yprime/ysigma)**2
    g = offset + amplitude * exp(-U/2)
    return g.ravel()

def hypergauss2D((x, y), amplitude, xm, ym, xsigma, ysigma, offset):
    """2D hyper gaussian
    
    :param tuple (x,y): evaluation grid    
    :param float amplitude: amplitude of 2D gaussian
    :param float xm: x peak position
    :param float ym: y peak position
    :param float xsigma: x standard deviation
    :param float ysigma: y standard deviation
    :param float offset: baseline offset
    :return: evaluated values at input grid points 
    """
    U = ((x-xm)/xsigma)**2 + ((y-ym)/ysigma)**2
    g = offset + amplitude * exp(-U**4/2)
    return g.ravel()

def hypergauss2D_tilt((x, y), amplitude, xm, ym, xsigma, ysigma, offset, theta):
    """ Tilted 2D hyper gaussian
    
    :param tuple (x,y): evaluation grid    
    :param float amplitude: amplitude of 2D gaussian
    :param float xm: x peak position
    :param float ym: y peak position
    :param float xsigma: x standard deviation
    :param float ysigma: y standard deviation
    :param float offset: baseline offset
    :param float theta: tilting angle
    :return: evaluated values at input grid points 
    """
    xprime = (x-xm)*cos(theta) - (y-ym)*sin(theta)
    yprime = (y-ym)*sin(theta) + (y-ym)*cos(theta)
    U = (xprime/xsigma)**2 + (yprime/ysigma)**2
    g = offset + amplitude * exp(-U**4/2)
    return g.ravel()

def GaussFit2D(dataGrid2d, hyper = True, tilt = False, crop = True):
    """Fits 2D gaussian onto input 2D data grid 
    
    :param ndarray dataGrid2d: input data
        
    """
    # setup grid
    (Ny, Nx) = dataGrid2d.shape
    xvec = linspace(0, Nx, Nx)
    yvec = linspace(0, Ny, Ny)
    xgrid, ygrid = meshgrid(xvec, yvec)
    # apply max of filtered image to initialise 2D gaussian fit
    sigma = 20
    (x0, y0) = unravel_index(nanargmax(gaussian_filter(dataGrid2d, sigma)), dataGrid2d.shape)
    if tilt:
        initial_guess = (1, x0, y0, 10, 10, 0, 0)
        if hyper:
            popt, pcov = curve_fit(hypergauss2D_tilt, (xgrid, ygrid), dataGrid2d.ravel(), p0=initial_guess)
            data_fitted = hypergauss2D_tilt((xgrid, ygrid), *popt)
        else:
            popt, pcov = curve_fit(gauss2D_tilt, (xgrid, ygrid), dataGrid2d.ravel(), p0=initial_guess)
            data_fitted = gauss2D_tilt((xgrid, ygrid), *popt)
    else:
        initial_guess = (1, x0, y0, 10, 10, 0)
        if hyper:
            popt, pcov = curve_fit(hypergauss2D, (xgrid, ygrid), dataGrid2d.ravel(), p0=initial_guess)
            data_fitted = hypergauss2D((xgrid, ygrid), *popt)
        else:
            popt, pcov = curve_fit(gauss2D, (xgrid, ygrid), dataGrid2d.ravel(), p0=initial_guess)
            data_fitted = gauss2D((xgrid, ygrid), *popt)
    # eventually crop FOV distribution (makes it more robust against outliers (eg. mountan ridge))
    if crop:
        # set outside (one 100th of amplitude) datapoints = 0
        data_fitted[data_fitted<popt[0]/100] = 0
    # reshape data_fitted as matrix instead of vector required for fitting
    data_fitted = data_fitted.reshape(Ny, Nx)
    # normalise fit result
    normsum = sum(data_fitted)
    data_fitted_norm = data_fitted/normsum
    popt[0] = popt[0]/normsum
    return [data_fitted_norm, popt] 

    
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
        self.err_fun = lambda p, x, y: (self.multi_gaussian_no_offset(x, *p) - y)**2
        
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
            #.  Determine 1st derivative of data (stored in ``self.data_gradient``)
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
        """Set fit bounds for gaussians
        """
        if not self.has_data:
            print "Could not init gauss bounds, no data available..."
            return 0
        if not self.auto_bounds:
            print "Automatic update of boundaries is deactivated..."
            return 1
        self.gauss_bounds["amp"][0] = 2*self.noise_amplitude
        self.gauss_bounds["amp"][1] = (self.y_range - self.offset) * 1.5
        self.gauss_bounds["mu"][0] = self.index[0]
        self.gauss_bounds["mu"][1] = self.index[-1]
        self.gauss_bounds["sigma"][0] = self.x_resolution/2.
        self.gauss_bounds["sigma"][1] = self.x_range/2.
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
                dat[ind - 3*w : ind + 3*w] = 0
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
        sub = [params[x:x+3] for x in range(0, len(params), 3)]
        newList = []
        #wrongGuess=[310,160,36]
        #bds = ([303,153,35], [320,180,38])
        for pl in sub:
            if self._in_bounds(pl):
                newList.extend(pl)
            else:
                print ("Gaussian in paramList with params: " + str(pl) + "out "
                    "of acceptance boundaries -> will be disregarded..:")
        lower, upper=[], []
        l,u=self._prep_bounds_single_gauss()
        for k in range(len(newList)/3):
            lower.extend(l)
            upper.extend(u)
        bds=(lower,upper)
        return newList,bds
        
    def do_fit(self, x = None, y = None, *guess):
        """Perform a least squares minimisation
        
        :param err_fun: Function (x,y,*guess) to be minimised
        :param array x: x argument for input function
        :param array y: y argument for input function
        :param list guess: initial guess of fit parameters
        """
        try:
            params, bds = self.prepare_fit_boundaries(*guess)
            print "Fitting data..."
            self._fit_result = res = least_squares(self.err_fun, params,\
                                                    args=(x, y), bounds=bds)
            #pL,ok=optimize.leastsq(self.err_fun, *guess, args=(x, y))
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
        guess = self.find_peak_positions_residual()
                
        y = self.data - self.offset
        if not self.do_fit(self.index, y, guess):
            return 0
        #self.write_fit_results()
        print self.gauss_info()
        add_params = self.find_peak_positions_residual()
        print "HEEERRRE"
        if add_params:
            print "Running optimisation..."
            if not self.run_optimisation():
                return 0
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
            print "\nITER OPTIMISATION\nCurrent k: " + str(k)
            if self.num_of_gaussians >= self.max_num_gaussians:
                print ("Maximum number of gaussians reached: abort optimisation")
                return 0
            if not self.optimise_result():
                print ("Optimisation aborted at iteration " + str(k))
                return 0
            
            residuals.append(self.get_residual())
            chis.append(residuals[-1].std() / residuals[-2].std())
            print "Last residual std: " + str(residuals[-2].std())
            print "This residual std: " + str(residuals[-1].std())
            print "Change factor residual std (Chi): " + str(chis[-1])
            if chis[-1] > 1:
                print "Optimisation stopped, residual increased... "
                self.params = last_params
                print chis
                return 1
            last_params = self.params
            if self.get_peak_to_peak_residual() < self.noise_amplitude or\
                                    0.9 < chis[-1] <= 1.0:
                print "Optimisation finished"
                return 1
        print "Optimisation aborted, maximum number of iterations reached..."
        return 0
        
    def optimise_result(self, addParams = None):
        #for i in range(maxIter):
        #lastRes=self.get_residual(self.params)
        #badIndices,residuals,addGaussians,xOffsets=self.check_result()
        print "\nOPTIMISATION\n---------------------------\n\n"
        if addParams is None:
            addParams = self.find_peak_positions_residual()
        if not len(addParams) > 0:
            print "No optimisation necessary..."
            return 1
        guess = list(self.params)
        guess.extend(addParams)
        y = self.data - self.offset
        print "Fit setup:"
        print guess
        print
        if not self.do_fit(self.index, y, guess):
            return 0
        return 1
    """
    Fit model functions
    """
    def gaussian_no_offset(self, x, ampl, mu, sigma):
        """1D gauss with baseline zero
        
        :param float x: x position of evaluation
        :param float ampl: Amplitude of gaussian
        :param float mu: center poistion
        :param float sigma: standard deviation
        :returns float: value at position x
        """
        #return float(ampl)*exp(-(x - float(mu))**2/(2*float(sigma)**2))
        return ampl * exp(-(x - mu)**2 / (2 * sigma**2))
        
    def gaussian(self, x, ampl, mu, sigma, offset):
        """1D gauss with arbitrary baseline
        
        :param float x: x position of evaluation
        :param float ampl: Amplitude of gaussian
        :param float mu: center poistion
        :param float sigma: standard deviation
        :param float offset: baseline of gaussian
        :returns float: value at position x
        """
        return self.gaussian_no_offset(x,ampl,mu,sigma) + offset
        
    def multi_gaussian_no_offset(self, x, *params):
        """
        Evaluate superposition of arbitrary amount of gaussians with baseline
        zero.
        
        :param array x: x array used for evaluation
        :param list *params: List of length L = 3xN were N corresponds to the 
            number of gaussians e.g.::
            
                [100,10,3,50,15,6]
                
            would correspond to 2 gaussians with the following characteristics:
            
                1. Peak amplitude: 100, Mu: 10, sigma: 3
                2. Peak amplitude: 50, Mu: 15, sigma: 6
        """
        res = 0
        num=len(params)/3
        for k in range(num):
            p=params[k*3:(k+1)*3]
            res=res+self.gaussian_no_offset(x,*p)
        return res #+ params[3*num]
    
    def multi_gaussian_same_offset(self,x, offset, *params):
        """
        Evaluate superposition of arbitrary amount of gaussians with same but
        arbitrary baseline.
        
        See :func:`multi_gaussian_no_offset` for instructions
        """
        return self.multi_gaussian_no_offset(x, *params) + offset
    
    """
    Fit analysis, results, post processing, etc..
    """
    def find_overlaps(self):
        """ For each gaussian in ``self.gaussians`` find all gaussians
        overlapping and write them in dictionary (``self.gaussiansOverlapIndices``) 
        where key is the respective index of the current gaussian and the values
        is a list containing all gaussians (param lists) within range of this 
        gaussian
        """
        info = []
        intVals = [] # integral values of overlapping gaussians
        for k in range(len(self.gaussians())):
            gs=self._get_all_gaussians_within_3sigma(k)[0]
            info.append(gs)
            intVal = 0
            for g in gs:
                intVal += self.integrate_gauss(*g)
            intVals.append(intVal)
            
        return info, intVals
            
    def analyse_fit_result(self):
        """This function analyses the fit result (individual gaussians) in order
        to find the main peak (using integral values of overlapping gaussians as
        measure of significance) and determines 
        """
        info, ints = self.find_overlaps()
        ind = argmax(ints)
        gs = info[ind]
        maxInt = ints[ind]
        mu0 = self.gaussians()[ind][1]
        sigmas = []
        weights = []
        mus = []
        delMus = []
        for g in gs:
            delMus.append(abs(g[1]-mu0))
            mus.append(g[1])
            weights.append(self.integrate_gauss(*g)/maxInt)
            sigmas.append(g[2])
        print "Mu array (used for averaging): " + str(mus)
        print "delMu array: " + str(delMus)
        print "Sigma array (used for averaging): " + str(sigmas)
        print "Weights (used for averaging): " + str(weights)
        meanMu=average(asarray(mus), weights=asarray(weights))
        meanSigma=average(asarray(sigmas), weights=asarray(weights))
        print "Mu, Sigma: " + str(meanMu) + ", " + str(meanSigma)
                
        return meanMu, meanSigma, mus, sigmas
        
    """
    Helpers
    """
    def gaussians(self):
        """Split self.params (i.e. parameters of all gaussians) into sublists
        containing information of individual gaussians
        """
        return [self.params[i:i + 3] for i in range(0, len(self.params), 3)]
        
    def integrate_gauss(self, amp, mu, sigma, start = -inf,stop = inf):
        """Return integral value of one gaussian
        :param float amp: amplitude of gaussian
        :param float mu: mean of gaussian
        :param float sigma: standard deviation
        :param float start (-inf): left integral border
        :param float stop (inf): right integral border
        """
        if start == -inf and stop == inf:
            return amp * sigma * sqrt(2 * pi)
        g=[amp,mu,sigma]
        return quad(lambda x: self.gaussian_no_offset(x, *g), start, stop)[0]
    
    def integrate_all_gaussians(self):
        """Determines the integral values of all gaussians in ``self.gaussians`` 
        :returns list: integral values for each gaussian
        """
        vals=[]
        for g in self.gaussians():
            vals.append(self.integrate_gauss(*g))
        return vals
    
    def create_test_data_singlegauss(self, addNoise=1):
        """Make a test data set with a single gaussian
        
        :param bool addNoise: add noise to test data
        
        """
        self.id="test_data"
        x = linspace(0, 400, 401)
        pL = [300, 150, 20]
        y = self.multi_gaussian_same_offset(x, 15, *pL)
        if addNoise:
            y=y+5*random.normal(0, 1, size=len(x))
        self.set_data(y, x)
        
    def create_test_data_multigauss(self, addNoise=1):
        """Make a test data set
        
        :param bool addNoise: add noise to test data
    
        """
        self.id="test_data"
        x=linspace(0,400,401)
        pL=[150,30,8,200,110,3,300,150,20,75,370,40,300,250,1]
        y=self.multi_gaussian_same_offset(x,45,*pL)
        if addNoise:
            y=y + 5*random.normal(0, 1, size = len(x))
        self.set_data(y, x)
        
    def create_noise_dataset(self):
        """Make pure noise and set as current data
        """
        self.id="noise"
        x=linspace(0,400,401)
        y=5*random.normal(0, 1, size=len(x))
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
    
    def first_derivative(self,data=None):
        """Apply discrete gradient to data
        """
        if data is None:
            data=self.data
        return gradient(data)        
        
    def set_noise_amplitude(self, ampl):
        """Set an amplitude threshold for the peak search
        """
        self.noise_amplitude=ampl
    
#==============================================================================
#     def estimate_noise_amplitude(self):
#==============================================================================
    def estimate_noise_amplitude(self, sigmaGauss = 1, stdFac = 2, cutOutFac = 20):
        b = full(len(self.data), True, dtype=bool)
        w = int(self.x_resolution*cutOutFac)
        #Smooth by width 3 and pick every 3rd entry of difference signal
        d = self.data - self.apply_binomial_filter(sigma = sigmaGauss)
        idxs = where(abs(d) > stdFac*d.std())[0]
        for idx in idxs:
             b[idx - w: idx + w] = False
        self.noise_amplitude = 3*d[b].std()
        return d, b, idxs
        
    def estimate_noise_amplitude_old(self, sigmaGauss = 1, medianwidth = 3,\
                                cutOutFac = 20, thresh = 0.75, maxIter = 10):
        """Estimate the amplitude of the noise in the data
        
        :param int sigmaGauss (3): width of gaussian smoothing kernel for high
            pass filter
        :param int medianwidth (3): width of median filter applied to low pass
            signal (for removal of narrow peaks)
        :param int cutOutFac (20): factor defining width of cutted out region
            around bad areas at each iteration
        :param float thresh (0.75): percentage threshold for std comparison of 
            target signals to fulfill "ok" condition 
        :param int maxIter (10): max number of iterations for optimisation
        
        Steps:

            1. Determine bool array ``B`` of ones (all data is considered initially)
            #. Determine high pass filtered data array (hp) by subtracting original
                data from smoothed data using gaussian kernel of input width
            #. Apply median filter to hp to reduce amplitudes of 
                remaining narrow and sharp peaks (e.g. salt & pepper peaks), 
                resulting in signal "hp_median"
            #. Run optimisation until maxIter is reached or "ok" condition is
                fulfilled. The ok condition is
                ``hp_median[B].std() / hp[B].std > thresh``
                and ``B`` is reduced by sharp peak areas at each iteration by 
                analysing target signal ``hp[B]`` for position of max 
                amplitude and cutting window around this region where the width 
                of this window is specified by ``self.x_resolution*cutOutFac``
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
        w = self.x_resolution*cutOutFac
        #Smooth by width 3 and pick every 3rd entry of difference signal
        d = self.data - self.apply_binomial_filter(sigma = sigmaGauss)
        dm = median_filter(d,medianwidth)
        #diff = dm - d
        ok = False
        for k in range(maxIter):
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
        """
        Return max value and x position
        """
        if self.has_results():
            vals = self.multi_gaussian_no_offset(self.index, *self.params) +\
                                                                    self.offset
            return max(vals), self.index[argmax(vals)]
        return [None, None]
    
    @property   
    def num_of_gaussians(self):
        """Get the current number of gaussians, which is the length 
        
        :returns float: ``len(self.params)/3`` 
        """
        return len(self.params) / 3
        
    @property
    def max_amp(self):
        """Get the max amplitude of the current fit results"""
        if self.has_results():
            return

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
        """Returns resolution of x data array
        """
        return self.x_range/(len(self.index)-1)
    
    def get_sub_intervals_bool_array(self, boolArr):
        """
        Get all subintervals of the input bool array
        
        .. note:: 
        
            Currently not in use, but might be helpful at any later stage
            
        """
        ind=where(diff(boolArr) == True)[0]
        if boolArr[0] == True:
            ind=insert(ind, 0, 0)
        if boolArr[-1] == True:
            ind=insert(ind, len(ind), len(boolArr) - 1)
        print "Found sub intervals: " + str(ind)
        return ind
        
    def get_residual(self, p = None, range_condition = None):
        """Get the current residual
        
        :param list p (None): Multi gauss parameters, if None, use ``self.params``
        :param logical rangeCondition (None): use only a certain range
        
        """
        if not self.has_results():
            print "No fit results available"
            return self.data - self.offset 
        if p is None:
            p = self.params
        x, y = self.index,self.data
        if range_condition is not None:
            x=x[range_condition]
            y=y[range_condition]
        dat = y - self.offset
        return dat - self.multi_gaussian_no_offset(x, *p)
        
    def get_peak_to_peak_residual(self, params=None):
        """Return peak to peak difference of fit residual
        """
        if params is None:
            params = self.params
        res=self.get_residual(params)
        return res.max()-res.min()
    
    def cut_sigma_range(self,x,y,params,nSigma=4):
        """
        Cut out a N x sigma environment around gaussian from data
        
        :param array x: x-data array
        :param array y: y-data array
        :param list params: gaussian fit parameters [ampl, mu, sigma, **kwargs]
        :param int nSigma: N (e.g. 3 => 3* sigma environment will be cutted)
        """
        data=[]
        mu, sigma=params[1],params[2]
        l,r=mu-nSigma*sigma,mu+nSigma*sigma
        x1,y1=x[x<l],y[x<l]
        x2,y2=x[x>r],y[x>r]
        if len(x1)>0:
            data.append([x1,y1])
        if len(x2)>0:
            data.append([x2,y2])
        print "Mu, sigma: " + str(mu) + ", " + str(sigma)
        print "Cutting out range: " + str(l) + "-" + str(r)
        return data
    
    def _prep_bounds_single_gauss(self):
        """Prepare two bounds arrays (lower, higher) from ``self.gauss_bounds``
        """
        bds = self.gauss_bounds
        low = [bds["amp"][0], bds["mu"][0], bds["sigma"][0]]
        high = [bds["amp"][1], bds["mu"][1], bds["sigma"][1]]
        return (low, high)
    
    def _in_bounds(self, gaussParams):
        """Checks if parameters (Amp, mu, sigma) of a single gaussian are within
        the acceptance bounds specified in ``self.gauss_bounds``
        """
        bds = self.gauss_bounds
        if not bds["amp"][0] <= gaussParams[0] <= bds["amp"][1]:
            print "Amplitude out of range, value: " + str(gaussParams[0]) 
            return 0
        if not bds["mu"][0] <= gaussParams[1] <= bds["mu"][1]:
            print "Mu out of range, value: " + str(gaussParams[1]) 
            return 0
        if not bds["sigma"][0] <= gaussParams[2] <= bds["sigma"][1]:
            print "Sigma out of range, value: " + str(gaussParams[2]) 
            return 0
        return 1
        
    def _value_range_single_gauss(self, x, p):
        """Return max amplitude of min/max of gaussian in x array
        """
        params = list(p)
        if len(params) == 3:
            params.append(0)
        vals = self.gaussian(x, *params)
        return abs(vals.max() - vals.min())     

    def _get_all_gaussians_within_3sigma(self, index):
        """Analyse ``self.params`` for all gaussians with mu within 3 sigma 
        range of one of the fitted gaussians specified by input number.
        
        :param int index: list index of considered gaussian
        """
        gs = self.gaussians()
        g = gs[index]
        l, r = g[1] - 3 * g[2], g[1] + 3 * g[2]
        gaussians = [g]
        rL = l
        rR = r
        for k in range(len(gs)):
            if not k == index:
                mu = gs[k][1]
                sig = gs[k][2]
                l1, r1 = mu - 3 * sig, mu + 3 * sig
                if l < mu < r:# or l1 < g[1] < r1:
                    gaussians.append(gs[k])
                    if l1 < rL:
                        rL = l1
                    if r1 > rR:
                        rR = r1
        return gaussians, rL, rR
    
    """
    Plotting / Visualisation etc..
    """
    def plot_signal_details(self):
        """Plot signal and derivatives both in original and smoothed version
        """
        if not self.has_data:
            print "No data available..."
            return 0 
        fig, ax=subplots(2,1)
        ax[0].plot(self.index, self.data, "--g", label="Signal " + self.id)
        ax[0].plot(self.index, self.apply_binomial_filter(sigma=3), "-r",\
                                                    label="Smoothed (width 3)")
        ax[0].legend(loc='best', fancybox=True, framealpha=0.5,\
                                        fontsize=self.plot_font_sizes["legend"])
        ax[0].set_title("Signal", fontsize=self.plot_font_sizes["titles"])
        ax[0].grid()
        ax[1].plot(self.index, self.first_derivative(), "--g", label="Grad signal")
        ax[1].plot(self.index, self.apply_binomial_filter(self.first_derivative(),\
                                    sigma=3), "-r", label="Smoothed (width 3)")
        ax[1].legend(loc='best', fancybox=True, framealpha=0.5,\
                                            fontsize=self.plot_font_sizes.legend)
        ax[1].set_title("Derivative")
        ax[1].grid()
        
    def plot_data(self, ax=None, subMin=False):
        """Plot the input data
        
        :param ax (None): matplotlib axes object
        :param bool subMin (False): if true, ``self.offset`` will be subtracted
            from data
            
        """
        if not self.has_data:
            print "No data available..."
            return 0 
        if ax is None:
            fig,ax=subplots(1,1)
        y=self.data
        lStr="Data"
        if subMin:
            y=self.data-self.offset
            lStr = lStr + " (submin)"
            
        ax.plot(self.index, y," x", lw=2, c='g', label=lStr)
        return ax

    def plot_multi_gaussian(self,x=None, params=None ,ax=None, **kwargs):
        """Plot multi gauss
        
        :param array x (None): x data array 
        :param list params (None): multi gauss parameter list (if ``None``, 
            use ``self.index``)
        :param axes ax (None): matplotlib axes object
        
        """
        if ax is None:
            fig,ax=subplots(1,1)
        if x is None:
            x=self.index
        if params is None:
            params=self.params
        model=self.multi_gaussian_no_offset(x,*params)+self.offset
        ax.plot(x, model,lw=3, c='r', ls='-', **kwargs) 
        return ax
                     
    def plot_gaussian(self,x, params, ax=None, **kwargs):
        """Plot gaussian
        
        :param array x: x data array 
        :param list params: single gauss parameter list ``[amp,mu,sigma]``
        :param axes ax (None): matplotlib axes object
        
        """
        if ax is None:
            fig,ax=subplots(1,1)
        params=list(params)            
        if len(params)==3:
            params.append(0)
        dat=self.gaussian(x, *params)+self.offset
        ax.plot(x, dat, lw=1, ls="--", marker=" ", **kwargs)
        return ax
    
    def plot_result(self, addSingleGaussians=False):
        """Plot the current fit result
        
        :param bool addSingleGaussians: if True, all individual gaussians are 
            plotted
            
        """
        if not self.has_data:
            print "Could not plot result, no data available.."
            return 0
        fig,axes=subplots(2,1)
        self.plot_data(subMin=0, ax=axes[0])
        x=linspace(self.index.min(), self.index.max(),200)
        self.plot_multi_gaussian(x,self.params,ax=axes[0], label="Superposition")        
        if addSingleGaussians:
            k=1
            for g in self.gaussians(): 
                self.plot_gaussian(self.index, g, ax=axes[0], label=(str(k) +\
                                                                ". Gauss fit"))
                k+=1
                
        axes[0].legend(loc = 'best', fancybox = True, framealpha  =0.5,\
                            fontsize = self.plot_font_sizes["legends"])
        axes[0].set_title("Fit result", fontsize = self.plot_font_sizes["titles"])
        res = self.get_residual(self.params)
        axes[1].plot(self.index, res)
        axes[1].set_title("Residual", fontsize = self.plot_font_sizes["titles"])
        
        return axes
    
    """
    I/O stuff, prints etc...
    """    
    def print_gauss(self, ind):
        """print info about gauss
        """
        g=self.gaussians()[ind]
        s=("Amplitude: " + str(g[0]) + "\nMu: " + str(g[1]) + "\nSigma: "
                + str(g[2]) + "\n")
        print s
    
    def gauss_str(self,g):
        """String representation of a gaussian
        
        :param list g: parameter list::
        
            [Amplitude, Mu, Sigma]
                        
        """
        return ("Amplitude: " + str(g[0]) + "\nMu: " + str(g[1]) + "\nSigma: "
                + str(g[2]) + "\n")
    
    def print_input_setup(self, params):
        num = len(params) / 3
        for k in range(num):     
            print self.gauss_str(params[k*3 : (k + 1)*3])
            print
            
    def gauss_info(self):
        """Print string representation"""
        print self.__str__()
    
    @property
    def has_data(self):
        """Returns True, if data available, else False"""
        if len(self.data) > 0:
            return True
        return False
    
    def has_results(self):
        """Check if fit results are available (i.e. if list ``self.gaussians``
        contains objects)
        """
        if self._fit_result is not None:
            return 1
        print "No multi gauss fit results available"
        return 0    
    """
    Magic methods
    """
    def __str__(self):
        """String representation
        """
        gs=self.gaussians()
        s=("MULTIGAUSSFIT INFO:\n--------------------------\n\n" + 
                                                "All current gaussians:\n\n")
        for k in range(len(gs)):
            g=gs[k]
            s=s+"Gaussian #" +str(k) + "\n"
            s=(s+"Amplitude: " + str(g[0]) + "\nMu: " + str(g[1]) + "\nSigma: "
                + str(g[2]) + "\n\n")
        s=s+("Current peak to peak residual: "
            + str(self.get_peak_to_peak_residual(self.params)) + "\nNoise amplitude: " 
            + str(self.noise_amplitude) + "\n\n")
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