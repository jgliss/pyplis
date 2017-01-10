# -*- coding: utf-8 -*-
from numpy import float, empty, unravel_index,min,arange,ogrid,asarray,\
         nanargmax, where, isnan, logical_or, floor, log10, abs, zeros,\
         inf, transpose, linspace, remainder, column_stack, ones,\
         meshgrid, exp, sin, cos, finfo, pi, e
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr  
from scipy.sparse.linalg import lsmr
from pandas import Series, concat
from scipy.signal import argrelmax
from scipy.ndimage.filters import gaussian_filter
from datetime import timedelta

from matplotlib.pyplot import subplots
from matplotlib.patches import Circle

from .imagelists import BaseImgList
from .processing import ImgStack
from .helpers import map_coordinates_sub_img, shifted_color_map
from .image import Img

class DoasFOV(object):
    """Class for storage of FOV information"""
    def __init__(self):
        self.search_settings = {}
        self.img_prep_settings = {}
        
        self.corr_img = None
        
        self.calib_curve = None
        
        #dictionaries containing FOV search results in relative coordinates
        #(i.e. image stack coordinates)
        self._results_rel = {"cx"           : None, # x-pos max corr_img
                             "cy"           : None, # y-pos max corr_img
                             "radius"       : None, # radius fov disk (pearson)
                             "corr_curve"   : None, # radius corr curve (pearson)
                             "tau_series"   : None, 
                             "g2d_popt"     : None,
                             "g2d_dat_fit"  : None}
      
    def fov_mask(self, abs_coords = False):
        """Returns FOV mask for data access
        
        :param bool abs_coords: if False, mask is created in stack 
            coordinates (i.e. corresponding to ROI and pyrlevel of stack).
            If True, the FOV parameters are converted into absolute 
            detector coordinates such that they can be used for original 
            images.
            
        """
        raise NotImplementedError    
        
    def save_fov_mask(self, file_type = "fts"):
        """Save the fov mask as image (in absolute detector coords)
        """
        raise NotImplementedError
        
    def draw(self):
        """Draw the current FOV position into the current correlation img"""
        fig, axes = subplots(1, 1)
        
        ax = axes[0]
        ax.hold(True)
        img = self.corr_img.img
        vmin, vmax = img.min(), img.max()
        cmap = shifted_color_map(vmin, vmax, cmap = "RdBu")
        
        disp = ax.imshow(img, vmin = vmin, vmax = vmax, cmap = cmap)#, cmap=plt.cm.jet)
        cb = fig.colorbar(disp, ax = ax)
        if self.search_settings["method"] == "ifr":
            cb.set_label(r"FOV fraction [$10^{-2}$ pixel$^{-1}$]",\
                                                         fontsize = 16)
            (Ny,Nx) = img.shape
            xvec = linspace(0, Nx, Nx)
            yvec = linspace(0, Ny, Ny)
            xgrid, ygrid = meshgrid(xvec, yvec)
            
        elif self.search_settings["method"] == "pearson":
            cb.set_label(r"Pearson corr. coeff.", fontsize = 16)
            
        #ax.plot(popt[1], popt[2], 'x')
        ax.contour(xgrid, ygrid, data_fitted_norm, (popt[0]/np.e, popt[0]/10), colors='k')
        ax.get_xaxis().set_ticks([popt[1]])
        ax.get_yaxis().set_ticks([popt[2]])
        ax.axhline(popt[2], ls="--", color="k")
        ax.axvline(popt[1], ls="--", color="k")
        #ax.set_axis_off()
        ax.set_title("A) LSMR routine (Parametr: tilted hypergauss)")
        
        ax=axes[1]
        ax.hold(True)
        #cmap = shifted_color_map(-1, corrIm.max(), cmap = plt.cm.RdBu)
        dispR = ax.imshow(corrIm, vmin = -1, vmax = 1, cmap=plt.cm.RdBu)
        cbR = fig.colorbar(dispR, ax = ax)
        cbR.set_label(r"Pearson corr. coeff.", fontsize = 16)
        ax.autoscale(False)
        #ax.plot(cx,cy, "x")
        c = Circle((cx, cy), radius, ec="k", fc="none")
        ax.add_artist(c)
        ax.set_title("B) Pearson routine (Parametr: circ. disk)")
        ax.get_xaxis().set_ticks([cx])
        ax.get_yaxis().set_ticks([cy])
        ax.axhline(cy, ls="--", color="k")
        ax.axvline(cx, ls="--", color="k")
        #ax.set_axis_off()
        # plot calibration curve
        tauLSMR = Series(tauValsLSMR, acqTimes)
        fig.tight_layout()
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        ax =axes[0]
        ts=tauDataPearson.index
        p1 = ax.plot(ts, tauDataPearson, "--x", label="tauPearson")
        p2 = ax.plot(ts, tauLSMR, "--x", label="tauLSMR")
        
class DoasFOVEngine(object):
    """Engine to perform DOAS FOV search"""
    def __init__(self, img_stack, doas_series, method = "pearson",\
                                                         **settings):
        
        self._settings = {"method"          :   "pearson",
                          "ifr_lambda"      :   1e-6,
                          "g2d_asym"        :   True,
                          "g2d_crop"        :   True,
                          "g2d_tilt"        :   False,
                          "smooth_corr_img" :   4,
                          "merge_type"      :   "average"}
        
        
        self.DATA_MERGED = False
        self.img_stack = img_stack
        self.doas_series = doas_series
        
        self.fov = DoasFOV()
        
        self.update_search_settings(**settings) 
        self.method = method
    
    def update_search_settings(self, **settings):
        """Update current search settings
        
        :param **settings: keyword args to be updated (only
            valid keys will be updated)
        """
        for k, v in settings.iteritems():
            if self._settings.has_key(k):
                print ("Updating FOV search setting %s, new value: %s" 
                       %(k, v))
                self._settings[k] = v
            
    @property
    def doas_data_vec(self):
        """Return DOAS CD vector (values of ``self.doas_series``)"""
        return self.doas_series.values
    
    @property
    def method(self):
        """Return current FOV search method"""
        return self._settings["method"]
    
    @method.setter
    def method(self, value):
        """Return current FOV search method"""
        if not value in ["ifr", "pearson"]:
            raise ValueError("Invalid search method: choose from ifr or"
                             " pearson")
        self._settings["method"] = value

    def _check_data(self):
        """Checks image stack and DOAS data"""
        if not isinstance(self.img_stack, ImgStack) or not\
                                    isinstance(self.doas_, Series):
            raise IOError("SearchCorrelation  could not be created wrong input"
                    ": %s, %s" %(type(self.img_stack), type(\
                                                     self.doas_series)))
    
    def perform_fov_search(self, **settings):
        """High level method for automatic FOV search
        
        Uses the current settings (``self._settings``) to perform the 
        following steps:
            
            1. Call :func:`merge_data`: Time merging of stack and DOAS 
            vector. This step is skipped if data was already merged within 
            this engine, i.e. if ``self.DATA_MERGED == True``
                
            #. Call :func:`det_correlation_image`: Determination of 
            correlation image using ``self.method`` ('ifr' or 'pearson')
                
            #. Call :func:`get_fov_shape`: Identification of FOV shape / 
            extend on image detector either using circular disk approach
            (if ``self.method == 'pearson'``) or 2D (super) Gauss fit 
            (if ``self.method == 'ifr').
        
        All relevant results are written into ``self.fov`` (
        :class:`DoasFOV` object)
        """
        self.update_search_settings(**settings)
        self.merge_data(merge_type = self._settings["merge_type"])
        self.det_correlation_image(search_type = self.method)
        self.get_fov_shape()
        self.fov.search_settings = self._settings
        
        return self.fov
        
                    
    def merge_data(self, merge_type = "average"):
        """Merge stack data and DOAS vector in time
        
        Wrapper for :func:`merge_with_time_series` of :class:`ImgStack`
        
        :param str merge_type: choose between ``average, interpolation, 
        nearest``
        
        .. note::
            
            Current data (i.e. ``self.img_stack`` and ``self.doas_series``)
            will be overwritten if merging succeeds
            
        """
        if self.DATA_MERGED:
            print ("Data merging aborted, img stack and DOAS vector are "
                   "already merged in time")
            return
        
        new_stack, new_doas_series = self.img_stack.merge_with_time_series(\
                            self.doas_series, method = merge_type)
        if len(new_doas_series) == new_stack.shape[0]:
            self.img_stack = new_stack
            self.doas_series = new_doas_series
            self._settings["merge_type"] = merge_type
            self.DATA_MERGED = True
            return True
        print "Data merging failed..."
        return False
    
    def det_correlation_image(self, search_type = "pearson", **kwargs):
        """Determines correlation image
        
        :param str search_type: updates current search type, available types
            ``["pearson", "ifr"]``
        """
        if not self.img_stack.shape[0] == len(self.doas_series):
            raise ValueError("DOAS correlation image object could not be "
                "determined: inconsistent array lengths, please perform time"
                "merging first")
        self.update_search_settings(**kwargs)
        if search_type == "pearson":
            corr_img, _ = self._det_correlation_image_pearson(\
                                                    **self._settings)
        elif search_type == "ifr":
            corr_img, _ = self._det_correlation_image_ifr_lsmr(\
                                                    **self._settings)
        else:
            raise ValueError("Invalid search type %s: choose from "
                             "pearson or ifr" %search_type)
        corr_img = Img(corr_img, pyrlevel =\
                               self.img_stack.img_prep["pyrlevel"])
        corr_img.pyr_up(self.img_stack.img_prep["pyrlevel"])
        self.fov.corr_img = corr_img
        
        return corr_img
        
    def _det_correlation_image_pearson(self, **kwargs):
        """Determine correlation image based on pearson correlation
        
        :returns: - correlation image (pix wise value of pearson corr coeff)
        """
        h,w = self.img_stack.shape[1:]
        corr_img = zeros((h,w), dtype = float)
        corr_img_err = zeros((h,w), dtype = float)
        doas_vec = self.doas_series.values
        for i in range(h):
            print "FOV search: current img row (y): " + str(i)
            for j in range(w):
                #get series from stack at current pixel
                corr_img[i,j], corr_img_err[i,j] = pearsonr(\
                        self.img_stack.stack[:, i, j], doas_vec)
        self._settings["method"] = "pearson"
        return corr_img, corr_img_err
    
    def _det_correlation_image_ifr_lsmr(self, ifr_lambda = 1e-6, **kwargs):
        """Apply LSMR algorithm to identify the FOV
        
        :param float ifr_lambda: tolerance parameter lambda
        """
        # some input data size checking
        (m,) = self.doas_data_vec.shape
        (m2, ny, nx) = self.img_stack.shape
        if m != m2:
            raise ValueError("Inconsistent array lengths, please perform "
                "time merging of image stack and doas vector first")
            
        # construct H-matrix through reshaping image stack
        #h_matrix = transpose(self.img_stack.stack, (2,0,1)).reshape(m, nx * ny)
        h_matrix = self.img_stack.stack.reshape(m, nx * ny)
        # and one-vector
        h_vec = ones((m,1), dtype = h_matrix.dtype)
        # and stacking in the end
        h = column_stack((h_vec, h_matrix))
        # solve using LSMR regularisation
        a = lsmr(h, self.doas_data_vec, atol = ifr_lambda, btol = ifr_lambda)
        c = a[0]
        # separate offset and image
        lsmr_offset = c[0]
        lsmr_image = c[1:].reshape(ny, nx)
        self._settings["method"] = "ifr"
        self._settings["ifr_lambda"] = ifr_lambda
        return lsmr_image, lsmr_offset
    
    def get_fov_shape(self, **settings):
        """Find shape of FOV based on correlation image
        
        Search pixel coordinate of highest correlation in 
        ``self.fov.corr_img`` (using :func:`get_img_maximum`) and based on 
        that finds FOV shape either using disk approach (if 
        ``self.method == 'pearson'``) calling :func:`fov_radius_search` or
        using 2D Gauss fit (if ``self.method == 'ifr'``) calling 
        :func:`fov_gauss_fit`. 
        Results are written into ``self.fov`` (:class:`DoasFOV` object)
        
        :param **settings: update current settings (keyword args passed 
            to :func:`update_search_settings`)
        
        """
        
        if self.fov.corr_img is None:
            raise Exception("Correlation image is not available")
            
        if self.method == "pearson":
            cy, cx = self.get_img_maximum(self.fov.corr_img, gaussian_blur =\
                                         self._settings["smooth_corr_img"])
            radius, corr_curve, tau_data = self.fov_radius_search(cx, cy)
            if not radius > 0:
                raise ValueError("Pearson FOV search failed")
            self.fov._results_rel["cx"] = cx
            self.fov._results_rel["cy"] = cy
            self.fov._results_rel["radius"] = radius
            self.fov._results_rel["corr_curve"] = corr_curve
            self.fov._results_rel["tau_series"] = tau_data
            return radius, corr_curve, tau_data
        
        elif self.method == "ifr":
            cx, cy, popt, data_fitted = self.fov_gauss_fit(\
                            self.fov.corr_img, **self._settings)
            self.fov._results_rel["cx"] = cx
            self.fov._results_rel["cy"] = cy
            self.fov._results_rel["g2d_popt"] = popt
            self.fov._results_rel["g2d_dat_fit"] = data_fitted
            
        else:
            raise ValueError("Invalid search method...")
            
    def fov_radius_search(self, cx, cy):
        """Search the FOV disk radius around center coordinate
        
        The search varies the radius around the center coordinate and extracts
        image data time series from average values of all pixels falling into 
        the current disk. These time series are correlated with spectrometer
        data to find the radius showing highest correlation.
        
        :param int cx: pixel x coordinate of center position
        :param int cy: pixel y coordinate of center position
            
        """
        stack = self.img_stack
        spec_data = self.doas_series.values
        if not len(spec_data) == stack.shape[0]:
            raise ValueError("Mismatch in lengths of input arrays")
        h, w =  stack.shape[1:]
        #find maximum radius (around CFOV pos) which still fits into the image
        #shape of the stack used to find the best radius
        max_rad = min([cx, cy, w - cx, h - cy])
        #radius array
        radii = arange(1, max_rad, 1)
        print "Maximum radius: " + str(max_rad - 1)
        #some variable initialisations
        coeffs, coeffs_err = [], []
        max_corr = 0
        tau_data = None
        radius = 0
        #loop over all radii, get tauSeries at each, (merge) and determine 
        #correlation coefficient
        for r in radii:
            print "current radius:" + str(r)
            #now get mean values of all images in stack in circular ROI around
            #CFOV
            tau_dat = stack.get_time_series(cx, cy, radius = r).values
    
            coeff, err = pearsonr(tau_dat, spec_data)
            coeffs.append(coeff)
            coeffs_err.append(err)
            #and append correlation coefficient to results
            if coeff > max_corr:
                radius = r
                max_corr = coeff
                tau_data = tau_dat
        corr_curve = Series(asarray(coeffs, dtype = float),radii)
        return radius, corr_curve, tau_data
        
    # define IFR model function (Super-Gaussian)    
    def _supergauss_2d(self, (x, y), amplitude, xm, ym, sigma, asym,\
                                                           shape, offset):
        """2D super gaussian without tilt
        
        :param tuple (x, y): position 
        :param float amplitude: amplitude of peak
        :param float xm: x position of maximum
        :param float ym: y position of maximum
        :param float asym: assymetry in y direction (1 is circle, smaller 
                means dillated in y direction)
        :param float shape: super gaussian shape parameter (2 is gaussian)
        :param float offset: base level of gaussian 
        """
        u = ((x - xm) / sigma) ** 2 + ((y - ym) * asym / sigma)**2
        g = offset + amplitude * exp(-u**shape)
        return g.ravel()
    
    def _supergauss_2d_tilt(self, (x, y), amplitude, xm, ym, sigma, asym,\
                                                   shape, offset, theta):
        """2D super gaussian without tilt
        
        :param tuple (x, y): position
        :param float amplitude: amplitude of peak
        :param float xm: x position of maximum
        :param float ym: y position of maximum
        :param float asym: assymetry in y direction (1 is circle, smaller 
                means dillated in y direction)
        :param float shape: super gaussian shape parameter (2 is gaussian)
        :param float offset: base level of gaussian 
        :param float theta: tilt angle (rad) of super gaussian
        
        """
        xprime = (x-xm) * cos(theta) - (y-ym) * sin(theta)
        yprime = (x-xm) * sin(theta) + (y-ym) * cos(theta)
        u = (xprime / sigma)**2 + (yprime * asym / sigma)**2
        g = offset + amplitude * exp(-u**shape)
        return g.ravel()
        
    def fov_gauss_fit(self, corr_img, g2d_asym = True,\
                      g2d_super_gauss = True, g2d_crop = True,\
                      g2d_tilt = False, smooth_corr_img = 4, **kwargs):
        """Apply 2D gauss fit to correlation image
        
        :param corr_img: correlation image
        :param bool asym: super gauss assymetry
        :param bool super_gauss
        :param int smooth_sigma_max_pos: width of gaussian smoothing kernel 
            convolved with correlation image in order to identify position of
            maximum
        
        """# setup grid
        (ny, nx) = corr_img.shape
        xvec = linspace(0, nx, nx)
        yvec = linspace(0, ny, ny)
        xgrid, ygrid = meshgrid(xvec, yvec)
        # apply maximum of filtered image to initialise 2D gaussian fit
        (cx, cy) = self.get_img_maximum(corr_img, smooth_corr_img)
        # constrain fit, if requested
        if g2d_asym:
            asym_lb = -inf
            asym_ub =  inf
        else:
            asym_lb = 1 - finfo(float).eps
            asym_ub = 1 + finfo(float).eps
        if g2d_super_gauss:
            shape_lb = -inf
            shape_ub =  inf
        else:
            shape_lb = 1 - finfo(float).eps
            shape_ub = 1 + finfo(float).eps
        if g2d_tilt and not g2d_asym:
            raise ValueError("With tilt and without asymmetry makes no sense")
        if g2d_tilt:
            guess = [1, cx,      cy,      20,       1,        1,       0,        0]
            lb = [-inf, -inf, -inf, -inf, asym_lb, shape_lb, -inf, -inf]
            ub = [ inf,  inf,  inf,  inf, asym_ub, shape_ub,  inf,  inf]
            if any(lb >= ub):
                print "Bound-Problem"
            popt, pcov = curve_fit(self._supergauss_2d_tilt, (xgrid, ygrid),\
                                corr_img.ravel(), p0 = guess, bounds = (lb,ub))
            popt[-1] = remainder(popt[-1], pi * 2)
            data_fitted = self._supergauss_2d_tilt((xgrid, ygrid), *popt)
        else:
            guess = [1, cx,      cy,      20,       1,        1,        0]
            lb = [-inf, -inf, -inf, -inf, asym_lb, shape_lb, -inf]
            ub = [ inf,  inf,  inf,  inf, asym_ub, shape_ub,  inf]
            popt, pcov = curve_fit(self._supergauss_2d, (xgrid, ygrid),\
                                   corr_img.ravel(), p0=guess, bounds=(lb,ub))
            data_fitted = self._supergauss_2d((xgrid, ygrid), *popt)
        # eventually crop FOV distribution (makes it more robust against outliers (eg. mountan ridge))
        if g2d_crop:
            # set outside (1/e amplitude) datapoints = 0
            data_fitted[data_fitted < popt[0] / e] = 0
        # reshape data_fitted as matrix instead of vector required for fitting
        data_fitted = data_fitted.reshape(ny, nx)
        # normalise
        return (cx, cy, popt, data_fitted)
        
    #function convolving the image stack with the obtained FOV distribution    
    def convolve_stack_fov(self, fov_fitted):
        """Normalize fov image and convolve stack
        
        :param ndarr
        :returns: - stack time series vector within FOV
        """
        # normalize data_fitted
        normsum = sum(fov_fitted)
        fov_fitted_norm = fov_fitted / normsum
        # convolve with image stack
        stack_data_conv = transpose(self.stac, (2,0,1)) * fov_fitted_norm
        return stack_data_conv.sum((1,2))
        
    def get_img_maximum(self, img_arr, gaussian_blur = 4):
        """Get coordinates of maximum in image
        
        :param array img_arr: numpy array with image data data
        :param int gaussian_blur: apply gaussian filter before max search
        
        """
        #replace nans with zeros
        #img_arr[where(isnan(img_arr))] = 0
        #print img_arr.shape
        img_arr = gaussian_filter(img_arr, gaussian_blur)
        return unravel_index(nanargmax(img_arr), img_arr.shape)
    
    
            
#==============================================================================
# class SearchFOVSpectrometer(object):
#     """Correlation based engine for FOV search
#     The base class for identifying the FOV of a spectrometer within the FOV
#     of a camera in order to determine the calibration curve of the camera using
#     spectral fitting results (slant column densities) of a tracer (e.g. SO2) in
#     the FOV of the camera which is sensitive to this tracer (e.g. SO2 camera).
#     
#     Input:
#         
#     :param SpectralResults specResults: piscope class containing spectral \
#         results (column densities) including start and stop times of the \
#         measurements performed        
#     :param ImgList imgList: :class:`ImgList` object containing the imagery \
#         file information, pre-processing information and handling features    
#     :param Img bgImgObj (None): background image used for :math:`\tau` image calculation
#     :param list subImRect (None): Rectangle defining ROI [[x0,y0],[x1,y1]], \
#         default => use whole image
#     
#     """
#     def __new__(cls, specData, imgList):
#         if not isinstance(imgList, BaseImgList) or not isinstance(specData, Series):
#             raise IOError("SearchCorrelation object could not be created:"
#                 "wrong input")
#         return super(SearchFOVSpectrometer, cls).__new__(cls, specData,\
#                                                 imgList)
#             
#     def __init__(self, specData, imgList):
#         #spectrometer results should be in pandas time series format
#         self.specData = specData
#         self.imgList = imgList
#         
#         self.mergeTypes = ["binning", "interpolation"]            
#         self.stack = None
#         
#         self.res_i = None
#         self.res_f = None
#         
#         self.fov = Bunch([("pos",   None),
#                           ("radius",None)])
#                           
#     @property
#     def maxCorr(self):
#         raise NotImplementedError
#         
#     def perform_fov_search(self, tauMode = 1, pyrLevel_i = 4, pyrLevel_f = 0,\
#                                                         mergeType = "binning"):
#         """High level implementation for automatic FOV search
#         
#         :param bool tauMode: determines tau image stack (bg data must be 
#             available in ``self.imgList``)
#         :param int pyrLevel_i (4): gaussian pyramide downscaling factor for
#             first step of search (which takes place in low pix res to get a 
#             rough idea of where the fov is located and how large it is)
#         :param int pyrLevel_f (0): gaussian pyramide downscaling factor for
#             second step of search (detailed search of fov position and radius).
#             It is recommended to leave this value at 0 (which means in full res)
#         :param str mergeType ("binning"): sets merge type for stack and 
#             spectrometer data, valid input: binning or interpolation (expl see
#             below)
#         
#         This algorithm performes an automatic search of the FOV of the 
#         soectrometer data within the camera FOV by determining pearson 
#         correlation between camera data time series (per pixel) and the 
#         spectrometer data vector. In order to do this, an image stack ``(x,y,t)`` 
#         (:class:`ImgListStack`) is determined from all images in 
#         ``self.imgList`` (if input param tauMode is active, a stack of tau 
#         images is determined). 
#         The next step is to pixelwise (i,j) determine pearson correlation coefficients 
#         between the pixel time series (ts_ij) in the stack ``ts_ij = stack[i,j,:]``
#         and the spectrometer data time series. 
#         In order to do this, the time series have to be merged to the same 
#         array length. The way to do this can be specified via input param 
#         mergeType using either:
#         
#             1. Binning
#             It is assumed that spectrum exposure times are longer than image
#             exposure times, i.e. that for most of the spectrometer data points
#             (each defined by their start and stop time) one or more images 
#             exist. All images corresponding to on spectrum are then averaged 
#             and a new image stack is determined which only includes the averaged
#             images. Spectra for which no images are available are then removed
#             from the sectrometer time series, such that the number of images in
#             the new stack matches the number of datapoints in the spectrometer
#             time series
#             
#             2. Interpolation (bit slower)
#             In this method, spectrometer time series are concatenated with the 
#             time series in each pixel of the initial stack. The concatenated
#             dataframe consists of an index grid merged from all time stamps of
#             the two input series and the datapoints are (linearly) interpolated
#             to fill the gaps. The concatenation and interplation happens while
#             looping over the pixels over the stack.
#         
#             
#         
#         """
#         itp = 0
#         if mergeType == "interpolation":
#             itp =1
#         self.make_img_stack(tauMode, pyrLevel_i)
#         xPos_i, yPos_i, corrIm_i, specData_i = self.find_viewing_direction(mergeType)
#         radius_i, corrCurve_i, imData_i, specData_i= self.find_fov_radius(\
#                             self.stack, xPos_i, yPos_i, specData_i, itp)
#         
#         
#         r0 = (radius_i + 3) * 2**(pyrLevel_i)
#         xAbs0, yAbs0 = self.stack.imgPrep.map_coordinates(xPos_i, yPos_i,\
#                                                                 inverse = 1)
#                                                                 
#         roi = [xAbs0 - int(r0), xAbs0 + int(r0), yAbs0 - int(r0), yAbs0 + int(r0)]
#         stack_i = self.stack
#         
#         #write the results of the "rough" search into dictionary
#         res_i = Bunch({"xAbs"        :   xAbs0,
#                        "yAbs"        :   yAbs0,
#                        "xRel"        :   xPos_i,
#                        "yRel"        :   yPos_i,
#                        "radius"      :   radius_i,
#                        "corrIm"      :   corrIm_i,
#                        "corrCurve"   :   corrCurve_i,
#                        "specData"    :   specData_i,
#                        "imData"      :   imData_i,
#                        "stack"       :   stack_i})
#                       
#         self.make_img_stack(tauMode, pyrLevel_f, roi)
#         xPos_f, yPos_f, corrIm_f, specData_f = self.find_viewing_direction(mergeType)
#         radius_f, corrCurve_f, imData_f, specData_f= self.find_fov_radius(\
#                             self.stack, xPos_f, yPos_f, specData_f, itp)
#         dx, dy = self.stack.imgPrep.get_subimg_offset_xy()
#         xAbs, yAbs = dx + xPos_f, dy + yPos_f
#         
#         #write the results of the "detailed" search into dictionary
#         res_f = Bunch({"xAbs"        :   xAbs,
#                        "yAbs"        :   yAbs,
#                        "xRel"        :   xPos_f,
#                        "yRel"        :   yPos_f,
#                        "radius"      :   radius_f,
#                        "corrIm"      :   corrIm_f,
#                        "corrCurve"   :   corrCurve_f,
#                        "specData"    :   specData_f,
#                        "imData"      :   imData_f,
#                        "stack"       :   self.stack})
#                        
#         return res_i, res_f
#         
#     def check_fov_search_results(self):
#         """Base check if all steps of the FOV search were succesful
#         """
#         ok=1
#         info=self.searchInfo
#         print "Checking results from automatic FOV search algorithm\n-----------------------------------------------\n"
#         for id, result in info.results.iteritems():        
#             if not result.ok:
#                 print (str(id) +" search failed")
#                 ok = 0
#         return ok
#      
#     def make_img_stack(self, tauMode = 1, pyrLevel = 0, roi = None):
#         """Make the image stack for one of the settings"""
#         if tauMode and not self.imgList.activate_tau_mode():
#             raise Exception("Failed to create image stack in FOV search engine")
#         
#         stack = ImgListStack(self.imgList)
#         #prepInfo = ImagePreparation({"pyrlevel": pyrLevel})
#         stack.imgPrep.update_settings(pyrlevel = pyrLevel, roi = roi)
#         stack.activate_tau_mode(tauMode)
#         if stack.make_stack():
#             self.stack = stack
#             return True
#     
#     def check_start_stop_times_spectrometer(self):
#         """Checks if start and stop time arrays in doas data are valid
#         
#         ``self.specData`` is supposed to be a 
#         :class:`pydoas.Analysis.DoasResults` object which IS a pandas.Series 
#         object with a bit more functionality, i.a. two arrays specifying 
#         start and stop times of spectra acquisition. Knowledge of these is 
#         important here if image and DOAS data is merged using the method binning
#         where all images within spectrum start/stop times are averaged.
#         
#         :returns: boolean stating whether start / stop time arrays of 
#             ``self.specData`` are valid
#         
#         """
#         try:
#             i,f = self.specData.startTimes, self.specData.stopTimes
#             diff = f - i
#             for td in diff:
#                 if not isinstance(td, timedelta) or td.total_seconds() < 0:
#                     print "Invalid value encountered for timedelta %s" %td
#                     return False
#             return True        
#         except Exception as e:
#             print ("Failed to read start / stop acq. time info in spectrometer"
#                 " data object %s: %s" %(type(self.specData), repr(e)))
#             return False
#             
#     def auto_set_mergetype(self, threshNum = 100):
#         """Checks the number of spectrum data points and updates the merge type
#         
#         :param int threshNum (100): integer specifying the threshold for 
#             number of spectrum datapoints (num). 
#             If num < threshNum => use interpolation, else use binning
#         :returns: mergetype
#         """
#         if len(self.specData) > threshNum and\
#                         self.check_start_stop_times_spectrometer():
#             return "binning"
#         return "interpolation"
#         
#     def find_viewing_direction(self, mergeType = "interpolation"):
#         """Find the pixel coordinates with highest correlation between image
#         stack data and spectral data time series.
#         
#         :param str id: id of search step (rough, fine)
#         
#         .. todo::
#         
#             include base check (i.e. if the detected maximum is clearly defined
#             and that there are no other areas with high correlation)
#             
#         """
#         if not isinstance(self.stack, ImgListStack):
#             raise Exception("Could not search viewing direction, image stack "
#                 " is not available, call :func:``make_img_stack`` first")
#         elif not mergeType in self.mergeTypes:
#             print ("Invalid input for mergeType %s, setting merge type  " 
#                 "automatically" %mergeType)
#             mergeType = self.auto_set_mergetype()
#             print ("New merge type: %s " %mergeType)
#         
#         h, w = self.stack.shape[:2]
#         corrIm = empty((h,w))
#         exp = int(floor(log10(abs(self.specData.mean()))))
#         specData = self.specData*10**(-exp)
#         if mergeType == "binning":
#             if not self.check_start_stop_times_spectrometer():
#                 print ("Error retrieving spectrometer start / stop times, "
#                 "changing merge type from binning to interpolation")
#                 mergeType = "interpolation"
#             else:
#                 #averages images in stack considering start and stop times
#                 #of spectra, returns list of spectrum indices for which no 
#                 #images were found
#                 badIndices = self.stack.make_stack_time_average(\
#                     self.specData.startTimes, self.specData.stopTimes)
#                 #throws out invalid spec indices, after this operation
#                 #both stack and specData
#                 specData = specData.drop(specData.index[badIndices])
#         stack = self.stack
#         
#         for i in range(h):
#             print "FOV search: current img row (y): " + str(i)
#             for j in range(w):
#                 #get series from stack at current pixel
#                 tauSeries = Series(stack.stack[i,j,:], stack.times)
#                 #create a dataframe
#                 if mergeType == "interpolation":
#                     df = concat([tauSeries, specData], axis = 1)
#                     df = df.interpolate()#"cubic")
#                     #throw all N/A values
#                     df = df.dropna()
#                     tauSeries, specData = df[0], df[1]
#                     #determine correlation coefficient
#                 corrIm[i,j] = tauSeries.corr(specData)
#         
#         #find position of maximum correlation and convert into absolute coordinates
#         yPos, xPos = self.find_local_maximum_img(corrIm)
#         
#         return xPos, yPos, corrIm, specData
#     
#     def find_fov_radius(self, stack, cx, cy, specData, interpolate = 0):
#         """Search the radius with highest correlation
#         """
#         h, w =  stack.shape[:2]
#         #find maximum radius (around CFOV pos) which still fits into the image
#         #shape of the stack used to find the best radius
#         max_rad = min([cx, cy, w - cx, h - cy])
#         #radius array
#         radii = arange(1, max_rad, 1)
#         s = stack.stack
#         print "Maximum radius: " + str(max_rad-1)
#         y ,x = ogrid[:h, :w]
#         coeffs = []
#     
#         #some variable initialisations
#         maxCorr = 0
#         tauData = None
#         radius = None
#         #loop over all radii, get tauSeries at each, (merge) and determine 
#         #correlation coefficient
#         for r in radii:
#             print "current radius:" + str(r)
#             #create mask to cut out ROI in stack
#             mask = (x - cx)**2 + (y - cy)**2 < r**2
#             means = []
#             #now get mean values of all images in stack in circular ROI around
#             #CFOV
#             for i in range(s.shape[2]):
#                 subim = s[:,:,i]
#                 means.append(subim[mask].mean())
#             #create pandas series
#             tauSeries = Series(means, stack.times)
#             #merge data (if applicable)
#             if interpolate:
#                 df = concat([tauSeries, specData], axis = 1)
#                 df = df.interpolate()#"cubic")
#                 #throw all N/A values
#                 df = df.dropna()
#                 tauSeries, specData = df[0], df[1]
#             #determine correlation coefficient at current radius
#             coeff = tauSeries.corr(specData)
#             coeffs.append(coeff)
#             #and append correlation coefficient to results
#             if coeff > maxCorr:
#                 radius = r
#                 maxCorr = coeff
#                 tauData = tauSeries
#         corrCurve = Series(asarray(coeffs, dtype = float),radii)
#         return radius, corrCurve, tauData, specData
# 
#     """Post processing / Helpers..."""
#     def find_local_maximum_img(self, imgArr, blur = 1):
#         """Get coordinates of maximum in image
#         
#         :param array imgArr: numpy array containing imagery data
#         :param int blur: apply gaussian filter (width==blur) before max search
#         """
#         #replace nans with zeros
#         imgArr[where(isnan(imgArr))]=0
#         imgArr=gaussian_filter(imgArr, blur)
#         return unravel_index(nanargmax(imgArr), imgArr.shape)
#     
#     def check_pixel_position(self, x, y, imgArr, minDistBorder = 20):
#         """Check if a pixel position is far enough away from image borders
#         :param int x: x position of pixel
#         :param int y: y position of pixel
#         :param array imgArr: the actual 2D numpy image array 
#         :param int minDistBorder (20): minimum distance from image border
#         :returns bool: True or false
#         """
#         if logical_or(x < minDistBorder,y < minDistBorder):
#             return 0
#         yf,xf=imgArr.shape
#         if x>xf-minDistBorder:
#             return 0
#         if y>yf-minDistBorder:
#             return 0
#         return 1
#         
#==============================================================================

        
