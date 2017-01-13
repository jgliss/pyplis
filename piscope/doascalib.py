# -*- coding: utf-8 -*-
from numpy import min, arange, asarray, zeros, linspace, column_stack,\
    ones, e, nan, float32, polyfit, poly1d, sqrt, isnan
from scipy.stats.stats import pearsonr  
from scipy.sparse.linalg import lsmr
from pandas import Series
from copy import deepcopy

from matplotlib.pyplot import subplots
from matplotlib.patches import Circle
from matplotlib.cm import RdBu
from .processing import ImgStack
from .helpers import shifted_color_map, mesh_from_img, get_img_maximum,\
        sub_img_to_detector_coords
from .optimisation import gauss_fit_2d, GAUSS_2D_PARAM_INFO
from .image import Img
from .inout import get_camera_info
from .setupclasses import Camera

class DoasCalibData(object):
    """Object representing DOAS calibration data"""
    def __init__(self, tau_vec = None, doas_vec = None, time_stamps = None, 
                     polyorder = 1):
        self.tau_vec = tau_vec #tau data vector within FOV
        self.doas_vec = doas_vec #doas data vector
        self.time_stamps = time_stamps
        
        self.coeffs = None
        self.cov = None
        self.polyorder = polyorder
    
    
    @property
    def calib_poly(self):
        """return poly1d object of current coefficients"""
        return poly1d(self.coeffs)
        
    @property
    def slope(self):
        """return current calib curve slope plus std"""
        return self.coeffs[-2], sqrt(self.cov[-2][-2])
    
    @property
    def y_offset(self):
        """return y axis offset of calib curve plus uncertainty"""
        return self.coeffs[-1], sqrt(self.cov[-1][-1])
    @property
    def doas_tseries(self):
        """Return pandas Series object of doas data"""
        return Series(self.doas_vec, self.time_stamps)
    
    @property
    def tau_tseries(self):
        """Return pandas Series object of tau data"""
        return Series(self.tau_vec, self.time_stamps)
    
    def fit_calib_polynomial(self, polyorder = None, plot = 1):
        """Fit calibration polynomial to current data
        
        :param int polyorder: update current polyorder        
        """
        if polyorder is None:
            polyorder = self.polyorder
    
        if sum(isnan(self.tau_vec)) + sum(isnan(self.doas_vec)) > 0:
            raise ValueError("Encountered nans in data")
        coeffs, cov = polyfit(self.tau_vec, self.doas_vec,\
                                        polyorder, cov = True)
        self.polyorder = polyorder
        self.coeffs = coeffs
        self.cov = cov
        if plot:
            self.plot()
        return self.calib_poly
    
    @property
    def tau_range(self):
        """Returns range of tau values extended by 5%"""
        tau = self.tau_vec
        taumin, taumax = tau.min(), tau.max()
        if taumin > 0:
            taumin = 0
        add = (taumax - taumin) * 0.05
        return taumin - add, taumax + add
    
    @property
    def doas_cd_range(self):
        """Returns range of DOAS cd values extended by 5%"""
        cds = self.doas_vec
        cdmin, cdmax = cds.min(), cds.max()
        if cdmin > 0:
            cdmin = 0
        add = (cdmax - cdmin) * 0.05
        return cdmin - add, cdmax + add
        
    def plot(self, ax = None):
        """Plots current calibration"""
        if ax is None:
            fig, ax = subplots(1,1, figsize=(16,6))
        ax.plot(self.tau_vec, self.doas_vec, " x", label = "Data")
        taumin, taumax = self.tau_range
        x = linspace(taumin, taumax, 100)
        ax.plot(x, self.calib_poly(x), "--r", label = "Fit %s" %self.calib_poly)
        ax.legend(loc='best', fancybox=True, framealpha=0.5)
        ax.grid()
        return ax
        
    def plot_data_tseries_overlay(self, ax = None):
        """Plot overlay of tau and DOAS time series"""
        if ax is None:
            fig, ax = subplots(1,1)
        s1 = self.tau_tseries
        s2 = self.doas_tseries
        p1 = ax.plot(s1, "--xb", label = r"$\tau$")
        ax.set_ylabel("tau")
        ax2 = ax.twinx()
            
        p2=ax2.plot(s2, "--xr", label="DOAS CDs")
        ax2.set_ylabel("SO2-SCD [cm-2]")
        ax.set_title("Time series overlay DOAS calib data")
        
        ps = p1 + p2
        labs = [l.get_label() for l in ps]
        ax.legend(ps, labs, loc="best",fancybox=True, framealpha=0.5)
        
        return ax, ax2
#==============================================================================
#         ax1 = self.tau_tseries.plot(label = "img tau data")
#         ax1.set_ylabel("tau data")
#         ax2 = self.doas_tseries.plot(label = "doas data", secondary_y = True)
#         ax2.set_ylabel("DOAS CD time series")
#==============================================================================
        
        
        
class DoasFOV(object):
    """Class for storage of FOV information"""
    def __init__(self, camera = None):
        self.search_settings = {}
        self.img_prep = {}
        self.roi = None
        self.camera = None
        self.calib_data = DoasCalibData()
        
        self.corr_img = None
        
        self.fov_mask = None
        
        self.result_pearson = {"cx_rel"     :   nan,
                               "cy_rel"     :   nan,
                               "radius_rel" :   nan,
                               "corr_curve" :   None}
        self.result_ifr = {"popt"           :   None,
                           "pcov"           :   None}
                           
        if isinstance(camera, Camera):
            self.camera = camera
    
    @property
    def method(self):
        """Returns search method"""
        return self.search_settings["method"]
        
    @property
    def cx(self):
        """Return center x coordinate of FOV (in relative coords)"""
        if self.method == "ifr":
            return self.result_ifr["popt"][1]
        else:
            return self.result_pearson["cx_rel"]
    @property
    def cy(self):
        """Return center x coordinate of FOV (in relative coords)"""
        if self.method == "ifr":
            return self.result_ifr["popt"][2]
        else:
            return self.result_pearson["cy_rel"]
    
    @property
    def radius(self):
        """Returns radius of FOV (in relative coords)

        :raises: TypeError if method == "ifr"
        """
        if self.method == "ifr":
            raise TypeError("Invalid value: method IFR does not have FOV "
                "parameter radius, call self.popt for relevant parameters")
        return self.result_pearson["radius_rel"]
        
    @property
    def popt(self):
        """Return super gauss optimisation parameters (in relative coords)
        
        :raises: TypeError if method == "pearson"        
        
        """
        if self.method == "pearson":
            raise TypeError("Invalid value: method pearson does not have FOV "
                "shape parameters, call self.radius to retrieve disk radius")
        return self.result_ifr["popt"]
        
    def transform_fov_mask_abs_coords(self, img_shape_orig = (), cam_id = ""):
        """Converts the FOV mask to absolute detector coordinates
        
        :param tuple img_shape_orig: image shape of original image data (can
            be extracted from an unedited image), or
        :param str cam_id: string ID of piscope default camera (e.g. "ecII")
            
        The shape of the FOV mask (and the represented pixel coordinates) 
        depends on the image preparation settings of the :class:`ImgStack` 
        object which was used to identify the FOV. 
        """
        if not len(img_shape_orig) == 2:
            try:
                info = get_camera_info(cam_id)
                img_shape_orig = (int(info["pixnum_y"]), int(info["pixnum_x"]))
            except:
                raise IOError("Image shape could not be retrieved...")
        mask = self.fov_mask.astype(float32)       
        return sub_img_to_detector_coords(mask, img_shape_orig,\
                    self.img_prep["pyrlevel"], self.roi).astype(bool)
        
#==============================================================================
#       
#     def fov_mask(self, abs_coords = False):
#         """Returns FOV mask for data access
#         
#         :param bool abs_coords: if False, mask is created in stack 
#             coordinates (i.e. corresponding to ROI and pyrlevel of stack).
#             If True, the FOV parameters are converted into absolute 
#             detector coordinates such that they can be used for original 
#             images.
#             
#         """
#         raise NotImplementedError    
#==============================================================================
        
    def save_as_fits(self, ):
        """Save the fov mask as image (in absolute detector coords)
        """
        raise NotImplementedError
    
    def __str__(self):
        """String representation"""
        s = "DoasFOV information\n------------------------\n"
        s += "\nImg stack preparation settings\n............................\n"
        for k, v in self.img_prep.iteritems():
            s += "%s: %s\n" %(k, v)
        s += "\nFOV search settings\n............................\n"
        for k, v in self.search_settings.iteritems():
            s += "%s: %s\n" %(k, v)
        if self.method == "ifr":
            s += "\nIFR search results \n.........................\n"
            s += "\nSuper gauss fit optimised params\n"
            popt = self.popt
            for k in range(len(popt)):
                s += "%s: %.3f\n" %(GAUSS_2D_PARAM_INFO[k], popt[k])
        elif self.method == "pearson":
            s += "\nPearson search results \n.......................\n"
            for k, v in self.result_pearson.iteritems():
                if not k == "corr_curve":
                    s += "%s: %s\n" %(k, v)
        return s
        
    def plot(self, ax = None):
        """Draw the current FOV position into the current correlation img"""
        if ax is None:        
            fig, ax = subplots(1, 1)
        
        img = self.corr_img.img
        vmin, vmax = img.min(), img.max()
        cmap = shifted_color_map(vmin, vmax, cmap = RdBu)
        h, w = img.shape
        disp = ax.imshow(img, vmin = vmin, vmax = vmax, cmap = cmap)
        cb = fig.colorbar(disp, ax = ax)
        if self.method == "ifr":
            popt = self.popt
            cb.set_label(r"FOV fraction [$10^{-2}$ pixel$^{-1}$]",\
                                                         fontsize = 16)
            
            xgrid, ygrid = mesh_from_img(img)
        
            ax.contour(xgrid, ygrid, self.fov_mask,\
                (popt[0] / e, popt[0] / 10), colors = 'k')
            ax.axhline(self.cy, ls="--", color = "k")
            ax.axvline(self.cx, ls="--", color = "k")
            ax.get_xaxis().set_ticks([0, self.cx, w])
            ax.get_yaxis().set_ticks([0, self.cy, h])
            #ax.set_axis_off()
            ax.set_title(r"IFR result $\lambda=%.1e$"\
                        %self.search_settings["ifr_lambda"])
        elif self.method == "pearson":
            cb.set_label(r"Pearson corr. coeff.", fontsize = 16)
            ax.autoscale(False)
            
            c = Circle((self.cx, self.cy), self.radius, ec = "k", fc = "none")
            ax.add_artist(c)
            ax.set_title("Pearson routine")
            ax.get_xaxis().set_ticks([0, self.cx, w])
            ax.get_yaxis().set_ticks([0, self.cy, h])
            ax.axhline(self.cy, ls="--", color="k")
            ax.axvline(self.cx, ls="--", color="k")
            
        
class DoasFOVEngine(object):
    """Engine to perform DOAS FOV search"""
    def __init__(self, img_stack, doas_series, method = "pearson",\
                                                         **settings):
        
        self._settings = {"method"              :   "pearson",
                          "pearson_max_radius"  :   80,
                          "ifr_lambda"          :   1e-6,
                          "ifr_g2d_asym"        :   True,
                          "ifr_g2d_super_gauss" :   True,
                          "ifr_g2d_crop"        :   True,
                          "ifr_g2d_tilt"        :   False,
                          "smooth_corr_img"     :   4,
                          "merge_type"          :   "average"}
        
        
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
        self.fov = DoasFOV() #includes DoasCalibData class
        self.update_search_settings(**settings)
        self.merge_data(merge_type = self._settings["merge_type"])
        self.det_correlation_image(search_type = self.method)
        self.get_fov_shape()
        self.fov.search_settings = deepcopy(self._settings)
        
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
            print ("Data merging unncessary, img stack and DOAS vector are "
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
        
        Determines correlation image either using IFR or Pearson method.
        Results are written into ``self.fov`` (:class:`DoasFOV`)
        
        :param str search_type: updates current search type, available types
            ``["pearson", "ifr"]``
        """
        if not self.img_stack.shape[0] == len(self.doas_series):
            raise ValueError("DOAS correlation image object could not be "
                "determined: inconsistent array lengths, please perform time"
                "merging first")
        self.update_search_settings(method = search_type, **kwargs)
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
        #corr_img.pyr_up(self.img_stack.img_prep["pyrlevel"])
        self.fov.corr_img = corr_img
        self.fov.img_prep = self.img_stack.img_prep
        self.fov.roi = self.img_stack.roi
        
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
        #THIS NORMALISATION IS NEW
        #lsmr_image = lsmr_image / abs(lsmr_image).max()
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
            cy, cx = get_img_maximum(self.fov.corr_img.img,\
                gaussian_blur = self._settings["smooth_corr_img"])
            print "Start radius search in stack around x/y: %s/%s" %(cx, cy)
            radius, corr_curve, tau_vec, doas_vec, fov_mask =\
                                    self.fov_radius_search(cx, cy)
            if not radius > 0:
                raise ValueError("Pearson FOV search failed")
    
#==============================================================================
#             cx_abs, cy_abs = map_coordinates_sub_img(cx, cy, roi =\
#                 self.img_stack.roi, pyrlevel = pyrlevel, inverse = True)
#==============================================================================
            self.fov.result_pearson["cx_rel"] = cx
            self.fov.result_pearson["cy_rel"] = cy
            self.fov.result_pearson["radius_rel"] = radius
            self.fov.result_pearson["corr_curve"] = corr_curve
            
            self.fov.fov_mask = fov_mask
            self.fov.calib_data.tau_vec = tau_vec
            self.fov.calib_data.doas_vec = doas_vec
            self.fov.calib_data.time_stamps = self.img_stack.time_stamps
            return 
        
        elif self.method == "ifr":
            #the fit is performed in absolute dectector coordinates
            #corr_img_abs = Img(self.fov.corr_img.img).pyr_up(pyrlevel).img
            popt, pcov, fov_mask = self.fov_gauss_fit(\
                            self.fov.corr_img.img, **self._settings)
            tau_vec = self.convolve_stack_fov(fov_mask)
            
            self.fov.result_ifr["popt"] = popt
            self.fov.fov_mask = fov_mask            
            self.fov.calib_data.tau_vec = tau_vec
            self.fov.calib_data.doas_vec = self.doas_data_vec
            self.fov.calib_data.time_stamps = self.img_stack.time_stamps
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
        doas_vec = self.doas_series.values
        if not len(doas_vec) == stack.shape[0]:
            raise ValueError("Mismatch in lengths of input arrays")
        h, w =  stack.shape[1:]
        #find maximum radius (around CFOV pos) which still fits into the image
        #shape of the stack used to find the best radius
        max_rad = min([cx, cy, w - cx, h - cy])
        if self._settings["pearson_max_radius"] < max_rad:
            max_rad = self._settings["pearson_max_radius"]
        else:
            self._settings["pearson_max_radius"] = max_rad
        #radius array
        radii = arange(1, max_rad, 1)
        print "Maximum radius: " + str(max_rad - 1)
        #some variable initialisations
        coeffs, coeffs_err = [], []
        max_corr = 0
        tau_vec = None
        mask = None
        radius = 0
        #loop over all radii, get tauSeries at each, (merge) and determine 
        #correlation coefficient
        for r in radii:
            print "current radius:" + str(r)
            #now get mean values of all images in stack in circular ROI around
            #CFOV
            tau_series, m = stack.get_time_series(cx, cy, radius = r)
            tau_dat = tau_series.values
            coeff, err = pearsonr(tau_dat, doas_vec)
            coeffs.append(coeff)
            coeffs_err.append(err)
            #and append correlation coefficient to results
            if coeff > max_corr:
                radius = r
                mask = m
                max_corr = coeff
                tau_vec = tau_dat
        corr_curve = Series(asarray(coeffs, dtype = float),radii)
        return radius, corr_curve, tau_vec, doas_vec, mask
        
    # define IFR model function (Super-Gaussian)    
        
    def fov_gauss_fit(self, corr_img, ifr_g2d_asym = True,\
                      ifr_g2d_super_gauss = True, ifr_g2d_crop = True,\
                      ifr_g2d_tilt = False, smooth_corr_img = 4, **kwargs):
        """Apply 2D gauss fit to correlation image
        
        :param corr_img: correlation image
        :param bool asym: super gauss assymetry
        :param bool super_gauss
        :param int smooth_sigma_max_pos: width of gaussian smoothing kernel 
            convolved with correlation image in order to identify position of
            maximum
        
        """
        xgrid, ygrid = mesh_from_img(corr_img)
        # apply maximum of filtered image to initialise 2D gaussian fit
        (cy, cx) = get_img_maximum(corr_img, smooth_corr_img)
        # constrain fit, if requested
        (popt, pcov, fov_mask) = gauss_fit_2d(corr_img, cx, cy, ifr_g2d_asym,\
            g2d_super_gauss = ifr_g2d_super_gauss, g2d_crop = ifr_g2d_crop,\
                                            g2d_tilt = ifr_g2d_tilt, **kwargs)
        # normalise
        return (popt, pcov, fov_mask)
    
    #function convolving the image stack with the obtained FOV distribution    
    def convolve_stack_fov(self, fov_mask):
        """Normalize fov image and convolve stack
        
        :param ndarr
        :returns: - stack time series vector within FOV
        """
        # normalize fov_mask
        print fov_mask.shape
        normsum = fov_mask.sum()
        print "Normsum %s" %normsum
        fov_mask_norm = fov_mask / normsum
        print "Sum FOV mask norm: %s" %fov_mask_norm.sum()
        # convolve with image stack
        #stack_data_conv = transpose(self.stac, (2,0,1)) * fov_fitted_norm
        stack_data_conv = self.img_stack.stack * fov_mask_norm
        return stack_data_conv.sum((1,2))
        
            
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

        
