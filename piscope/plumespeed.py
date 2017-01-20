# -*- coding: utf-8 -*-
"""
Classes for plume speed retrievals
----------------------------------
"""
from time import time
from numpy import mgrid,vstack,int32,sqrt,arctan2,rad2deg, asarray,\
    logical_and, histogram, nan, ceil, ones, roll, argmax, arange

from copy import deepcopy
#from scipy.misc import bytescale
from os.path import exists
from os import mkdir
from traceback import format_exc

from collections import OrderedDict as od
from matplotlib.pyplot import subplots, figure, close, Figure

from matplotlib.patches import Rectangle
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.stats.stats import pearsonr
    
from pandas import Series

from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN,\
    cvtColor,COLOR_GRAY2BGR,line,circle,VideoCapture,COLOR_BGR2GRAY,\
    waitKey, imshow

from .helpers import bytescale, check_roi, map_roi, roi2rect
from .processing import LineOnImage, ImgStack
from .optimisation import MultiGaussFit
from .image import Img

def determine_ica_cross_correlation(icas_first_line, icas_second_line,\
        time_stamps, reg_grid_tres = None, cut_border = 0,\
                                            sigma_smooth = 1, plot = False):
    """Determines ICA cross correlation from two ICA time series
    :param ndarray icaValsPCS1: time series values of first ICA
    :param ndarray icaValsPCS1: time series values of second ICA
    :param ndarray timeStamps: array with image acquisition time stamps 
        (datetime objects)
    """
    if reg_grid_tres is None:
        delts = asarray([delt.total_seconds() for delt in\
                        (time_stamps[1:] - time_stamps[:-1])])
        reg_grid_tres = ceil(delts.mean()) - 1 #time resolution for re gridded data
    
    delt_str = "%dS" %(reg_grid_tres)

    s1 = Series(icas_first_line, time_stamps).resample(delt_str).\
            interpolate().dropna()
    s2 = Series(icas_second_line, time_stamps).resample(delt_str).\
                                                interpolate().dropna()
    if cut_border > 0:
        s1 = s1[cut_border:-cut_border]
        s2 = s2[cut_border:-cut_border]
    s1_vec = gaussian_filter(s1, sigma_smooth) 
    s2_vec = gaussian_filter(s2, sigma_smooth) 
    
    fig, ax = subplots(1,1)
    
    coeffs = []
    max_coeff = -10
    max_coeff_signal = None
    for k in range(len(s1_vec)):
        shift_s1 = roll(s1_vec, k)
        coeffs.append(pearsonr(shift_s1, s2_vec)[0])
        if coeffs[-1] > max_coeff:
            max_coeff_signal = Series(shift_s1, s1.index)
    coeffs = asarray(coeffs)
    ax = None
    if plot:
        fig, ax = subplots(1, 2, figsize = (18,6))
        x = arange(0, len(coeffs), 1) * reg_grid_tres
        s1.plot(ax = ax[0], label="First line")
        s2.plot(ax = ax[0], label="Second line")
        ax[1].plot(x, coeffs)
        ax[1].set_xlabel("Delta t [%s]" %delt_str)
        ax[1].grid()
        #ax[1].set_xlabel("Shift")
        ax[1].set_ylabel("Correlation coeff")
       
    lag = argmax(coeffs) * reg_grid_tres
    return lag, coeffs, s1, s2, max_coeff_signal, ax
    
class OpticalFlowFarnebackSettings(object):
    """Settings for optical flow Farneback calculations and visualisation"""
    def __init__(self, **settings):
        """Initiation of settings object"""
        self._contrast = od([("i_min"  ,   -9999.0),
                             ("i_max"  ,   9999.0)])
        
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
    def __init__(self, first_img = None, next_img = None, name = "",\
                                                            **settings):        
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
        raise AttributeError("Pleas use attribute roi_abs to change the "
            "current ROI")
        
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
            raise ValueError("Error setting images for optical flow calc: "
                "input images are cropped, please use uncropped images")
        
        i_min, i_max = self.current_contrast_range() 
        if any([abs(int(x)) == 9999 for x in [i_min, i_max]]) or\
                                            self.auto_update_contrast:
            roi = map_roi(self.roi_abs, this_img.edit_log["pyrlevel"])
            sub = this_img.img[roi[1]:roi[3], roi[0]:roi[2]]
            i_min, i_max = sub.min(), sub.max()
            self.update_contrast_range(i_min, i_max)
            if this_img.edit_log["is_tau"]:
                print "Image is tau, setting i_min = 0.0"
                self.settings._contrast["i_min"] = 0.0   #exclude negative tau 
                                                         #areas for flow

        self.images_prep["this"] = bytescale(this_img.img, cmin = i_min,\
                                                            cmax = i_max)
        self.images_prep["next"] = bytescale(next_img.img, cmin = i_min,\
                                                            cmax = i_max)
        
    
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
                                                    upper_val = 9999):
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
        
    def get_flow_orientation_image(self, in_roi = False):
        """Returns image corresponding to flow orientation values in each pixel"""
        if in_roi:
            fl = self.get_flow_in_roi()
        else:
            fl = self.flow
        fx, fy = fl[:,:,0], fl[:,:,1]
        return rad2deg(arctan2(fx, -fy))
      
    def get_flow_vector_length_image(self, in_roi = False):        
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
    
    def flow_orientation_histo(self, bin_res_degrees = 6, multi_gauss_fit = 1,\
            exclude_short_vecs = True, cond_mask_flat = None):
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
        """
        lens, angles = self.flow_len_and_angle_vectors
        cond = ones(len(lens))
        if cond_mask_flat is not None:
            cond = cond * cond_mask_flat
        if exclude_short_vecs:
            cond = cond * (lens > self.settings.min_length)
        angles = angles[cond.astype(bool)]
        
        num_bins = int(round(360 / bin_res_degrees))
        count, bins = histogram(angles, num_bins)
        fit = None
        if multi_gauss_fit:
            x = asarray([0.5 * (bins[i] + bins[i + 1]) for\
                                                i in xrange(len(bins) - 1)])
            fit = MultiGaussFit(count, x)
        return count, bins, angles, fit
    
    def flow_length_histo(self, multi_gauss_fit = 1, exclude_short_vecs =\
                                                True, cond_mask_flat = None):
        """Get histogram of displacement length distribution of flow field
        
        :param bool multi_gauss_fit (True): apply multi gauss fit to histo
        :param bool exclude_short_vecs: don't include flow vectors which are 
            shorter than ``self.settings.min_length``
        :param ndarray cond_mask_flat: additional conditional boolean vector
            applied to flattened orientation array (for instance all pixels
            in original imaged that exceed a certain tau value, see also
            :func:`prepare_intensity_condition_mask`)
            
        """
        lens, angles = self.flow_len_and_angle_vectors
        cond = ones(len(lens))
        if cond_mask_flat is not None:
            cond = cond * cond_mask_flat
        if exclude_short_vecs:
            cond = cond * (lens > self.settings.min_length)
        lens = lens[cond.astype(bool)]
        count, bins = histogram(lens, int(ceil(lens.max())))
        fit = None
        if multi_gauss_fit:
            x = asarray([0.5 * (bins[i] + bins[i + 1]) for\
                                                i in xrange(len(bins) - 1)])
            fit = MultiGaussFit(count, x)
        return count, bins, lens, fit
    
    def get_main_flow_field_params(self, min_length = None,\
                sigma_tol_mean_dir = None, cond_mask_flat = None):
        """
        Try to fit a gaussian to the angular distribution and determine mean
        flow direction (+/- sigma) and the average displacement length from
        statistical using masking operations
        """
        #print "Applying length thresh to histo: " + str(lenThresh)
        count, bins, angles, fit = self.flow_orientation_histo()
        if not fit.has_results():
            print ("Could not retrieve main flow field parameters..probably "
            "due to failure of multi gaussian fit to angular distribution "
            "histogram")
            return 0
        mu, sigma, add_gaussians =  fit.analyse_fit_result()
        ang_min, ang_max = mu - sigma * self.settings.sigma_tol_mean_dir,\
                mu + sigma * self.settings.sigma_tol_mean_dir
        lens, angles = self.flow_len_and_angle_vectors
        cond1 = logical_and(angles > ang_min, angles < ang_max)
        cond2 = lens > self.settings.min_length
        cond = cond1 * cond2
        if cond_mask_flat is not None:
            cond = cond * cond_mask_flat
        
        good_lens = lens[cond]
        good_angles = angles[cond]
        
        return good_lens, good_angles
        
        
    def apply_median_filter(self, shape = (3,3)):
        """Apply a median filter to flow field, i.e. to both flow images (dx, dy
        stored in self.flow) individually
        
        :param tuple shape (3,3): size of the filter
        """
        
        self.flow[:,:,0] = median_filter(self.flow[:,:,0],shape)
        self.flow[:,:,1] = median_filter(self.flow[:,:,1],shape)
    
    @property
    def del_t(self):
        """Return time difference in s between both images"""
        t0, t1 = self.get_img_acq_times()
        return (t1 - t0).total_seconds()
        
    def get_img_acq_times(self):
        """Return acquisition times of current input images
        
        :return:
            - datetime, acquisition time of first image
            - datetime, acquisition time of next image
            
        """
        t0 = self.images_input["this"].meta["start_acq"]
        t1 = self.images_input["next"].meta["start_acq"]
        return t0, t1
    """
    Plotting / visualisation etc...
    """        
    def plot_flow_histograms(self, multi_gauss_fit = 1,\
                exclude_short_vecs = True, cond_mask_flat = None, for_app = 0):
        """Plot histograms of flow field within roi
        """
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
        self.draw_flow(0,ax=ax1)
        self.draw_flow(1,ax=ax11)
        #load and draw the length and angle image
        angle_im = self.get_flow_orientation_image(True)#rad2deg(arctan2(fx,-fy))
        len_im = self.get_flow_vector_length_image(True)#sqrt(fx**2+fy**2)
        angle_im_disp = ax4.imshow(angle_im, interpolation='nearest')
        ax4.set_title("Displacment orientation", fontsize=11)        
        fig.colorbar(angle_im_disp, ax = ax4)
        
        len_im_disp = ax5.imshow(len_im, interpolation='nearest')
        fig.colorbar(len_im_disp, ax = ax5)
        ax5.set_title("Displacement lengths", fontsize=11)        
        
        #prepare the histograms
        n1, bins1, angles1, fit1 = self.flow_orientation_histo(\
            multi_gauss_fit = multi_gauss_fit, exclude_short_vecs =\
                    exclude_short_vecs, cond_mask_flat = cond_mask_flat)
        tit = "Flow angle histo"
        if multi_gauss_fit and fit1.has_results():
            mu, sigma,_ = fit1.analyse_fit_result()
        
            info_str=" mu (sigma) = %.1f (+/- %.1f)" %(mu, sigma)
            tit += info_str
        if exclude_short_vecs:
            #lenThresh=mu1+3*sigma1
            thresh = self.settings.min_length
            ax3.axvline(thresh, linestyle="--", color="r")
            tit += "\nOnly vectors longer than %d" %thresh
            
        #n, bins,angles,m=self._ang_dist_histo(fx,fy,gaussFit=1)
        w = bins1[1] - bins1[0]
        ax2.bar(bins1[:-1], n1, width = w, label = "Histo")
        ax2.set_title(tit, fontsize=11)
        if multi_gauss_fit and fit1.has_results():
            fit1.plot_multi_gaussian(ax=ax2, label="Multi-gauss fit")
        ax2.set_xlim([-180,180])    
        ax2.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)                
        #now the length histogram
        n2, bins2, lens2, fit2 = self.flow_length_histo(\
            multi_gauss_fit = multi_gauss_fit, exclude_short_vecs =\
                    exclude_short_vecs, cond_mask_flat = cond_mask_flat)
                    
        tit="Flow length histo"
        if multi_gauss_fit and fit2.has_results():
            mu, sigma,_ = fit2.analyse_fit_result()
        
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
    
    def draw_flow(self, in_roi = False, ax = None):
        """Draw the current optical flow field
        
        :param bool in_roi: if True, the flow field is plotted in a
            cropped image area (using current ROI), else, the whole image is 
            drawn and the flow field is plotted within the ROI which is 
            indicated with a rectangle
        :param ax (None): matplotlib axes object
        """
        if ax is None:
            fig, ax = subplots(1,1)
        
        x0, y0 = 0, 0
        img = self.images_input["this"]
        if in_roi:
            img = img.crop(roi_abs = self.roi_abs, new_img = True)
        i_min, i_max = self.current_contrast_range()
    
        disp = cvtColor(bytescale(img.img, cmin = i_min, cmax = i_max),\
                                                            COLOR_GRAY2BGR)    
        if self.flow is None:
            print "Could not draw flow, no flow available"
            return
        lines = self.calc_flow_lines()
        
        if not in_roi:
            x0, y0, w, h = roi2rect(self.roi)
            ax.add_patch(Rectangle((x0, y0), w, h, fc = "none", ec = "c"))
    
        for (x1, y1), (x2, y2) in lines:
            line(disp, (x0+ x1, y0 + y1),\
                        (x0 + x2,y0 + y2),(0, 255, 255), 1)
            circle(disp, (x0 + x2, y0 + y2), 1, (255, 0, 0), -1)
        ax.imshow(disp)
        
        tit = "%s (delta t = %.2f s)" %(self.get_img_acq_times()[0],\
                                                            self.del_t)
        ax.set_title(tit, fontsize = 10)
        return ax, disp
        
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
            #flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
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
            
class OpticalFlowAnalysis(object):
    """A class for analysis of optical flow characteristics considering
    all images in an :class: ImgList` object. The analysis of the flow field 
    is based on one or more ROIs within the original images. These ROIs have to 
    be added manually using:: 
    
        self.add_roi(rect=None,id)
    
    """
    def __init__(self, lst = None, line = None, settings = None, imgPrepDict = None):
        self.line = None
        self.imgList = None
        
        self.optFlowEdit = None
        
        #:Main ROIs, i.e. ROIs for which the optical flow field will be determined
        #:(size needs to be large than 80x80 pixels)
        self.mainROI = None
        #: Sub ROIs: ROIs within main ROIs
        self.subRois=Bunch()
        
        self.meanFlowFieldInfo=None
        if lst:
            self.set_imglist(lst,imgPrepDict)        
        if line is not None:
            self.set_line(line)
        if settings is not None:
            self.optFlowEdit.add_settings(settings)
            self.optFlowEdit.change_current_settings_object(settings.id)
    
    def set_imglist(self, lst, imgPrepDict = None):
        """Set a deepcopy of the input list and if applicable, change image 
        preparation settings
        
        :param ImgListStatic lst: the image list
        :param dict imgPrepDict (None): img preparation settings
        
        """
        try:
            if lst.numberOfFiles>0:
                self.imgList=deepcopy(lst)
                self.optFlowEdit=self.imgList.optFlowEdit=OpticalFlowFarneback(id=self.imgList.id)
                if isinstance(imgPrepDict, dict):
                    self.imgList.update_img_prep_settings(imgPrepDict)
        except:
            raise TypeError(format_exc())

    def img_prep_settings(self):
        """Return info about the image preparation setting
        """
        return self.imgList.current_edit()
        
    def add_settings(self, optFlowSettings):
        self.optFlowEdit.add_settings(optFlowSettings)
    
    def change_current_settings_object(self, key):
        self.optFlowEdit.change_current_settings_object(key)
        self.imgList.update_img_prep_settings(self.optFlowEdit.settings.imgPrepSettings)
        
    def set_save_path(self, p):
        if exists(p):
            self.savePath=p
            
    def set_line(self, line):
        if not isinstance(line, LineOnImage):
            raise TypeError("Input is not a piscope.Processing.LineOnImage object")
        self.line=line
        
    def set_main_roi(self,rect):
        """Add one ROI for optical flow analysis
        """
        self.mainROI=self.rect_2_roi(rect)
    
    def _in_roi(self, subRoi, roi):
        """Test if subRoi lies in roi
        """
        s,r=subRoi,roi
        if not(s[0]>=r[0] and s[1]<=r[1] and s[2]>=r[2] and s[3]<=r[3]):
            raise ValueError("Sub-ROI exceeds borders of parent ROI")
        return 1
        
        
    def add_sub_roi(self, rect, id):
        """Add sub ROI to existing main ROI
        
        :param list rect: rectangle defining ROI
        :param str id: id of sub-roi (e.g. "zoom")
        
        If main ROI exists and sub-ROI is within main ROI then add subROI        
        """
        subRoi=self.rect_2_roi(rect)
        mainRoi=self.mainROI
        if self._in_roi(subRoi, mainRoi):
            self.subRois[id]=subRoi
        
    def rect_2_roi(self,rect,inverse=0):
        return self.optFlowEdit.settings.imgShapePrep.rect_2_roi(rect, inverse)
        
    def set_main_roi_from_line(self,**kwargs):
        if self.line is None:
            raise KeyError("No line found")
        self.line.set_roi(self.imgList.loadedImages.this.img,**kwargs)
        self.mainROI=self.line.roi()
        self.update_flow_roi(self.line.id)
    
    def update_flow_roi(self, roiId):
        self.optFlowEdit.set_roi(self.rect_2_roi(self.mainROI,1))
    
    def estimate_mean_flow(self, plotLengthProfile=0):
        """Estimates the mean values of the flow field along the line
        """
        p=self.optFlowEdit.get_main_flow_field_params()
        if p == 0:
            return 0
        lenThresh, mu, sigma, v, vErr, goodLens, goodAngles,vmax=p
        fx,fy=self.optFlowEdit.flow[:,:,0],self.optFlowEdit.flow[:,:,1]
        lx=self.line.get_line_profile(fx,key="roi")
        ly=self.line.get_line_profile(fy,key="roi")
        #totNum=len(lx)
        lens=sqrt(lx**2+ly**2)
        if plotLengthProfile:
            fig,ax=subplots(1,1)
            ax.plot(lens, ' x')
            ax.set_title("Flow length vector distribution along line")
        angles=rad2deg(arctan2(lx,-ly))
        cond1=lens>lenThresh
        #cond2=logical_and(mu-3*sigma<angles,mu+3*sigma>angles)
        cond2=logical_and(mu-sigma<angles,mu+sigma>angles)
        cond=cond1*cond2
        #get all lens on line pointing in the right direction and having
        #an acceptable length
        mask=lens[cond]
        if len(mask)>0:
#==============================================================================
#         if not len(mask) > 0.1*totNum or len(mask) < 20:
#             print ("Mean flow could not be estimated too little amount of"
#                 " datapoints on the line")
#             return 0
#==============================================================================
            fit = MultiGaussFit(datY=lens, id="Lengths on PCS")
            fit.set_noise_amplitude(lenThresh)
            if not fit.fit_multiple_gaussian():
                #and if this does not work, try to fit a single gaussian (based
                #on position of maximum count)
                fit.init_results()
                fit.init_data()
                fit.fit_single_gaussian()
            if fit.got_results():   
                fit.run_optimisation()
                mu1, sigma1,_,_=fit.get_main_gauss_info()
                mask=lens[logical_and(mu1-sigma1<fit.x,mu1+sigma1>fit.x)]
                print fit.gauss_info()
                #v,vErr=mask.mean(),mask.std()
                v,vErr=mask.max(),mask.std()
                print "\nSuccesfully estimated mean flow"
                print v, vErr, lenThresh
            else:
                print ("Mean flow velocity along PCS could not be determined")
                v, vErr=nan,nan
        else:
            v, vErr=nan,nan
            
        return lenThresh, mu, sigma, v, vErr, fit

    def flow_field_mean_analysis(self, lenThresh=4):
        """Histogram based analysis of optical flow in image time series of 
        `self.imgList`.
        """
        self.optFlowEdit.active=1
        self.imgList.goto_im(0)
        num=self.imgList.numberOfFiles
        #bad=FlowFieldAnalysisResults(self.mainROI)
        good=FlowFieldAnalysisResults(self.mainROI)
        h,w=self.optFlowEdit.flow.shape[:2]
        good.init_stacks(h,w,num-1)
        times=self.imgList.get_img_times()
        lastParams=self.optFlowEdit.get_main_flow_field_params()
        if lastParams == 0:
            print("Mean analysis of flow field failed, significant flow direction"
                " could not be initialised, gaussian fit insignificant, check"
                " optical flow input and contrast input settings for ROI: " +
                str(self.mainROI))
            return 0
        lenThresh, mu, sigma, v, vErr, goodLens, goodAngles, vmax=lastParams
#==============================================================================
#         blInfo=self.optFlowEdit.estimate_mean_displacement_from_blobs(mu=mu,\
#                                             sigma=sigma, lenThresh=lenThresh)
#         if blInfo is not 0:
#             v1, v1Err=blInfo[0], blInfo[1]
#         else:
#             v1,v1Err=nan,nan
#==============================================================================
        good.append_result(times[0],v,vErr,lenThresh, mu, sigma,vmax)
        good.stacks.angleImgs.set_img(self.optFlowEdit.get_flow_angle_image(),0)
        good.stacks.lenImgs.set_img(self.optFlowEdit.get_flow_vector_length_image(),0)
        for k in range(1,num-1):
            self.imgList.next_im()
            good.stacks.angleImgs.set_img(self.optFlowEdit.get_flow_angle_image(),k)
            good.stacks.lenImgs.set_img(self.optFlowEdit.get_flow_vector_length_image(),k)
            lastParams=self.optFlowEdit.get_main_flow_field_params()
            if lastParams !=0:
                lenThresh, mu, sigma, v, vErr, goodLens, goodAngles, vmax=lastParams
#==============================================================================
#                 blInfo=self.optFlowEdit.estimate_mean_displacement_from_blobs(\
#                                         mu=mu, sigma=sigma, lenThresh=lenThresh)
#                 if blInfo is not 0:
#                     v1, v1Err=blInfo[0], blInfo[1]
#                 else:
#                     v1,v1Err=nan,nan
#==============================================================================
            else:
                print "Failed to estimate mean flow at image num: " + str(k)
                lenThresh, mu, sigma, v, vErr,vmax =[nan,nan,nan,nan,nan,nan]
            good.append_result(times[k],v,vErr,lenThresh, mu, sigma,vmax)
        good.make_pandas_series()
        self.meanFlowFieldInfo=good
        return good
        
    def draw_current_flow(self, includeBlobs=1, disp=1):
        if disp:
            fig=figure(figsize=(18,8))
        else:
            fig=Figure(figsize=(18,8))
        axes=[]
        axes.append(fig.add_subplot(1,2,1))
        axes.append(fig.add_subplot(1,2,2))
        axes[0], img=self.optFlowEdit.draw_flow(showInROI=0, ax=axes[0])
        axes[1], roiImg=self.optFlowEdit.draw_flow(showInROI=1, ax=axes[1])
        if includeBlobs:
            self.optFlowEdit.draw_blobs(ax=axes[1])
        if isinstance(self.line, LineOnImage):
            l=self.line
            axes[0].plot([l.start[0],l.stop[0]], [l.start[1],l.stop[1]],'co-')
            roi=l.roi()
            if roi is not None:
                dx,dy=roi[1]-roi[0], roi[3]-roi[2]
                axes[0].add_patch(Rectangle((roi[0],roi[2]),dx,dy,fc="none",ec="c"))
        
            x0,y0=self.optFlowEdit.settings.imgShapePrep.map_coordinates(l.start[0], l.start[1])
            x1,y1=self.optFlowEdit.settings.imgShapePrep.map_coordinates(l.stop[0], l.stop[1])
            axes[1].plot([x0,x1], [y0,y1],'co-')
            
        axes[0].set_xlim([0,img.shape[1]])
        axes[0].set_ylim([img.shape[0],0])
        axes[1].set_xlim([0,roiImg.shape[1]])
        axes[1].set_ylim([roiImg.shape[0],0])
        #axis('image')
        s=self.imgList.current_time().strftime("%Y.%m.%d %H:%M:%S")
#==============================================================================
#         try:
#==============================================================================
        lenThresh, mu, sigma, v, vErr, _,_,vmax=self.\
                    optFlowEdit.get_main_flow_field_params()
        s=(s+"\nMean displacement: " + "{:.1f}".format(v) + " (+/- " + "{:.1f}".format(vErr) + "), Max: " + "{:.1f}".format(vmax) + "pix\n"
        "Mean direction: " + "{:.1f}".format(mu) + " (+/- " + "{:.1f}".format(sigma) + ") deg")
#==============================================================================
#         except:
#             raise ValueError()
#==============================================================================
        axes[0].set_title(s,fontsize=12)
        return fig, axes
    
    def determine_and_save_all_flow_images(self, folderName, startNum=None, stopNum=None):
        if startNum is None:
            startNum=0
        if stopNum is None:
            stopNum=self.imgList.numberOfFiles-1
        if not self.savePath:
            print "Error"
            
            return
        p=self.savePath + folderName + "/"
        if exists(p):
            print "Path already exists, choose another name"
            return
        mkdir(p)
        self.optFlowEdit.active=1
        self.imgList.goto_im(startNum)
        for k in range(startNum+1,stopNum):
            fig, ax=self.draw_current_flow(disp=1)
            fig.savefig(p+str(k)+".png")
            close(fig) 
            del fig, ax
            self.imgList.next_im()
        
        

class FlowFieldAnalysisResults(object):
    """This object stores results (mean direction +/-, mean length, thresholds
    etc...) for an image time series (:class:`ImgList`) in a certain ROI
    of the image stack
    """
    def __init__(self, roi):
        self.roi=roi
        self.times=[]
        self.stacks=Bunch({"lenImgs" : ImgStack("lenImgs"),
                           "angleImgs": ImgStack("angleImgs")})
                           
        self.results=Bunch({"lenThreshs"    :   [],
                            "meanDirs"      :   [],
                            "meanDirErrs"   :   [],
                            "meanVelos"     :   [],
                            "meanVeloErrs"  :   [],
                            "maxVelos"      :   []})
        
        self.pandasSeries=Bunch()
        
        
    def init_stacks(self,h,w,d):
        for stack in self.stacks.values():
            stack.init_stack(h,w,d)
    
    def plot_overview(self):
        fig, axes=subplots(2,1)
        errs=self.pandasSeries.meanDirErrs
        self.pandasSeries.meanDirs.plot(ax=axes[0],yerr=errs.values)        
        axes[0].set_title("Mean flow direction [deg]")
        errs=self.pandasSeries.meanVeloErrs
        self.pandasSeries.meanVelos.plot(ax=axes[1], label="Mean (histogram analysis)",yerr=errs.values)
        blobErrs=self.pandasSeries.blobVeloErrs.values
        self.pandasSeries.blobVelos.plot(ax=axes[1], label="Mean (Blob analysis)",yerr=blobErrs)
        axes[1].set_title("Flow diplacement length [pix]")
        self.pandasSeries.maxVelos.plot(ax=axes[1], label="Max")
        axes[1].legend(loc='best', fancybox=True, framealpha=0.5, fontsize=10)
        
    def append_result(self,time,v,verr,lenThresh, mu, sigma, vmax):
        self.times.append(time)
        self.results.lenThreshs.append(lenThresh)
        self.results.meanVelos.append(v)
        self.results.meanVeloErrs.append(verr)
        self.results.maxVelos.append(vmax)
        self.results.meanDirs.append(mu)
        self.results.meanDirErrs.append(sigma)

    
    def make_pandas_series(self):
        """Make pandas time series objects of all results
        """
        for key,val in self.results.iteritems():
            self.pandasSeries[key]=Series(val,self.times)
    def prepare_pandas_series(self,):
        """
        """
        
#==============================================================================
# class WindRetrievalCollection(object):
#     """The base class for storing any information about the wind field and
#     standard algorithms to retrieve wind information using different methods. 
#     
#     In the current version, this is mainly:
#     
#         1. Manually setting average displacement and average orientation angle
#         #. Optical flow farneback analysis
#             1. Use output as is 
#             #. Do meanFlowField analysis (link..)
#     
#     """
#     def __init__(self, imgList=None, measGeometry=None):
#         self.imgList = imgList
#         self.measGeometry = measGeometry
#         
#         self.optFlowAnalysis = OpticalFlowAnalysis()
#         #this dictionary sets the image preparation info for optical flow
#         #calculations (see also :func:`self.set_imglist`)
#         self.optFlowImgPrep = Bunch([("DarkCorr"    ,   1),
#                                      ("Blurring"    ,   1)])
#         
#         
#         #:In this dictionary, global displacements (however measured, e.g. in
#         #: GUI) can be added, please use :func:`self.add_glob_displacement` to
#         #: do so. 
#         self._glob_displacements = Bunch()
#         self.warnings=["No warning"]
#     
#     def get_opt_flow_settings(self):
#         """Get current optical flow settings
#         """
#         try:
#             return self.optFlowAnalysis.optFlowEdit.settings
#         except:
#             msg=("Could not retrieve optical flow settings, OpticalFlowAnalysis "
#                 "tool might not be set")
#             self._warning(msg)
#             return 0
#             
#     def set_opt_flow_settings(self, settings):
#         """Set settings object for optical flow calculations
#         """
#         try:
#             if self.optFlowAnalysis.optFlowEdit.moreSettings.has_key(settings.id):
#                 msg=("Opt flow settings with id: " + str(settings.id) + " were "
#                     "overwritten")
#                 self._warning(msg)
#             self.optFlowAnalysis.add_settings(settings)
#             self.optFlowAnalysis.change_current_settings_object(settings.id)
#         except:
#             msg=("Could not retrieve optical flow settings, OpticalFlowAnalysis "
#                 "tool might not be set")
#             self._warning(msg)
#             return 0
#         
#     def add_glob_displacement(self,timeStamp, delt, len, lenErr, orientation,\
#                                                                 orientationErr):
#         """Set global values for displacement vector
#         
#         :param datetime timeStamp: datetime to which the displacment corresponds
#         :param float delt: time difference between the two images used to 
#             measure displacement
#         :param float len: length of displacement in pix
#         :param float len: uncertainty of length of displacement in pix
#         :param float orientation: orientation angle in deg 
#             0: upwards (-y direction)
#             180 and -180: downwards (+ y direction)
#             90: to the right (+ x direction)
#             -90: to the left (- x direction)
#         
#         Writes the data into `self._glob_displacements` in the following format::
#         
#             self._glob_displacements[timeStamp]=[delt, len, lenErr, orientation,\
#                 orientationErr]
#         
#         .. note::
#         
#             if you use this, make sure, to use the right datatypes, no control
#             of input performed here
#             
#         """
#         self._glob_displacements[timeStamp]=[delt, len, lenErr, orientation,\
#                                                                 orientationErr]
#     
#     def get_wind_info_glob(self, timeStamp=None):            
#         """Get global information about wind velocity and vetor orientation from
#         information in `self._glob_displacements`
#         :param datetime timeStamp: the time at which the data is supposed to be
#             retrieved (searches closest available info in 
#             `self._glob_displacements` dictionary)
#         """
#         if timeStamp is None:
#             try:
#                 timeStamp=self.imgList.current_time()
#             except:
#                 self.warning()
#         t0=min(self._glob_displacements.keys(), key=lambda x: abs(x - timeStamp))
#         dx=self.measGeometry.calcs["pixLengthX"] #the pixel lenght in m
#         info=self._glob_displacements[t0]
#         v, vErr=float(info[1])*dx/info[0],float(info[2])*dx/info[0]
#         return WindVector2D(v, vErr, info[3], info[4])
#     
#     def update_imprep_optflow(self, dictLike):
#         """Update valid entries in image preparation dict for optical flow 
#         calculations
#         :param dict dictLike: dictionary with variables
#         
#         """
#         for key, val in dictLike.iteritems():
#             if self.optFlowImgPrep.has_key(key):
#                 self.optFlowImgPrep[key]=val
#                 
#     def set_imglist(self,imgList, imgPrepDict=None):
#         """Try setting imgList object used for wind field analysis
#         """
#         try: 
#             if isinstance(imgPrepDict, dict):
#                 self.update_imprep_optflow(imgPrepDict)
#             if imgList.numberOfFiles>0:
#                 self.imgList=imgList
#                 self.optFlowAnalysis.set_imglist(imgList,self.optFlowImgPrep)
#             return 1
#         except:
#             msg=("Could not set imgList, probably wrong input type or empty list")
#             self._warning(msg)
#             return 0
# 
#     def set_and_init_pcs_line_optflow(self, line, **kwargs):
#         """Set the line of interest for optical flow calculations
#         
#         :param LineOnImage line: line object
#         :param **kwargs: accepted keys (addTop, addBottom, addLeft, addRight)
#         
#         .. note::
#         
#             a ROI is determined automatically such that it includes the 
#             rectangle spanned by the line on the image using 
#             :func:`LineOnImage.set_roi`, **kwargs (expand line borders) are 
#             passed to :func:`LineOnImage.set_roi`
#         
#         """
#         self.optFlowAnalysis.set_line(line)
#         self.optFlowAnalysis.set_main_roi_from_line(**kwargs)
#         self.optFlowAnalysis.update_flow_roi(line.id)
#         
#         
#     def set_meas_geometry(self, measGeometry):
#         """Try setting :class:`piscope.Utils.MeasGeometry` object used 
#         for wind field analysis
#         """
#         try: 
#             if not measGeometry.basicDataAvailable:
#                 msg=("Could not set measGeometry, basic data not available")
#                 self._warning(msg)
#                 return 0
#                 
#             self.measGeometry=measGeometry
#             return 1
#         except:
#             msg=("Could not set measGeometry, probably wrong input type")
#             self._warning(msg)
#             return 0
#            
#     def _warning(self,msg):
#         self.warnings.append(msg)
#         print msg
#         
# class WindVector2D(object):
#     """Object representing a 2D wind vector"""
#     def __init__(self,v,vErr,orientation, orientationErr,unit="m/s"):
#         self.v=v
#         self.vErr=vErr
#         self.orientation=orientation
#         self.orientationErr=orientationErr
#         self.unit=unit
#     
#     def __call__(self):
#         """On call, return velocity information
#         """
#         return (self.v,self.vErr)
#         
#     def __str__(self):
#         s = "Wind vector info\n------------------\n"
#         s = s + "v (+/-): " + str(self.v) + "+/-" + str(self.vErr) + " " + self.unit + "\n"
#         s = s + "Angle [deg] (+/-) " + str(self.orientation) + "+/-" + str(self.orientationErr) + "\n"
#         return s
#==============================================================================