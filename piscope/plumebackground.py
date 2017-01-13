# -*- coding: utf-8 -*-
"""
Test script for development of new background image treatment

Created on Tue Sep 06 14:38:54 2016

@author: jg
"""
from numpy import polyfit, poly1d, linspace, logical_and, log, full, argmin,\
    gradient, average, nan, exp, ndarray, arange, ones
from matplotlib.patches import Rectangle
from matplotlib.pyplot import GridSpec, figure, subplots_adjust, subplot,\
    subplots, setp
import matplotlib.colors as colors
from collections import OrderedDict as od
from scipy.ndimage.filters import gaussian_filter

from .exceptions import ImgMetaError
from .image import Img
from .processing import LineOnImage
from .optimisation import PolySurfaceFit
from .helpers import shifted_color_map

class PlumeBackgroundModel(object):
    """Class for plume background modelling and tau image determination"""
    def __init__(self, bg_raw = None, plume_init = None, **kwargs):
        """Initialisation of object
        
        :param (Img) bg_raw: sky background radiance raw image data
        :param (Img) plume_init: initial plume image data (is used to
            estimate default clear sky areas for bg modelling)
        :param **kwargs: additional class items (e.g. for modelling, valid keys
            are all keys in self.__dict__.keys())
        """        
        if isinstance(bg_raw, ndarray):
            bg_raw = Img(bg_raw)
        if isinstance(plume_init, ndarray):
            plume_init = Img(plume_init)
            
        self._current_imgs = {"plume"    :   plume_init,
                              "bg_raw"   :   bg_raw,
                              "tau"      :   None}        
        
        #: Correction mode
        self.CORR_MODE = 1
        
        self.vign_mask = None
        
        #: settings for poly surface fit (corr mode: 0)
        self.surface_fit_mask = None
        self.surface_fit_pyrlevel = 4
        self.surface_fit_polyorder = 2
            
        #: Rectangle for scaline of background image
        #: corr modes: 1 - 6
        self.scale_rect = None
        
        #: Rectangle for linear based correction of vertical gradient
        #: corr modes: 2, 4
        self.ygrad_rect = None
        
        #: Settings for quadratic correction of vertical gradient (along line)
        #: corr modes: 3, 5, 6
        self.ygrad_line_colnum = None # detector column of vertical line
        self.ygrad_line_polyorder = 2
        self.ygrad_line_startrow = 0 # start row for profile fit
        self.ygrad_line_stoprow = None # stop row for profile fit
        self.ygrad_line_mask = None # mask specifying rows for profile fit
        
        #: Rectangle for linear based correction of horizontal gradient (applied
        #: before ygradient correction is performed)
        #: corr modes: 4, 5
        self.xgrad_rect = None
        
        #: Settings for quadratic correction of horizontal gradient (along line)
        #: corr modes: 6
        self.xgrad_line_rownum = None
        self.xgrad_line_polyorder = 2
        self.xgrad_line_startcol = 0
        self.xgrad_line_stopcol = None
        self.xgrad_line_mask = None
        
        #initialisations        
        self.update(**kwargs)        

        if isinstance(plume_init, Img):
            self.guess_missing_settings(plume_init.img)
            self.surface_fit_mask = ones(plume_init.img.shape,\
                                                         dtype = bool)
    
    def get_current(self, key = "tau"):
        """Returns current image, specify type via input key
        
        :param str key: choose from plume, bg_raw, bg_model, tau
        """
        return self._current_imgs[key]
    
    @property
    def current_plume_background(self):
        """Retrieve the current plume background from modelled tau image and
        plume image, i.e::

            bg_img = Img(exp(tau_img) * plume_img)
        """
        return Img(exp(self._current_imgs["tau"]) *\
                                self._current_imgs["plume"])
        
    def check_settings(self):
        for value in self.__dict__.values():
            if value is None:
                return False
        return True
        #return ~any([x is None for x in self.__dict__.values()])
    
    def update(self, **kwargs):
        """Update class attributes
        :param **kwargs:
        """
        for k, v in kwargs.iteritems():
            self.__setitem__(k, v)
        
    def _check_rect(self, rect, img):
        """Check if rect is not None and if it is within image borders
        :param list r: rectangular area ``[x0, y0, x1, y1]``
        :param ndarray img: exemplary image
        :return bool: 
        """
        if rect == None:
            return False
        h, w = img.shape
        if rect[0] < 0 or rect[1] < 0 or rect[2] >= w or rect[3] >= h:
            return False
        return True
   
    def set_vignetting_mask(self, mask):
        """Sets mask to correct for vignetting"""
        plume = self.get_current("plume")
        bg = self.get_current("bg_raw")
        
        if isinstance(plume, Img):
            if not mask.shape == plume.img.shape:
                raise ValueError("Could not set vignetting mask in background "
                    "model, dimension mismatch with current plume image")
        elif isinstance(bg, Img):
            if not mask.shape == bg.img.shape:            
                raise ValueError("Could not set vignetting mask in background "
                    "model, dimension mismatch with current bg image")
        self.vign_mask = mask
        
    def set_vignetting_mask_from_bg(self, sigma_blur = 4):
        """Determines a vignetting mask from current background image
        
        :param int sigma_blur: width of gaussian kernel used for blurring of 
            ``self.bg_raw`` before mask is determined
        :return ndarray: the actual mask wich was set in ``self.vign_mask``
            
        .. note::
        
            The sky background radiance might vary over the image area, this is
            superimposed to the mask
            
        """
        try:
            bg = self.get_current("bg_model")
        except:
            bg = self.get_current("bg_raw")
        if not isinstance(bg, Img):
            print "Background image not available"
            return False
        bg_blur = gaussian_filter(bg.img, sigma_blur)
        mask = bg_blur / bg_blur.max()
        self.set_vignetting_mask(mask)
        return mask
            
    def guess_missing_settings(self, plume_img):
        """Checks current settings and inits defaults based on image dimension
        
        :param Img plume_img: input plume image (should be representative for
            a whole dataset)        
            
        Based on the input plume image, the clear sky reference areas for sky
        radiance image based tau modelling are estimated, i.e.::
        
            1. The rectangle areas for scaling and linear gradient corrections:
                ``self.scale_rect, self.ygrad_rect, self.xgrad_rect``
            2. Information for horizontal and vertical profile lines 
                for quadratic gradient corrections: 
                ``self.ygrad_line_colnum, self.xgrad_line_rownum`` 
                (i.e. positions and start / stop pixel coordinates)
                
        The estimation is performed based on a brightness analysis for left and
        right image area. 
        
        .. todo::
        
            1. Set as integers
        """   
        if not isinstance(plume_img, Img):
            raise TypeError("Invalid, input type: need Img object...")
        plume = plume_img.img
        if self.check_settings():
            return
        if self.surface_fit_mask == None:
            self.surface_fit_mask = full(plume.shape, True, dtype = bool)
        h, w = plume.shape
        
        res = find_sky_reference_areas(plume)
        if self.ygrad_line_colnum == None:
            self.ygrad_line_colnum = res["ygrad_line_colnum"]
            self.ygrad_line_stoprow = res["ygrad_line_stoprow"]
            self.ygrad_line_startrow = res["ygrad_line_startrow"]
        if self.xgrad_line_rownum == None:
            self.xgrad_line_rownum = res["xgrad_line_rownum"]
            self.xgrad_line_startcol = res["xgrad_line_startcol"]
            self.xgrad_line_stopcol = res["xgrad_line_stopcol"]
        if not self._check_rect(self.scale_rect, plume):
            self.scale_rect = res["scale_rect"]
        if not self._check_rect(self.ygrad_rect, plume):
            self.ygrad_rect = res["ygrad_rect"]
        if not self._check_rect(self.xgrad_rect, plume):
            self.xgrad_rect = res["xgrad_rect"]
        
        
    def bg_from_poly_surface_fit(self, plume, mask = None, polyorder = 2,\
                                                                pyrlevel = 4):
        """Applies poly surface fit to plume image for bg retrieval
        
        :param ndarray plume: plume image
        :param ndarray mask (None): mask specifying gas free areas (if None, 
            use all pixels)
        :param int polyorder (2): order of polynomial used for fit
        :param int pyrlevel (4): scale space level in which fit is performed (
            e.g. 4 => image size for fit is reduced by factor 2^4 = 16)
        :return tuple: 1st entry: fitted background image
            second: ``PolySurfaceFit`` object 
        
        Example::
        
            from piscope.PlumeBackground import ModelPlumeBackground as bgModel
            from numpy import ones, full, log
            from matplotlib.pyplot import imshow
            
            boringExampleImg = ones((512,512)) 
            boringExampleImg[100:200,150:250] = 15
            
            testMask = full((512,512), True, dtype = bool)
            testMask[boringExampleImg > 2] = False #exclude values exceeding 2
            
            m = bgModel()
            bg = m.bg_from_poly_surface_fit(boringExampleImg,\
                mask = testMask, polyorder = 1,\
                pyrlevel = 3)
            #plot tau image
            imshow(log(bg/plume))
        """
        #update settings from input keyword args
        if mask == None or not mask.shape == plume.shape:
            print ("Warning: invalid mask for poly surface fit (bg modelling)"
                    " considering all image pixels for retrieval")
            mask = full(plume.shape, True, dtype = bool)   
    
        fit = PolySurfaceFit(plume, mask.astype(float), polyorder = polyorder,\
                                                pyrlevel = pyrlevel)
        return (fit.model, fit)
        
    def subtract_tau_offset(self, tau0, rect):
        """Subtract offset in tau image with based on mean val input rectangle
        
        Performs background scaling after tau image was determined
        
        :param ndarray tau0: initial tau image
        :param list rect: rectangular area ``[x0, y0, x1, y1]``
        :return ndarray: modified tau image
        """
        offs,_ = _mean_in_rect(tau0, rect)
        return tau0 - offs
    
    def get_tau_image(self, plume_img, bg_img = None, **kwargs):
        """Determine current tau image for input plume image
        
        :param Img plume_img: plume image 
        :param Img bg_img: sky radiance image (for ``self.CORR_MODE = 1 - 5``)
        :param **kwargs: keyword arguments for updating current settings
            (valid input keywords (strings): CORR_MODE, ygrad_rect, 
            ygrad_line_colnum, ygrad_line_startrow, ygrad_line_stoprow
        :returns: - :class:`Img`, plume tau image    
          
        """
        if not isinstance(plume_img, Img):
            raise TypeError("Invalid, input type: need Img object...")
        #update current settings
        for k, v in kwargs.iteritems():
            self.__setitem__(k, v)
        if not plume_img.edit_log["darkcorr"]:
            print ("Warning in PlumeBackgroundModel: plume image is not "
             " corrected for dark current" )
        if plume_img.is_tau:
            raise IOError("Input image is already tau image")
        plume = plume_img.img
        if self.CORR_MODE != 0:
            if not isinstance(bg_img, Img):
                bg_img = self.get_current("bg_raw")
            if not bg_img.edit_log["darkcorr"] == plume_img.edit_log["darkcorr"]:
                raise ImgMetaError("Error in PlumeBackgroundModel: plume and"
                    " background image have different dark corr states")
            bg = bg_img.img

        tau = None
        mode = self.CORR_MODE
        if mode == 0: #no sky radiance image, poly surface fit
            (bg, fit) = self.bg_from_poly_surface_fit(plume,\
                self.surface_fit_mask, self.surface_fit_polyorder,\
                                            self.surface_fit_pyrlevel)
            tau = log(bg / plume)
        
        else:
            bg_norm = scale_bg_img(bg, plume, self.scale_rect)
            tau = log(bg_norm / plume)
            if mode > 1:
                tau = self.correct_tau_curvature_ref_areas(tau)
            
        tau_img = plume_img.duplicate()
        tau_img.meta["bit_depth"] = nan
        tau_img.edit_log["is_tau"] = 1
        tau_img.img = tau
        self.set_current_images(plume_img, bg_img, tau_img)
        
        return tau_img
    
    def correct_tau_curvature_ref_areas(self, tau_init):
        """Correct an initial tau image such that tau is zero in sky reference
        areas
        
        The method used is based on the current ``CORR_MODE``. This method only
        applies for correction modes 2-6.
        
        :param (ndarray, Img) tau_init: inital tau image
        
        """
        mode = self.CORR_MODE
        tau = None
        
        if not 2 <= mode <= 6:
            raise ValueError("This method only works for background model"
                "modes (param CORR_MODE) 2-6")
        if isinstance(tau_init, Img):
            tau_init = tau_init.img
        if mode == 2:
                 tau = corr_tau_curvature_vert_two_rects(tau_init,\
                                             self.scale_rect, self.ygrad_rect)
        elif mode == 3:
            tau, _ = corr_tau_curvature_vert_line(tau_init,\
                self.ygrad_line_colnum, self.ygrad_line_startrow,\
                    self.ygrad_line_stoprow, self.ygrad_line_mask,\
                                            self.ygrad_line_polyorder)

        elif mode == 4:
            tau = corr_tau_curvature_vert_two_rects(tau_init,\
                                            self.scale_rect, self.ygrad_rect)
            tau = corr_tau_curvature_hor_two_rects(tau,\
                                            self.scale_rect, self.xgrad_rect)
            
        elif mode == 5:
            tau, _ = corr_tau_curvature_vert_line(tau_init,\
                self.ygrad_line_colnum, self.ygrad_line_startrow,\
                    self.ygrad_line_stoprow, self.ygrad_line_mask,\
                                            self.ygrad_line_polyorder)
            tau = corr_tau_curvature_hor_two_rects(tau,\
                self.scale_rect, self.xgrad_rect)
            

        elif mode == 6:
            tau, _ = corr_tau_curvature_vert_line(tau_init,\
                    self.ygrad_line_colnum, self.ygrad_line_startrow,\
                        self.ygrad_line_stoprow, self.ygrad_line_mask,\
                                                self.ygrad_line_polyorder)
            tau, _ = corr_tau_curvature_hor_line(tau,\
                    self.xgrad_line_rownum, self.xgrad_line_startcol,\
                        self.xgrad_line_stopcol, self.xgrad_line_mask,\
                                                self.xgrad_line_polyorder)
        return tau
        
    def get_aa_image(self, plume_on, plume_off, bg_on, bg_off):
        """Method to retrieve apparent absorbance image from on and off imgs
                
        :param Img plume_on: on band plume image
        :param Img plume_off: off band plume image
        :param Img bg_on: on band sky radiance image
        :param Img bg_off: off band sky radiance image
        
        """            
        aa_init = log(bg_on.img / plume_on.img) -\
                            log(bg_off.img / plume_off.img)
                        
        aa = self.correct_tau_curvature_ref_areas(aa_init)
        
#==============================================================================
#         aa2_init = log(plume_off.img * bg_on.img / (plume_on.img * bg_off.img))
#         aa2 = self.correct_tau_curvature_ref_areas(aa2_init)
#==============================================================================
        aa_img = plume_on.duplicate()
        aa_img.meta["bit_depth"] = nan
        aa_img.edit_log["is_tau"] = 1
        aa_img.img = aa
        #self.set_current_images(plume_img, bg_img, tau_img)
        
        return aa_img#, Img(aa2)
        
                                
        
    def _prep_img_type(self, img):
        """Checks input images and converts them into ndarrays if they are Img"""
        if isinstance(img, Img):
            return img.img
        return img
        
    def set_current_images(self, plume, bg_raw, tau):
        """Write the input images into ``self._current_imgs`` dict
        
        This method is called at the end of :func:`get_tau_image`
        
        :param Img plume: current plume image
        :param Img bg_raw: input background image
        :param Img tau: the modelled tau image
        
        """
        self._current_imgs["plume"] = plume
        self._current_imgs["bg_raw"] = bg_raw
        self._current_imgs["tau"] = tau
            
    """Plotting"""
    def plot_sky_reference_areas(self, plume):
        """Plot the current sky ref areas into a plume image"""
        d = self.sky_ref_areas_to_dict()
        return plot_sky_reference_areas(plume, d)
        
    def plot_tau_result(self, tau_img = None, tau_min = None, tau_max = None,\
                                    edit_profile_labels = True, **add_lines):
        """Plot current tau image including all gas free reference sections and 
        the horizontal and vertical profile line
        
        :param Img tau_img: the tau image to be displayed
        :param float tau_min: lower tau boundary for colormap
        :param float tau_max: upper tau boundary for colormap
        :param bool edit_profile_labels: beta version of smart layout for 
            axis labels from profile subplots
        :param **kwargs: 
            - additional lines to be plotted, e.g.:: 
                pcs = [300, 400, 500, 600]
        """
        tau = tau_img
        if not isinstance(tau, Img):
            tau = self._current_imgs["tau"]
        if not isinstance(tau, Img):
            raise AttributeError("No tau image available in background model")
            
        if tau_max == None:
            tau_max = tau.max()
        if tau_min == None:
            tau_min = - tau_max
            
        h0, w0 = tau.shape
        cmap = shifted_color_map(tau_min, tau_max)
        fig = figure()
        gs = GridSpec(2, 2, width_ratios = [w0, w0 * .3],\
                            height_ratios = [h0 * .3, h0])
        ax = [subplot(gs[2]),]
        ax.append(subplot(gs[3]))
        ax.append(subplot(gs[0]))
        
        if self.CORR_MODE == 0:
            ax.append(subplot(gs[1]))
            palette = colors.ListedColormap(['white', 'lime'])
            norm = colors.BoundaryNorm([0, .5, 1], palette.N)
    
            ax[3].imshow(self.surface_fit_mask, cmap = palette, norm = norm,\
                                                                    alpha = .7)
            ax[3].set_title("Mask", fontsize = 10)
            ax[3].set_xticklabels([])
            ax[3].set_yticklabels([])
        
        ax[0].imshow(tau.img, cmap = cmap, vmin = tau_min, vmax = tau_max)
        
        ax[0].plot([self.ygrad_line_colnum,self.ygrad_line_colnum], [0, h0],\
                                            "-b", label = "vert profile")
        ax[0].plot([0, w0],[self.xgrad_line_rownum,self.xgrad_line_rownum],\
                                            "-c", label = "hor profile")
        for k, l in add_lines.iteritems():
            ax[0].plot([l[0],l[2]],[l[1],l[3]], "--",c = "g", label = k)
    
        ax[0].set_xlim([0, w0 - 1])
        ax[0].set_ylim([h0 - 1, 0])
        
        
        xs, ys, ws, hs = _roi_coordinates(self.scale_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="g",fc="none",\
                                                label = "scale_rect"))
        
        xs, ys, ws, hs = _roi_coordinates(self.ygrad_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="g",fc="none",\
                                                label = "ygrad_rect"))
        
        xs, ys, ws, hs = _roi_coordinates(self.xgrad_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="g",fc="none",\
                                                label = "xgrad_rect"))
                                                
        ax[2].set_xticklabels([])
        ax[1].set_yticklabels([])
        
        
        
        #plot vertical profile
        lvert = LineOnImage(self.ygrad_line_colnum, 0,\
                        self.ygrad_line_colnum, h0 - 1, id = "vert")
        p_vert = lvert.get_line_profile(tau.img) 
            
        ax[1].plot(p_vert, arange(0, len(p_vert), 1), "-b" ,\
                                            label = "vert profile")
        ax[1].yaxis.tick_right()   
        ax[1].set_ylim([h0 - 1, 0])
        setp( ax[1].xaxis.get_majorticklabels(), rotation = 15)
        ax[1].yaxis.tick_right()   
        
        #plot horizontal profile
        line_hor = LineOnImage(0, self.xgrad_line_rownum, w0 - 1,\
                                    self.xgrad_line_rownum, id = "hor")
        p_hor = line_hor.get_line_profile(tau.img)
        ax[2].plot(arange(0, len(p_hor), 1), p_hor, "-c" , label = "hor profile")
        #ax[2].get_yaxis().set_ticks(horYLabels)
        #ax[2].set_ylim([-.05,.25])
        ax[2].set_xlim([0, w0 - 1])
    
        subplots_adjust(wspace = 0.02, hspace =  0.02)
        ax[2].axhline(0, ls = "--", color = "k")
        ax[1].axvline(0, ls = "--", color = "k")
        
        if edit_profile_labels:
            _range = p_vert.max() - p_vert.min()
            lbls = [0, p_vert.max() - _range * .05]
            if p_vert.min() < 0 and abs(p_vert.min())/p_vert.max() > 0.5:
                lbls.insert(0, p_vert.min() + _range*.05)
            ax[1].get_xaxis().set_ticks(lbls)
            lbl_str = ["%.2f" %lbl for lbl in lbls]
            ax[1].set_xticklabels(lbl_str)                
            
            _range = p_hor.max() - p_hor.min()
            lbls = [0, p_hor.max() - _range*.05]
            if p_hor.min() < 0 and abs(p_hor.min())/p_hor.max() > 0.5:
                lbls.insert(0, p_hor.min() + _range *.05)
            ax[2].get_yaxis().set_ticks(lbls)
            lbl_str = ["%.2f" %lbl for lbl in lbls]
            ax[2].set_yticklabels(lbl_str)         
            
        ax[1].set_xlabel(r"$\tau$", fontsize = 16)
        ax[2].set_ylabel(r"$\tau$", fontsize = 16)  
        fig.suptitle("CORR_MODE: %s" %self.CORR_MODE, fontsize = 16)
        ax[0].legend(loc = "best", fancybox = True, framealpha = 0.5,\
                                                            fontsize = 10)
        return fig
    """Helpers"""
    def sky_ref_areas_to_dict(self):
        """Create a dictionary with the current sky reference area settings"""
        results = {}

        results["ygrad_line_colnum"] = self.ygrad_line_colnum
        results["ygrad_line_stoprow"] = self.ygrad_line_stoprow
        results["ygrad_line_startrow"] = self.ygrad_line_startrow
        
        results["xgrad_line_rownum"] = self.xgrad_line_rownum
        results["xgrad_line_startcol"] = self.xgrad_line_startcol
        results["xgrad_line_stopcol"] = self.xgrad_line_stopcol
        results["scale_rect"] = self.scale_rect
        results["ygrad_rect"] = self.ygrad_rect
        results["xgrad_rect"] = self.xgrad_rect
        return results
            
    @property
    def mode_info(self):                         
        return od( [[0 , "No additional BG image: poly surface fit using plume"
                        " image pixels specified with mask"],
                    [1 , "Scaling of bg image in rect scale_rect"],
                    [2 , "Scaling (mode 1, scale_rect) and linear y gradient "
                        "correction using rects scale_rect and ygrad_rect"],
                    [3 , "Scaling (mode 1, scale_rect) and quadratic y "
                         "gradient correction using vertical profile line"],
                    [4 , "Like 2, including linear x gradient correction using "
                            "rect xgrad_rect"],
                    [5 , "Like 3, including linear x gradient correction using "
                            "rect xgrad_rect"],
                    [6 , "Like 3, including quadratic x gradient correction "
                            "using horizontal profile line"]])
    
    def print_mode_info(self):
        """Print information about the different correction modes"""
        print "Available modes for automatic plume background retrieval"
        for k, v in self.mode_info.iteritems():
            print "Mode %s: %s" %(k, v)
        
    def __setitem__(self, key, value):
        """Update class item"""
        if self.__dict__.has_key(key):
            print "Updating %s in background model" %key
            self.__dict__[key] = value
            
    def __call__(self, plume, bg, **kwargs):
        return self.get_model(plume, bg, **kwargs)

def _mean_in_rect(img_array, rect = None):
    """Helper to get mean and standard deviation of pixels within rectangle
    
    :param ndarray imgarray: the image data
    :param rect: rectanglular area ``[x0, y0, x1, y1]` where x0 < x1, y0 < y1
    """
    if rect is None:
        sub = img_array
    else:
        sub = img_array[rect[1] : rect[3], rect[0] : rect[2]]
    return sub.mean(), sub.std()

def scale_bg_img(bg, plume, rect):
    """Normalise background image to plume image intensity in input rect
    
    :param (ndarray, Img) bg: background image
    :param (ndarray, Img) plume: plume image
    :param list rect: rectangular area ``[x0, y0, x1, y1]``
    :return ndarray: modified background image
    """
    #extract data if input is image
    if isinstance(bg, Img):
        bg = bg.img
    if isinstance(plume, Img):
         plume = plume.img
    #bg, plume = [x.img for x in [bg, plume] if isinstance(x, Img)]
    mean_bg,_ = _mean_in_rect(bg, rect)
    mean_img,_ = _mean_in_rect(plume, rect)
    del_rad = mean_img / float(mean_bg)
    return bg * del_rad

def corr_tau_curvature_vert_two_rects(tau0, r0, r1):
    """Applies linear backround curvature correction in tau img based on 
    two rectangular areas
    
    :param (ndarray, Img) tau0: inital tau image
    :param list r0: 1st rectanglular area ``[x0, y0, x1, y1]`
    :param list r1: 2nd rectanglular area ``[x0, y0, x1, y1]`
    :return ndarray: modified tau image

    Retrieves pixel mean value in both rectangles and from the determines
    linear offset function based on the vertical positions of the rectangle
    center coordinates. The corresponding offset for each image row is then
    subtracted from the input tau image
    
    """
    if isinstance(tau0, Img):
        tau0 = tau0.img
    y0, y1 = 0.5 * (r0[1] + r0[3]), 0.5*(r1[1] + r1[3])
    max_y = tau0.shape[0]
    
    mean_r0,_ = _mean_in_rect(tau0, r0)
    mean_r1,_ = _mean_in_rect(tau0, r1)
    
    slope = float(mean_r0 - mean_r1) / float(y0 - y1)
    offs = mean_r1 - slope * y1
    
    ygrid = linspace(0, max_y - 1, max_y, dtype = int)
    poly_vals = offs + slope * ygrid 
    tau_mod = (tau0.T - poly_vals).T
    return tau_mod#, vert_poly
    
def corr_tau_curvature_hor_two_rects(tau0, r0, r1):
    """Applies linear backround curvature correction in tau img based on 
    two rectangular areas
    
    :param (ndarray, Img) tau0: inital tau image
    :param list r0: 1st rectanglular area ``[x0, y0, x1, y1]`
    :param list r1: 2nd rectanglular area ``[x0, y0, x1, y1]`
    :return ndarray: modified tau image

    Retrieves pixel mean value in both rectangles and from the determines
    linear offset function based on the horizontal positions of the 
    rectangle center coordinates. The corresponding offset for each image 
    row is then subtracted from the input tau image
    """
    if isinstance(tau0, Img):
        tau0 = tau0.img
    x0, x1 = 0.5 * (r0[0] + r0[2]), 0.5 * (r1[0] + r1[2])
    max_x = tau0.shape[1]
    
    
    mean_r0,_ = _mean_in_rect(tau0, r0)
    mean_r1,_ = _mean_in_rect(tau0, r1)

    slope = float(mean_r0 - mean_r1) / float(x0 - x1)
    offs = mean_r1 - slope * x1
    
    xgrid = linspace(0, max_x - 1, max_x, dtype = int)
    poly_vals = offs + slope * xgrid 
    tau_mod = tau0 - poly_vals
    return tau_mod#, vert_poly
    
def corr_tau_curvature_vert_line(tau0, pos_x, start_y = 0,\
                        stop_y = None, row_mask = None, polyorder = 2):
    """Correction of vertical tau curvature using selected row indices of 
    vertical line.
    
    :param (ndarray, Img) tau0: inital tau image
    :param int pos_x: x position of line (column number)
    :param int start_y: first considered vertical index for fit (0)
    :param int stop_y: last considered vertical index for fit (is set 
        to last row number if unspecified)
    :param ndarray row_mask: boolean mask specifying considered row indices 
        (if valid, params start_y, stop_y are not considered)
    :param int polyorder: order of polynomial to fit curvature
    return tuple: 1st entry: modified tau image, second: fitted polynomial
    """
    if isinstance(tau0, Img):
        tau0 = tau0.img
    max_y = tau0.shape[0]
    line_vert = LineOnImage(pos_x, 0, pos_x, max_y)
    vert_profile = line_vert.get_line_profile(tau0)
    
    if stop_y == None:
        stop_y = max_y
    
    ygrid = linspace(0, max_y - 1, max_y, dtype = int)
    try:
        if len(row_mask) == max_y:
            mask = row_mask
    except:
        mask = logical_and(ygrid >= start_y, ygrid <= stop_y)
    
    
    vert_poly = poly1d(polyfit(ygrid[mask], vert_profile[mask], polyorder))
    
    poly_vals = vert_poly(ygrid)
    tau_mod = (tau0.T - poly_vals).T
    return (tau_mod, vert_poly)

def corr_tau_curvature_hor_line(tau0, pos_y, start_x = 0,\
                        stop_x = None, col_mask = None, polyorder = 2):
    """Correction of vertical tau curvature using selected row indices of 
    vertical line.
    
    :param (ndarray, Img) tau0: inital tau image
    :param int pos_y: y position of line (row number)
    :param int start_x: first considered horizontal index for fit (0)
    :param int stop_y: last considered horizontal index for fit (is 
        set to last col number if unspecified)
    :param ndarray col_mask: boolean mask specifying considered column 
        indices (if valid, params start_x, stop_x are not considered)
    :param int polyorder: order of polynomial to fit curvature
    return tuple: 1st entry: modified tau image, second: fitted polynomial
    """
    if isinstance(tau0, Img):
        tau0 = tau0.img
    max_x = tau0.shape[1]
    line_hor = LineOnImage(0, pos_y, max_x, pos_y)
    hor_profile = line_hor.get_line_profile(tau0)
    
    if stop_x == None:
        stop_x = max_x
    
    xgrid = linspace(0, max_x - 1, max_x, dtype = int)
    try:
        if len(col_mask) == max_x:
            mask = col_mask
    except:
        mask = logical_and(xgrid >= start_x, xgrid <= stop_x)
    
    
    hor_poly = poly1d(polyfit(xgrid[mask], hor_profile[mask], polyorder))
    
    poly_vals = hor_poly(xgrid)
    tau_mod = tau0 - poly_vals
    return (tau_mod, hor_poly)

def _roi_coordinates(roi):
    """Convert roi coordinates into start point, height and width

    :param list roi: region of interest, i.e. ``[x0, y0, x1, y1]``
    """
    return roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]
        
def find_sky_reference_areas(plume_img, plot = False):
    """Takes an input plume image and identifies suited sky reference areas"""
    if isinstance(plume_img, Img):
        plume = plume_img.img    
    else:
        plume = plume_img

    h, w = plume.shape
    results = {}
    vert_mag, hor_mag = int(h * 0.05), int(w * 0.05)
    
    #estimate mean intensity in left image part (without flank pixels)
    y0_left = argmin(gradient(average(plume[vert_mag : h - vert_mag,\
        hor_mag : hor_mag * 2], axis = 1)))
    
    avg_left = plume[\
            vert_mag : y0_left - vert_mag, hor_mag:hor_mag * 2].mean()
    print "Left: %s, %s" %(y0_left, avg_left)
    #estimate mean intensity in right image part (without flank pixels
    y0_right = argmin(gradient(average(plume[vert_mag : h - vert_mag,\
                            w - 2 * hor_mag : w - hor_mag], axis=1)))
    
    avg_right = plume[vert_mag : y0_right - vert_mag,\
                        w - 2 * hor_mag : w - hor_mag].mean()
    print "Right: %s, %s" %(y0_right, avg_right)
    if avg_right > avg_left: #brighter on the right image side (assume this is clear sky)
        print "set vert line right corner"
        results["ygrad_line_colnum"] = w - hor_mag
        results["ygrad_line_stoprow"] = y0_right - 0.2 * vert_mag
    else:
        print "set vert line left corner"
        results["ygrad_line_colnum"] = hor_mag
        results["ygrad_line_stoprow"] = y0_left - 0.2 * vert_mag
    results["ygrad_line_startrow"] = 0.1 * vert_mag
    
    results["xgrad_line_rownum"] = vert_mag
    results["xgrad_line_startcol"] = hor_mag
    results["xgrad_line_stopcol"] = w - hor_mag
    x0, y0 = int(results["ygrad_line_colnum"] - hor_mag * 0.5),\
                                                    0.5 * vert_mag
    x1, y1 = x0 + hor_mag, y0 + vert_mag
    results["scale_rect"] = [x0, y0, x1, y1]
    
    x0, y0 = int(results["ygrad_line_colnum"] - hor_mag * 0.5),\
                        int(results["ygrad_line_stoprow"] - vert_mag)
    x1, y1 = x0 + hor_mag, y0 + vert_mag
    results["ygrad_rect"] = [x0, y0, x1, y1]
    x0, y0 = int(w*.5 - hor_mag * .5), 0.5 * vert_mag
    x1, y1 = x0 + hor_mag, y0 + vert_mag
    results["xgrad_rect"] = [x0, y0, x1, y1]
    if plot:
        plot_sky_reference_areas(plume, results)
    return results

def plot_sky_reference_areas(plume_img, settings_dict, ax = None):
    """Plot provided sky reference areas into a plume image
    
    :param (ndarray, Img) plume_img: plume image data
    :param dict settings_dict: dictionary containing settings (e.g. retrieved
        using :func:`find_sky_reference_areas`)
    """
    if isinstance(plume_img, Img):
        plume = plume_img.img    
    else:
        plume = plume_img
    if ax is None:
        fig, ax = subplots(1,1)
    r = settings_dict
    h0, w0 = plume.shape[:2]
    ax.imshow(plume, cmap = "gray")
    ax.plot([r["ygrad_line_colnum"], r["ygrad_line_colnum"]],\
                [r["ygrad_line_startrow"], r["ygrad_line_stoprow"]],\
                                "-", c="lime", label = "vert profile")
    ax.plot([r["xgrad_line_startcol"], r["xgrad_line_stopcol"]],\
                [r["xgrad_line_rownum"], r["xgrad_line_rownum"]],\
                                "-", c="lime", label = "hor profile")
    ax.set_xlim([0, w0 - 1])
    ax.set_ylim([h0 - 1, 0])
    
    
    xs, ys, ws, hs = _roi_coordinates(r["scale_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="lime",fc="none",\
                                            label = "scale_rect"))
    
    xs, ys, ws, hs = _roi_coordinates(r["ygrad_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="lime",fc="none",\
                                            label = "ygrad_rect"))
    
    xs, ys, ws, hs = _roi_coordinates(r["xgrad_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="lime",fc="none",\
                                            label = "xgrad_rect"))
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
        
    return ax
        
#==============================================================================
# if __name__ == "__main__": 
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os
# 
#     """
#     4 images from 16/09/2016 Etna campaign containing one plume and 3 bg images
#     
#         0. plume image recorded at 06:40 UTC 
#         1. bg image recorded at 06:52 UTC (same elev, only upper half is sky)
#         2. bg image recorded at 07:02 UTC (same azim that previous, elev up)
#         3. bg image recorded at 08:05 UTC (may be clouds / gas in lower end)
#     """
#     plt.close("all")
# ### Start of example script
# ### Set image paths
#     p = r'D:/Jonas/Research/Data/Campaigns/201509_Etna/DATA_RAW_CURRENT_WRK/ECII/images/259/'
#     onP = p + 'EC2_1106307_1R02_2015091606402173_F01_Etnaxxxxxxxxxxxx.fts'
#     bgOnP = p + 'EC2_1106307_1R02_2015091607022602_F01_Etnaxxxxxxxxxxxx.fts'
#     offP = p + 'EC2_1106307_1R02_2015091606402416_F02_Etnaxxxxxxxxxxxx.fts'
#     bgOffP = p + 'EC2_1106307_1R02_2015091607022820_F02_Etnaxxxxxxxxxxxx.fts'
#     savePath = r'D:/Dropbox/Python27/jgliss/modules/piscope/_private/lib_dev/plots/'
#     savePath = r'D:/Dropbox/Dokumente/2016/jgliss_publications/piscope/graphics/'
#     
# ### Some options for background retrieval using poly surface fit
#     maskThresh = 3550 #threshold to init mask for bg surface fit
#     vignCorr = 1
#     polyorder = 1
#     pyrlevel = 4
#     
# ### Some options for plotting
#     tau_min, tau_max = -0.30, 1.8
#     hspace,wspace = 0.02, 0.02
#     hPl=.3
#     saveFigs  = 0
#     w, h = 40, 40
#     cmap = shifted_color_map(tau_min, tau_max)
#     #x0,y0,x1,y1
# ### Define rectangles
#     scale   =   [1200, 80 , 1200 + w, 80 + h]
#     xgrad   =   [760 , 80 , 760 + w , 80 + h]
#     ygrad   =   [1200, 520, 1200 + w, 520 + h]
# 
# ### Define plume cross section line
#     pcs     =   [420, 655, 860, 200]
# ### Define coordinates of horizontal and vertical profile lines 
#     horLineY = 100
#     vertLineX = 1220
# 
# ### Load the image objects
#     plume, bg = Img(onP), Img(bgOnP)
#     plumeOff, bgOff = Img(offP), Img(bgOffP)
#     
# ### Blur the images (sigma = 1)
#     plume.add_gaussian_blurring(1)
#     bg.add_gaussian_blurring(1)
#     
#     plumeOff.add_gaussian_blurring(1)
#     bgOff.add_gaussian_blurring(1)
# 
# ### Create vignetting correction mask from background image 
#     vign = bg.img / bg.img.max() #NOTE: potentially includes y and x gradients
#     plumeCorr = plume.img / vign
#     
# ### Create tau0 image (i.e. log(bg/plume))
#     tau0 = log(bg.img/plume.img) #numpy array
# 
#     m = PlumeBackgroundModel()
#     m1 = PlumeBackgroundModel()
#     m1.guess_missing_settings(plume)
# ### First method: scale background image to plume image in "scale" rect
#     tau1 = m.get_tau_image(plume, bg, CORR_MODE = 1, scale_rect = scale).img
# ### Second method: Linear correction for radiance differences based on two rectangles (scale, ygrad)
#     tau2 = m.get_tau_image(plume, bg, CORR_MODE = 2, scale_rect = scale,\
#         ygrad_rect = ygrad).img
# ### Third method: 2nd order polynomial fit along vertical profile line
# ### For this method, determine tau on tau off and AA image
#     tau3 = m.get_tau_image(plume, bg, CORR_MODE = 3, ygrad_line_colnum = vertLineX,\
#         ygrad_line_startrow = 10, ygrad_line_stoprow = 590, ygrad_line_polyorder = 2).img
#         
#     tau3Off = m.get_tau_image(plumeOff, bgOff, CORR_MODE = 3, ygrad_line_colnum = vertLineX,\
#         ygrad_line_startrow = 10, ygrad_line_stoprow = 590, ygrad_line_polyorder = 2).img
#     aa = tau3 - tau3Off
#     
# ### Create mask for poly surface fit  
#     mask = np.ones(plumeCorr.shape, dtype = np.float32)   
#     mask[plumeCorr < maskThresh] = 0
# ### Fourth Method: retrieve tau image using poly surface fit
#     tau4 = m.get_tau_image(Img(plumeCorr), CORR_MODE = 0, surface_fit_mask = mask,\
#         surface_fit_polyorder = 1).img
# 
# 
# ### Create :class:`LineOnImage` objects for line profile retrievals 
#     lvert = LineOnImage(vertLineX, 0, vertLineX, 1023, id = "vert")
#     line_hor = LineOnImage(0, horLineY, 1343, horLineY, id = "hor")
#     pcsL = LineOnImage(*pcs, id="pcs")
# 
# 
#     p1_0 = lvert.get_line_profile(tau1)
#     p1_1 = lvert.get_line_profile(tau2)
#     p1_2 = lvert.get_line_profile(tau3)
#     p1_2Off = lvert.get_line_profile(tau3Off)
#     p1_2aa = lvert.get_line_profile(aa)
#     p1_3 = lvert.get_line_profile(tau4)
#     
#     p2_0 = line_hor.get_line_profile(tau1)
#     p2_1 = line_hor.get_line_profile(tau2)
#     p2_2 = line_hor.get_line_profile(tau3)
#     p2_2Off = line_hor.get_line_profile(tau3Off)
#     p2_2aa = line_hor.get_line_profile(aa)    
#     p2_3 = line_hor.get_line_profile(tau4)
#     
#     horYLabels = [0.0,.10,.20]
#     vertXLabels = [.0,.7,1.4]
#     
#     h0,w0 = bg.img.shape
# 
# 
# ### Plot PCS profiles for method 3 (on, off, aa)
#     fig0, ax0 = plt.subplots(1,1)
#     x=np.arange(0,633,1)
#     pOn = pcsL.get_line_profile(tau3)
#     pOff = pcsL.get_line_profile(tau3Off)
#     pAA = pcsL.get_line_profile(aa)    
#     
#     ax0.plot(pOn,"--", label=r"On: $\phi=%.3f$"%(sum(pOn)/633))
#     ax0.plot(pOff,"--", label=r"Off: $\phi=%.3f$"%(sum(pOff)/633))
#     ax0.plot(pAA,"-", lw = 2, label=r"AA: $\phi=%.3f$"%(sum(pAA)/633))
#     #ax0.fill_between(x, pOn, pOff, facecolor="r", alpha = 0.1)
#     
#     ax0.legend(loc=3, fancybox=True, framealpha=0.5, fontsize=12)
#     ax0.grid()
#     ax0.set_ylabel(r"$\tau$", fontsize=20)
#     ax0.set_xlim([0,632])
#     ax0.set_xticklabels([])
#     ax0.set_xlabel("PCS", fontsize=16)
#     #ax1.plot(pcsL.get_line_profile(tau1))
# 
#     
# ### Plot PCS profiles for all 4 methods (Fig. XX in manuscript)
#     fig1, ax1 = plt.subplots(1,1)
#     p1 = pcsL.get_line_profile(tau1)
#     p2 = pcsL.get_line_profile(tau2)
#     p3 = pcsL.get_line_profile(tau3)
#     p4 = pcsL.get_line_profile(tau4)
#     ax1.plot(p1,"-", label=r"A) scaled: $\phi=%.3f$"%(sum(p1)/633))
#     ax1.plot(p2,"-", label=r"B) lin. corr: $\phi=%.3f$"%(sum(p2)/633))
#     ax1.plot(p3,"-", label=r"C) quad. corr: $\phi=%.3f$"%(sum(p3)/633))
#     ax1.plot(p4,"-", label=r"D) 2D surface fit: $\phi=%.3f$"%(sum(p4)/633))
#     
#     ax1.legend(loc=3, fancybox=True, framealpha=0.5, fontsize=12)
#     ax1.grid()
#     ax1.set_ylabel(r"$\tau$", fontsize=20)
#     ax1.set_xlim([0,632])
#     ax1.set_xticklabels([])
#     ax1.set_xlabel("PCS", fontsize=16)
#     #ax1.plot(pcsL.get_line_profile(tau1))
# 
# ### Plot result first method
#     fig2 = plt.figure()
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
#     ax = [plt.subplot(gs[2]),]
#     ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
#     ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
#     ax[0].imshow(tau1, cmap = cmap, vmin = tau_min, vmax = tau_max)
#     ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
#     ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
#     ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
#     #ax[0].set_title("Tau img after scaling", fontsize=14)
#     #fig.suptitle("Tau img after scaling", fontsize=14)
#     ax[0].add_patch(Rectangle((scale[0], scale[1]), w, h, ec="g",fc="none",\
#                                                 label = "scale_rect"))
#     
#     ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#     
#     #ax[0].xaxis.tick_top()   
#     ax[2].set_xticklabels([])
#     ax[1].set_yticklabels([])
#     
#     #plot vertical profile
#     ax[1].plot(p1_0, np.arange(0, len(p1_0),1), "-b" , label = "vert profile")
#     ax[1].set_xlabel(r"$\tau$", fontsize=16)
#     ax[2].set_ylabel(r"$\tau$", fontsize=16)
#     ax[1].yaxis.tick_right()   
#     ax[1].set_ylim([1023,0])
#     ax[1].get_xaxis().set_ticks(vertXLabels)
#     ax[1].set_xlim([-.2,1.6])
#     
#     #plot horizontal profile
#     ax[2].plot(np.arange(0, len(p2_0),1),p2_0, "-c" , label = "hor profile")
#     ax[2].get_yaxis().set_ticks(horYLabels)
#     ax[2].set_ylim([-.05,.25])
#     ax[2].set_xlim([0,1343])
# 
#     plt.subplots_adjust(wspace=wspace, hspace=hspace)
#     
#     ax[2].axhline(0, ls="--", color="k")
#     ax[1].axvline(0, ls="--", color="k")
#     
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023,0])
# 
# ### Plot result seond method   
#     fig3 = plt.figure()
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*.2], height_ratios=[h0,h0*.2])
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
#     ax = [plt.subplot(gs[2]),]
#     ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
#     ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
#     
#     #plot image    
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023,0])
# 
#     ax[0].imshow(tau2, cmap = cmap, vmin = tau_min, vmax = tau_max)
#     #ax[0].imshow(tau2, cmap=cmap)
#     ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
#     ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
#     ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
#     #fig.suptitle("Tau image after scaling + ygrad corr", fontsize=14)
#     ax[0].add_patch(Rectangle((scale[0], scale[1]), w, h, ec="g",fc="none",\
#                                                 label = "scale_rect"))
#     ax[0].add_patch(Rectangle((ygrad[0], ygrad[1]), w, h, ec="g",fc="none",\
#                                                 label="ygrad_rect"))
#     
#     ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#     
#     ax[2].set_xticklabels([])
#     ax[1].set_yticklabels([])
# 
#     
#     #plot vertical profile
#     ax[1].plot(p1_1, np.arange(0, len(p1_1),1), "-b" , label = "vert profile")
#     ax[1].yaxis.tick_right()   
#     ax[1].set_ylim([1023,0])
#     ax[1].get_xaxis().set_ticks(vertXLabels)
#     ax[1].set_xlim([-.2,1.6])
#     
#     #plot horizontal profile
#     ax[2].plot(np.arange(0, len(p2_1),1),p2_1, "-c" , label = "hor profile")
#     ax[2].get_yaxis().set_ticks(horYLabels)
#     ax[2].set_ylim([-.05,.25])
#     ax[2].set_xlim([0,1343])
# 
#     plt.subplots_adjust(wspace=wspace, hspace=hspace)
#     ax[2].axhline(0, ls="--", color="k")
#     ax[1].axvline(0, ls="--", color="k")
#     
#     ax[1].set_xlabel(r"$\tau$", fontsize=16)
#     ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result third method    
#     fig4 = plt.figure()
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
#     ax = [plt.subplot(gs[2]),]
#     ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
#     ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
#     
#     #plot image    
#     ax[0].imshow(tau3, cmap = cmap, vmin = tau_min, vmax = tau_max)
#     #ax[0].imshow(tau3)
#     ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
#     ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
#     ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023,0])
#     ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#     
#     ax[2].set_xticklabels([])
#     ax[1].set_yticklabels([])
# 
#     #plot vertical profile
#     ax[1].plot(p1_2, np.arange(0, len(p1_2),1), "-b" , label = "vert profile")
#     ax[1].yaxis.tick_right()   
#     ax[1].set_ylim([1023,0])
#     ax[1].get_xaxis().set_ticks(vertXLabels)
#     ax[1].set_xlim([-.2,1.6])
#    
#     
#     #plot horizontal profile
#     ax[2].plot(np.arange(0, len(p2_2), 1),p2_2, "-c" , label = "hor profile")
#     ax[2].get_yaxis().set_ticks(horYLabels)
#     ax[2].set_ylim([-.05,.25])
#     ax[2].set_xlim([0,1343])
# 
#     plt.subplots_adjust(wspace=wspace, hspace=hspace)
#     ax[2].axhline(0, ls="--", color="k")
#     ax[1].axvline(0, ls="--", color="k")
#     
#     ax[1].set_xlabel(r"$\tau$", fontsize=16)
#     ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result third method (off band) 
#     fig4_off = plt.figure()
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
#     ax = [plt.subplot(gs[2]),]
#     ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
#     ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
#     
#     #plot image    
#     ax[0].imshow(tau3Off, cmap = cmap, vmin = tau_min, vmax = tau_max)
#     #ax[0].imshow(tau3)
#     ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
#     ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
#     ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023,0])
#     ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#     
#     ax[2].set_xticklabels([])
#     ax[1].set_yticklabels([])
# 
#     #plot vertical profile
#     ax[1].plot(p1_2Off, np.arange(0, len(p1_2Off),1), "-b" , label = "vert profile")
#     ax[1].yaxis.tick_right()   
#     ax[1].set_ylim([1023,0])
#     ax[1].get_xaxis().set_ticks(vertXLabels)
#     ax[1].set_xlim([-.2,1.6])
#    
#     
#     #plot horizontal profile
#     ax[2].plot(np.arange(0, len(p2_2Off), 1),p2_2Off, "-c" , label = "hor profile")
#     ax[2].get_yaxis().set_ticks(horYLabels)
#     ax[2].set_ylim([-.05,.25])
#     ax[2].set_xlim([0,1343])
# 
#     plt.subplots_adjust(wspace=wspace, hspace=hspace)
#     ax[2].axhline(0, ls="--", color="k")
#     ax[1].axvline(0, ls="--", color="k")
#     
#     ax[1].set_xlabel(r"$\tau$", fontsize=16)
#     ax[2].set_ylabel(r"$\tau$", fontsize=16)
# 
# ### Plot result third method (off band) 
#     fig4_aa = plt.figure()
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
#     ax = [plt.subplot(gs[2]),]
#     ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
#     ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
# 
#     
#     #plot image    
#     ax[0].imshow(aa, cmap = cmap, vmin = tau_min, vmax = tau_max)
#     #ax[0].imshow(tau3)
#     ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
#     ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
#     ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023,0])
#     ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#     
#     ax[2].set_xticklabels([])
#     ax[1].set_yticklabels([])
# 
#     #plot vertical profile
#     ax[1].plot(p1_2aa, np.arange(0, len(p1_2aa),1), "-b" , label = "vert profile")
#     ax[1].yaxis.tick_right()   
#     ax[1].set_ylim([1023,0])
#     ax[1].get_xaxis().set_ticks(vertXLabels)
#     ax[1].set_xlim([-.2,1.6])
#    
#     
#     #plot horizontal profile
#     ax[2].plot(np.arange(0, len(p2_2aa), 1),p2_2aa, "-c" , label = "hor profile")
#     ax[2].get_yaxis().set_ticks(horYLabels)
#     ax[2].set_ylim([-.05,.25])
#     ax[2].set_xlim([0,1343])
# 
#     plt.subplots_adjust(wspace=wspace, hspace=hspace)
#     ax[2].axhline(0, ls="--", color="k")
#     ax[1].axvline(0, ls="--", color="k")
#     
#     ax[1].set_xlabel(r"$\tau$", fontsize=16)
#     ax[2].set_ylabel(r"$\tau$", fontsize=16)
#     
# ### Plot result fourth method    
#     fig5 = plt.figure()
#     gs = plt.GridSpec(2, 2, width_ratios=[w0,w0*hPl], height_ratios=[h0*hPl,h0])
#     ax = [plt.subplot(gs[2]),]
#     ax.append(plt.subplot(gs[3]))#, sharey=ax[0]))
#     ax.append(plt.subplot(gs[0]))#, sharex=ax[0]))
#     ax.append(plt.subplot(gs[1]))
# 
#     palette = colors.ListedColormap(['white', 'lime'])
#     norm = colors.BoundaryNorm([0,.5,1], palette.N)
# 
#     ax[3].imshow(mask, cmap=palette, norm=norm, alpha = .7)
#     ax[3].set_title("Mask", fontsize=10)
#     ax[3].set_xticklabels([])
#     ax[3].set_yticklabels([])
#     
#     ax[0].imshow(tau4, cmap = cmap, vmin = tau_min, vmax = tau_max)
#     
#     ax[0].plot([vertLineX,vertLineX],[0,1023], "-b", label = "vert profile")
#     ax[0].plot([0,1343],[horLineY,horLineY], "-c", label = "hor profile")
#     ax[0].plot([pcs[0],pcs[2]],[pcs[1],pcs[3]], "--",c="g", label="PCS")
# 
#     ax[0].set_xlim([0,1343])
#     ax[0].set_ylim([1023,0])
#     ax[0].legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
#     
#     ax[2].set_xticklabels([])
#     ax[1].set_yticklabels([])
#     
#     
#     #plot vertical profile
#     ax[1].plot(p1_3, np.arange(0, len(p1_3),1), "-b" , label = "vert profile")
#     ax[1].yaxis.tick_right()   
#     ax[1].set_ylim([1023,0])
#     ax[1].get_xaxis().set_ticks(vertXLabels)
#     ax[1].set_xlim([-.2,1.6])
#    
#     
#     #plot horizontal profile
#     ax[2].plot(np.arange(0, len(p2_3), 1),p2_3, "-c" , label = "hor profile")
#     ax[2].get_yaxis().set_ticks(horYLabels)
#     ax[2].set_ylim([-.05,.25])
#     ax[2].set_xlim([0,1343])
# 
#     plt.subplots_adjust(wspace=wspace, hspace=hspace)
#     ax[2].axhline(0, ls="--", color="k")
#     ax[1].axvline(0, ls="--", color="k")
#     
#     ax[1].set_xlabel(r"$\tau$", fontsize=16)
#     ax[2].set_ylabel(r"$\tau$", fontsize=16)    
#     if saveFigs:
#         fig1.savefig(os.path.join(savePath, "bg_corr_ICAs.png"))
#         fig2.savefig(os.path.join(savePath, "bg_corr_0.png"))
#         fig3.savefig(os.path.join(savePath, "bg_corr_1.png"))
#         fig4.savefig(os.path.join(savePath, "bg_corr_2.png"))
#         fig5.savefig(os.path.join(savePath, "bg_corr_3.png"))
# 
#     fig6 = plt.figure()
#     plt.imshow(plume.img, cmap = "gray")
#     
#==============================================================================
