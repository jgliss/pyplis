# -*- coding: utf-8 -*-
"""
Module containing features related to plume background analysis and tau
image determination
"""
from numpy import polyfit, poly1d, linspace, logical_and, log, full, argmin,\
    gradient, nan, exp, ndarray, arange, ones, finfo, asarray
from matplotlib.patches import Rectangle
from matplotlib.pyplot import GridSpec, figure, subplots_adjust, subplot,\
    subplots, setp
import matplotlib.colors as colors
from collections import OrderedDict as od
from scipy.ndimage.filters import gaussian_filter
from warnings import warn

from .image import Img
from .processing import LineOnImage
from .optimisation import PolySurfaceFit
from .helpers import shifted_color_map, _roi_coordinates

class PlumeBackgroundModel(object):
    """Class for plume background modelling and tau image determination
    
    Parameters
    ----------
    bg_raw : Img
        sky background radiance raw image data
    plume_init : Img
        initial plume image data (is used to estimate default clear sky areas 
        for bg modelling)
    **kwargs : 
        additional class attributes (e.g. for modelling, valid keys
        are all keys in self.__dict__.keys())    
    """
    def __init__(self, bg_raw=None, plume_init=None, **kwargs):
        """Initialisation of object
        
        
        """        
        if isinstance(bg_raw, ndarray):
            bg_raw = Img(bg_raw)
        if isinstance(plume_init, ndarray):
            plume_init = Img(plume_init)
            
        self._current_imgs = {"plume"    :   plume_init,
                              "bg_raw"   :   bg_raw,
                              "tau"      :   None}        
        
        #: Correction mode
        self._mode = 0
        
        #: settings for poly surface fit (corr mode: 0)
        self.surface_fit_mask = None
        self.surface_fit_mask_vmin = None
        self.surface_fit_mask_vmax = None
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
            self.mode = 1
            self.guess_missing_settings(plume_init)
            self.surface_fit_mask = ones(plume_init.img.shape, dtype=bool)
    
    @property
    def mode(self):
        """Current modelling mode"""
        return self._mode

    @mode.setter
    def mode(self, val):
        if not val in self.all_modes:
            raise ValueError("Invalid mode %d, choose from %s" 
                             %(val, self.all_modes))
        self._mode = val
        
    @property
    def CORR_MODE(self):
        """Current background modelling mode"""
        return self.mode
        
    @CORR_MODE.setter
    def CORR_MODE(self, val):
        self.mode = val
        
    @property
    def current_plume_background(self):
        """Retrieve the current plume background from modelled tau image and
        plume image, i.e::

            bg_img = Img(exp(tau_img) * plume_img)
            
        """
        return Img(exp(self._current_imgs["tau"].img) * 
                    self._current_imgs["plume"].img)
    
    def get_current(self, key="tau"):
        """Returns current image, specify type via input key
        
        Parameters
        ----------
        key : str
            choose from plume, bg_raw, bg_model, tau
            
        Returns
        -------
        Img
            the specified image
            
        Raises
        ------
        KeyError
            in case input ``key`` is invalid
        """
        return self._current_imgs[key]
        
    def check_settings(self):
        for value in self.__dict__.values():
            if value is None:
                return False
        return True
        
    def mean_in_rects(self, img):
        """Determine ``(mean, min, max)`` intensity in all reference rectangles
        
        Parameters
        ----------
        img : array
            image data array (can also be :class:`Img`)
          
        Returns
        -------
        tuple
            2-element tuple containing mean value and error
        """
        try:
            img = img.img
        except:
            pass
        a = []
        a.append(_mean_in_rect(img, self.scale_rect)[0])
        a.append(_mean_in_rect(img, self.ygrad_rect)[0])
        a.append(_mean_in_rect(img, self.xgrad_rect)[0])
        a = asarray(a)
        return (a.mean(), a.min(), a.max())
        
    
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
        if rect is None:
            return False
        h, w = img.shape
        if rect[0] < 0 or rect[1] < 0 or rect[2] >= w or rect[3] >= h:
            return False
        return True
          
    def guess_missing_settings(self, plume_img):
        """Wrapper for :func:`set_missing_ref_areas`
        
        Note
        ----
        This is the previous name of the method 
        :func:`set_missing_ref_areas`
        """  
        self.set_missing_ref_areas(plume_img)
        
    def set_missing_ref_areas(self, plume_img):
        """Find and set missing default sky reference areas for modelling
        
        Based on the input plume image, the clear sky reference areas for sky
        radiance image based tau modelling are estimated, i.e.:
        
            1. The rectangle areas for scaling and linear gradient corrections:
                ``self.scale_rect, self.ygrad_rect, self.xgrad_rect``
            2. Coordinates of horizontal and vertical profile lines 
            for quadratic gradient corrections: 
            ``self.ygrad_line_colnum, self.xgrad_line_rownum`` 
            (i.e. positions and start / stop pixel coordinates)
                
        The estimation is performed based on a brightness analysis for left and
        right image area. 
        
        Parameters
        ----------
        plume_img : Img
            exemplary plume image (should be representative for a whole 
            dataset) 
        """   
        if not isinstance(plume_img, Img):
            raise TypeError("Invalid, input type: need Img object...")
        plume = plume_img.duplicate().img
        if self.check_settings():
            return
        if self.surface_fit_mask is None:
            self.surface_fit_mask = full(plume.shape, True, dtype=bool)
        h, w = plume.shape
        
        res = find_sky_reference_areas(plume)
        if self.ygrad_line_colnum is None:
            self.ygrad_line_colnum = res["ygrad_line_colnum"]
            self.ygrad_line_stoprow = res["ygrad_line_stoprow"]
            self.ygrad_line_startrow = res["ygrad_line_startrow"]
        if self.xgrad_line_rownum is None:
            self.xgrad_line_rownum = res["xgrad_line_rownum"]
            self.xgrad_line_startcol = res["xgrad_line_startcol"]
            self.xgrad_line_stopcol = res["xgrad_line_stopcol"]
        if not self._check_rect(self.scale_rect, plume):
            self.scale_rect = res["scale_rect"]
        if not self._check_rect(self.ygrad_rect, plume):
            self.ygrad_rect = res["ygrad_rect"]
        if not self._check_rect(self.xgrad_rect, plume):
            self.xgrad_rect = res["xgrad_rect"]
    
    def settings_dict(self):
        """Write current sky reference areas and masks into dictionary"""
        d = {}
        d["mode"] = self.mode
        d["surface_fit_mask"] = self.surface_fit_mask 
        d["surface_fit_pyrlevel"] = self.surface_fit_pyrlevel
        d["surface_fit_polyorder"] = self.surface_fit_polyorder
        d["scale_rect"] = self.scale_rect
        d["ygrad_rect"] = self.ygrad_rect
        
        d["ygrad_line_colnum"] = self.ygrad_line_colnum
        d["ygrad_line_polyorder"] = self.ygrad_line_polyorder
        d["ygrad_line_startrow"] = self.ygrad_line_startrow
        d["ygrad_line_stoprow"] = self.ygrad_line_stoprow
        d["ygrad_line_mask"] = self.ygrad_line_mask
        
        d["xgrad_rect"] = self.xgrad_rect
    
        d["xgrad_line_rownum"] = self.xgrad_line_rownum
        d["xgrad_line_polyorder"] = self.xgrad_line_polyorder
        d["xgrad_line_stopcol"] = self.xgrad_line_stopcol
        d["xgrad_line_startcol"] = self.xgrad_line_startcol
        d["xgrad_line_mask"] = self.xgrad_line_mask
        return d
        
    def bg_from_poly_surface_fit(self, plume, mask=None, polyorder=2,
                                 pyrlevel=4):
        """Applies poly surface fit to plume image for bg retrieval
        
        :param ndarray plume: plume image
        :param ndarray mask (None): mask specifying gas free areas (if None, 
            use all pixels)
        :param int polyorder (2): order of polynomial used for fit
        :param int pyrlevel (4): scale space level in which fit is performed (
            e.g. 4 => image size for fit is reduced by factor 2^4 = 16)
        :return tuple: 1st entry: fitted background image
            second: ``PolySurfaceFit`` object 
    
        """
        #update settings from input keyword args
        if mask is None or not mask.shape == plume.shape:
            print ("Warning: invalid mask for poly surface fit (bg modelling)"
                    " considering all image pixels for retrieval")
            mask = full(plume.shape, True, dtype = bool)   
    
        fit = PolySurfaceFit(plume, mask.astype(float), polyorder=polyorder,
                             pyrlevel=pyrlevel)
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
    
    def get_surface_fit_mask(self, plume_img, vmin, vmax, base_mask=None):
        """ Derive a mask from a plume_img based on thresholds (vmin,vmax)
        
        Parameters
        ----------
        plume_img : Img
            plume image in intensity space
        vmin : float
            minimum value to be flaged as 1
        vmax : float
            maximum value to be flaged as 1
        base_mask : numpy.array, optional
            additonal maks, all values which are 0 in this mask will be 0 in
            0 in the resulting mask; has to be of same shape as plume_img.img
        
        Returns
        -------
        surface_fit_mask : numpy.array
            mask which is 1 for background and 0 else
        
        """

        # BLur the image to get homogenous values
        # ? maybe do a sharpening instead up bluring?
        from numpy import float32
        Img = plume_img.duplicate() # make a deep copy 
        Img.add_gaussian_blurring(3)    
        
        mask = ones(Img.img.shape)#, dtype=float32)   #np.ones
        
        if not isinstance(vmin, float):
            warn("Invalid input for vmin in get_surface_fit_mask.")
            return mask
        
        if not isinstance(vmax, float):
            warn("Invalid input for vmax in get_surface_fit_mask.")
            return mask
            
        mask[Img.img < vmin] = 0
        mask[Img.img > vmax] = 0
        
        if isinstance(base_mask, ndarray):
            mask[base_mask==0]=0
        return mask
        
    
    def get_tau_image(self, plume_img, bg_img=None, update_imgs=False, 
                      **kwargs):
        """Determine current tau image for input plume image
        
        Parameters
        ----------
        plume_img : Img
            plume image in intensity space
        bg_img : :obj:`Img`, optional
            sky radiance image (for ``self.CORR_MODE = 1 - 6``)
        update_imgs : bool
            if True, the internal images within this class are updated 
            (stored in priv. attr :attr:`_current_imgs`)
        **kwargs : 
            additional keyword arguments for updating current settings
            (valid input keywords (strings): CORR_MODE, ygrad_rect, 
            ygrad_line_colnum, ygrad_line_startrow, ygrad_line_stoprow
        
        Returns
        -------
        Img
            plume tau image    
            
        Raises
        ------
        AttributeError
            if input image is already a tau or AA image or if input plume 
            image and the current background image have different states
            with regard to vignetting correction.
          
        """
        if not isinstance(plume_img, Img):
            raise TypeError("Invalid, input type: need Img object...")
        #update current settings
        for k, v in kwargs.iteritems():
            self.__setitem__(k, v)
            
        mode = self.CORR_MODE
        if not plume_img.edit_log["darkcorr"]:
            warn("plume image is not corrected for dark current")
        if plume_img.is_tau:
            raise AttributeError("Input image is already tau image")
        plume = plume_img.img
        if mode != 0:
            if not isinstance(bg_img, Img):
                bg_img = self.get_current("bg_raw")
            bg = bg_img.img
            if not bg_img.edit_log["darkcorr"]:
                warn("Sky BG image is not corrected for dark current")

            if not plume_img.is_vigncorr is bg_img.is_vigncorr:
                raise AttributeError("Cannot model tau image: plume img and "
                                     "sky radiance image have different "
                                     "vignetting correction states.")
        tau = None
        if mode == 0: #no sky radiance image, poly surface fit
            # retrieve an individual mask
            self.surface_fit_mask = self.get_surface_fit_mask(plume_img,
                                                              self.surface_fit_mask_vmin,
                                                              self.surface_fit_mask_vmax)
            
            (bg, fit)=self.bg_from_poly_surface_fit(plume,
                                                    self.surface_fit_mask,
                                                    self.surface_fit_polyorder,
                                                    self.surface_fit_pyrlevel)
            r = bg / plume
            #make sure no 0 values or neg. numbers are in the image
            r[r<=0] = finfo(float).eps
            tau = log(r)
    
        else:
            #bg_norm = scale_bg_img(bg, plume, self.scale_rect)
            r = bg / plume
            #make sure no 0 values or neg. numbers are in the image
            r[r<=0] = finfo(float).eps
            tau = log(r)
            if mode != 99:
                tau = self.correct_tau_curvature_ref_areas(tau)
            
        tau_img = plume_img.duplicate()
        tau_img.meta["bit_depth"] = nan
        tau_img.edit_log["is_tau"] = 1
        tau_img.img = tau
        if update_imgs:
            self.set_current_images(plume_img, bg_img, tau_img)
        
        return tau_img
    
    def get_aa_image(self, plume_on, plume_off, bg_on=None, bg_off=None, 
                     update_imgs=False, **kwargs):
        """Method to retrieve apparent absorbance image from on and off imgs
          
        Determines an initial AA image based on input plume and background 
        images and
        
        Parameters
        ----------
        plume_on : Img
            on-band plume image
        plume_off : Img
            off-band plume image
        bg_on : :obj:`Img`, optional
            on-band sky radiance image (for ``self.CORR_MODE = 1 - 6``)
        bg_off : :obj:`Img`, optional
            off-band sky radiance image (for ``self.CORR_MODE = 1 - 6``)
        update_imgs : bool
            if True, the internal images within this class are updated 
            (stored in priv. attr :attr:`_current_imgs`), defaults to False
        **kwargs : 
            additional keyword arguments for updating current settings
            (valid input keywords (strings), e.g. ``surface_fit_mask`` if
            ``CORR_MODE == 0``
        
        Returns
        -------
        Img
            plume AA image
        
        """ 
        for k, v in kwargs.iteritems():
            self.__setitem__(k, v)   
            
        mode = self.CORR_MODE
        if mode == 0:
            mask = self.surface_fit_mask
            po = self.surface_fit_polyorder
            pyr = self.surface_fit_pyrlevel
            (bg_on, fit_on) = self.bg_from_poly_surface_fit(plume_on, 
                                                            mask, po, pyr)
            (bg_off, fit_off) = self.bg_from_poly_surface_fit(plume_off,
                                                              mask, po, pyr)
            r_on = bg_on / plume_on.img
            #make sure no 0 values or neg. numbers are in the image
            r_on[r_on <= 0] = finfo(float).eps
            
            r_off = bg_off / plume_off.img
            #make sure no 0 values or neg. numbers are in the image
            r_off[r_off <= 0] = finfo(float).eps
            aa = log(r_on) - log(r_off)
        else:
            r1 = bg_on.img / plume_on.img
            r1[r1<=0] = finfo(float).eps
            r2 = bg_off.img / plume_off.img
            r2[r2<=0] = finfo(float).eps
            aa = log(r1) - log(r2)
            if mode != 99:               
                aa = self.correct_tau_curvature_ref_areas(aa)
        
        aa_img = plume_on.duplicate()
        aa_img.meta["bit_depth"] = nan
        aa_img.edit_log["is_tau"] = 1
        aa_img.edit_log["is_aa"] = 1
        aa_img.img = aa
        if update_imgs:
            try:
                self.set_current_images(plume_on.duplicate(), 
                                    bg_on.duplicate(), aa_img)
            except:
                self.set_current_images(plume_on.duplicate(), 
                                        Img(bg_on), aa_img)
        #self.set_current_images(plume_img, bg_img, tau_img)
        
        return aa_img#, Img(aa2)
        
    def correct_tau_curvature_ref_areas(self, tau_init):
        """Scale and correct curvature in initial tau image
                
        The method used is depends on the current ``CORR_MODE``. This method 
        only applies for correction modes 1-6.
        
        Parameters
        ----------
        tau_init : :obj:`array`, :obj:`Img`
            inital tau image
        
        Returns
        -------
        array
            modelled tau image
        
        """
        mode = self.CORR_MODE
        tau = None
        
        if not 1 <= mode <= 6:
            raise ValueError("This method only works for background model"
                "modes (param CORR_MODE) 1-6")
        try:
            tau_init = tau_init.img
        except:
            pass
        if mode == 1:
            tau = scale_tau_img(tau_init, self.scale_rect)  
        if mode == 2:
            tau = corr_tau_curvature_vert_two_rects(tau_init,
                                                    self.scale_rect,
                                                    self.ygrad_rect)
        elif mode == 3:
            tau, _ = corr_tau_curvature_vert_line(tau_init,
                                                  self.ygrad_line_colnum,
                                                  self.ygrad_line_startrow,
                                                  self.ygrad_line_stoprow, 
                                                  self.ygrad_line_mask,
                                                  self.ygrad_line_polyorder)

        elif mode == 4:
            tau = corr_tau_curvature_vert_two_rects(tau_init,
                                                    self.scale_rect,
                                                    self.ygrad_rect)
            tau = corr_tau_curvature_hor_two_rects(tau,
                                                   self.scale_rect,
                                                   self.xgrad_rect)
            
        elif mode == 5:
            tau, _ = corr_tau_curvature_vert_line(tau_init,
                                                  self.ygrad_line_colnum,
                                                  self.ygrad_line_startrow,
                                                  self.ygrad_line_stoprow,
                                                  self.ygrad_line_mask,
                                                  self.ygrad_line_polyorder)
            
            tau = corr_tau_curvature_hor_two_rects(tau,
                                                   self.scale_rect,
                                                   self.xgrad_rect)
            

        elif mode == 6:
            tau, _ = corr_tau_curvature_vert_line(tau_init,
                                                  self.ygrad_line_colnum,
                                                  self.ygrad_line_startrow,
                                                  self.ygrad_line_stoprow,
                                                  self.ygrad_line_mask,
                                                  self.ygrad_line_polyorder)
            
            tau, _ = corr_tau_curvature_hor_line(tau,
                                                 self.xgrad_line_rownum, 
                                                 self.xgrad_line_startcol,
                                                 self.xgrad_line_stopcol, 
                                                 self.xgrad_line_mask,
                                                 self.xgrad_line_polyorder)
        return tau
            
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
        try:
            self._current_imgs["plume"] = plume.duplicate()
        except:
            pass
        try:
            self._current_imgs["bg_raw"] = bg_raw.duplicate()
        except:
            pass
        try:
            self._current_imgs["tau"] = tau
        except:
            pass
        
            
    """Plotting"""
    def plot_sky_reference_areas(self, plume):
        """Plot the current sky ref areas into a plume image"""
        d = self.sky_ref_areas_to_dict()
        return plot_sky_reference_areas(plume, d)
        
    def plot_tau_result(self, tau_img=None, tau_min=None, tau_max=None,
                        edit_profile_labels=True, legend_loc=3, **add_lines):
        """Plot current tau image including all reference areas 
        
        Parameters
        ----------
        tau_img : Img
            the tau image to be displayed
        tau_min : :obj:`float`, optional
            lower tau boundary to be displayed
        tau_max : :obj:`float`, optional
            upper tau boundary for colormap
        edit_profile_labels : bool
            beta version of smart layout for axis labels from profile subplots
        **kwargs: 
            additional lines to be plotted, e.g.:: 
                pcs = [300, 400, 500, 600]
        """
        tau = tau_img
        if not isinstance(tau, Img):
            tau = self._current_imgs["tau"]
        if not isinstance(tau, Img):
            raise AttributeError("No tau image available in background model")
        tau = tau.duplicate().to_pyrlevel(0)
        tmin = tau_min
        tmax = tau_max
        if tau_max is None:
            tau_max = tau.max()
        if tau_min is None:
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
    
            ax[3].imshow(self.surface_fit_mask, cmap=palette, norm=norm,
                         alpha=.7)
            ax[3].set_title("Mask", fontsize = 10)
            ax[3].set_xticklabels([])
            ax[3].set_yticklabels([])
        
        ax[0].imshow(tau.img, cmap=cmap, vmin=tau_min, vmax=tau_max)
        
        ax[0].plot([self.ygrad_line_colnum, self.ygrad_line_colnum],
                   [0, h0],"-b", label="vert profile")
        ax[0].plot([0, w0],[self.xgrad_line_rownum, self.xgrad_line_rownum],
                    "-c", label="hor profile")
        for k, l in add_lines.iteritems():
            try:
                x0, y0, x1, y1 = l.to_list()
                c = l.color
            except:
                x0, y0, x1, y1 = l
                c="g"
        
            ax[0].plot([x0, x1],[y0,y1], "-", lw=2, c=c, label=k)
    
        ax[0].set_xlim([0, w0 - 1])
        ax[0].set_ylim([h0 - 1, 0])
        
        
        xs, ys, ws, hs = _roi_coordinates(self.scale_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="lime",fc="lime",
                        label="scale_rect", alpha=0.3))
        
        xs, ys, ws, hs = _roi_coordinates(self.ygrad_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="b",fc="b",
                        label="ygrad_rect", alpha=0.3))
        
        xs, ys, ws, hs = _roi_coordinates(self.xgrad_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="c",fc="c",
                        label="xgrad_rect", alpha=0.3))
                                                
        ax[2].set_xticklabels([])
        ax[1].set_yticklabels([])
        
        
        
        #plot vertical profile
        lvert = LineOnImage(self.ygrad_line_colnum, 0, self.ygrad_line_colnum,
                            h0 - 1, line_id="vert")
        p_vert = lvert.get_line_profile(tau.img) 
            
        ax[1].plot(p_vert, arange(0, len(p_vert), 1), "-b",
                   label="vert profile")
        ax[1].yaxis.tick_right()   
        ax[1].set_ylim([h0 - 1, 0])
        setp(ax[1].xaxis.get_majorticklabels(), rotation = 15)
        ax[1].yaxis.tick_right()   
        
        #plot horizontal profile
        line_hor = LineOnImage(0, self.xgrad_line_rownum, w0 - 1,
                               self.xgrad_line_rownum, line_id="hor")
        p_hor = line_hor.get_line_profile(tau.img)
        ax[2].plot(arange(0, len(p_hor), 1), p_hor, "-c",
                   label="hor profile")
        #ax[2].get_yaxis().set_ticks(horYLabels)
        #ax[2].set_ylim([-.05,.25])
        ax[2].set_xlim([0, w0 - 1])
    
        subplots_adjust(wspace=0.02, hspace=0.02)
        ax[2].axhline(0, ls = "--", color = "k")
        ax[1].axvline(0, ls = "--", color = "k")
        
        if edit_profile_labels:
            low, high = tmin, tmax
            if low is None:
                low = p_vert.min()
            if high is None:
                high = p_vert.max()
            _range = high - low
            lbls = [0]
            if high > 0 and high/_range > 0.2:
                lbls.append(high - _range * .05)
            if low < 0 and abs(low)/high > 0.5:
                lbls.insert(0, low + _range*.05)
            ax[1].get_xaxis().set_ticks(lbls)
            lbl_str = ["%.2f" %lbl for lbl in lbls]
            ax[1].set_xlim([low, high])
            ax[1].set_xticklabels(lbl_str)                
            
            low, high = tmin, tmax
            if low is None:
                low = p_hor.min()
            if high is None:
                high = p_hor.max()
            _range = high - low
            lbls = [0]
            if high > 0 and high / _range > 0.2:
                lbls.append(high - _range*.05)
            if low < 0 and abs(low) / high > 0.5:
                lbls.insert(0, low + _range *.05)
            ax[2].get_yaxis().set_ticks(lbls)
            lbl_str = ["%.2f" %lbl for lbl in lbls]
            ax[2].set_ylim([low, high])
            ax[2].set_yticklabels(lbl_str)         
            
        ax[1].set_xlabel(r"$\tau$", fontsize=16)
        ax[2].set_ylabel(r"$\tau$", fontsize=16)  
        fig.suptitle("CORR_MODE: %s" %self.CORR_MODE, fontsize=16)
        ax[0].legend(loc=legend_loc, fancybox=True, framealpha=0.7, fontsize=11)
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
    def mode_info_dict(self):
        """Dictionary containing information about available bg modelling modes
        """
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
                            "using horizontal profile line"],
                    [99, "USE AS IS: no background modelling performed"]])
    
    @property
    def all_modes(self):
        """List containing valid modelling modes"""
        return self.mode_info_dict.keys()        
    
    def mode_info(self, mode_num):
        """Return short information about one of the available modelling modes
        
        Parameters
        ----------
        mode_num : int
            the background modelling mode for which information is required
            
        Returns
        -------
        str
            short description of mode
        """    
        try:
            return self.mode_info_dict[mode_num]                     
        except KeyError:
            return "Mode does not exist"
    
    def print_mode_info(self):
        """Print information about the different correction modes"""
        print "Available modes for automatic plume background retrieval"
        for k, v in self.mode_info_dict.iteritems():
            print "Mode %s: %s" %(k, v)
        
    def __setitem__(self, key, value):
        """Update class item"""
        if self.__dict__.has_key(key):
            #print "Updating %s in background model" %key
            self.__dict__[key] = value
        elif key == "mode":
            "Updating %s in background model" %key
            self.mode = value
        elif key == "CORR_MODE":
            warn("Got input key CORR_MODE which is out-dated in versions 0.10+)"
                ". Updated background modelling mode accordingly")
            self.mode = value
        else:
            warn("Updating of %s in background model failed" %key)
            
    def __call__(self, plume, bg, **kwargs):
        return self.get_model(plume, bg, **kwargs)

def _mean_in_rect(img_array, rect=None):
    """Helper to get mean and standard deviation of pixels within rectangle
    
    :param ndarray imgarray: the image data
    :param rect: rectanglular area ``[x0, y0, x1, y1]` where x0 < x1, y0 < y1
    """
    if rect is None:
        sub = img_array
    else:
        sub = img_array[rect[1] : rect[3], rect[0] : rect[2]]
    return sub.mean(), sub.std()

def scale_tau_img(tau, rect):
    """Scale tau image such that it fulfills tau==0 in reference area"""
    try:
        tau = tau.img
    except:
        pass
    avg,_ = _mean_in_rect(tau, rect)
    return tau - avg
    
def scale_bg_img(bg, plume, rect):
    """Normalise background image to plume image intensity in input rect
    
    :param (ndarray, Img) bg: background image
    :param (ndarray, Img) plume: plume image
    :param list rect: rectangular area ``[x0, y0, x1, y1]``
    :return ndarray: modified background image
    """
    #extract data if input is image
    try:
        bg = bg.img
    except:
        pass
    try:
        plume = plume.img
    except:
        pass
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
    try:
        tau0 = tau0.img
    except:
        pass
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
    try:
        tau0 = tau0.img
    except:
        pass
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
    
def corr_tau_curvature_vert_line(tau0, pos_x, start_y=0, stop_y=None,
                                 row_mask=None, polyorder=2):
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
    try:
        tau0 = tau0.img
    except:
        pass
    max_y = tau0.shape[0]
    
    line_vert = LineOnImage(pos_x, 0, pos_x, max_y)
    vert_profile = line_vert.get_line_profile(tau0)
    
    if stop_y is None:
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

def corr_tau_curvature_hor_line(tau0, pos_y, start_x=0, stop_x=None,
                                col_mask=None, polyorder=2):
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
    try:
        tau0 = tau0.img
    except:
        pass
    max_x = tau0.shape[1]
    line_hor = LineOnImage(0, pos_y, max_x, pos_y)
    hor_profile = line_hor.get_line_profile(tau0)
    
    if stop_x is None:
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
        
def find_sky_reference_areas(plume_img, sigma_blur=2, plot=False):
    """Takes an input plume image and identifies suited sky reference areas"""
    try:
        plume = plume_img.img    
    except:
        plume = plume_img
    plume = gaussian_filter(plume, sigma_blur)
    h, w = plume.shape
    results = {}
    vert_mag, hor_mag = int(h * 0.005) + 1, int(w * 0.005) + 1
    
    #estimate mean intensity in left image part (without flank pixels)
    y0_left = argmin(gradient(plume[vert_mag : h - vert_mag, hor_mag]))
    
    avg_left = plume[vert_mag : y0_left - vert_mag, hor_mag:hor_mag * 2].mean()
    #estimate mean intensity in right image part (without flank pixels
    grad = gradient(plume[vert_mag : h - vert_mag, w - hor_mag])
    y0_right = argmin(grad)
    avg_right = plume[vert_mag : y0_right - vert_mag,
                      w - 2 * hor_mag : w - hor_mag].mean()
    results["xgrad_line_rownum"] = vert_mag
    if avg_right > avg_left: #brighter on the right image side (assume this is clear sky)
        results["ygrad_line_colnum"] = w - hor_mag
        results["ygrad_line_stoprow"] = int(y0_right - 0.2 * vert_mag)
        results["xgrad_line_startcol"] = int(w / 2.0)    
        results["xgrad_line_stopcol"] = int(w - 1)
        results["scale_rect"] = [int(w - 5 * hor_mag), int(vert_mag), 
                                 int(w - hor_mag), int(5 * vert_mag)]
    else:
        results["ygrad_line_colnum"] = 1
        results["ygrad_line_stoprow"] = int(y0_left - 2 * vert_mag)
        results["xgrad_line_startcol"] = hor_mag    
        results["xgrad_line_stopcol"] = int(w / 2.0)
        results["scale_rect"] = [int(hor_mag), int(vert_mag), 
                                 int(5 * hor_mag), int(5 * vert_mag)]
    results["ygrad_line_startrow"] = 1
    
    x0, y0, x1, y1 = results["scale_rect"]
    ymax = results["ygrad_line_stoprow"]
    
    results["ygrad_rect"] = [x0, int(ymax - 8 * hor_mag),
                             x1, int(ymax - 4 * hor_mag)]
                             
    results["xgrad_rect"] = [int(w / 2.0 - 2 * hor_mag), y0,
                             int(w / 2.0 + 2 * hor_mag), y1]
    if plot:
        plot_sky_reference_areas(plume, results)
    return results

def plot_sky_reference_areas(plume_img, settings_dict, ax=None):
    """Plot provided sky reference areas into a plume image
    
    :param (ndarray, Img) plume_img: plume image data
    :param dict settings_dict: dictionary containing settings (e.g. retrieved
        using :func:`find_sky_reference_areas`)
    """
    try:
        plume = plume_img.img    
    except:
        plume = plume_img
    if ax is None:
        fig, ax = subplots(1,1)
    r = settings_dict
    h0, w0 = plume.shape[:2]
    ax.imshow(plume, cmap = "gray")
    ax.plot([r["ygrad_line_colnum"], r["ygrad_line_colnum"]],
            [r["ygrad_line_startrow"], r["ygrad_line_stoprow"]],
            "-", c="lime", label = "vert profile")
    ax.plot([r["xgrad_line_startcol"], r["xgrad_line_stopcol"]], 
            [r["xgrad_line_rownum"], r["xgrad_line_rownum"]],
            "-", c="lime", label = "hor profile")
    ax.set_xlim([0, w0 - 1])
    ax.set_ylim([h0 - 1, 0])
    
    
    xs, ys, ws, hs = _roi_coordinates(r["scale_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="lime",fc="lime", alpha = 0.3,
                           label = "scale_rect"))
    
    xs, ys, ws, hs = _roi_coordinates(r["ygrad_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="b",fc="b", alpha = 0.3,
                           label = "ygrad_rect"))
    
    xs, ys, ws, hs = _roi_coordinates(r["xgrad_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="c",fc="c", alpha = 0.3,
                           label = "xgrad_rect"))
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)
        
    return ax