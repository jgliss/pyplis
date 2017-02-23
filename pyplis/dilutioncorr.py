# -*- coding: utf-8 -*-
"""
pyplis module for image based light dilution correction
"""
from numpy import asarray, linspace, exp, ones, nan
from matplotlib.pyplot import subplots
from collections import OrderedDict as od
from warnings import warn
from pandas import Series

from .processing import LineOnImage
from .image import Img
from .optimisation import dilution_corr_fit
from .model_functions import dilutioncorr_model
from .geometry import MeasGeometry

class DilutionCorr(object):
    """Class for management of dilution correction
    
    The class provides functionality to retrieve topographic distances from
    meas geometry, to manage lines in the image used for the retrieval, to
    perform the actual dilution fit (i.e. retrieval of atmospheric scattering
    coefficients) and to apply the dilution correction.
    
    This class does not store any results related to individual images.
    """
    def __init__(self, lines, meas_geometry, **settings):
        """Class initialisation
        
        :param lines: can be :class:`LineOnImage` or a list containing 
            such objects. Pixels on these lines are used to retrieve distances 
            to topography and radiances from plume images, respectively
        :param MeasGeometry meas_geometry: the measurement geometry
        :param **settings: settings for topo distance retrievals
        """
        if isinstance(lines, LineOnImage):
            lines = [lines]
        if not isinstance(lines, list):
            raise TypeError("Invalid input type for parameter lines, need "
                "LineOnGrid class or a python list containing LineOnGrid "
                "objects")
        if not isinstance(meas_geometry, MeasGeometry):
            raise TypeError("Invalid input type for parameter meas_geometry, "
                "need MeasGeometry class")
        self.meas_geometry = meas_geometry
        self.lines = od()

        self.settings = {"skip_pix"        :   5,
                         "min_slope_angle" :   5.0,
                         "topo_res_m"      :   5.0}
        
        self._masks = od()
        self._dists = od()
        self._skip_pix = od()
        self._geopoints = od()
        
        for line in lines:
            self.lines[line.line_id] = line
            
        self.update_settings(**settings)
     
    @property
    def line_ids(self):
        """Return IDs of all LineOnImage objects attached to this class"""
        return self.lines.keys()
        
    def update_settings(self, **settings):
        """Update settings dict"""
        for k, v in settings.iteritems():
            if self.settings.has_key(k):
                self.settings[k] = v
                
    def det_topo_dists_line(self, line_id, **settings):
        """Estimate distances to pixels on current lines

        Retrieves distances to all :class:`LineOnImage` objects  in
        ``self.lines`` using ``self.meas_geometry`` (i.e. camera position
        and viewing direction).
        
        :param **settings: update ``self.settings`` dict before search is 
            applied
        """     
        if not line_id in self.lines.keys():
            raise KeyError("No line with ID %s available" %line_id)
        self.update_settings(**settings)
        
        l = self.lines[line_id].to_list() #line coords as list
        res = self.meas_geometry.get_distances_to_topo_line(l, **self.settings)
        dists = res["dists"] * 1000. #convert to m
        self._geopoints[line_id] = res["geo_points"]
        self._dists[line_id] = dists
        self._masks[line_id] = res["ok"]
        self._skip_pix[line_id] = self.settings["skip_pix"]
        return dists
            
    def get_data(self, img, line_ids = []):
        """Returns array with all available distances
        
        :param Img img: vignetting corrected plume image
        :param list line_ids: if desired, the data can also be accessed for
            specified line ids, which have to be provided in a list. If empty
            (default), all lines are considered
        """
        if not isinstance(img, Img) or not img.edit_log["vigncorr"]:
            raise ValueError("Invalid input, need Img class and Img needs to "
                "be corrected for vignetting")
        if len(line_ids) == 0:
            line_ids = self.line_ids
            
        dists, rads = [], []
        for line_id in line_ids:
            if self._dists.has_key(line_id):
                skip = int(self._skip_pix[line_id])
                l = self.lines[line_id]
                mask = self._masks[line_id]
                dists.extend(self._dists[line_id][mask])
                rads.extend(l.get_line_profile(img)[::skip][mask])
            else:
                warn("Distances to line %s not available, please apply "
                    "distance retrieval first using class method "
                    "det_topo_dists_line")
        return asarray(dists), asarray(rads)
    
    def apply_dilution_fit(self, img, rad_ambient, i0_guess=None,
                      i0_min=0, i0_max=None, ext_guess=1e-4, ext_min=0,
                      ext_max=1e-3, line_ids =[], plot=True, **kwargs):
        """Perform dilution correction fit to retrieve extinction coefficient
        
        Uses :func:`dilution_corr_fit` of :mod:`optimisation` which is a 
        bounded least square fit based on the following model function
        
        .. math::
        
            I_{meas}(\lambda) = I_0(\lambda)e^{-\epsilon(\lambda)d} + 
            I_A(\lambda)(1-e^{-\epsilon(\lambda)d})
        
        :param Img img: vignetting corrected image for radiance extraction
        :param float rad_ambient: ambient intensity (:math:`I_A` in model)
        :param i0_guess: guess value for initial intensity of topographic 
            features, i.e. the reflected radiation before entering scattering 
            medium (:math:`I_0` in model, if None, then it is set 5% of the 
            ambient intensity ``rad_ambient``)
        :param float i0_min: minimum initial intensity of topographic features
        :param float i0_max: maximum initial intensity of topographic features
        :param float ext_guess: guess value for atm. extinction coefficient
            (:math:`\epsilon` in model)
        :param float ext_min: minimum value for atm. extinction coefficient
        :param float ext_max: maximum value for atm. extinction coefficient
        :param list line_ids: if desired, the data can also be accessed for
            specified line ids, which have to be provided in a list. If empty
            (default), all lines are considered        
        :param bool plot: if True, the result is plotted
        :param **kwargs: keyword args passed to plotting function (e.g. to 
            pass an axes object)
        """    
        dists, rads = self.get_data(img, line_ids)
        fit_res = dilution_corr_fit(rads, dists, rad_ambient, i0_guess,
                                    i0_min, i0_max, ext_guess, ext_min, ext_max)
        i0, ext = fit_res.x
        ax = None
        if plot:
            ax = self.plot_fit_result(dists, rads, rad_ambient, i0, ext, 
                                      **kwargs)
        return ext, i0, fit_res, ax
    
    def correct_img(self, plume_img, ext, plume_bg_img, plume_dist_img,\
                                                        plume_pix_mask):
        """Perform dilution correction for a plume image
        
        Corresponds to Eq. 4 in in `Campion et al., 2015 <http://
        www.sciencedirect.com/science/article/pii/S0377027315000189>`_.
        
        :param Img plume_img: vignetting corrected plume image
        :param float ext: atmospheric extinction coefficient
        :param Img plume_bg_img: vignetting corrected plume background image
            (can be, for instance retrieved using :mod:`plumebackground`)
        :param ndarray plume_dist_img: plume distance image (pixel values
            correspond to plume distances in m), can also be type :class:`Img`
        :param ndarray plume_pix_mask: mask specifying plume pixels (only those are
            corrected), can also be type :class:`Img`
        """
        for im in [plume_img, plume_bg_img]:
            if not isinstance(im, Img) or im.edit_log["vigncorr"] == False:
                raise ValueError("Plume and background image need to Img objects"
                " and vignetting corrected")
        
        try:
            plume_dist_img = plume_dist_img.img
        except:
            pass
        try:
            plume_pix_mask = plume_pix_mask.img
        except:
            pass

        dists = plume_pix_mask.astype(float) * plume_dist_img 
        corr_img = plume_img.duplicate()
        corr_img.img = (corr_img.img - plume_bg_img.img *\
                            (1 - exp(-ext * dists))) / exp(-ext * dists)
        corr_img.edit_log["dilcorr"] = True
        return corr_img
        
    def plot_fit_result(self, dists, rads, rad_ambient, i0, ext, ax = None):
        """Plot result of dilution fit"""
        if ax is None:
            fig, ax = subplots(1,1)
        x = linspace(0, dists.max(), 100)
        ints = dilutioncorr_model(x, rad_ambient, i0, ext)
        ax.plot(dists, rads, " x", label = "Data")
        lbl_fit = r"Fit result: $I_0$=%.1f DN, $\epsilon$ = %.2e" %(i0, ext)
        ax.plot(x, ints, "--c", label = lbl_fit)
        ax.set_xlabel("Distance [m]", fontsize=14)
        ax.set_ylabel("Radiances [DN]", fontsize=14)
        ax.set_title(r"$I_A$ = %.1f" %rad_ambient, fontsize=16)
        ax.grid()
        ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=13)
        return ax
    
    def get_extinction_coeffs_imglist(self, imglist, ambient_roi_abs,
                                      darkcorr=True, line_ids=[],
                                      **fit_settings):
                                            
        """Retrieve extinction coefficients for all imags in list
        
        .. note::
        
            Alpha version: not yet tested
            
        """
        imglist.aa_mode = False
        imglist.tau_mode = False
        imglist.auto_reload = False
        imglist.darkcorr_mode = True
        if imglist.gaussian_blurring and imglist.pyrlevel == 0:
            print ("Adding gaussian blurring of 2 for topographic radiance "
                "retrieval")
            imglist.gaussian_blurring = 2
        if imglist.pyrlevel != self.lines.values()[0].pyrlevel:
            raise ValueError("Mismatch in pyramid level of lines and imglist")
        if len(line_ids) == 0:
            line_ids = self.line_ids
        imglist.vigncorr_mode = True
        imglist.goto_img(0)
        imglist.auto_reload = True
        num = imglist.nof
        i0s, exts, acq_times = ones(num)*nan, ones(num)*nan, [nan]*num
        for k in range(num):
            img = imglist.current_img()
            rad_ambient = img.crop(ambient_roi_abs, True).mean()
            ext, i0, _, _ = self.apply_dilution_fit(img, rad_ambient,
                                                    line_ids=line_ids,
                                                    plot=False,
                                                    **fit_settings)
            acq_times[k] = img.meta["start_acq"]
            i0s[k] = i0
            exts[k] = ext
        
        return Series(exts, acq_times), Series(i0s, acq_times)
        
    def plot_distances_3d(self, draw_cam=1, draw_source=1, draw_plume=0,
                          draw_fov=0, cmap_topo="Oranges", axis_off=True,
                          line_ids=[], **kwargs):
        """Draw 3D map of scene including geopoints of distance retrievals
        
        :param bool draw_cam: insert camera position into map
        :param bool draw_source: insert source position into map
        :param bool draw_plume: insert plume vector into map
        :param bool draw_fov: insert camera FOV (az range) into map 
        :param cmap_topo: colormap for topography plot (default="Oranges")
        :param bool axis_off: if True, then the rendering of axes is excluded
        :param list line_ids: if desired, the data can also be accessed for
            specified line ids, which have to be provided in a list. If empty
            (default), all lines are considered
        """
        map3d = self.meas_geometry.draw_map_3d(draw_cam, draw_source, 
                                               draw_plume, draw_fov, 
                                               cmap_topo)
        if len(line_ids) == 0:
            line_ids = self.line_ids
        for line_id in self.line_ids:
            if self._dists.has_key(line_id):
                line = self.lines[line_id]
                mask = self._masks[line_id]
                pts = self._geopoints[line_id][mask]
                map3d.add_geo_points_3d(pts, color=line.color)
        if axis_off:
            map3d.ax.set_axis_off()
        return map3d
        
def get_topo_dists_lines(lines, geom, img = None, skip_pix = 5,\
        topo_res_m = 5.0, min_slope_angle = 5.0, plot = False,\
                                                    line_color = "lime"):

    if isinstance(lines, LineOnImage):
        lines = [lines]
    
    ax = None
    map3d = None
    
    pts, dists, mask = [], [], []
    for line in lines:
        l = line.to_list() #line coords as list
        res = geom.get_distances_to_topo_line(l, skip_pix, topo_res_m, 
                                              min_slope_angle)
        pts.extend(res["geo_points"])
        dists.extend(res["dists"])
        mask.extend(res["ok"])
        
    pts, dists = asarray(pts), asarray(dists) * 1000. 
    if plot:
        if isinstance(img, Img):
            ax = img.show()
            h, w = img.img.shape
            for line in lines:
                line.plot_line_on_grid(ax = ax, color = line_color, marker = "")
            ax.set_xlim([0, w - 1])
            ax.set_ylim([h - 1, 0])
        
        map3d = geom.draw_map_3d(0, 0, 0, 0, cmap_topo = "gray")
        #insert camera position into 3D map
        map3d.add_geo_points_3d(pts, color = line_color)
        geom.cam_pos.plot_3d(map = map3d, add_name = True, dz_text = 40)
        map3d.ax.set_axis_off()
        
    return dists, asarray(mask), map3d, ax

def perform_dilution_correction(plume_img, ext, plume_bg_img, plume_dist_img,\
                                                        plume_pix_mask):
    dists = plume_pix_mask.astype(float) * plume_dist_img 
    return (plume_img - plume_bg_img * (1 - exp(-ext * dists))) /\
                                                    exp(-ext * dists)
                                                
def get_extinction_coeff(rads, dists, rad_ambient, plot = True, **kwargs):
    """Perform dilution correction fit to retrieve extinction coefficient
    
    :param ndarray rads: radiances retrieved for topographic features
    :param ndarray dists: distances corresponding to ``rads``
    :param rad_ambient: ambient sky intensity
    :param bool plot: if True, the result is plotted
    :param **kwargs: additional keyword arguments for fit settings (passed
        to :func:`dilution_corr_fit` of module :mod:`optimisation`)
    """    
    
    fit_res = dilution_corr_fit(rads, dists, rad_ambient, **kwargs)
    i0, ext = fit_res.x
    ax = None
    if plot:
        x = linspace(0, dists.max(), 100)
        ints = dilutioncorr_model(x, rad_ambient, i0, ext)
        fig, ax = subplots(1,1)
        ax.plot(dists, rads, " x", label = "Data")
        lbl_fit = r"Fit result: $I_0$=%.1f DN, $\epsilon$ = %.2e" %(i0, ext)
        ax.plot(x, ints, "--c", label = lbl_fit)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Radiances [DN]")
        ax.legend(loc = "best", fancybox = True, framealpha = 0.5,\
                                                        fontsize = 12)
    return ext, i0, fit_res, ax
#==============================================================================
# class DilutionCorrection(object):
#     
#     def __init__(self, onlist, geometry, offlist = None, vign_mask = None,\
#                                                             *lines, **kwargs):
#                                                         
#         if not isinstance(onlist, ImgList):
#             raise TypeError("Invalid input: need ImgList...")
#         if not isinstance(geometry, MeasGeometry):
#             raise TypeError("Invalid input type: need MeasGeometry...")
#             
#         self.lists = {"on"  :   None, 
#                       "off" :   None,
#                       "aa"  :   None}
#     
#         self.geometry = geometry
#         
#         self.lines = {}
#         
#         self.settings = {"skip_pix"    :   10,
#                          "bg_corr_mode":   6}
#         
#         for line in lines:
#             if isinstance(line, LineOnImage):
#                 self.lines[line.line_id] = line
#         
#         for key, val in kwargs.iteritems():
#             if self.settings.has_key(key):
#                 self.settings[key] = val
#                 
#         self._check_lists(onlist, offlist)
#     
#     @property
#     def on_list(self):
#         """Get / set onband list"""
#         return self.lists["on"]
#     
#     @on_list.setter
#     def on_list(self, val):
#         dc, tau = val.dark_corr_mode, val.tau_mode
#         val.bg_model.CORR_MODE = self.bg_corr_mode
#         val.dark_corr_mode = 1
#         val.tau_mode = 1
#         val.tau_mode = tau
#         val.dark_corr_mode = dc
#         self.lists["on"] = val
#         
#     @property
#     def off_list(self): 
#         try:
#             return self.on_list.get_off_list(self._off_id)
#         except:
#             return None
#         
#     @property
#     def bg_corr_mode(self):
#         """Get / set bg corr mode in on and off image list"""
#         return self.settings["bg_corr_mode"]
#     
#     @bg_corr_mode.setter
#     def bg_corr_mode(self, value):
#         """Change background correction mode"""        
#         if int(value) in arange(7):
#              self.on_list.bg_model.CORR_MODE = value
#              self.off_list.bg_model.CORR_MODE = value
#              self.on_list.load()
#              self.off_list.load()
#              return
#         raise ValueError("Invalid input for background correction mode")
#         
#     def _check_lists(self, onlist, offlist):
#         """Check input and initiate working environment"""
#         #first check if on list is prepared for dark correction and tau img
#         #calculation
#         self.on_list = onlist
#         try:
#             dc, tau = offlist.dark_corr_mode, offlist.tau_mode
#             offlist.bg_model.CORR_MODE = self.bg_corr_mode
#             offlist.dark_corr_mode = 1
#             offlist.tau_mode = 1
#             self.on_list.link_imglist(offlist)
#             self._off_id = offlist.list_id
#             offlist.tau_mode = tau
#             offlist.dark_corr_mode = dc
#         except:
#             offlist = self.on_list.get_off_list()
#             offlist.bg_model.CORR_MODE = self.bg_corr_mode
#             dc, tau = offlist.dark_corr_mode, offlist.tau_mode
#             offlist.dark_corr_mode = 1
#             offlist.tau_mode = 1
#             offlist.tau_mode = tau
#             offlist.dark_corr_mode = dc
#             self._off_id = offlist.list_id
#         
#         self.lists["off"] = offlist
#         aa_list = deepcopy(onlist)
#         aa_list.aa_mode = True
#         aa_list.list_id = "aa"
#         self.on_list.link_imglist(aa_list)
#         self.lists["aa"] = aa_list
#==============================================================================
            
        
            
        
        
                    
                
            
        