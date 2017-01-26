# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 09:15:51 2016

@author: jg
"""
from numpy import asarray, linspace, exp
from matplotlib.pyplot import subplots

from .processing import LineOnImage
from .image import Img
from .optimisation import dilution_corr_fit
from .model_functions import dilutioncorr_model

def get_topo_dists_lines(lines, geom, img = None, skip_pix = 5, plot = False,\
                                                        line_color = "lime"):
    if isinstance(lines, LineOnImage):
        lines = [lines]
    
    ax = None
    map3d = None
    
    #Create 3D map of scen
    pts, dists, access_mask = [], [], []
    for line in lines:
        l = line.to_list() #line coords as list
        res = geom.get_distances_to_topo_line(l, skip_pix = skip_pix)
        pts.extend(res["geo_points"])
        dists.extend(res["dists"])
        access_mask.extend(res["ok"])
        
    pts, dists = asarray(pts), asarray(dists) * 1000. 
    access_mask = asarray(access_mask)
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
        map3d.add_geo_points_3d(pts[access_mask], color = line_color)
        geom.cam_pos.plot_3d(map = map3d, add_name = True, dz_text = 40)
        map3d.ax.set_axis_off()
        
    return dists, access_mask, map3d, ax

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
    
    fit_res = dilution_corr_fit(rads, dists, rad_ambient)
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
            
        
            
        
        
                    
                
            
        