# -*- coding: utf-8 -*-
"""This module contains features related to DOAS calibration including
FOV search engines"""
from numpy import min, arange, asarray, zeros, linspace, column_stack,\
    ones, nan, float32, polyfit, poly1d, sqrt, isnan, round
from scipy.stats.stats import pearsonr 
from datetime import datetime 
from scipy.sparse.linalg import lsmr
from pandas import Series
from copy import deepcopy
from astropy.io import fits
from os import remove
from os.path import join, exists, isdir, abspath, basename, dirname
from traceback import format_exc
from warnings import warn

from matplotlib.pyplot import subplots, rcParams
from matplotlib.patches import Circle, Ellipse
from matplotlib.cm import RdBu
from matplotlib.dates import DateFormatter

from .glob import SPECIES_ID, CALIB_ID_STRINGS
from .processing import ImgStack
from .helpers import shifted_color_map, mesh_from_img, get_img_maximum,\
        sub_img_to_detector_coords, map_coordinates_sub_img, exponent,\
        rotate_xtick_labels
from .optimisation import gauss_fit_2d, GAUSS_2D_PARAM_INFO
from .image import Img
from .inout import get_camera_info
from .setupclasses import Camera


class DoasCalibData(object):
    """Class containing DOAS calibration data
    
    Parameters
    ----------
    tau_vec : ndarray
        tau data vector for calibration data
    doas_vec : ndarray
        DOAS-CD data vector for calibration data
    doas_vec_err : ndarray
        Fit errors of DOAS-CDs
    time_stamps : ndarray
        array with datetime objects containing time stamps 
        (e.g. start acquisition) of calibration data
    calib_id : str
        calibration ID (e.g. "aa", "tau_on", "tau_off")
    camera : Camera
        camera object (not necessarily required). A camera can be assigned 
        in order to convert the FOV extend from pixel coordinates into 
        decimal degrees
        
    """
    def __init__(self, tau_vec=[], doas_vec=[], doas_vec_err=[], 
                 time_stamps=[], calib_id="", fov=None, camera=None, 
                 polyorder=1):
        
        #tau data vector within FOV
        self.tau_vec = asarray(tau_vec) 
        #doas data vector
        self.doas_vec = asarray(doas_vec) 
        self.doas_vec_err = asarray(doas_vec_err)
        
        self.time_stamps = time_stamps
        self.calib_id = calib_id
        
        self.camera = None
        
        if not isinstance(fov, DoasFOV):
            fov = DoasFOV(camera)
        self.fov = fov
        
        self.poly = None
        self.cov = None
        self.polyorder = polyorder
        if isinstance(camera, Camera):
            self.camera = Camera
    
    @property
    def start(self):
        """Return start datetime"""
        try:
            return self.time_stamps[0]
        except TypeError:
            return self.fov.start_search
    
    @property
    def stop(self):
        """Return start datetime"""
        try:
            return self.time_stamps[-1]
        except TypeError:
            return self.fov.stop_search
    
    @property
    def calib_id_str(self):
        """Return plot string for calibration ID"""
        try:
            return CALIB_ID_STRINGS[self.calib_id.split("_")[0]]
        except:
            return self.calib_id.split("_")[0]
            
    @property
    def coeffs(self):
        """return poly1d object of current coefficients"""
        return self.poly.coeffs 
        
    @property
    def slope(self):
        """returns slope of current calib curve"""
        return self.coeffs[-2]
        
    @property
    def slope_err(self):
        """returns slope error of current calib curve"""
        return sqrt(self.cov[-2][-2])
    
    @property
    def y_offset(self):
        """return y axis offset of calib curve"""
        return self.coeffs[-1]
    
    @property
    def y_offset_err(self):
        """return error of y axis offset of calib curve"""
        return sqrt(self.cov[-1][-1])
        
    @property
    def doas_tseries(self):
        """Return pandas Series object of doas data"""
        return Series(self.doas_vec, self.time_stamps)
    
    @property
    def tau_tseries(self):
        """Return pandas Series object of tau data"""
        return Series(self.tau_vec, self.time_stamps)
    
    @property
    def tau_range(self):
        """Returns range of tau values extended by 5%
        
        :return float tau_min: lower end of tau range
        :return float tau_max: upper end of tau range
        """
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
    
    @property
    def residual(self):
        """Residual of calibration curve"""
        return self.poly(self.tau_vec) - self.tau_vec
        
    def has_calib_data(self):
        """Checks if calibration data is available"""
        if not all([len(x) > 0 for x in [self.doas_vec, self.tau_vec]]):
            return False
        if not len(self.tau_vec) == len(self.doas_vec):
            return False
        return True
        
    def fit_calib_polynomial(self, polyorder=None, weighted=True, 
                             weights_how="rel", plot=False):
        """Fit calibration polynomial to current data
        
        Parameters
        ----------
        polyorder : :obj:`int`, optional
            update current polyorder
        weighted : bool
            performs weighted fit based on DOAS errors in ``doas_vec_err``
            (if available), defaults to True
        weights_how : str
            use "rel" if relative errors are supposed to be used (i.e.
            w=CD/CD_sigma) or "abs" if absolute error is supposed to be 
            used (i.e. w=1/CD_sigma).
        plot : bool
            If True, the calibration curve and the polynomial are plotted
        
        Returns
        -------
        poly1d
            calibration polynomial
        """
        if not weights_how in ["rel", "abs"]:
            raise IOError("Invalid input for parameter weights_how:"
                          "Use rel for relative errors or abs for absolute"
                          "errors for calculation of weights")
        if not self.has_calib_data():
            raise ValueError("Calibration data is not available")
            
        if polyorder is None:
            polyorder = self.polyorder
    
        if sum(isnan(self.tau_vec)) + sum(isnan(self.doas_vec)) > 0:
            raise ValueError("Encountered nans in data")
        
        exp = exponent(self.doas_vec.max())
        ws = ones(len(self.doas_vec))
        if weighted:
            if not len(self.doas_vec) == len(self.doas_vec_err):
                warn("Could not perform weighted calibration fit: "
                     "Length mismatch between DOAS data vector"
                     " and corresponding error vector")
            else:
                try:
                    if weights_how == "abs":
                        ws = 1 / self.doas_vec_err
                    else:
                        ws = self.doas_vec / self.doas_vec_err
                    ws = ws / max(ws)
                except:
                    warn("Failed to calculate weights")
        coeffs, cov = polyfit(self.tau_vec, self.doas_vec/10**exp, 
                              polyorder, w=ws, cov=True)
        self.polyorder = polyorder
        self.poly = poly1d(coeffs * 10**exp)
        self.cov = cov * 10**(2*exp)
        if plot:
            self.plot()
        return self.poly
    
    def save_as_fits(self, save_dir=None, save_name=None):
        """Save stack as FITS file
        
                
        """
        if not len(self.doas_vec) == len(self.tau_vec):
            raise ValueError("Could not save calibration data, mismatch in "
                " lengths of data arrays")
        if not len(self.time_stamps) == len(self.doas_vec):
            self.time_stamps = asarray([datetime(1900,1,1)] *\
                                                len(self.doas_vec))
        
        save_dir = abspath(save_dir) #returns abspath of current wkdir if None
        if not isdir(save_dir): #save_dir is a file path
            save_name = basename(save_dir)
            save_dir = dirname(save_dir)
        if save_name is None:
            save_name = "pyplis_doascalib_id_%s_%s_%s_%s.fts" %(\
                self.calib_id, self.start.strftime("%Y%m%d"),\
                self.start.strftime("%H%M"), self.stop.strftime("%H%M"))
        else:
            save_name = save_name.split(".")[0] + ".fts"
        fov_mask = fits.PrimaryHDU()
        fov_mask.data = self.fov.fov_mask
        fov_mask.header.update(self.fov.img_prep)
        fov_mask.header.update(self.fov.search_settings)
        fov_mask.header["calib_id"] = self.calib_id
        fov_mask.header.append()
        
        rd = self.fov.result_pearson
        try:
            fov_mask.header.update(cx_rel=rd["cx_rel"], cy_rel=rd["cy_rel"],\
                                   rad_rel=rd["rad_rel"])
        except:
            print "(Saving calib data): Position of FOV not available"
        
        try:
            hdu_cim = fits.ImageHDU(data = self.fov.corr_img.img)        
        except:
            hdu_cim = fits.ImageHDU()
            print "(Saving calib data): FOV search correlation image not available"
        
        tstamps = [x.strftime("%Y%m%d%H%M%S%f") for x in self.time_stamps]
        col1 = fits.Column(name = "time_stamps", format = "25A", array =\
            tstamps)
        col2 = fits.Column(name = "tau_vec", format = "D", array =\
                                                        self.tau_vec)
        col3 = fits.Column(name = "doas_vec", format = "D", array =\
                                                        self.doas_vec)
        cols = fits.ColDefs([col1, col2, col3])
        arrays = fits.BinTableHDU.from_columns(cols)
                                            
        roi = fits.BinTableHDU.from_columns([fits.Column(name = "roi",\
                                format = "I", array = self.fov.roi_abs)])
        #==============================================================================
        # col1 = fits.Column(name = 'target', format = '20A', array=a1)
        # col2 = fits.Column(name = 'V_mag', format = 'E', array=a2)
        #==============================================================================
        
        hdulist = fits.HDUList([fov_mask, hdu_cim, arrays, roi])
        fpath = join(save_dir, save_name)
        try:
            remove(fpath)
        except:
            pass
        hdulist.writeto(fpath)

    def load_from_fits(self, file_path):
        """Load stack object (fits)
        
        :param str file_path: file path of stack
        """
        if not exists(file_path):
            raise IOError("DoasCalibData object could not be loaded, "
                "path does not exist")
        hdu = fits.open(file_path)
        try:
            self.fov.fov_mask = hdu[0].data.byteswap().newbyteorder()
        except:
            print ("(Warning loading DOAS calib data): FOV mask not "
                "available")
        
        prep_keys = Img().edit_log.keys()
        search_keys = DoasFOVEngine()._settings.keys()
        self.calib_id = hdu[0].header["calib_id"]
        for key, val in hdu[0].header.iteritems():
            k = key.lower()
            if k in prep_keys:
                self.fov.img_prep[k] = val
            elif k in search_keys:
                self.fov.search_settings[k] = val
            elif k in self.fov.result_pearson.keys():
                self.fov.result_pearson[k] = val
        try:
            self.fov.corr_img = Img(hdu[1].data.byteswap().newbyteorder())
        except:
            print ("(Warning loading DOAS calib data): FOV search correlation "
                "image not available")
        try:
            times = hdu[2].data["time_stamps"].byteswap().newbyteorder()
            self.time_stamps = [datetime.strptime(x, "%Y%m%d%H%M%S%f")
                                for x in times]
        except:
            print ("(Warning loading DOAS calib data): Failed to import "
                        "time stamps")
        try:
            self.tau_vec = hdu[2].data["tau_vec"].byteswap().newbyteorder()
        except:
            print "Failed to import calibration tau data vector"
        try:
            self.doas_vec = hdu[2].data["doas_vec"].byteswap().newbyteorder()
        except:
            print "Failed to import calibration doas data vector"
            
        self.fov.roi_abs = hdu[3].data["roi"].byteswap().newbyteorder()
    
    @property
    def poly_str(self):
        """Return custom string representation of polynomial"""
        exp = exponent(self.poly.coeffs[0])
        p = poly1d(round(self.poly / 10**(exp - 2))/10**2)
        s = "(%s)E%+d" %(p, exp)
        return s.replace("x", r"$\tau$")
        
    def plot(self, add_label_str="", shift_yoffset=False, ax=None, **kwargs):
        """Plot calibration data and fit result
        
        Parameters
        ----------
        add_label_str : str
            additional string added to label of plots for legend
        shift_yoffset : bool
            if True, the data is plotted without y-offset
        ax : 
            matplotlib axes object, if None, a new one is created
        """
        if not "color" in kwargs:
            kwargs["color"] = "b"
            
        if ax is None:
            fig, ax = subplots(1,1, figsize=(10,8))
        
        taumin, taumax = self.tau_range
        x = linspace(taumin, taumax, 100)
        
        cds = self.doas_vec
        cds_poly = self.poly(x)
        if shift_yoffset:
            try:
                cds -= self.y_offset
                cds_poly -= self.y_offset
            except:
                warn("Failed to subtract y offset")
                
        ax.plot(self.tau_vec, cds, ls="", marker=".",
                label="Data %s" %add_label_str, **kwargs)
        try:
            ax.errorbar(self.tau_vec, cds, yerr=self.doas_vec_err, 
                        fmt=None, color="#919191")
        except:
            warn("No DOAS-CD errors available")
        try:
            ax.plot(x, cds_poly, ls="-", marker="",
                    label="Fit result", **kwargs)
                    
        except TypeError:
            print "Calibration poly probably not fitted"
        
        ax.set_title("DOAS calibration data, ID: %s" %self.calib_id_str)
        ax.set_ylabel(r"$S_{%s}$ [cm$^{-2}$]" %SPECIES_ID)
        ax.set_xlabel(r"$\tau_{%s}$" %self.calib_id_str)
        ax.grid()
        ax.legend(loc='best', fancybox=True, framealpha=0.7)
        return ax
        
    def plot_data_tseries_overlay(self, date_fmt=None, ax=None):
        """Plot overlay of tau and DOAS time series"""
        if ax is None:
            fig, ax = subplots(1,1)
        s1 = self.tau_tseries
        s2 = self.doas_tseries
        p1 = ax.plot(s1.index.to_pydatetime(), s1.values, "--xb", 
                     label = r"$\tau$")
        ax.set_ylabel("tau")
        ax2 = ax.twinx()
            
        p2 = ax2.plot(s2.index.to_pydatetime(), s2.values,"--xr", 
                      label="DOAS CDs")
        ax2.set_ylabel(r"$S_{%s}$ [cm$^{-2}$]" %SPECIES_ID)
        ax.set_title("Time series overlay DOAS calib data")
        
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
            
        ps = p1 + p2
        labs = [l.get_label() for l in ps]
        ax.legend(ps, labs, loc="best",fancybox=True, framealpha=0.5)
        ax.grid()
        rotate_xtick_labels(ax)
        return (ax, ax2)
    
    def err(self, value):
        """Returns measurement error of tau value based on slope error"""
        val = self(value)
        r = self.slope_err / self.slope
        return val * r
        
    def __call__(self, value,  **kwargs):
        """Define call function to apply calibration
        
        :param float value: tau or AA value
        :return: corresponding column density
        """
        if not isinstance(self.poly, poly1d):
            self.fit_calib_polynomial()
        if isinstance(value, Img):
            calib_im = value.duplicate()
            calib_im.img = self.poly(calib_im.img) - self.y_offset
            calib_im.edit_log["gascalib"] = True
            return calib_im
        elif isinstance(value, ImgStack):
            try:
                value = value.duplicate()
            except MemoryError:
                warn("Stack cannot be duplicated, applying calibration to "
                "input stack")
            value.stack = self.poly(value.stack) - self.y_offset
            value.img_prep["gascalib"] = True
            return value
        return self.poly(value) - self.y_offset
        
class DoasFOV(object):
    """Class for storage of FOV information"""
    def __init__(self, camera=None):
        self.search_settings = {}
        self.img_prep = {}
        self.roi_abs = None
        self.camera = None
        
        self.start_search = datetime(1900, 1, 1)
        self.stop_search = datetime(1900, 1, 1)
        
        self.corr_img = None
        
        self.fov_mask = None
        
        self.result_pearson = {"cx_rel"     :   nan,
                               "cy_rel"     :   nan,
                               "rad_rel"    :   nan,
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
    def pyrlevel(self):
        """Return pyramide level at which FOV search was performed"""
        try:
            return self.img_prep["pyrlevel"]
        except KeyError:
            raise KeyError("Image preparation data is not available: %s"
                           %format_exc())
        
    @property
    def cx_rel(self):
        """Return center x coordinate of FOV (in relative coords)"""
        if self.method == "ifr":
            return self.result_ifr["popt"][1]
        else:
            return self.result_pearson["cx_rel"]
            
    @property
    def cy_rel(self):
        """Return center x coordinate of FOV (in relative coords)"""
        if self.method == "ifr":
            return self.result_ifr["popt"][2]
        else:
            return self.result_pearson["cy_rel"]
    
    @property
    def radius_rel(self):
        """Returns radius of FOV (in relative coords)

        :raises: TypeError if method == "ifr"
        """
        if self.method == "ifr":
            raise TypeError("Invalid value: method IFR does not have FOV "
                "parameter radius, call self.popt for relevant parameters")
        return self.result_pearson["rad_rel"]
    
    @property
    def popt(self):
        """Return super gauss optimisation parameters (in relative coords)
        
        :raises: TypeError if method == "pearson"        
        
        """
        if self.method == "pearson":
            raise TypeError("Invalid value: method pearson does not have FOV "
                "shape parameters, call self.radius to retrieve disk radius")
        return self.result_ifr["popt"]
    
    def _max_extend_rel(self):
        """Returns maximum pixel extend of FOV
        
        For method pearson this is the radius (trivial), for an elliptical 
        super gauss (i.e. method IFR) this is the longer axis
        """
        if self.method == "pearson":
            return self.radius_rel
        else: 
            return max([self.popt[3], self.popt[3] / self.popt[4]])
         
    def pixel_extend(self, abs_coords=True):
        """Return pixel extend of FOV on image
        
        :param bool abs_coords: return value in absolute or relative 
            coordinates (considering pyrlevel and roi)        
        """
        ext_rel = self._max_extend_rel()
        if not abs_coords:
            return ext_rel
        return ext_rel*2**2
    
    @property
    def pos_abs(self):
        """Returns center coordinates of FOV (in absolute detector coords)"""
        return self.pixel_position_center(True)
        
    def pixel_position_center(self, abs_coords=False):
        """Return pixel position of center of FOV
        
        :param bool abs_coords: return position in absolute or relative 
            coordinates (considering pyrlevel and roi) 
            
        :return: 
            - tuple, ``(cx, cy)``
        """
        try:
            cx, cy = self.cx_rel, self.cy_rel
        except:
            warn("Could not access information about FOV position")
        if not abs_coords:
            return (cx, cy)
        return map_coordinates_sub_img(cx, cy, self.roi_abs, self.pyrlevel,
                                       inverse=True)
                                                                
        
    def transform_fov_mask_abs_coords(self, img_shape_orig=(), cam_id=""):
        """Converts the FOV mask to absolute detector coordinates
        
        :param tuple img_shape_orig: image shape of original image data (can
            be extracted from an unedited image), or
        :param str cam_id: string ID of pyplis default camera (e.g. "ecII")
            
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
        return sub_img_to_detector_coords(mask, img_shape_orig,
                                          self.img_prep["pyrlevel"],
                                          self.roi_abs).astype(bool)
        
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
        
    def save_as_fits(self, **kwargs):
        """Save the fov as fits file
        
        Saves this object as DoasCalibData::
        
            d = DoasCalibData(fov = self)
            d.save_as_fits(**kwargs)
            
        """
        d = DoasCalibData(fov=self)
        d.save_as_fits(**kwargs)
    
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
    
    def plot(self, ax=None):
        """Draw the current FOV position into the current correlation img"""
        if ax is None:        
            fig, ax = subplots(1, 1, figsize=(12, 8))
        else:
            fig = ax.figure
        img = self.corr_img.img
        vmin, vmax = img.min(), img.max()
        cmap = shifted_color_map(vmin, vmax, cmap=RdBu)
        h, w = img.shape
        disp = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        cb = fig.colorbar(disp, ax=ax, shrink=0.9)
        cx, cy = self.pixel_position_center(1)
        if self.method == "ifr":
            popt = self.popt
            cb.set_label(r"FOV fraction [$10^{-2}$ pixel$^{-1}$]")
            
            xgrid, ygrid = mesh_from_img(img)
            if len(popt) == 7:
                ell = Ellipse(xy = (popt[1], popt[2]), width=popt[3],
                              height=popt[3]/popt[4], color="k", lw=2,
                              fc="lime", alpha=.5)
            else:
                ell = Ellipse(xy = (popt[1], popt[2]), width=popt[3],
                              height=popt[3]/popt[4], angle=popt[7], color="k",
                              lw=2, fc="lime", alpha=.5)
                    
            ax.add_artist(ell)
            ax.axhline(self.cy_rel, ls="--", color="k")
            ax.axvline(self.cx_rel, ls="--", color="k")

            ax.get_xaxis().set_ticks([0, self.cx_rel, w])
            ax.get_yaxis().set_ticks([0, self.cy_rel, h])
            
            #ax.set_axis_off()
            ax.set_title(r"Corr img (IFR), pos abs (x,y): (%d, %d), "
                "lambda=%.1e" %(cx, cy, self.search_settings["ifrlbda"]))
                        
        elif self.method == "pearson":
            cb.set_label(r"Pearson corr. coeff.")
            ax.autoscale(False)
            
            c = Circle((self.cx_rel, self.cy_rel), self.radius_rel, ec="k",
                       lw=2, fc="lime", alpha=.5)
            ax.add_artist(c)
            ax.set_title("Corr img (pearson), pos abs (x,y): (%d, %d)" 
                            %(cx, cy))
            ax.get_xaxis().set_ticks([0, self.cx_rel, w])
            ax.get_yaxis().set_ticks([0, self.cy_rel, h])
            ax.axhline(self.cy_rel, ls="--", color="k")
            ax.axvline(self.cx_rel, ls="--", color="k")
        ax.set_xlabel("Pixel row")
        ax.set_ylabel("Pixel column")    
        return ax

class DoasFOVEngine(object):
    """Engine to perform DOAS FOV search"""
    def __init__(self, img_stack=None, doas_series=None, method="pearson",
                 **settings):
        
        self._settings = {"method"              :   "pearson",
                          "maxrad"              :   80,
                          "ifrlbda"             :   1e-6, #lambda val IFR
                          "g2dasym"             :   True, #elliptic FOV
                          "g2dsuper"            :   True, #super gauss fit (IFR)
                          "g2dcrop"             :   True,
                          "g2dtilt"             :   False,
                          "blur"                :   4,
                          "mergeopt"            :   "average"}
        
        
        self.DATA_MERGED = False
        self.img_stack = img_stack
        self.doas_series = doas_series
        
        self.calib_data = DoasCalibData() #includes DoasFOV class
        
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
        
        All relevant results are written into ``self.calib_data`` (which 
        includes :class:`DoasFOV` object)
        
        
        """
        self.calib_data = DoasCalibData() #includes DoasCalibData class
        self.update_search_settings(**settings)
        self.merge_data(merge_type=self._settings["mergeopt"])
        self.det_correlation_image(search_type=self.method)
        self.get_fov_shape()
        self.calib_data.fov.search_settings = deepcopy(self._settings)
        
        return self.calib_data
        
                    
    def merge_data(self, merge_type="average"):
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
        
        new_stack, new_doas_series = self.img_stack.merge_with_time_series(
                                        self.doas_series, 
                                        method=merge_type)
        if len(new_doas_series) == new_stack.shape[0]:
            self.img_stack = new_stack
            self.doas_series = new_doas_series
            self._settings["mergeopt"] = merge_type
            self.DATA_MERGED = True
            return True
        print "Data merging failed..."
        return False
    
    def det_correlation_image(self, search_type="pearson", **kwargs):
        """Determines correlation image
        
        Determines correlation image either using IFR or Pearson method.
        Results are written into ``self.calib_data.fov`` (:class:`DoasFOV`)
        
        :param str search_type: updates current search type, available types
            ``["pearson", "ifr"]``
        """
        if not self.img_stack.shape[0] == len(self.doas_series):
            raise ValueError("DOAS correlation image object could not be "
                "determined: inconsistent array lengths, please perform time"
                "merging first")
        self.update_search_settings(method=search_type, **kwargs)
        if search_type == "pearson":
            corr_img, _ = self._det_correlation_image_pearson(
                                                    **self._settings)
        elif search_type == "ifr":
            corr_img, _ = self._det_correlation_image_ifr_lsmr(
                                                    **self._settings)
        else:
            raise ValueError("Invalid search type %s: choose from "
                             "pearson or ifr" %search_type)
        corr_img = Img(corr_img, pyrlevel=
                       self.img_stack.img_prep["pyrlevel"])
        #corr_img.pyr_up(self.img_stack.img_prep["pyrlevel"])
        self.calib_data.fov.corr_img = corr_img
        self.calib_data.fov.img_prep = self.img_stack.img_prep
        self.calib_data.fov.roi_abs = self.img_stack.roi_abs
        self.calib_data.fov.start_search = self.img_stack.start
        self.calib_data.fov.stop_search = self.img_stack.stop
        self.calib_data.calib_id = self.img_stack.stack_id
        
        return corr_img
        
    def _det_correlation_image_pearson(self, **kwargs):
        """Determine correlation image based on pearson correlation
        
        :returns: - correlation image (pix wise value of pearson corr coeff)
        """
        h,w = self.img_stack.shape[1:]
        corr_img = zeros((h,w), dtype = float)
        corr_img_err = zeros((h,w), dtype = float)
        doas_vec = self.doas_series.values
        exp = int(10**exponent(h) / 4.0)
        for i in range(h):
            try:
                if i % exp == 0:
                    print "FOV search: current img row (y): " + str(i)
            except:
                pass
            for j in range(w):
                #get series from stack at current pixel
                corr_img[i,j], corr_img_err[i,j] = pearsonr(\
                        self.img_stack.stack[:, i, j], doas_vec)
        self._settings["method"] = "pearson"
        return corr_img, corr_img_err
    
    def _det_correlation_image_ifr_lsmr(self, ifrlbda=1e-6, **kwargs):
        """Apply LSMR algorithm to identify the FOV
        
        :param float ifrlbda: tolerance parameter lambda
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
        a = lsmr(h, self.doas_data_vec, atol = ifrlbda, btol=ifrlbda)
        c = a[0]
        # separate offset and image
        lsmr_offset = c[0]
        lsmr_image = c[1:].reshape(ny, nx) / max(c[1:])
        #THIS NORMALISATION IS NEW
        #lsmr_image = lsmr_image / abs(lsmr_image).max()
        self._settings["method"] = "ifr"
        self._settings["ifrlbda"] = ifrlbda
        return lsmr_image, lsmr_offset
    
    def get_fov_shape(self, **settings):
        """Find shape of FOV based on correlation image
        
        Search pixel coordinate of highest correlation in 
        ``self.calib_data.fov.corr_img`` (using :func:`get_img_maximum`) and 
        based on that finds FOV shape either using disk approach (if 
        ``self.method == 'pearson'``) calling :func:`fov_radius_search` or
        using 2D Gauss fit (if ``self.method == 'ifr'``) calling 
        :func:`fov_gauss_fit`. Results are written into ``self.calib_data.fov`` 
        (:class:`DoasFOV` object)
        
        :param **settings: update current settings (keyword args passed 
            to :func:`update_search_settings`)
        
        """
        
        if not isinstance(self.calib_data.fov.corr_img, Img):
            raise Exception("Could not access correlation image")
        if self.method == "pearson":
            cy, cx = get_img_maximum(self.calib_data.fov.corr_img.img,
                                     gaussian_blur=self._settings["blur"])
            print "Start radius search in stack around x/y: %s/%s" %(cx, cy)
            radius, corr_curve, tau_vec, doas_vec, fov_mask =\
                                    self.fov_radius_search(cx, cy)
            if not radius > 0:
                raise ValueError("Pearson FOV search failed")
    
            self.calib_data.fov.result_pearson["cx_rel"] = cx
            self.calib_data.fov.result_pearson["cy_rel"] = cy
            self.calib_data.fov.result_pearson["rad_rel"] = radius
            self.calib_data.fov.result_pearson["corr_curve"] = corr_curve
            
            self.calib_data.fov.fov_mask = fov_mask
            self.calib_data.tau_vec = tau_vec
            self.calib_data.doas_vec = doas_vec
            try:
                self.calib_data.doas_vec_err = self.doas_series.fit_errs
            except:
                pass
            self.calib_data.time_stamps = self.img_stack.time_stamps
            return 
        
        elif self.method == "ifr":
            #the fit is performed in absolute dectector coordinates
            #corr_img_abs = Img(self.fov.corr_img.img).pyr_up(pyrlevel).img
            popt, pcov, fov_mask = self.fov_gauss_fit(
                            self.calib_data.fov.corr_img.img, **self._settings)
            tau_vec = self.convolve_stack_fov(fov_mask)
            
            self.calib_data.fov.result_ifr["popt"] = popt
            self.calib_data.fov.result_ifr["pcov"] = pcov
            self.calib_data.fov.fov_mask = fov_mask            
            self.calib_data.tau_vec = tau_vec
            self.calib_data.doas_vec = self.doas_data_vec
            try:
                self.calib_data.doas_vec_err = self.doas_series.fit_errs
            except:
                pass
            self.calib_data.time_stamps = self.img_stack.time_stamps
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
        if self._settings["maxrad"] < max_rad:
            max_rad = self._settings["maxrad"]
        else:
            self._settings["maxrad"] = max_rad
        #radius array
        radii = arange(1, max_rad + 1, 1)
        print "Maximum radius: " + str(max_rad - 1)
        #some variable initialisations
        coeffs, coeffs_err = [], []
        max_corr = 0
        tau_vec = None
        mask = zeros((h, w)).astype(float32)
        radius = 0
        #loop over all radii, get tauSeries at each, (merge) and determine 
        #correlation coefficient
        for r in radii:
            print "current radius:" + str(r)
            #now get mean values of all images in stack in circular ROI around
            #CFOV
            tau_series, m = stack.get_time_series(cx, cy, radius=r)
            tau_dat = tau_series.values
            coeff, err = pearsonr(tau_dat, doas_vec)
            coeffs.append(coeff)
            coeffs_err.append(err)
            #and append correlation coefficient to results
            if coeff > max_corr:
                radius = r
                mask = m.astype(float32)
                max_corr = coeff
                tau_vec = tau_dat
        corr_curve = Series(asarray(coeffs, dtype = float),radii)
        return radius, corr_curve, tau_vec, doas_vec, mask
        
    # define IFR model function (Super-Gaussian)    
        
    def fov_gauss_fit(self, corr_img, g2dasym=True, g2dsuper=True,
                      g2dcrop=True, g2dtilt=False, blur=4, **kwargs):
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
        (cy, cx) = get_img_maximum(corr_img, blur)
        # constrain fit, if requested
        (popt, pcov, fov_mask) = gauss_fit_2d(corr_img, cx, cy, g2dasym,\
            g2d_super_gauss = g2dsuper, g2d_crop = g2dcrop,\
                                            g2d_tilt = g2dtilt, **kwargs)
        # normalise
        return (popt, pcov, fov_mask)
    
    #function convolving the image stack with the obtained FOV distribution    
    def convolve_stack_fov(self, fov_mask):
        """Normalize fov image and convolve stack
        
        :returns: - stack time series vector within FOV
        """
        # normalize fov_mask
        normsum = fov_mask.sum()
        fov_mask_norm = fov_mask / normsum
        # convolve with image stack
        #stack_data_conv = transpose(self.stac, (2,0,1)) * fov_fitted_norm
        stack_data_conv = self.img_stack.stack * fov_mask_norm
        return stack_data_conv.sum((1,2))
        
            
