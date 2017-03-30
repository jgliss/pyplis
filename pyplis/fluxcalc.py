# -*- coding: utf-8 -*-
"""Module containing high level functionality for emission rate analysis
"""
from warnings import warn
from numpy import dot, sqrt, mean, nan, isnan, asarray, nanmean, nanmax,\
    nanmin
from matplotlib.dates import DateFormatter
from collections import OrderedDict as od
from matplotlib.pyplot import subplots
from os.path import join, isdir
from os import getcwd

from pandas import Series, DataFrame
try:
    from scipy.constants import N_A
except:
    N_A = 6.022140857e+23

MOL_MASS_SO2 = 64.0638 #g/mol

from .imagelists import ImgList
from .plumespeed import LocalPlumeProperties  
from .processing import LineOnImage  
from .exceptions import ImgMetaError
from .helpers import map_roi, check_roi

class EmissionRateSettings(object):
    """Class for management of settings for emission rate retrievals"""
    def __init__(self, velo_glob=nan, velo_glob_err=None, **settings):
        self.velo_modes = od([("glob"               ,   True),
                              ("farneback_raw"      ,   False),
                              ("farneback_histo"    ,   False)])
        
        self.velo_glob_err = velo_glob_err
        self.velo_glob = velo_glob
        
        self.senscorr = True #apply AA sensitivity correction
        self.min_cd = -1e30 #minimum required column density for retrieval [cm-2]
        self.mmol = MOL_MASS_SO2
        
        for key, val in settings.iteritems():
            self[key] = val
        
        if self.velo_modes["glob"]:
            if velo_glob is None or isnan(velo_glob):
                warn("Deactivating velocity retrieval mode glob, since global"
                    " velocity was not provided")
                self.velo_modes["glob"] = False
        if not sum(self.velo_modes.values()) > 0:
            warn("All velocity retrieval modes are deactivated")
    
    @property
    def farneback_required(self):
        """Checks if current velocity mode settings require farneback algo"""
        for k, v in self.velo_modes.iteritems():
            if "farneback" in k and v:
                return True
        return False
        
    @property
    def velo_glob(self):
        """Get / set global velocity"""
        return self._velo_glob
        
    @velo_glob.setter
    def velo_glob(self, val):
        """Set global velocity"""
        if val < 0:
            raise ValueError("Velocity must be larger than 0")
        elif val > 40:
            warn("Large value warning: input velocity exceeds 40 m/s")
        self._velo_glob = val
        if self.velo_glob_err is None or isnan(self.velo_glob_err):
            warn("No input for global velocity error, assuming 20% of "
                "velocity")
            self.velo_glob_err = val * 0.20
        
    def __str__(self):
        """String representation"""
        s = "\npyplis settings for emission rate retrieval\n"
        s+= "--------------------------------------------\n\n"
        s+= "Velocity retrieval:\n"
        for k, v in self.velo_modes.iteritems():
            s += "%s: %s\n" %(k,v)
        s+= "\nGlobal velocity: v = (%2f +/- %.2f) m/s" %(self.velo_glob,
                                                        self.velo_glob_err)
        s+= "\nAA sensitivity corr: %s\n" %self.senscorr
        s+= "Minimum considered CD: %s cm-2\n" %self.min_cd
        s+= "Molar mass: %s g/mol\n" %self.mmol
        return s
    
    def __setitem__(self, key, val):
        """Set item method"""
        if self.__dict__.has_key(key):
            self.__dict__[key] = val
        elif self.velo_modes.has_key(key):
            self.velo_modes[key] = val
     
class EmissionRateResults(object):
    """Class to store results from emission rate analysis"""
    def __init__(self, pcs_id, velo_mode="glob", settings=None):
        self.pcs_id = pcs_id
        self.settings = settings
        self.velo_mode = velo_mode
        self._start_acq = []
        self._phi = [] #array containing emission rates
        self._phi_err = [] #emission rate errors
        self._velo_eff = [] #effective velocity through cross section
        self._velo_eff_err = [] #error effective velocity 
        
        self.pix_dist_mean = None
        self.pix_dist_mean_err = None
        self.cd_err = None
    
    
    @property
    def meta_header(self):
        """Return string containing available meta information
        
        Returns
        -------
        str
            string containing relevant meta information (e.g. for txt export)
        """
        
        date, i, f = self.get_date_time_strings()
        s = ("pcs_id=%s\ndate=%s\nstart=%s\nstop=%s\nvelo_mode=%s\n"
             "pix_dist_mean=%s m\npix_dist_mean_err=%s m\ncd_err=%s cm-2"
             %(self.pcs_id, date, i, f, self.velo_mode, self.pix_dist_mean, 
               self.pix_dist_mean_err, self.cd_err))
        return s
    
    def get_date_time_strings(self):
        """Returns string reprentations of date and start / stop times
        
        Returns
        -------
        tuple
            3-element tuple containing
            
            - date string
            - start acq. time string
            - stop acq. time string
        """
        try:
            date = self.start.strftime("%d-%m-%Y")
            start = self.start.strftime("%H:%M")
            stop = self.stop.strftime("%H:%M")
        except:
            date, start, stop = "", "", ""
        return date, start, stop
        
    def to_dict(self):
        """Write all data attributes into dictionary 
        
        Keys of the dictionary are the private class names
        
        Returns
        -------
        dict
            Dictionary containing results 
        """
        return dict(_phi            =   self.phi,
                    _phi_err        =   self.phi_err,
                    _velo_eff       =   self.velo_eff,
                    _velo_eff_err   =   self.velo_eff_err,
                    _start_acq      =   self.start_acq)
            
    def to_pandas_dataframe(self):
        """Converts object into pandas dataframe
        
        This can, for instance be used to store the data as csv (cf.
        :func:`from_pandas_dataframe`)        
        """
        d = self.to_dict()
        del d["_start_acq"]
        try:
            df = DataFrame(d, index=self.start_acq)
            return df
        except:
            warn("Failed to convert EmissionRateResults into pandas DataFrame")
    
    @property
    def default_save_name(self):
        """Returns default name for txt export"""
        try:
            d = self.start.strftime("%Y%m%d")
            i = self.start.strftime("%H%M")
            f = self.stop.strftime("%H%M")
        except:
            d, i, f = "", "", ""    
        return "pyplis_EmissionRateResults_%s_%s_%s.txt" %(d, i, f)
        
    def save_txt(self, path=None):
        """Save this object as text file"""       
        
        try:
            if isdir(path): # raises exception in case path is not valid loc
                path = join(path, self.default_save_name)
        except:
            path = join(getcwd(), self.default_save_name)
            
        self.to_pandas_dataframe().to_csv(path)
        
    def from_pandas_dataframe(self, df):
        """Import results from pandas :class:`DataFrame` object
        
        Parameters
        ----------
        df : DataFrame
            pandas dataframe containing emisison rate results
        
        Returns
        -------
        EmissionRateResults
            this object
        """
        self._start_acq = df.index.to_pydatetime()
        for key in df.keys():
            if self.__dict__.has_key(key):
                self.__dict__[key] = df[key].values
        return self
        
    @property
    def start(self):
        """Returns acquisistion time of first image"""
        return self.start_acq[0]
        
    @property
    def stop(self):
        """Returns start acqusition time of last image"""
        return self.start_acq[-1]
        
    @property
    def start_acq(self):
        """Return array containing acquisition time stamps"""
        return asarray(self._start_acq)
    
    @property
    def phi(self):
        """Return array containing emission rates"""
        return asarray(self._phi)
    
    @property
    def phi_err(self):
        """Return array containing emission rate errors"""
        return asarray(self._phi_err)
    
    @property
    def velo_eff(self):
        """Return array containing effective plume velocities"""
        return asarray(self._velo_eff)
    
    @property
    def velo_eff_err(self):
        """Return array containing effective plume velocitie errors"""
        return asarray(self._velo_eff_err)
        
    @property
    def as_series(self):
        """Return emission rate as pandas Series"""
        return Series(self.phi, self.start_acq)
    
    def plot_velo_eff(self, yerr=True, label=None, ax=None, date_fmt=None, 
                      **kwargs):
        """Plots emission rate time series
                
        Parameters
        ----------
        yerr : bool
            Include uncertainties
        label : str
            optional, string argument specifying label 
        ax
            optional, matplotlib axes object
        date_fmt : str
            optional, x label datetime formatting string, passed to 
            :class:`DateFormatter` (e.g. "%H:%M")
        **kwargs
            additional keyword args passed to plot function of :class:`Series`
            object
            
        Returns
        -------
        axes
            matplotlib axes object
            
        """
        if ax is None:
            fig, ax = subplots(1,1)
            
        if not "color" in kwargs:
            kwargs["color"] = "b" 
        if label is None:
            label = ("velo_mode: %s" %(self.velo_mode))
        
        v, verr = self.velo_eff, self.velo_eff_err
    
        s = Series(v, self.start_acq)
        try:
            s.index = s.index.to_pydatetime()
        except:
            pass
        
        s.plot(ax=ax, label=label, **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
        
        if yerr:
            phi_upper = Series(v + verr, self.start_acq)
            phi_lower = Series(v - verr, self.start_acq)
        
            ax.fill_between(s.index, phi_lower, phi_upper, alpha=0.1,
                            **kwargs)
        ax.set_ylabel(r"$v_{eff}$ [m/s]", fontsize=16)
        ax.grid()
        return ax
        
    def plot(self, yerr=True, label=None, ax=None, date_fmt=None, **kwargs):
        """Plots emission rate time series
        
        Parameters
        ----------
        yerr : bool
            Include uncertainties
        label : str
            optional, string argument specifying label 
        ax
            optional, matplotlib axes object
        date_fmt : str
            optional, x label datetime formatting string, passed to 
            :class:`DateFormatter` (e.g. "%H:%M")
        **kwargs
            additional keyword args passed to plot function of :class:`Series`
            object
            
        Returns
        -------
        axes
            matplotlib axes object
            
        """
        if ax is None:
            fig, ax = subplots(1,1)
        if not "color" in kwargs:
            kwargs["color"] = "b" 
        if label is None:
            label = ("velo_mode: %s" %(self.velo_mode))
        
        phi, phierr = self.phi, self.phi_err
        s = self.as_series
        try:
            s.index = s.index.to_pydatetime()
        except:
            pass
        ax.plot(s.index, s.values, label=label, **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
        if yerr:
            phi_upper = Series(phi + phierr, self.start_acq)
            phi_lower = Series(phi - phierr, self.start_acq)
        
            ax.fill_between(s.index, phi_lower, phi_upper, alpha=0.1,
                            **kwargs)
        ax.set_ylabel(r"$\Phi$ [g/s]", fontsize=16)
        ax.grid()
        return ax
        
    def __str__(self):
        """String representation"""
        s = "pyplis EmissionRateResults\n--------------------------------\n\n"
        s += self.meta_header
        s += ("\nphi_min=%.2f g/s\nphi_max=%.2f g/s\n"
              %(nanmin(self.phi), nanmax(self.phi)))
        s += "phi_err=%.2f g/s\n" %nanmean(self.phi_err)
        s += ("v_min=%.2f m/s\nv_max=%.2f m/s\n"
              %(nanmin(self.velo_eff), nanmax(self.velo_eff)))
        s += "v_err=%.2f m/s" %nanmean(self.velo_eff_err)
        return s
        
class EmissionRateAnalysis(object):
    """Class to perform emission rate analysis
    
    The analysis is performed by looping over images in an image list which
    is in ``calib_mode``, i.e. which loads images as gas CD images. 
    Emission rates can be retrieved for an arbitrary amount of plume cross 
    sections (defined by a list of :class:`LineOnImage` objects which can be 
    provided on init or added later). The image list needs to include a valid
    measurement geometry (:class:`MeasGeometry`) object which is used to 
    determine pixel to pixel distances (on a pixel column basis) and 
    corresponding uncertainties. 
    
    Parameters
    ----------
    imglist : ImgList
        onband image list prepared such, that at least ``aa_mode`` and 
        ``calib_mode`` can be activated. If emission rate retrieval is supposed 
        to be performed using optical flow, then also ``optflow_mode`` needs to 
        work. Apart from setting these modes, no further changes are applied to 
        the list (e.g. dark correction, blurring or choosing the pyramid level) 
        and should therefore be set before. A warning is given, in case dark 
        correction is not activated.
    
    pcs_lines : list
        python list containing :class:`LineOnImage` objects supposed to be used 
        for retrieval of emission rates (can also be a :class:`LineOnImage` 
        object directly)
    velo_glob : float
        global plume velocity in m/s (e.g. retrieved using cross correlation 
        algorithm)
    velo_glob_err : float
        uncertainty in global plume speed estimate
    bg_roi : list
        region of interest specifying gas free area in the images. It is used 
        to extract mean, max, min values from each of the calibrated images 
        during the analysis as a quality check for the performance of the plume 
        background retrieval or to detect disturbances in this region (e.g. due 
        to clouds). If unspecified, the ``scale_rect`` of the plume background 
        modelling class is used (i.e. ``self.imglist.bg_model.scale_rect``).
    **settings : 
        analysis settings (passed to :class:`EmissionRateSettings`)
        
    Todo
    ----

        1. Include light dilution correction - automatic correction for light 
        dilution is currently not supported in this object. If you wish
        to perform light dilution, for now, please calculate dilution
        corrected on and offband images first (see example script ex11) and 
        save them locally. The new set of images can then be used normally
        for the analysis by creating a :class:`Dataset` object and an 
        AA image list from that (see example scripts 1 and 4). 
            
    """
    def __init__(self, imglist, pcs_lines=[], velo_glob=nan,
                 velo_glob_err=None, bg_roi=None, **settings):

        if not isinstance(imglist, ImgList):
            raise TypeError("Need ImgList, got %s" %type(imglist))
           
        self.imglist = imglist
        self.settings = EmissionRateSettings(velo_glob, velo_glob_err,
                                             **settings)
    
        self.pcs_lines = od()        
        try:
            len(pcs_lines)
        except:
            pcs_lines = [pcs_lines]
        
        for line in pcs_lines:
            self.add_pcs_line(line)

        
        #Retrieved emission rate results are written into the following 
        #dictionary, keys are the line_ids of all PCS lines
        self.results = od()
        
        #Local plume properties (from optical flow histogram analysis), will
        #be determined for each PCS individually
        self.local_plume_props = od()
    
        if not check_roi(bg_roi):
            bg_roi = map_roi(imglist.bg_model.scale_rect,
                          pyrlevel_rel=imglist.pyrlevel)
            if not check_roi(bg_roi):
                raise ValueError("Fatal: check scale rectangle in background "
                    "model of image list...")
        
        self.bg_roi = bg_roi
        self.bg_roi_info = {"min"   :   None, 
                            "max"   :   None,
                            "mean"  :   None}
        
        self.warnings = []
            
        if not self.pcs_lines:
            self.warnings.append("No PCS analysis lines available for emission" 
                                 " rate analysis")
        try:
            self.check_and_init_list()
        except:
            self.warnings.append("Failed to initate image list for analysis "
                "check previous warnings...")
        for warning in self.warnings:
            warn(warning)
    
    @property
    def velo_glob(self):
        """Get current global velocity"""
        return self.settings.velo_glob
    
    @property
    def velo_glob_err(self):
        """Return error of current global velocity"""
        return self.settings.velo_glob_err
     
    def get_results(self, line_id=None, velo_mode=None):
        """Return emission rate results (if available)
        
        :param str line_id: ID of PCS line 
        :param str velo_mode: velocity retrieval mode (see also
            :class:`EmissionRateSettings`)
        :return: - EmissionRateResults, class containing emission rate 
            results for specified line and velocity retrieval
        :raises: - KeyError, if result for the input constellation cannot be
            found
        """
        if line_id is None:
            try:
                line_id = self.results.keys()[0]
                print "Input line ID unspecified, using: %s" %line_id
            except IndexError:
                raise IndexError("No emission rate results available...")
        if velo_mode is None:
            try:
                velo_mode = self.results[line_id].keys()[0]
                print "Input velo_mode unspecified, using: %s" %velo_mode
            except:
                raise IndexError("No emission rate results available...")
        if not self.results.has_key(line_id):
            raise KeyError("No results available for pcs with ID %s" %line_id)
        elif not self.results[line_id].has_key(velo_mode):
            raise KeyError("No results available for line %s and velocity mode"
                " %s" %(line_id, velo_mode))
        return self.results[line_id][velo_mode]
        
    def check_and_init_list(self):
        """Checks if image list is ready and includes all relevant info"""
        
        lst = self.imglist
        
        if not lst.darkcorr_mode:
            self.warnings.append("Dark image correction is not activated in "
                "image list")
        lst.auto_reload = False
        if self.settings.senscorr:
            # activate sensitivity correcion mode: images are divided by 
            try:
                lst.sensitivity_corr_mode = True
            except:
                self.warnings.append("AA sensitivity correction was deactivated"
                    "because it could not be succedfully activated in imglist")
                self.settings.senscorr = False
        
        # activate calibration mode: images are calibrated using DOAS calibration 
        # polynomial. The fitted curve is shifted to y axis offset 0 for the retrieval
        lst.calib_mode = True
        lst.auto_reload = True
        
        if self.settings.velo_glob:
            try:
                float(self.velo_glob)
            except:
                self.warnings.append("Global velocity is not available, try "
                    " activating optical flow")
                lst.optflow_mode = True
                lst.optflow.plot_flow_histograms()
                
                self.settings.velo_farneback_histo = True
        try:
            lst.meas_geometry.get_all_pix_to_pix_dists(pyrlevel=lst.pyrlevel)
        except ValueError:
            raise ValueError("measurement geometry in image list is not ready"
                "for pixel distance access")
        
    def get_pix_dist_info_all_lines(self):
        """Retrieve pixel distances and uncertainty for all pcs lines
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`dict`, keys are line ids, vals are arrays with pixel dists
            - :obj:`dict`, keys are line ids, vals are distance uncertainties
            
        """
        lst = self.imglist
        PYR = self.imglist.pyrlevel
        # get pixel distance image
        dist_img = lst.meas_geometry.get_all_pix_to_pix_dists(pyrlevel=PYR)[0]
        #init dicts
        dists, dist_errs = {}, {}
        for line_id, line in self.pcs_lines.iteritems():
            dists[line_id] = line.get_line_profile(dist_img)
            col = line.center_pix[0] #pixel column of center of PCS
            dist_errs[line_id] = lst.meas_geometry.pix_dist_err(col, PYR)
            
        return dists, dist_errs
    
    def init_results(self):
        """Reset results
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`dict`, keys are line ids, vals are empty result classes
            - :obj:`dict`, keys are line ids, vals are empty \
                :class:`LocalPlumeProperties` objects
        """
        if sum(self.settings.velo_modes.values()) == 0:
            raise ValueError("Cannot initiate result structure: no velocity "
                "retrieval mode is activated, check self.settings.velo_modes "
                "dictionary.")
    
        res = od()
        plume_props = od()
        for line_id, line in self.pcs_lines.iteritems():
            res[line_id] = od()
            plume_props[line_id] = LocalPlumeProperties(roi_id=line_id)
            for mode, val in self.settings.velo_modes.iteritems():
                if val:
                    res[line_id][mode] = EmissionRateResults(line_id, mode)
        self.results = res
        self.local_plume_props = plume_props
        self.bg_roi_info = {"min"   :   None, 
                            "max"   :   None,
                            "mean"  :   None}
        return res, plume_props
     
    def _write_meta(self, dists, dist_errs, cd_err):
        """Write meta info in result classes"""
        for line_id, mode_dict in self.results.iteritems():
            for mode, resultclass in mode_dict.iteritems():
                resultclass.pix_dist_mean = mean(dists[line_id])
                resultclass.pix_dist_mean_err = dist_errs[line_id]
                resultclass.cd_err = cd_err
        
    def calc_emission_rate(self, start_index=0, stop_index=None,
                           check_list=False):
        """Calculate emission rate based on current settings
        
        Performs emission rate analysis for each line in ``self.pcs_lines`` 
        and for all plume velocity retrievals activated in 
        ``self.settings.velo_modes``, the results for each line and 
        velocity mode are stored within :class:`EmissionRateResults` objects
        which are saved in ``self.results[line_id][velo_mode]``, e.g.::
        
            res = self.results["bla"]["farneback_histo"]
            
        would yield emission rate results for line with ID "bla" using 
        histogram based plume speed analysis. 
        
        The results can also be easily accessed using :func:`get_results`.
        
        Parameters
        ----------
        start_index : int
            index of first considered image in ``self.imglist``, defaults to 0
        stop_index : int
            index of last considered image in ``self.imglist``, defaults to 
            last image in list
        check_list : bool
            if True, :func:`check_and_init_list` is called before analysis
        
        Returns
        -------
        tuple
            2-element tuple containing
            
            - :obj:`dict`, keys are line ids, vals are corresponding results
            - :obj:`dict`, keys are line ids, vals are \
                :class:`LocalPlumeProperties` objects
                
        """
        if check_list:
            self.check_and_init_list()
        lst = self.imglist
        if stop_index is None:
            stop_index = lst.nof - 1 
        results, plume_props = self.init_results()
        dists, dist_errs = self.get_pix_dist_info_all_lines()
        lst.goto_img(start_index)
        try:
            cd_err = lst.calib_data.slope_err
        except:
            cd_err = None
            
        self._write_meta(dists, dist_errs, cd_err)
        
        # init parameters for main loop
        mmol = self.settings.mmol    
        if self.settings.farneback_required:
            lst.optflow_mode = True
        else:
            lst.optflow_mode = False #should be much faster
        vglob, vglob_err = self.velo_glob, self.velo_glob_err
        ts, bg_min, bg_mean, bg_max = [], [], [], []
        roi_bg = self.bg_roi
        velo_modes = self.settings.velo_modes
        min_cd = self.settings.min_cd
        lines = self.pcs_lines
        for k in range(start_index, stop_index):
            print "Progress: %d (%d)" %(k, stop_index)
            img = lst.current_img()
            t = lst.current_time()
            ts.append(t)
            sub = img.img[roi_bg[1] : roi_bg[3], roi_bg[0] : roi_bg[2]]
            
            bg_min.append(sub.min())
            bg_max.append(sub.max())
            bg_mean.append(sub.mean())

            for pcs_id, pcs in lines.iteritems():
                res = results[pcs_id]
                n = pcs.normal_vector
                cds = pcs.get_line_profile(img)
                cond = cds > min_cd
                cds = cds[cond]
                distarr = dists[pcs_id][cond]
                disterr = dist_errs[pcs_id]
                
                if velo_modes["glob"]:
                    phi, phi_err = det_emission_rate(cds, vglob, distarr,
                                                     cd_err, vglob_err, 
                                                     disterr, mmol)
                    if isnan(phi):
                        print cds
                        raise ValueError
                    res["glob"]._start_acq.append(t)
                    res["glob"]._phi.append(phi)
                    res["glob"]._phi_err.append(phi_err)
                    res["glob"]._velo_eff.append(vglob)
                    res["glob"]._velo_eff_err.append(vglob_err)
                    
                    
                if lst.optflow_mode:
                    delt = lst.optflow.del_t

                    if velo_modes["farneback_raw"]:
                        dx = pcs.get_line_profile(lst.optflow.flow[:,:,0])
                        dy = pcs.get_line_profile(lst.optflow.flow[:,:,1])
                        veff_arr = dot(n, (dx, dy))[cond] * distarr / delt
                        
                        phi, phi_err = det_emission_rate(cds, veff_arr,
                                                         distarr, cd_err, 
                                                         None, disterr, mmol)
                        res["farneback_raw"]._start_acq.append(t)                                
                        res["farneback_raw"]._phi.append(phi)
                        res["farneback_raw"]._phi_err.append(phi_err)

                        veff = veff_arr.mean()
                        #note that the velocity is likely underestimated due to
                        #low contrast regions (e.g. out of the plume, this can
                        #be accounted for by setting an appropriate CD minimum
                        #threshold in settings, such that the retrieval is
                        #only applied to pixels exceeding a certain column 
                        #density)
                        res["farneback_raw"]._velo_eff.append(veff)
                        res["farneback_raw"]._velo_eff_err.append(veff * .60)
                        
                    if velo_modes["farneback_histo"]:
                        props = plume_props[pcs_id]
                        # get mask specifying plume pixels
                        mask = lst.get_thresh_mask(min_cd)
                        props.get_and_append_from_farneback(lst.optflow,
                                                            line=pcs,
                                                            pix_mask=mask)
                        v, verr = props.get_velocity(-1, distarr.mean(),
                                                     disterr, 
                                                     pcs.normal_vector)
                        
                        phi, phi_err = det_emission_rate(cds, v, distarr, 
                                                         cd_err, verr, disterr, 
                                                         mmol)
                                                         
                        res["farneback_histo"]._start_acq.append(t)                                
                        res["farneback_histo"]._phi.append(phi)
                        res["farneback_histo"]._phi_err.append(phi_err)
                        res["farneback_histo"]._velo_eff.append(v)
                        res["farneback_histo"]._velo_eff_err.append(verr)
                        
            lst.next_img()  
        self.plume_properties = plume_props
        self.bg_roi_info["mean"] = Series(bg_mean, ts)
        self.bg_roi_info["max"] = Series(bg_max, ts)
        self.bg_roi_info["min"] = Series(bg_min, ts)
        
        return self.results, plume_props

    def add_pcs_line(self, line):
        """Add one analysis line to this list
        
        :param LineOnImage line: the line object
        """
        if not isinstance(line, LineOnImage):
            raise TypeError("Invalid input type for PCS line, need "
                "LineOnImage...")
        elif self.pcs_lines.has_key(line.line_id):
            raise KeyError("A PCS line with ID %s already exists in list %s"
                            %(line.line_id, self.list_id))
        elif line.pyrlevel != self.imglist.pyrlevel:
            raise ImgMetaError("Pyramid level of PCS line %s and image list %s"
                " do not match.." %(line.line_id, self.imglist.list_id))
        self.pcs_lines[line.line_id] = line
    
    def plot_pcs_lines(self):
        """Plots all current PCS lines onto current list image"""
        # plot current image in list and draw line into it
        ax = self.imglist.show_current()
        for line_id, line in self.pcs_lines.iteritems():
            line.plot_line_on_grid(ax=ax, include_normal=True, label=line_id)
        ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=12)
        return ax
    
    def plot_bg_roi_vals(self, ax=None, date_fmt=None, **kwargs):
        """Plots emission rate time series
        
        Parameters
        ----------
        ax
            optional, matplotlib axes object
        date_fmt : str
            optional, x label datetime formatting string, passed to 
            :class:`DateFormatter` (e.g. "%H:%M")
        **kwargs
            additional keyword args passed to plot function of :class:`Series`
            object
            
        Returns
        -------
        axes
            ax, matplotlib axes object
            
        """
        if ax is None:
            fig, ax = subplots(1,1)
        if not "color" in kwargs:
            kwargs["color"] = "r" 
        
        s = self.bg_roi_info["mean"]
        try:
            s.index = s.index.to_pydatetime()
        except:
            pass
        s.plot(ax=ax, label="mean", **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except:
            pass
        lower, upper = self.bg_roi_info["min"], self.bg_roi_info["max"]
        ax.fill_between(s.index, lower, upper, alpha=0.1, **kwargs)
        ax.set_ylabel(r"CD [cm-2]", fontsize=16)
        ax.grid()
        return ax
        
def det_emission_rate(cds, velo, pix_dists, cds_err=None, velo_err=None,
                      pix_dists_err=None, mmol=MOL_MASS_SO2):
    """Determine emission rate
    
    :param cds: column density in units cm-2 (float or ndarray)
    :param velo: effective plume velocity in units of m/s (float or ndarray)
        Effective means, that it is with respect to the direction of the normal 
        vector of the plume cross section used (e.g. by performing a scalar 
        product of 2D velocity vectors with normal vector of the PCS)
    :param pix_dists: pixel to pixel distances in units of m (float or ndarray)
    
    """
    if cds_err is None:
        print ("Uncertainty in column densities unspecified, assuming 20 % of "
                "mean CD")
        cds_err = mean(cds) * 0.20
    if velo_err is None:
        print ("Uncertainty in plume velocity unspecified, assuming 20 % of "
                "mean velocity")
        velo_err = mean(velo) * 0.20
        
    if pix_dists_err is None:
        print ("Uncertainty in pixel distance unspecified, assuming 10 % of "
                "mean pixel distance")
        pix_dists_err = mean(pix_dists) * 0.10
        
    C = 100**2 * mmol / N_A
    phi = sum(cds * velo * pix_dists) * C
    dphi1 = sum(velo * pix_dists * cds_err)**2
    dphi2 = sum(cds * pix_dists * velo_err)**2
    dphi3 = sum(cds * velo *pix_dists_err)**2
    phi_err = C * sqrt(dphi1 + dphi2 + dphi3)
    return phi, phi_err