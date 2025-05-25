# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""Pyplis module containing methods and classes for emission-rate retrievals.
"""
from numpy import (dot, sqrt, mean, nan, isnan, asarray, nanmean, nanmax,
                   nanmin, sum, arctan2, rad2deg, logical_and, ones, arange,
                   nanstd)
from matplotlib.dates import DateFormatter
from collections import OrderedDict as od
from matplotlib.pyplot import subplots, rcParams, Rectangle
from os.path import join, isdir
from os import getcwd
from traceback import format_exc

from pyplis import logger
from pyplis.utils import LineOnImage
from pyplis.imagelists import ImgList
from pyplis.plumespeed import LocalPlumeProperties
from pyplis.helpers import check_roi, exponent, roi2rect, map_roi

import pandas as pd
from pandas import Series, DataFrame
from scipy.constants import N_A

MOL_MASS_SO2 = 64.0638  # g/mol

class EmissionRateSettings(object):
    """Class for management of settings for emission rate retrievals.

    Parameters
    ----------
    pcs_lines :
        :class:`LineOnImage` object or list containing :class:`LineOnImage`
        objects along which emission rates are retrieved.
    velo_glob : float
        optional, global velocity estimate (e.g. retrieved from cross
        correlation analysis). Please note, that global velocities can also
        be assigned directly to :class:`LineOnImage` objects (see prev. inp.
        param), hence, this input velocity is only used for lines, which do
        not have an explicit global velocity assigned. In any case, these
        velocities (whether assigned in :class:`LineOnImage`objects or here)
        are only used if ``self.velo_mode["glob"] is True``.
    velo_glob_err : float
        optional, error on prev. parameter
    bg_roi_abs : list
        background region of interest used for logging of retrieved CDs in an
        area out of the plume (can later be used for an assessment of the
        performance of the plume background retrieval for each image) since the
        CDs are expected to be zero.
    ref_check_lower_lim : float
        lower required limit for CDs in ``bg_roi_abs`` area in case
        ``ref_check_mode`` is active. All images which show average CDs lower
        than this thresh within ``bg_roi_abs`` are disregarded for the analysis
    ref_check_upper_lim : float
        upper required limit for CDs in ``bg_roi_abs`` area in case
        ``ref_check_mode`` is active. All images which show average CDs larger
        than this thresh within ``bg_roi_abs`` are disregarded for the analysis

    """

    def __init__(self, pcs_lines=None, velo_glob=nan, velo_glob_err=nan,
                 bg_roi_abs=None, ref_check_lower_lim=None,
                 ref_check_upper_lim=None, **settings):

        # allow input for older version attributes
        if pcs_lines is None:
            pcs_lines = []
        if "bg_roi" in settings and bg_roi_abs is None:
            bg_roi_abs = settings["bg_roi"]
            del settings["bg_roi"]

        self.velo_modes = od([("glob", True),
                              ("flow_raw", False),
                              ("flow_histo", False),
                              ("flow_hybrid", False)])

        # empirically determined intrinsic error of farneback optical flow
        # with respect to the effective velocities (i.e. dot product of optical
        # flow vector with PCS normal).
        self.optflow_err_rel_veff = 0.15
        self.pcs_lines = od()

        # Dictionary that will be filled with flags (in method add_pcs_line)
        # specifying whether or not local plume displacement information
        # (class LocalPlumeProperties, retrieved using optical flow histogram
        # analysis) are available in within the provided LineOnImage objects
        # along which the emission rates are retrieved.
        self.plume_props_available = od()

        self._velo_glob = nan
        self._velo_glob_err = nan
        self.velo_glob = velo_glob
        self.velo_glob_err = velo_glob_err

        self._bg_roi_abs = bg_roi_abs
        self._ref_check_mode = False
        self.ref_check_lower_lim = ref_check_lower_lim
        self.ref_check_upper_lim = ref_check_upper_lim

        self.velo_dir_multigauss = True
        self.senscorr = True  # apply AA sensitivity correction
        self.dilcorr = False
        self.live_calib = False
        self.min_cd = -1e30  # min required column density for retrieval [cm-2]
        self.min_cd_flow = nan
        self.mmol = MOL_MASS_SO2

        try:
            len(pcs_lines)
        except BaseException:
            pcs_lines = [pcs_lines]

        for line in pcs_lines:
            self.add_pcs_line(line)

        for key, val in settings.items():
            self[key] = val

        if self.velo_modes["glob"]:
            if not self._check_velo_glob_access():
                logger.warning("Deactivating velocity retrieval mode glob, since global"
                     " velocity was not provided")
                self.velo_modes["glob"] = False
        if not sum(list(self.velo_modes.values())) > 0:
            logger.warning("All velocity retrieval modes are deactivated")

    @property
    def bg_roi_abs(self):
        """Return current background reference ROI."""
        return self._bg_roi_abs

    @bg_roi_abs.setter
    def bg_roi_abs(self, value):
        if not check_roi(value):
            raise ValueError("Invalid ROI: %s, need list: [x0,y0,x1,y1]"
                             % value)
        self._bg_roi_abs = value

    @property
    def ref_check_mode(self):
        """Activate / deactivate reference area control mode."""
        return self._ref_check_mode

    @ref_check_mode.setter
    def ref_check_mode(self, value):
        try:
            value = bool(value)
        except BaseException:
            raise ValueError("Need bool or similar")
        if value:
            if not check_roi(self.bg_roi_abs):
                raise ValueError("Cannot activate ref_check_mode: bg_roi is "
                                 "not set")
            try:
                self.ref_check_lower_lim = float(self.ref_check_lower_lim)
            except BaseException:
                raise ValueError("Please assign a valid lower value (type "
                                 "float) for reference check in bg_roi using "
                                 "attr. ref_check_lower_lim")
            try:
                self.ref_check_upper_lim = float(self.ref_check_upper_lim)
            except BaseException:
                raise ValueError("Please assign a valid upper value (type "
                                 "float) for reference check in bg_roi using "
                                 "attr. ref_check_upper_lim")
        self._ref_check_mode = value

    @property
    def velo_mode_glob(self):
        """Attribute velo_glob for velocity analysis retrieval."""
        return self.velo_modes["glob"]

    @velo_mode_glob.setter
    def velo_mode_glob(self, val):
        self.velo_modes["glob"] = bool(val)

    @property
    def velo_mode_flow_raw(self):
        """Attribute velo_glob for velocity analysis retrieval."""
        return self.velo_modes["flow_raw"]

    @velo_mode_flow_raw.setter
    def velo_mode_flow_raw(self, val):
        self.velo_modes["flow_raw"] = bool(val)

    @property
    def velo_mode_flow_histo(self):
        """Attribute for velocity analysis retrieval."""
        return self.velo_modes["flow_histo"]

    @velo_mode_flow_histo.setter
    def velo_mode_flow_histo(self, val):
        self.velo_modes["flow_histo"] = bool(val)

    @property
    def velo_mode_flow_hybrid(self):
        """Attribute for velocity analysis retrieval."""
        return self.velo_modes["flow_hybrid"]

    @velo_mode_flow_hybrid.setter
    def velo_mode_flow_hybrid(self, val):
        self.velo_modes["flow_hybrid"] = bool(val)

    @property
    def velo_glob(self):
        """Global velocity in m/s, assigned to this line.

        Raises
        ------
        AttributeError
            if current value is not of type float

        """
        return self._velo_glob

    @velo_glob.setter
    def velo_glob(self, val):
        try:
            val = float(val)
        except BaseException:
            raise ValueError("Invalid input, need float or int...")
        if val < 0:
            raise ValueError("Velocity must be larger than 0")
        elif val > 40:
            logger.warning("Large value warning: input velocity exceeds 40 m/s")
        self._velo_glob = val
        if self.velo_glob_err is None or isnan(self.velo_glob_err):
            logger.warning("Global velocity error not assigned, assuming 50% of "
                 "velocity")
            self.velo_glob_err = val * 0.50

    @property
    def velo_glob_err(self):
        """Error of global velocity in m/s, assigned to this line."""
        return self._velo_glob_err

    @velo_glob_err.setter
    def velo_glob_err(self, val):
        try:
            val = float(val)
        except BaseException:
            raise ValueError("Invalid input, need float or int...")
        if not isnan(val):
            self._velo_glob_err = val

    def _check_velo_glob_access(self):
        """Check if global velocity information is accessible for all lines."""
        vglob = self.velo_glob
        if not isnan(float(vglob)):
            return True
        for l in self.pcs_lines.values():
            try:
                l.velo_glob
            except BaseException:
                return False
        return True

    def add_pcs_line(self, line):
        """Add one analysis line to this list.

        Parameters
        ----------
        line : LineOnImage
            emission rate retrieval line

        """
        if not isinstance(line, LineOnImage):
            raise TypeError("Invalid input type for PCS line, need "
                            "LineOnImage...")
        elif line.line_id in self.pcs_lines:
            raise KeyError("A PCS line with ID %s already exists"
                           % (line.line_id))
        try:
            line.velo_glob  # raises exception if not assigned
        except BaseException:
            try:
                line.velo_glob = self.velo_glob
                err = self.velo_glob_err
                if isinstance(err, float) and not isnan(err):
                    line.velo_glob_err = err
            except BaseException:
                pass
        try:
            line.plume_props  # raises exception if not assigned
            self.plume_props_available[line.line_id] = 1
        except BaseException:
            logger.info("Creating new LocalPlumeProperties object in line %s"
                  % line.line_id)
            line.plume_props = LocalPlumeProperties(roi_id=line.line_id)
            self.plume_props_available[line.line_id] = 0

        self.pcs_lines[line.line_id] = line

    def __str__(self):
        s = "\npyplis settings for emission rate retrieval\n"
        s += "--------------------------------------------\n\n"
        s += "Retrieval lines:\n"
        if bool(self.pcs_lines):
            for v in self.pcs_lines.values():
                s += "%s\n" % (v)
        else:
            s += "No PCS lines assigned yet ...\n"
        s += "\nVelocity retrieval:\n"
        for k, v in self.velo_modes.items():
            s += "%s: %s\n" % (k, v)
        s += "\nGlobal velocity: v = (%2f +/- %.2f) m/s" % (self.velo_glob,
                                                            self.velo_glob_err)
        s += "\nAA sensitivity corr: %s\n" % self.senscorr
        s += "Dilution correction: %s\n" % self.dilcorr
        s += "Minimum considered CD: %s cm-2\n" % self.min_cd
        s += "Molar mass: %s g/mol\n" % self.mmol
        return s

    def __setitem__(self, key, val):
        """Set item method."""
        if key in self.__dict__:
            self.__dict__[key] = val
        elif key in self.velo_modes:
            self.velo_modes[key] = val


class EmissionRates(object):
    """Class to store results from emission rate analysis."""

    def __init__(self, pcs_id, velo_mode="glob", settings=None, color="b"):
        self.pcs_id = pcs_id
        self.settings = settings
        self.velo_mode = velo_mode
        self._start_acq = []
        self._phi = []  # array containing emission rates
        self._phi_err = []  # emission rate errors
        self._velo_eff = []  # effective velocity through cross section
        self._velo_eff_err = []  # error effective velocity
        # fraction of reliable optical flow vectors along the retrieval line
        # only relevant for histogram based optical flow retrieval
        # (e.g. 0.6 means that 40% of the optical flow vectors were replaced)
        # with average vector derived from histograms
        self._frac_optflow_ok = []
        # fraction / impact of reliable optical flow vectors on integrated
        # column amount (ICA) along retrieval line L. E.g. 0.8 means that 80%
        # of the accumulated ICA along L corresponds to vectors for which the
        # optical flow is reliable. This information is also only relevant
        # for flow_hybrid velocity method and may be used to assess the
        # impact of erroneous flow vectors (which are replaced with result
        # from histogram analysis) relative to the actual flux
        self._frac_optflow_ok_ica = []

        self.pix_dist_mean = None
        self.pix_dist_mean_err = None
        self.cd_err = None

        self.color = color

    @property
    def start(self):
        """Acquisistion time of first image."""
        return self.start_acq[0]

    @property
    def stop(self):
        """Start acqusition time of last image."""
        return self.start_acq[-1]

    @property
    def start_acq(self):
        """Array containing acquisition time stamps."""
        return asarray(self._start_acq)

    @property
    def phi(self):
        """Array containing emission rates."""
        return asarray(self._phi)

    @property
    def phi_err(self):
        """Array containing emission rate errors."""
        return asarray(self._phi_err)

    @property
    def velo_eff(self):
        """Array containing effective plume velocities."""
        return asarray(self._velo_eff)

    @property
    def velo_eff_err(self):
        """Array containing effective plume velocitie errors."""
        return asarray(self._velo_eff_err)

    @property
    def as_series(self):
        """Emission rates as pandas Series."""
        return Series(self.phi, self.start_acq)

    @property
    def meta_header(self):
        """Return string containing available meta information.

        Returns
        -------
        str
            string containing relevant meta information (e.g. for txt export)

        """
        date, i, f = self.get_date_time_strings()
        s = ("pcs_id=%s\ndate=%s\nstart=%s\nstop=%s\nvelo_mode=%s\n"
             "pix_dist_mean=%s m\npix_dist_mean_err=%s m\ncd_err=%s cm-2"
             % (self.pcs_id, date, i, f, self.velo_mode, self.pix_dist_mean,
                self.pix_dist_mean_err, self.cd_err))
        return s

    @property
    def default_save_name(self):
        """Return default name for txt export."""
        try:
            d = self.start.strftime("%Y%m%d")
            i = self.start.strftime("%H%M")
            f = self.stop.strftime("%H%M")
        except BaseException:
            d, i, f = "", "", ""
        return "pyplis_EmissionRates_%s_%s_%s.txt" % (d, i, f)

    def mean(self):
        """Mean of emission rate time series."""
        return self.phi.mean()

    def nanmean(self):
        """Mean of emission rate time series excluding NaNs."""
        return nanmean(self.phi)

    def std(self):
        """Mean of emission rate time series."""
        return self.phi.std()

    def nanstd(self):
        """Mean of emission rate time series excluding NaNs."""
        return nanstd(self.phi)

    def min(self):
        """Minimum value of emission rate time series."""
        return self.phi.min()

    def nanmin(self):
        """Minimum value of emission rate time series excluding NaNs."""
        return nanmin(self.phi)

    def max(self):
        """Maximum value of emission rate time series."""
        return self.phi.max()

    def nanmax(self):
        """Maximum value of emission rate time series excluding NaNs."""
        return nanmax(self.phi)

    def get_date_time_strings(self):
        """Return string reprentations of date and start / stop times.

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
        except BaseException:
            date, start, stop = "", "", ""
        return date, start, stop

    def to_dict(self):
        """Write all data attributes into dictionary.

        Keys of the dictionary are the private class names

        Returns
        -------
        dict
            Dictionary containing results

        """
        return dict(_phi=self.phi,
                    _phi_err=self.phi_err,
                    _velo_eff=self.velo_eff,
                    _velo_eff_err=self.velo_eff_err,
                    _frac_optflow_ok=self._frac_optflow_ok,
                    _frac_optflow_ok_ica=self._frac_optflow_ok_ica,
                    _start_acq=self.start_acq)

    def _fill_missing_data(self):
        """Check length of all data arrays and fill nans where data is missing.
        """
        d = self.to_dict()
        num = len(self.start_acq)
        for k, v in d.items():
            if not len(v) == num:
                self.__dict__[k] = [nan] * num

    def to_pandas_dataframe(self):
        """Convert object into pandas dataframe.

        This can, for instance be used to store the data as csv (cf.
        :func:`from_pandas_dataframe`)
        """
        self._fill_missing_data()
        d = self.to_dict()
        del d["_start_acq"]
        try:
            df = DataFrame(d, index=self.start_acq)
            return df
        except BaseException:
            logger.warning("Failed to convert EmissionRates into pandas DataFrame")

    def from_pandas_dataframe(self, df):
        """Import results from pandas :class:`DataFrame` object.

        Parameters
        ----------
        df : DataFrame
            pandas dataframe containing emisison rate results

        Returns
        -------
        EmissionRates
            this object

        """
        self._start_acq = df.index.to_pydatetime()
        for key in df.keys():
            if key in self.__dict__:
                self.__dict__[key] = df[key].values
        return self

    def plot_velo_eff(self, yerr=True, label=None, ax=None, date_fmt=None,
                      **kwargs):
        """Plot emission rate time series.

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
            fig, ax = subplots(1, 1)

        if "color" not in kwargs:
            kwargs["color"] = "b"
        if label is None:
            label = ("velo_mode: %s" % (self.velo_mode))

        v, verr = self.velo_eff, self.velo_eff_err

        s = Series(v, self.start_acq)
        try:
            s.index = s.index.to_pydatetime()
        except BaseException:
            pass

        s.plot(ax=ax, label=label, **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except BaseException:
            pass

        if yerr:
            phi_upper = Series(v + verr, self.start_acq)
            phi_lower = Series(v - verr, self.start_acq)

            ax.fill_between(s.index, phi_lower, phi_upper, alpha=0.1,
                            **kwargs)
        ax.set_ylabel(r"$v_{eff}$ [m/s]")
        ax.grid()
        return ax

    def plot(self, yerr=True, label=None, ax=None, date_fmt=None, ymin=None,
             ymax=None, alpha_err=0.1, in_kg=True, **kwargs):
        """Plot emission rate time series.

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
        ymin : :obj:`float`, optional
            lower limit of y-axis
        ymax : :obj:`float`, optional
            upper limit of y-axis
        alpha_err : float
            transparency of uncertainty range
        in_kg : bool
            if True, emission rates are plotted in units of kg / s
        **kwargs
            additional keyword args passed to plot call

        Returns
        -------
        axes
            matplotlib axes object

        """
        if ax is None:
            fig, ax = subplots(1, 1)
            ax.grid()
        if "color" not in kwargs:
            kwargs["color"] = self.color
        if label is None:
            label = ("velo: %s" % (self.velo_mode))

        phi, phierr = self.phi, self.phi_err
        s = self.as_series
        unit = "g/s"
        if in_kg:
            s /= 1000.0
            unit = "kg/s"
        try:
            s.index = s.index.to_pydatetime()
        except BaseException:
            pass

        pl = ax.plot(s.index, s.values, label=label, **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except BaseException:
            pass
        if yerr:
            phi_upper = Series(phi + phierr, self.start_acq)
            phi_lower = Series(phi - phierr, self.start_acq)
            if in_kg:
                phi_lower /= 1000.0
                phi_upper /= 1000.0
            ax.fill_between(s.index, phi_lower, phi_upper, alpha=alpha_err,
                            color=pl[0].get_color())
        ax.set_ylabel(r"$\Phi$ [%s]" % unit)

        ylim = list(ax.get_ylim())
        if ymin is not None:
            ylim[0] = ymin
        if ymax is not None:
            ylim[1] = ymax
        ax.set_ylim(ylim)
        return ax

    def save_txt(self, path=None):
        """Save this object as text file."""
        try:
            if isdir(path):  # raises exception in case path is not valid loc
                path = join(path, self.default_save_name)
        except BaseException:
            path = join(getcwd(), self.default_save_name)

        self.to_pandas_dataframe().to_csv(path)

    def load_txt(self, path):
        """Load results from text file.

        Parameters
        ----------
        path : str
            valid file location

        Returns
        -------
        EmissionRates
            loaded result data class

        """
        df = pd.read_csv(path)
        return self.from_pandas_dataframe(df)

    def __add__(self, other):
        """Add emission rate results from two result classes.

        The values of the emission rates ``phi`` are added, the other data
        (``phi_err, velo_eff, velo_eff_err``) are averaged between this and
        the other time series.

        Parameters
        ----------
        other : EmissionRates
            emission rate results from a different position in the image

        Returns
        -------
        EmissionRates
            added results

        """
        if not isinstance(other, EmissionRates):
            raise ValueError("Invalid input, need EmissionRates class")
        df = self.to_pandas_dataframe()
        df1 = other.to_pandas_dataframe()
        df["_phi"] += df1["_phi"]
        df["_phi_err"] = (df["_phi_err"] + df1["_phi_err"]) / 2.
        df["_velo_eff"] = (df["_velo_eff"] + df1["_velo_eff"]) / 2.
        df["_velo_eff"] = (df["_velo_eff_err"] + df1["_velo_eff_err"]) / 2.
        new_id = "%s + %s" % (self.pcs_id, other.pcs_id)

        new = EmissionRates(new_id)
        new.from_pandas_dataframe(df)
        try:
            pdm_diff = abs(self.pix_dist_mean - other.pix_dist_mean)
            new.pix_dist_mean = nanmean(
                [self.pix_dist_mean, other.pix_dist_mean])
            pdm_err = nanmean(
                [self.pix_dist_mean_err, other.pix_dist_mean_err])
            new.pix_dist_mean_err = max([pdm_diff, pdm_err])
        except BaseException:
            logger.warning("Could not access meta info pix_dist_mean in flux results")
        try:
            new.cd_err = nanmean([self.cd_err, other.cd_err])
        except BaseException:
            logger.warning("Could not access meta cd_err in flux results")
        return new

    def __sub__(self, other):
        """Subtract emission rate results from two result classes.

        The values of the emission rates ``phi`` are subtracted, the other data
        (``phi_err, velo_eff, velo_eff_err``) are averaged between this and
        the other time series.

        Parameters
        ----------
        other : EmissionRates
            emission rate results from a different position in the image

        Returns
        -------
        EmissionRates
            added results

        """
        if not isinstance(other, EmissionRates):
            raise ValueError("Invalid input, need EmissionRates class")
        df = self.to_pandas_dataframe()
        df1 = other.to_pandas_dataframe()
        df["_phi"] -= df1["_phi"]
        df["_phi_err"] = (df["_phi_err"] + df1["_phi_err"]) / 2.
        df["_velo_eff"] = (df["_velo_eff"] + df1["_velo_eff"]) / 2.
        df["_velo_eff"] = (df["_velo_eff_err"] + df1["_velo_eff_err"]) / 2.
        new_id = "%s - %s" % (self.pcs_id, other.pcs_id)

        new = EmissionRates(new_id)
        new.from_pandas_dataframe(df)
        try:
            pdm_diff = abs(self.pix_dist_mean - other.pix_dist_mean)
            new.pix_dist_mean = nanmean(
                [self.pix_dist_mean, other.pix_dist_mean])
            pdm_err = nanmean(
                [self.pix_dist_mean_err, other.pix_dist_mean_err])
            new.pix_dist_mean_err = max([pdm_diff, pdm_err])
        except BaseException:
            logger.warning("Could not access meta info pix_dist_mean in flux results")
        try:
            new.cd_err = nanmean([self.cd_err, other.cd_err])
        except BaseException:
            logger.warning("Could not access meta cd_err in flux results")

        return new

    def __truediv__(self, other):
        """Divide other emission rate results.

        The values of the emission rates ``phi`` are divided, the other data
        (``phi_err, velo_eff, velo_eff_err``) are averaged between this and
        the other time series.

        Parameters
        ----------
        other : EmissionRates
            emission rate results from a different position in the image

        Returns
        -------
        EmissionRates
            added results

        """
        if not isinstance(other, EmissionRates):
            raise ValueError("Invalid input, need EmissionRates class")
        df = self.to_pandas_dataframe()
        df1 = other.to_pandas_dataframe()
        df["_phi"] /= df1["_phi"]
        df["_phi_err"] = df["_phi"] * sqrt((df["_phi_err"] / df["_phi"])**2 +
                                           (df1["_phi_err"] / df1["_phi"]**2))

        df["_velo_eff"] = (df["_velo_eff"] + df1["_velo_eff"]) / 2.
        df["_velo_eff"] = (df["_velo_eff_err"] + df1["_velo_eff_err"]) / 2.
        new_id = "%s / %s" % (self.pcs_id, other.pcs_id)

        new = EmissionRateRatio(new_id)
        new.from_pandas_dataframe(df)
        try:
            pdm_diff = abs(self.pix_dist_mean - other.pix_dist_mean)
            new.pix_dist_mean = nanmean(
                [self.pix_dist_mean, other.pix_dist_mean])
            pdm_err = nanmean(
                [self.pix_dist_mean_err, other.pix_dist_mean_err])
            new.pix_dist_mean_err = max([pdm_diff, pdm_err])
        except BaseException:
            logger.warning("Could not access meta info pix_dist_mean in flux results")
        try:
            new.cd_err = nanmean([self.cd_err, other.cd_err])
        except BaseException:
            logger.warning("Could not access meta cd_err in flux results")

        return new

    def __str__(self):
        s = "pyplis EmissionRates\n--------------------------------\n\n"
        s += self.meta_header
        s += ("\nphi_min=%.2f g/s\nphi_max=%.2f g/s\n"
              % (nanmin(self.phi), nanmax(self.phi)))
        s += "phi_err=%.2f g/s\n" % nanmean(self.phi_err)
        s += ("v_min=%.2f m/s\nv_max=%.2f m/s\n"
              % (nanmin(self.velo_eff), nanmax(self.velo_eff)))
        s += "v_err=%.2f m/s" % nanmean(self.velo_eff_err)
        return s


class EmissionRateRatio(EmissionRates):
    """Time series ratio of two emission rates.

    This class is new and still in Beta status
    """

    def __init__(self, *args, **kwargs):
        super(EmissionRateRatio, self).__init__(*args, **kwargs)

    @property
    def dphi(self):
        """Return attr. phi, as this class represents ratios."""
        return self.phi

    @property
    def dphi_err(self):
        """Return for attr. phi_err, as this class represents ratios."""
        return self.phi_err

    def plot(self, yerr=False, label=None, ax=None, date_fmt=None, ymin=None,
             ymax=None, alpha_err=0.1, **kwargs):
        ax = super(EmissionRateRatio, self).plot(yerr, label, ax, date_fmt,
                                                 ymin, ymax, alpha_err,
                                                 in_kg=False,
                                                 **kwargs)
        ax.set_ylabel("")
        return ax


class EmissionRateAnalysis:
    """Class to perform emission rate analysis.

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

    def __init__(self, imglist, **settings):

        if not isinstance(imglist, ImgList):
            raise TypeError("Need ImgList, got %s" % type(imglist))

        self.imglist = imglist
        self._imglist_optflow = None
        self.settings = EmissionRateSettings(**settings)

        # Retrieved emission rate results are written into the following
        # dictionary, keys are the line_ids of all PCS lines
        self.results = od()

        if not check_roi(self.settings.bg_roi_abs):
            try:
                bg_roi_abs = imglist.bg_model.scale_rect
                if not check_roi(bg_roi_abs):
                    raise ValueError("Fatal: check scale rectangle in "
                                     "background model of image list...")
            except BaseException:
                logger.warning("Failed to access scale rectangle in background model "
                     "of image list, setting bg_roi to lower left image "
                     "corner")
                bg_roi_abs = [5, 5, 20, 20]
            self.settings.bg_roi_abs = bg_roi_abs
        self.bg_roi_info = {"mean": None,
                            "std": None}

        self.warnings = []

        if not self.pcs_lines:
            self.warnings.append("No PCS analysis lines available for emission"
                                 " rate analysis")
        try:
            self.check_and_init_list()
        except BaseException:
            self.warnings.append("Failed to initate image list for analysis "
                                 "check previous warnings...")
        for warning in self.warnings:
            logger.warning(warning)

    @property
    def imglist_optflow(self):
        """Image list supposed to be used for optical flow retrieval.

        Is required to have the same number of images than analysis list. If
        this list is not set explicitely, then the optical flow is calculated
        from the analysis list (default setting).

        This feature was introduced, since it was empirically found, that
        images that are dilution corrected, often cause problems with the
        optical flow retrieval, due to the applied threshold

        .. note:

            Beta version

        """
        if not isinstance(self._imglist_optflow, ImgList):
            return self.imglist
        return self._imglist_optflow

    @imglist_optflow.setter
    def imglist_optflow(self, val):
        if val is self.imglist:
            raise IOError("Input list for optical flow retrieval is the same"
                          " as current analysis list")
        if isinstance(val, ImgList) and val.nof == self.imglist.nof:
            val.goto_img(self.imglist.cfn)
            logger.info("Setting list for optflow retrieval, current mode status:")
            for k, v in val._list_modes.items():
                logger.info("%s: %s" % (k, v))

            self._imglist_optflow = val
            self.imglist.link_imglist(self._imglist_optflow)
        else:
            raise IOError("Failed to assign optical flow image list")

    @property
    def pcs_lines(self):
        """Return dict containing PCS retrieval lines assigned to settings class.
        """
        return self.settings.pcs_lines

    @property
    def velo_glob(self):
        """Global velocity."""
        return self.settings.velo_glob

    @property
    def velo_glob_err(self):
        """Return error of current global velocity."""
        return self.settings.velo_glob_err

    @property
    def flow_required(self):
        """Check if current velocity mode settings require flow algo."""
        s = self.settings
        if s.velo_modes["flow_raw"] or s.velo_modes["flow_hybrid"]:
            return True
        elif s.velo_modes["flow_histo"]:
            d = s.plume_props_available
            if not sum(list(d.values())) == len(d):
                return True
        return False

    def get_results(self, line_id=None, velo_mode=None):
        """Return emission rate results (if available).

        :param str line_id: ID of PCS line
        :param str velo_mode: velocity retrieval mode (see also
            :class:`EmissionRateSettings`)
        :return: - EmissionRateResults, class containing emission rate
            results for specified line and velocity retrieval
        :raises: - KeyError, if result for the input constellation cannot be
            found
        """
        if line_id is None:
            if len(self.results) > 0:
                line_id = list(self.results)[0]
                logger.info(f"Input line ID unspecified, using: {line_id}")
            else:
                raise ValueError("No emission rate results available...")
        if velo_mode is None:
            if len(self.results[line_id]) > 0:
                velo_mode = list(self.results[line_id])[0]
                logger.info(f"Input velo_mode unspecified, using: {velo_mode}")
            else:
                raise ValueError("No emission rate results available...")
        if line_id not in self.results:
            raise ValueError(f"No results available for pcs with ID {line_id}")
        elif velo_mode not in self.results[line_id]:
            raise ValueError(f"No results available for line {line_id} and velocity mode {velo_mode}")
        return self.results[line_id][velo_mode]

    def check_and_init_list(self):
        """Check if image list is ready and include all relevant info."""
        lst = self.imglist
        # activate calibration mode: images are calibrated using DOAS
        # calibration polynomial. The fitted curve is shifted to y axis
        # offset 0 for the retrieval
        lst.calib_mode = True

        if self.settings.velo_glob:
            try:
                float(self.velo_glob)
            except BaseException:
                self.warnings.append("Global velocity is not available, try "
                                     " activating optical flow")
                self.imglist_optflow.optflow_mode = True
                self.imglist_optflow.optflow.plot_flow_histograms()

                self.settings.velo_flow_histo = True
        try:
            lst.meas_geometry.compute_all_integration_step_lengths(
                pyrlevel=lst.pyrlevel)
        except ValueError:
            raise ValueError("measurement geometry in image list is not ready"
                             "for pixel distance access")
        if not lst.darkcorr_mode:
            self.warnings.append("Dark image correction is not activated in "
                                 "image list")
        if self.settings.senscorr:
            # activate sensitivity correcion mode: images are divided by
            try:
                lst.sensitivity_corr_mode = True
            except BaseException:
                self.warnings.append("AA sensitivity correction was "
                                     "deactivated because it could not be "
                                     "succedfully activated in imglist")
                self.settings.senscorr = False
        if self.settings.dilcorr:
            lst.dilcorr_mode = True

    def get_pix_dist_info_all_lines(self):
        """Retrieve pixel distances and uncertainty for all pcs lines.

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
        dist_img = lst.meas_geometry.compute_all_integration_step_lengths(
            pyrlevel=PYR)[0]
        # init dicts
        dists, dist_errs = {}, {}
        for line_id, line in self.pcs_lines.items():
            dists[line_id] = line.get_line_profile(dist_img)
            col = line.center_pix[0]  # pixel column of center of PCS
            dist_errs[line_id] = lst.meas_geometry.pix_dist_err(col, PYR)

        return (dists, dist_errs)

    def init_results(self):
        r"""Reset results.

        Returns
        -------
        tuple
            2-element tuple containing

            - :obj:`dict`, keys are line ids, vals are empty result classes
            - :obj:`dict`, keys are line ids, vals are empty \
                :class:`LocalPlumeProperties` objects

        """
        if sum(list(self.settings.velo_modes.values())) == 0:
            raise ValueError("Cannot initiate result structure: no velocity "
                             "retrieval mode is activated, check "
                             "self.settings.velo_modes "
                             "dictionary.")

        res = od()
        for line_id in self.pcs_lines:
            res[line_id] = od()
            for mode, val in self.settings.velo_modes.items():
                if val:
                    res[line_id][mode] = EmissionRates(line_id, mode)
        self.results = res
        self.check_pcs_plume_props()
        self.bg_roi_info = {"mean": None,
                            "std": None}
        return res

    def check_pcs_plume_props(self):
        """Check if plume displacement information is available for all PCS.

        Tries to access :class:`LocalPlumeProperties` objects in each of the
        assigned plume cross section retrieval lines (:attr:`pcs_lines`). If
        so and if a considerable datetime index overlap is given in the
        corresponding object (with datetime indices in :attr:`imglist`), then
        the object is interpolated onto the time stamps of the list and the
        corresponding displacement information is used (and not re-calculated)
        while performing emission rate retrieval when using
        ``velo_mode = flow_histo``. If no significant overlap can be
        detected, the :class:`LocalPlumeProperties` object in the corresponding
        :class:`LineOnImage` object is initiated and filled while performing
        the analysis.
        """
        lst = self.imglist
        span = (lst.stop - lst.start).total_seconds()

        for key, line in self.pcs_lines.items():
            try:
                p = line.plume_props
                dt0 = (p.start - lst.start).total_seconds()
                if dt0 > 0 and dt0 / span > 0.05:
                    raise ValueError("Insufficient overlap of time stamps in "
                                     "plume properties of line %s with "
                                     "timestamps in list...")
                dt1 = (lst.stop - p.stop).total_seconds()
                if dt1 > 0 and dt1 / span > 0.05:
                    raise ValueError("Insufficient overlap of time stamps in "
                                     "plume properties of line %s with "
                                     "timestamps in list...")
                line.plume_props = p.interpolate(time_stamps=lst.start_acq,
                                                 how="time")
                self.settings.plume_props_available[key] = 1
            except Exception as e:
                if isinstance(e, ValueError):
                    logger.warning(format_exc(e))
                self.settings.plume_props_available[key] = 0
                line.plume_props = LocalPlumeProperties(roi_id=key)

    def _write_meta(self, dists, dist_errs, cd_err):
        """Write meta info in result classes."""
        for line_id, mode_dict in self.results.items():
            for mode, resultclass in mode_dict.items():
                resultclass.pix_dist_mean = mean(dists[line_id])
                resultclass.pix_dist_mean_err = dist_errs[line_id]
                resultclass.cd_err = cd_err

    def calc_emission_rate(self, **kwargs):
        """Old name of :func:`run_retrieval`."""
        logger.warning("Old name of method run_retrieval")
        return self.run_retrieval(**kwargs)

    def run_retrieval(self, start_index=0, stop_index=None, check_list=True):
        r"""Calculate emission rates of image list.

        Performs emission rate analysis for each line in ``self.pcs_lines``
        and for all plume velocity retrievals activated in
        ``self.settings.velo_modes``. The results for each line and
        velocity mode are stored within :class:`EmissionRates` objects
        which are saved in ``self.results[line_id][velo_mode]``, e.g.::

            res = self.results["bla"]["flow_histo"]

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

        num = lst._iter_num(start_index, stop_index)
        flow = self.imglist_optflow.optflow
        s = self.settings
        results = self.init_results()
        dists, dist_errs = self.get_pix_dist_info_all_lines()
        lst.goto_img(start_index)
        try:
            cd_err = lst.calib_data.err()
        except ValueError as e:
            logger.warning("Calibration error could not be accessed: {}".format(repr(e)))
            cd_err = None

        self._write_meta(dists, dist_errs, cd_err)

        # init parameters for main loop
        mmol = s.mmol
        if self.flow_required:
            self.imglist_optflow.optflow_mode = True
        else:
            lst.optflow_mode = False  # should be much faster
        ts, bg_mean, bg_std = [], [], []
        counter = 0
        fl_sigma_tol = flow.settings.hist_sigma_tol
        fl_min_len = flow.settings.min_length
        roi_bg_abs = self.settings.bg_roi_abs
        velo_modes = s.velo_modes
        min_cd = s.min_cd
        min_cd_flow = min_cd if isnan(s.min_cd_flow) else s.min_cd_flow
        gauss_fit = s.velo_dir_multigauss
        lines = self.pcs_lines
        pnum = int(10**exponent(num) / 4.0)
        imin, imax = s.ref_check_lower_lim, s.ref_check_upper_lim
        for k in range(num):
            img = lst.current_img()
            t = lst.current_time()
            ts.append(t)
            ok = True
            try:
                sub = img.crop(roi_bg_abs, new_img=True)
                # sub = img.img[roi_bg[1] : roi_bg[3], roi_bg[0] : roi_bg[2]]
                avg = sub.mean()
                bg_mean.append(avg)
                bg_std.append(sub.std())
                if self.settings.ref_check_mode:
                    if not imin < avg < imax:
                        ok = False
            except BaseException:
                logger.warning("Failed to retrieve data within background ROI (bg_roi)"
                     "writing NaN")
                bg_std.append(nan)
                bg_mean.append(nan)
                if self.settings.ref_check_mode:
                    ok = False
            if ok:
                for pcs_id, pcs in lines.items():
                    res = results[pcs_id]
                    n = pcs.normal_vector
                    cds = pcs.get_line_profile(img)
                    cond = cds > min_cd
                    cds = cds[cond]
                    distarr = dists[pcs_id][cond]
                    disterr = dist_errs[pcs_id]

                    if velo_modes["glob"]:
                        try:
                            vglob, vglob_err = pcs.velo_glob, pcs.velo_glob_err
                        except BaseException:
                            vglob, vglob_err = self.velo_glob,\
                                self.velo_glob_err
                        phi, phi_err = det_emission_rate(cds, vglob, distarr,
                                                         cd_err, vglob_err,
                                                         disterr, mmol)
                        if isnan(phi):
                            logger.info(cds)
                            raise ValueError
                        res["glob"]._start_acq.append(t)
                        res["glob"]._phi.append(phi)
                        res["glob"]._phi_err.append(phi_err)
                        res["glob"]._velo_eff.append(vglob)
                        res["glob"]._velo_eff_err.append(vglob_err)
                    dx, dy = None, None
                    if velo_modes["flow_raw"]:
                        delt = flow.del_t

                        # retrieve diplacement vectors along line
                        dx = pcs.get_line_profile(flow.flow[:, :, 0])
                        dy = pcs.get_line_profile(flow.flow[:, :, 1])

                        # detemine array containing effective velocities
                        # through the line using dot product with line normal
                        veff_arr = dot(n, (dx, dy))[cond] * distarr / delt

                        # Calculate mean of effective velocity through l and
                        # uncertainty using 2 sigma confidence of standard
                        # deviation
                        veff_avg = veff_arr.mean()
                        veff_err = veff_avg * \
                            self.settings.optflow_err_rel_veff

                        phi, phi_err = det_emission_rate(cds, veff_arr,
                                                         distarr, cd_err,
                                                         veff_err, disterr,
                                                         mmol)
                        res["flow_raw"]._start_acq.append(t)
                        res["flow_raw"]._phi.append(phi)
                        res["flow_raw"]._phi_err.append(phi_err)

                        # note that the velocity is likely underestimated due
                        # to low contrast regions (e.g. out of the plume, this
                        # can be accounted for by setting an appropriate CD
                        # minimum threshold in settings, such that the
                        # retrieval is only applied to pixels exceeding a
                        # certain column density)
                        res["flow_raw"]._velo_eff.append(veff_avg)
                        res["flow_raw"]._velo_eff_err.append(veff_err)

                    props = pcs.plume_props
                    verr = None
                    if velo_modes["flow_histo"]:
                        if s.plume_props_available[pcs_id]:
                            idx = k
                        else:
                            # get mask specifying plume pixels
                            mask = lst.get_thresh_mask(min_cd_flow)
                            props.\
                                get_and_append_from_farneback(
                                    flow,
                                    line=pcs,
                                    pix_mask=mask,
                                    dir_multi_gauss=gauss_fit)
                            idx = -1

                        # logger.info("IMGLIST CTIME: %s "
                        #       % self.imglist.current_time())
                        # get effective velocity through the pcs based on
                        # results from histogram analysis
                        (v,
                         verr) = props.get_velocity(idx, distarr.mean(),
                                                    disterr,
                                                    pcs.normal_vector,
                                                    sigma_tol=fl_sigma_tol)
                        # logger.info("HISTO VEFF: %.2f m/s" %v)
                        phi, phi_err = det_emission_rate(cds, v, distarr,
                                                         cd_err, verr, disterr,
                                                         mmol)

                        res["flow_histo"]._start_acq.append(t)
                        res["flow_histo"]._phi.append(phi)
                        res["flow_histo"]._phi_err.append(phi_err)
                        res["flow_histo"]._velo_eff.append(v)
                        res["flow_histo"]._velo_eff_err.append(verr)

                    if velo_modes["flow_hybrid"]:
                        # get results from local plume properties analysis
                        if not velo_modes["flow_histo"]:
                            if s.plume_props_available[pcs_id]:
                                idx = k
                            else:
                                # get mask specifying plume pixels
                                mask = lst.get_thresh_mask(min_cd_flow)
                                props.\
                                    get_and_append_from_farneback(
                                        flow,
                                        line=pcs,
                                        pix_mask=mask,
                                        dir_multi_gauss=gauss_fit)
                                idx = -1

                        if dx is None:
                            # extract raw diplacement vectors along line
                            dx = pcs.get_line_profile(flow.flow[:, :, 0])
                            dy = pcs.get_line_profile(flow.flow[:, :, 1])

                        if verr is None:
                            # get effective velocity through the pcs based on
                            # results from histogram analysis
                            (_,
                             verr) = props.get_velocity(idx, distarr.mean(),
                                                        disterr,
                                                        pcs.normal_vector,
                                                        sigma_tol=fl_sigma_tol)
                        # determine orientation angles and magnitudes along
                        # raw optflow output
                        phis = rad2deg(arctan2(dx, -dy))[cond]
                        mag = sqrt(dx**2 + dy**2)[cond]

                        # get expectation values of predominant displacement
                        # vector
                        min_len = (props.len_mu[idx] - props.len_sigma[idx])

                        min_len = max([min_len, fl_min_len])

# ==============================================================================
#                         print "LEN_MU: %.2f" %props.len_mu[idx]
#                         print "LEN_SIGMA: %.2f" %props.len_sigma[idx]
#                         print "MIN LENGTH: %s" %min_len
# ==============================================================================
                        dir_min = (props.dir_mu[idx] -
                                   fl_sigma_tol * props.dir_sigma[idx])
                        dir_max = (props.dir_mu[idx] +
                                   fl_sigma_tol * props.dir_sigma[idx])

                        # get bool mask for indices along the pcs
                        bad = ~ (logical_and(phis > dir_min, phis < dir_max) *
                                 (mag > min_len))

                        frac_bad = sum(bad) / float(len(bad))
                        indices = arange(len(bad))[bad]
                        # now check impact of ill-constraint motion vectors
                        # on ICA
                        ica_fac_ok = sum(cds[~bad] / sum(cds))

                        vec = props.displacement_vector(idx)

                        flc = flow.replace_trash_vecs(displ_vec=vec,
                                                      min_len=min_len,
                                                      dir_low=dir_min,
                                                      dir_high=dir_max)

                        delt = flow.del_t
                        dx = pcs.get_line_profile(flc.flow[:, :, 0])
                        dy = pcs.get_line_profile(flc.flow[:, :, 1])
                        veff_arr = dot(n, (dx, dy))[cond] * distarr / delt

                        # Calculate mean of effective velocity through l and
                        # uncertainty using 2 sigma confidence of standard
                        # deviation
                        veff_avg = veff_arr.mean()
                        fl_err = veff_avg * self.settings.optflow_err_rel_veff

                        # logger.info("Assumed intrinsic optflow error veff=%.2f m/s"
                        #       % fl_err)
                        # neglect uncertainties in the successfully constraint
                        # flow vectors along the pcs by initiating an zero
                        # array ...
                        veff_err_arr = ones(len(veff_arr)) * fl_err
                        # ... and set the histo errors for the indices of
                        # ill-constraint flow vectors on the pcs (see above)
                        veff_err_arr[indices] = verr

                        phi, phi_err = det_emission_rate(cds, veff_arr,
                                                         distarr, cd_err,
                                                         veff_err_arr,
                                                         disterr, mmol)
                        veff_err_avg = veff_err_arr.mean()


# ==============================================================================
#                         logger.info("Fraction of bad vectors along %s): %.3f"
#                               % (pcs_id, frac_bad))
#                         logger.info("Kappa: %.3f %%" %(ica_fac_ok))
# ==============================================================================
                        # logger.info("Avg. eff. velocity (hybrid) = %.2f +/- %.2f"
                        #       %(veff_avg, veff_err_avg))
                        res["flow_hybrid"]._start_acq.append(t)
                        res["flow_hybrid"]._phi.append(phi)
                        res["flow_hybrid"]._phi_err.append(phi_err)
                        res["flow_hybrid"]._velo_eff.append(veff_avg)
                        res["flow_hybrid"]._velo_eff_err.append(veff_err_avg)
                        res["flow_hybrid"]._frac_optflow_ok.append(
                            1 - frac_bad)
                        res["flow_hybrid"]._frac_optflow_ok_ica.append(
                            ica_fac_ok)
                counter += 1
            else:
                logger.warning("Skipped image no. %d" % k)
            try:
                if k % pnum == 0:
                    logger.info("Progress: %d (%d)" % (k, num))
            except:
                pass
            lst.goto_next()

        self.bg_roi_info["mean"] = Series(bg_mean, ts)
        self.bg_roi_info["std"] = Series(bg_std, ts)

        if not counter > 0:
            raise ValueError("Emission rate retrieval failed for all images "
                             "in image list...")
        logger.info("Emission rates could be successfully retrieved for %d of %d"
              "images in image list" % (counter, (stop_index - start_index)))
        return self.results

    def add_pcs_line(self, line):
        """Add one analysis line to this list.

        :param LineOnImage line: the line object
        """
        self.settings.add_pcs_line(line)

    def plot_pcs_lines(self, ax=None, **kwargs):
        """Plot all current PCS lines onto current list image."""
        # plot current image in list and draw line into it
        if ax is None:
            ax = self.imglist.show_current(**kwargs)
        for line_id, line in self.pcs_lines.items():
            line.plot_line_on_grid(ax=ax, include_normal=True, label=line_id)
        ax.legend(loc='best', fancybox=True, framealpha=0.5)
        return ax

    def plot_bg_roi_rect(self, ax=None, to_pyrlevel=0):
        """Plot rectangular area used for background check."""
        roi = map_roi(self.settings.bg_roi_abs, to_pyrlevel)
        x, y, w, h = roi2rect(roi)
        r = Rectangle(xy=(x, y), width=w, height=h, fc="none", ec="r")
        ax.add_artist(r)
        return ax

    def plot_bg_roi_vals(self, ax=None, date_fmt=None, labelsize=None,
                         **kwargs):
        """Plot emission rate time series.

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
            fig, ax = subplots(1, 1)
        if "color" not in kwargs:
            kwargs["color"] = "r"
        if labelsize is None:
            labelsize = rcParams["font.size"]
        s = self.bg_roi_info["mean"]
        try:
            s.index = s.index.to_pydatetime()
        except BaseException:
            pass
        err = self.bg_roi_info["std"]
        lower = s - err
        upper = s + err
        exp = exponent(upper.values.max())

        s_disp = s / 10**exp
        lower_disp = lower / 10**exp
        upper_disp = upper / 10**exp

        s_disp.plot(ax=ax, label="mean", **kwargs)
        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except BaseException:
            pass

        ax.fill_between(s.index, lower_disp, upper_disp, alpha=0.1, **kwargs)
        ax.set_ylabel(r"$ROI_{BG}\,[E%d\,cm^{-2}]$" % exp, fontsize=labelsize)
        ax.grid()
        return ax


def det_emission_rate(cds, velo, pix_dists, cds_err=None, velo_err=None,
                      pix_dists_err=None, mmol=MOL_MASS_SO2):
    """Determine emission rate.

    :param cds: column density in units cm-2 (float or ndarray)
    :param velo: effective plume velocity in units of m/s (float or ndarray)
        Effective means, that it is with respect to the direction of the normal
        vector of the plume cross section used (e.g. by performing a scalar
        product of 2D velocity vectors with normal vector of the PCS)
    :param pix_dists: pixel to pixel distances in units of m (float or ndarray)

    """
    if cds_err is None:
        logger.info("Uncertainty in column densities unspecified, assuming 20 % of "
              "mean CD")
        cds_err = mean(cds) * 0.20
    if velo_err is None:
        logger.info("Uncertainty in plume velocity unspecified, assuming 20 % of "
              "mean velocity")
        velo_err = mean(velo) * 0.20

    if pix_dists_err is None:
        logger.info("Uncertainty in pixel distance unspecified, assuming 10 % of "
              "mean pixel distance")
        pix_dists_err = mean(pix_dists) * 0.10

    C = 100**2 * mmol / N_A
    phi = sum(cds * velo * pix_dists) * C
    dphi1 = sum(velo * pix_dists * cds_err)**2
    dphi2 = sum(cds * pix_dists * velo_err)**2
    dphi3 = sum(cds * velo * pix_dists_err)**2
    phi_err = C * sqrt(dphi1 + dphi2 + dphi3)
    return phi, phi_err

