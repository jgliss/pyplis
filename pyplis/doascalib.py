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
"""Pyplis module for DOAS calibration including FOV search engines."""
from __future__ import (absolute_import, division)
from numpy import (min, arange, asarray, zeros, column_stack,
                   ones, nan, float64)
from scipy.stats.stats import pearsonr
from scipy.sparse.linalg import lsmr

from datetime import datetime
from pandas import Series
from copy import deepcopy
from astropy.io import fits
from traceback import format_exc

import six

from pyplis import logger
from matplotlib.pyplot import subplots
from matplotlib.patches import Circle, Ellipse
from matplotlib.cm import RdBu
from matplotlib.dates import DateFormatter

from .glob import SPECIES_ID
from .helpers import (shifted_color_map, mesh_from_img, get_img_maximum,
                      sub_img_to_detector_coords, map_coordinates_sub_img,
                      exponent, rotate_xtick_labels)

from .optimisation import gauss_fit_2d, GAUSS_2D_PARAM_INFO
from .image import Img
from .inout import get_camera_info
from .setupclasses import Camera
from .calib_base import CalibData
from .helpers import make_circular_mask


class DoasCalibData(CalibData):
    """Class containing DOAS calibration data.

    Parameters
    ----------
    tau_vec : ndarray
        tau data vector for calibration data
    cd_vec : ndarray
        DOAS-CD data vector for calibration data
    cd_vec_err : ndarray
        Fit errors of DOAS-CDs
    time_stamps : ndarray
        array with datetime objects containing time stamps
        (e.g. start acquisition) of calibration data
    calib_fun : function
        optimisation function used for fitting of calibration data
    calib_coeffs : ;obj:`list`, optional
        optimisation parameters for calibration curve.
    senscorr_mask : :obj:`ndarray`or :obj:`Img`, optional
        sensitivity correction mask that was normalised relative to the
        pixel position where the calibration data was retrieved (i.e.
        position of DOAS FOV in case of DOAS calibration data, or image pixel
        position, where cell calibration data was retrieved)
    calib_id : str
        calibration ID (e.g. "aa", "tau_on", "tau_off")
    camera : Camera
        camera object (not necessarily required). A camera can be assigned
        in order to convert the FOV extend from pixel coordinates into
        decimal degrees
    fov : DoasFOV
        information about position and shape of the FOV of the DOAS within
        the camera images

    """

    def __init__(self, tau_vec=None, cd_vec=None, cd_vec_err=None, time_stamps=None,
                 calib_fun=None, calib_coeffs=None, senscorr_mask=None,
                 polyorder=1, calib_id="", camera=None, fov=None):
        super(DoasCalibData, self).__init__(tau_vec, cd_vec, cd_vec_err,
                                            time_stamps, calib_fun,
                                            calib_coeffs, senscorr_mask,
                                            polyorder, calib_id, camera)
        if tau_vec is None:
            tau_vec = []
        if cd_vec is None:
            cd_vec = []
        if cd_vec_err is None:
            cd_vec_err = []
        if time_stamps is None:
            time_stamps = []
        if calib_coeffs is None:
            calib_coeffs = []
        self.type = "doas"
        if not isinstance(fov, DoasFOV):
            fov = DoasFOV(camera)
        self.fov = fov

    def save_as_fits(self, save_dir=None, save_name=None,
                     overwrite_existing=True):
        """Save calibration data as FITS file.

        Parameters
        ----------
        save_dir : str
            save directory, if None, the current working directory is used
        save_name : str
            filename of the FITS file (if None, use pyplis default naming)

        """
        # hdulist containing calibration data and senscorr_mask
        hdulist = self._prep_fits_save()

        # add DOAS FOV information (if applicable)
        hdulist.extend(self.fov.prep_hdulist())
        # returns abspath of current wkdir if None
        hdulist.writeto(self._prep_fits_savepath(save_dir, save_name),
                        clobber=overwrite_existing)

    def load_from_fits(self, file_path):
        """Load stack object (fits).

        Parameters
        ----------
        file_path : str
            file path of calibration data

        """
        # loads senscorr_mask and calibration data (tau and cd vectors,
        # timestamps)
        hdu = super(DoasCalibData, self).load_from_fits(file_path)

        self.fov.import_from_hdulist(hdu, first_idx=2)
        hdu.close()

    def plot_data_tseries_overlay(self, date_fmt=None, ax=None):
        """Plot overlay of tau and DOAS time series."""
        if ax is None:
            fig, ax = subplots(1, 1)
        s1 = self.tau_tseries
        s2 = self.cd_tseries
        p1 = ax.plot(s1.index.to_pydatetime(), s1.values, "--xb",
                     label=r"$\tau$")
        ax.set_ylabel("tau")
        ax2 = ax.twinx()

        p2 = ax2.plot(s2.index.to_pydatetime(), s2.values, "--xr",
                      label="DOAS CDs")
        ax2.set_ylabel(r"$S_{%s}$ [cm$^{-2}$]" % SPECIES_ID)
        ax.set_title("Time series overlay DOAS calib data")

        try:
            if date_fmt is not None:
                ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        except BaseException:
            pass

        ps = p1 + p2
        labs = [l.get_label() for l in ps]
        ax.legend(ps, labs, loc="best", fancybox=True, framealpha=0.5)
        ax.grid()
        rotate_xtick_labels(ax)
        return (ax, ax2)


class DoasFOV(object):
    """Class for storage of FOV information."""

    def __init__(self, camera=None):
        self.search_settings = {}
        self.img_prep = {}
        self.roi_abs = None
        self.img_shape_orig = None
        self.camera = None

        self.start_search = datetime(1900, 1, 1)
        self.stop_search = datetime(1900, 1, 1)

        self.corr_img = None

        self.fov_mask_rel = None

        self.result_pearson = {"cx_rel": nan,
                               "cy_rel": nan,
                               "rad_rel": nan,
                               "corr_curve": None}
        self.result_ifr = {"popt": None,
                           "pcov": None}

        if isinstance(camera, Camera):
            self.camera = camera
            self.img_shape_orig = (camera.pixnum_y, camera.pixnum_x)

    @property
    def method(self):
        """Return search method."""
        try:
            return self.search_settings["method"]
        except KeyError:
            raise ValueError("No information about FOV search available")

    @property
    def pyrlevel(self):
        """Return pyramide level at which FOV search was performed."""
        try:
            return self.img_prep["pyrlevel"]
        except KeyError:
            raise KeyError("Image preparation data is not available: %s"
                           % format_exc())

    @property
    def cx_rel(self):
        """Return center x coordinate of FOV (in relative coords)."""
        if self.method == "ifr":
            return self.result_ifr["popt"][1]
        else:
            return self.result_pearson["cx_rel"]

    @property
    def cy_rel(self):
        """Return center x coordinate of FOV (in relative coords)."""
        if self.method == "ifr":
            return self.result_ifr["popt"][2]
        else:
            return self.result_pearson["cy_rel"]

    @property
    def radius_rel(self):
        """Return radius of FOV (in relative coords).

        :raises: TypeError if method == "ifr"
        """
        if self.method == "ifr":
            raise TypeError("Invalid value: method IFR does not have FOV "
                            "parameter radius, call self.popt for relevant "
                            "parameters")
        return self.result_pearson["rad_rel"]

    @property
    def popt(self):
        """Return super gauss optimisation parameters (in relative coords).

        :raises: TypeError if method == "pearson"

        """
        if self.method == "pearson":
            raise TypeError("Invalid value: method pearson does not have "
                            "FOV shape parameters, call self.radius to "
                            "retrieve disk radius")
        return self.result_ifr["popt"]

    @property
    def x_abs(self):
        return self.pos_abs[0]

    @property
    def y_abs(self):
        return self.pos_abs[1]

    @property
    def sigma_x_abs(self):
        if self.method == "pearson":
            return self.radius_rel * 2**self.pyrlevel
        return self.popt[3] * 2**self.pyrlevel

    @property
    def sigma_y_abs(self):
        if self.method == "pearson":
            return self.radius_rel * 2**self.pyrlevel
        return (self.popt[3] / self.popt[4]) * 2 ** self.pyrlevel

    @property
    def pos_abs(self):
        """Return center coordinates of FOV (in absolute detector coords)."""
        return self.pixel_position_center(True)

    def _max_extend_rel(self):
        """Return maximum pixel extend of FOV.

        For method pearson this is the radius (trivial), for an elliptical
        super gauss (i.e. method IFR) this is the longer axis
        """
        if self.method == "pearson":
            return self.radius_rel
        else:
            return max([self.popt[3], self.popt[3] / self.popt[4]])

    def pixel_extend(self, abs_coords=True):
        """Return pixel extend of FOV on image.

        :param bool abs_coords: return value in absolute or relative
            coordinates (considering pyrlevel and roi)
        """
        ext_rel = self._max_extend_rel()
        if not abs_coords:
            return ext_rel
        return ext_rel * 2**self.pyrlevel

    def pixel_position_center(self, abs_coords=False):
        """Return pixel position of center of FOV.

        :param bool abs_coords: return position in absolute or relative
            coordinates (considering pyrlevel and roi)

        :return:
            - tuple, ``(cx, cy)``
        """
        try:
            cx, cy = self.cx_rel, self.cy_rel
        except BaseException:
            logger.warning("Could not access information about FOV position")
        if not abs_coords:
            return (cx, cy)
        return map_coordinates_sub_img(cx, cy, self.roi_abs, self.pyrlevel,
                                       inverse=True)

    def fov_mask_abs(self, img_shape_orig=(), cam_id=""):
        """Convert the FOV mask to absolute detector coordinates.

        The shape of the FOV mask (and the represented pixel coordinates)
        depends on the image preparation settings of the :class:`ImgStack`
        object which was used to identify the FOV.

        Parameters
        ----------
        img_shape_orig : tuple
            image shape of original image data (can be extracted from an
            unedited image)
        cam_id : str
            string ID of pyplis default camera (e.g. "ecII")

        """
        if not len(img_shape_orig) == 2:
            try:
                info = get_camera_info(cam_id)
                img_shape_orig = (int(info["pixnum_y"]), int(info["pixnum_x"]))
            except BaseException:
                raise IOError("Image shape could not be retrieved...")
        mask = self.fov_mask_rel.astype(float64)
        return sub_img_to_detector_coords(mask, img_shape_orig,
                                          self.pyrlevel,
                                          self.roi_abs).astype(bool)

# ==============================================================================
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
# =============================================================================

    def import_from_hdulist(self, hdu, first_idx=0):
        """Import FOV information from FITS HDU list.

        Parameters
        ----------
        hdu : HDUList
            HDU list containing a list of HDUs created using
            :func:`prep_hdulist` starting at index :param:`first_idx`
            (e.g. first_idx==2 if the method :func:`save_as_fits` from
            the :class:`DoasCalibData` class is used, since the first 2
            indices are used for saving the acutal calibration data)
        first_idx : int
            index specifying the first entry of the FOV info in the
            provided HDU list

        """
        i = first_idx
        try:
            self.fov_mask_rel = hdu[i].data.byteswap().newbyteorder()
        except BaseException:
            logger.info("(Warning loading DOAS calib data): FOV mask not "
                  "available")

        prep_keys = Img().edit_log.keys()
        search_keys = DoasFOVEngine()._settings.keys()

        for key, val in six.iteritems(hdu[i].header):
            k = key.lower()
            if k in prep_keys:
                self.img_prep[k] = val
            elif k in search_keys:
                self.search_settings[k] = val
            elif k in self.result_pearson.keys():
                self.result_pearson[k] = val

        try:
            self.corr_img = Img(hdu[i + 1].data.byteswap().newbyteorder())
        except BaseException:
            logger.info("(Warning loading DOAS calib data): FOV search correlation "
                  "image not available")
        self.roi_abs = hdu[i + 2].data["roi"].byteswap().newbyteorder()
        try:
            self.result_ifr["popt"] =\
                hdu[i + 3].data["ifr_res"].byteswap().newbyteorder()
        except BaseException:
            logger.info("Failed to import array containing IFR optimisation "
                  " results from FOV search")

    def prep_hdulist(self):
        """Prepare and return :class:`HDUList` object for saving as FITS."""
        fov_mask = fits.ImageHDU(self.fov_mask_rel)
        fov_mask.header.update(self.img_prep)
        fov_mask.header.update(self.search_settings)

        ifr_res = []
        if self.method == "pearson":
            rd = self.result_pearson
            try:
                fov_mask.header.update(cx_rel=rd["cx_rel"],
                                       cy_rel=rd["cy_rel"],
                                       rad_rel=rd["rad_rel"])
            except BaseException:
                logger.warning("Position of FOV (pearson method) not available")

        elif self.method == "ifr":
            ifr_res = self.result_ifr["popt"]

        try:
            hdu_cim = fits.ImageHDU(data=self.corr_img.img)
        except BaseException:
            hdu_cim = fits.ImageHDU()
            logger.warning("FOV search correlation image not available")

        roi = fits.BinTableHDU.from_columns([fits.Column(name="roi",
                                                         format="I",
                                                         array=self.roi_abs)])
        col_ifr = fits.Column(name="ifr_res", format="D", array=ifr_res)
        res_ifr = fits.BinTableHDU.from_columns([col_ifr])

        return fits.HDUList([fov_mask, hdu_cim, roi, res_ifr])

    def save_as_fits(self, **kwargs):
        """Save the fov as fits file.

        Saves this object as DoasCalibData::

            d = DoasCalibData(fov = self)
            d.save_as_fits(**kwargs)
        """
        d = DoasCalibData(fov=self)
        d.save_as_fits(**kwargs)

    def __str__(self):
        s = "DoasFOV information\n------------------------\n"
        s += "\nImg stack preparation settings\n............................\n"
        for k, v in six.iteritems(self.img_prep):
            s += "%s: %s\n" % (k, v)
        s += "\nFOV search settings\n............................\n"
        for k, v in six.iteritems(self.search_settings):
            s += "%s: %s\n" % (k, v)
        if self.method == "ifr":
            s += "\nIFR search results \n.........................\n"
            s += "\nSuper gauss fit optimised params\n"
            popt = self.popt
            for k in range(len(popt)):
                s += "%s: %.3f\n" % (GAUSS_2D_PARAM_INFO[k], popt[k])
        elif self.method == "pearson":
            s += "\nPearson search results \n.......................\n"
            for k, v in six.iteritems(self.result_pearson):
                if not k == "corr_curve":
                    s += "%s: %s\n" % (k, v)
        return s

    def plot(self, ax=None):
        """Draw the current FOV position into the current correlation img."""
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
                ell = Ellipse(xy=(popt[1], popt[2]), width=popt[3],
                              height=popt[3] / popt[4], color="k", lw=2,
                              fc="lime", alpha=.5)
            else:
                ell = Ellipse(xy=(popt[1], popt[2]), width=popt[3],
                              height=popt[3] / popt[4], angle=popt[7],
                              color="k", lw=2, fc="lime", alpha=.5)

            ax.add_artist(ell)
            ax.axhline(self.cy_rel, ls="--", color="k")
            ax.axvline(self.cx_rel, ls="--", color="k")

            ax.get_xaxis().set_ticks([0, self.cx_rel, w])
            ax.get_yaxis().set_ticks([0, self.cy_rel, h])

            # ax.set_axis_off()
            ax.set_title(r"Corr img (IFR), pos abs (x,y): (%d, %d), "
                         "lambda=%.1e"
                         % (cx, cy, self.search_settings["ifrlbda"]))

        elif self.method == "pearson":
            cb.set_label(r"Pearson corr. coeff.")
            ax.autoscale(False)

            c = Circle((self.cx_rel, self.cy_rel), self.radius_rel, ec="k",
                       lw=2, fc="lime", alpha=.5)
            ax.add_artist(c)
            ax.set_title("Corr img (pearson), pos abs (x,y): (%d, %d)"
                         % (cx, cy))
            ax.get_xaxis().set_ticks([0, self.cx_rel, w])
            ax.get_yaxis().set_ticks([0, self.cy_rel, h])
            ax.axhline(self.cy_rel, ls="--", color="k")
            ax.axvline(self.cx_rel, ls="--", color="k")
        ax.set_xlabel("Pixel row")
        ax.set_ylabel("Pixel column")
        return ax


class DoasFOVEngine(object):
    """Engine to perform DOAS FOV search."""

    def __init__(self, img_stack=None, doas_series=None, method="pearson",
                 **settings):

        self._settings = {"method": "pearson",
                          "maxrad": 80,
                          "ifrlbda": 1e-6,  # lambda val IFR
                          "g2dasym": True,  # elliptic FOV
                          "g2dsuper": True,  # super gauss fit (IFR)
                          "g2dcrop": True,
                          "g2dtilt": False,
                          "blur": 4,
                          "mergeopt": "average"}

        self.data_merged = False
        self.img_stack = img_stack
        self.doas_series = doas_series

        self.calib_data = DoasCalibData()  # includes DoasFOV class

        self.update_search_settings(**settings)
        self.method = method

    @property
    def maxrad(self):
        """For Pearson method: maximum expected disk radius of FOV.

        Note
        ----
        this radius is considered independent of the current pyramid level
        of the image stack, hence, if it is set 20 and the pyramid level of
        the stack is 2, then, the FOV disk radius (in detector coords) may
        be 80.
        """
        return self._settings["maxrad"]

    @maxrad.setter
    def maxrad(self, val):
        logger.info("Updating seeting for maximum radius of FOV, new value: %s"
              % val)
        self._settings["maxrad"] = int(val)

    @property
    def ifrlbda(self):
        """For IFR method: allow asymmetric 2d gauss fit."""
        return self._settings["ifrlbda"]

    @ifrlbda.setter
    def ifrlbda(self, val):
        self._settings["ifrlbda"] = val

    @property
    def g2dasym(self):
        """For IFR method: allow asymmetric 2d gauss fit."""
        return self._settings["g2dasym"]

    @g2dasym.setter
    def g2dasym(self, val):
        if val not in [True, False]:
            raise ValueError("Invalid input value: require boolean")
        self._settings["g2dasym"] = val

    @property
    def g2dsuper(self):
        """For IFR method: use supergauss parametrisation."""
        return self._settings["g2dsuper"]

    @g2dsuper.setter
    def g2dsuper(self, val):
        if val not in [True, False]:
            raise ValueError("Invalid input value: require boolean")
        self._settings["g2dsuper"] = val

    @property
    def g2dcrop(self):
        """For IFR method: crop gaussian FOV parametrisation at sigma."""
        return self._settings["g2dcrop"]

    @g2dcrop.setter
    def g2dcrop(self, val):
        if val not in [True, False]:
            raise ValueError("Invalid input value: require boolean")
        self._settings["g2dcrop"] = val

    @property
    def g2dtilt(self):
        """For IFR method: allow supergauss-fit to be tilted."""
        return self._settings["g2dtilt"]

    @g2dtilt.setter
    def g2dtilt(self, val):
        if val not in [True, False]:
            raise ValueError("Invalid input value: require boolean")
        self._settings["g2dtilt"] = val

    @property
    def blur(self):
        """Sigma of gaussian blurring filter applied to correlation image.

        The filter is applied to the correlation image before finding the
        position of the maximum correlation. This is only relevant for
        method IFR, since this method parameterises the FOV by fitting a
        2D Gaussian to the correlation image. Defaults to 4.
        """
        return self._settings["blur"]

    @blur.setter
    def blur(self, val):
        self._settings["blur"] = val

    @property
    def mergeopt(self):
        """Option for temporal merging of stack and DOAS vector.

        Choose from ``average, nearest, interpolation``
        """
        return self._settings["mergeopt"]

    @mergeopt.setter
    def mergeopt(self, val):
        if val not in ["average", "nearest", "interpolation"]:
            raise ValueError("Invalid method: choose from average, "
                             "nearest or interpolation")
        self._settings["mergeopt"] = val

    @property
    def method(self):
        """Return method used for FOV search (e.g. pearson, ifr)."""
        return self._settings["method"]

    @method.setter
    def method(self, val):
        if val not in ["pearson", "ifr"]:
            raise ValueError("Invalid method: choose from pearson or ifr")
        self._settings["method"] = val

    def update_search_settings(self, **settings):
        """Update current search settings.

        :param **settings: keyword args to be updated (only
            valid keys will be updated)
        """
        for k, v in six.iteritems(settings):
            if k in self._settings:
                logger.info("Updating FOV search setting %s, new value: %s"
                      % (k, v))
                self._settings[k] = v

    @property
    def doas_data_vec(self):
        """Return DOAS CD vector (values of ``self.doas_series``)."""
        return self.doas_series.values

    @property
    def method(self):
        """Return current FOV search method."""
        return self._settings["method"]

    @method.setter
    def method(self, value):
        """Return current FOV search method."""
        if value not in ["ifr", "pearson"]:
            raise ValueError("Invalid search method: choose from ifr or"
                             " pearson")
        self._settings["method"] = value

    def perform_fov_search(self, **settings):
        """High level method for automatic FOV search.

        Uses the current settings (``self._settings``) to perform the
        following steps:

            1. Call :func:`merge_data`: Time merging of stack and DOAS
            vector. This step is skipped if data was already merged within
            this engine, i.e. if ``self.data_merged == True``

            #. Call :func:`det_correlation_image`: Determination of
            correlation image using ``self.method`` ('ifr' or 'pearson')

            #. Call :func:`get_fov_shape`: Identification of FOV shape /
            extend on image detector either using circular disk approach
            (if ``self.method == 'pearson'``) or 2D (super) Gauss fit
            (if ``self.method == 'ifr').

        All relevant results are written into ``self.calib_data`` (which
        includes :class:`DoasFOV` object)


        """
        self.calib_data = DoasCalibData()  # includes DoasCalibData class
        self.update_search_settings(**settings)
        self.merge_data(merge_type=self._settings["mergeopt"])
        self.det_correlation_image(search_type=self.method)
        self.get_fov_shape()
        self.calib_data.fov.search_settings = deepcopy(self._settings)

        return self.calib_data

    def run_fov_fine_search(self, img_list, doas_series, extend_fac=3,
                            **settings):
        """Get FOV position in full resolution.

        Note
        ----
        1. Only works if FOV search (i.e. :func:`perform_fov_search`) was
        already performed.
        #. This method requires some time as it needs to
        recompute a cropped image stack in full resolution from the
        provided img_list.
        #. This method deletes the current image stack in this objects.
        #. Uses the same search settings as set in this class (i.e. method,
        etc.)

        Parameters
        ----------
        img_list : BaseImgList
            image list used to calculate cropped stack
        doas_series : DoasResults
            original DOAS time series (i.e. not merged in time with image
            data, needs to be provided since the one stored within this
            class is modified during the first FOV search)
        extend_fac : int
            factor determining crop ROI based on the current pixel extend
            of the FOV

        Returns
        -------
        DoasFOVEngine
            new instance containing results from fine search

        """
        self.update_search_settings(**settings)
        try:
            extend = self.calib_data.fov.pixel_extend(abs_coords=True)
            (pos_x, pos_y) = self.calib_data.fov.pixel_position_center(abs_coords=True)  # noqa: E501

            self.img_stack = None  # make space for new stack
            # create ROI around center position of FOV
            roi = [pos_x - extend_fac * extend, pos_y - extend_fac * extend,
                   pos_x + extend_fac * extend, pos_y + extend_fac * extend]

            self.img_stack = stack = img_list.make_stack(pyrlevel=0,
                                                         roi_abs=roi)
            s = DoasFOVEngine(stack, self.doas_series, **self._settings)
            calib = s.perform_fov_search()
            calib.fit_calib_data()
            return s

        except Exception as e:
            raise Exception("Failed to perform fine search: %s" % repr(e))

    def merge_data(self, merge_type=None):
        """Merge stack data and DOAS vector in time.

        Wrapper for :func:`merge_with_time_series` of :class:`ImgStack`

        :param str merge_type: choose between ``average, interpolation,
        nearest``

        Note
        ----

        Current data (i.e. ``self.img_stack`` and ``self.doas_series``)
        will be overwritten if merging succeeds.

        Parameters
        ----------
        merge_type : :obj:`str`, optional,
            one of the available merge types, see :attr:`mergeopt` for
            valid options

        Raises
        ------
        RuntimeError
            if merging of data fails

        """
        if self.data_merged:
            logger.info("Data merging unncessary, img stack and DOAS vector are "
                  "already merged in time")
            return
        if merge_type is None:
            merge_type = self._settings["mergeopt"]
        new_stack, new_doas_series = self.img_stack.merge_with_time_series(
            self.doas_series,
            method=merge_type)
        if len(new_doas_series) == new_stack.shape[0]:
            self.img_stack = new_stack
            self.doas_series = new_doas_series
            self._settings["mergeopt"] = merge_type
            self.data_merged = True
            return
        raise RuntimeError("Temporal merging of image and DOAS data failed...")

    def det_correlation_image(self, search_type="pearson", **kwargs):
        """Determine correlation image.

        Determines correlation image either using IFR or Pearson method.
        Results are written into ``self.calib_data.fov`` (:class:`DoasFOV`)

        :param str search_type: updates current search type, available types
            ``["pearson", "ifr"]``
        """
        if not self.img_stack.shape[0] == len(self.doas_series):
            raise ValueError("DOAS correlation image object could not be "
                             "determined: inconsistent array lengths, please "
                             "perform timemerging first")
        self.update_search_settings(method=search_type, **kwargs)
        if search_type == "pearson":
            corr_img, _ = self._det_correlation_image_pearson(
                **self._settings)
        elif search_type == "ifr":
            corr_img, _ = self._det_correlation_image_ifr_lsmr(
                **self._settings)
        else:
            raise ValueError("Invalid search type %s: choose from "
                             "pearson or ifr" % search_type)
        corr_img = Img(corr_img,
                       pyrlevel=self.img_stack.img_prep["pyrlevel"])
        corr_img.add_gaussian_blurring(self._settings["blur"])
        # corr_img.pyr_up(self.img_stack.img_prep["pyrlevel"])
        self.calib_data.fov.corr_img = corr_img
        self.calib_data.fov.img_prep = self.img_stack.img_prep
        self.calib_data.fov.roi_abs = self.img_stack.roi_abs
        self.calib_data.fov.start_search = self.img_stack.start
        self.calib_data.fov.stop_search = self.img_stack.stop
        try:
            if self.img_stack.img_prep["is_aa"]:
                cid = "AA"
            else:
                raise Exception
        except:
            cid = self.img_stack.stack_id
        self.calib_data.calib_id = cid

        return corr_img

    def _det_correlation_image_pearson(self, **kwargs):
        """Determine correlation image based on pearson correlation.

        :returns: - correlation image (pix wise value of pearson corr coeff)
        """
        h, w = self.img_stack.shape[1:]
        corr_img = zeros((h, w), dtype=float64)
        corr_img_err = zeros((h, w), dtype=float64)
        cd_vec = self.doas_series.values
        exp = int(10**exponent(h) / 4.0)
        for i in range(h):
            try:
                if i % exp == 0:
                    logger.info("FOV search: current img row (y): " + str(i))
            except BaseException:
                pass
            for j in range(w):
                # get series from stack at current pixel
                corr_img[i, j], corr_img_err[i, j] = pearsonr(
                    self.img_stack.stack[:, i, j], cd_vec)
        self._settings["method"] = "pearson"
        return corr_img, corr_img_err

    def _det_correlation_image_ifr_lsmr(self, ifrlbda=1e-6, **kwargs):
        """Apply LSMR algorithm to identify the FOV.

        :param float ifrlbda: tolerance parameter lambda
        """
        # some input data size checking
        (m,) = self.doas_data_vec.shape
        (m2, ny, nx) = self.img_stack.shape
        if m != m2:
            raise ValueError("Inconsistent array lengths, please perform time "
                             "merging of image stack and doas vector first")

        # construct H-matrix through reshaping image stack
        # h_matrix = transpose(self.img_stack.stack, (2,0,1)).reshape(m, nx*ny)
        h_matrix = self.img_stack.stack.reshape(m, nx * ny)
        # and one-vector
        h_vec = ones((m, 1), dtype=h_matrix.dtype)
        # and stacking in the end
        h = column_stack((h_vec, h_matrix))
        # solve using LSMR regularisation
        a = lsmr(h, self.doas_data_vec, atol=ifrlbda, btol=ifrlbda)
        c = a[0]
        # separate offset and image
        lsmr_offset = c[0]
        lsmr_image = c[1:].reshape(ny, nx) / max(c[1:])
        # THIS NORMALISATION IS NEW
        # lsmr_image = lsmr_image / abs(lsmr_image).max()
        self._settings["method"] = "ifr"
        self._settings["ifrlbda"] = ifrlbda

        return lsmr_image, lsmr_offset

    def get_fov_shape(self, **settings):
        """Find shape of FOV based on correlation image.

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
            cy, cx = get_img_maximum(self.calib_data.fov.corr_img.img)

            logger.info("Start radius search in stack around x/y: %s/%s" % (cx, cy))
            (radius,
             corr_curve,
             tau_vec,
             cd_vec,
             fov_mask) = self.fov_radius_search(cx, cy)

            if not radius > 0:
                raise ValueError("Pearson FOV search failed")

            self.calib_data.fov.result_pearson["cx_rel"] = cx
            self.calib_data.fov.result_pearson["cy_rel"] = cy
            self.calib_data.fov.result_pearson["rad_rel"] = radius
            self.calib_data.fov.result_pearson["corr_curve"] = corr_curve

            self.calib_data.fov.fov_mask_rel = fov_mask
            self.calib_data.tau_vec = tau_vec.astype(float64)
            self.calib_data.cd_vec = cd_vec.astype(float64)
            try:
                self.calib_data.cd_vec_err = self.doas_series.fit_errs
            except BaseException:
                pass
            self.calib_data.time_stamps = self.img_stack.time_stamps
            return

        elif self.method == "ifr":
            # the fit is performed in absolute dectector coordinates
            # corr_img_abs = Img(self.fov.corr_img.img).pyr_up(pyrlevel).img
            popt, pcov, fov_mask = self._fov_gauss_fit(
                self.calib_data.fov.corr_img,
                **self._settings)
            tau_vec = self.convolve_stack_fov(fov_mask)

            self.calib_data.fov.result_ifr["popt"] = popt
            self.calib_data.fov.result_ifr["pcov"] = pcov
            self.calib_data.fov.fov_mask_rel = fov_mask
            self.calib_data.tau_vec = tau_vec
            self.calib_data.cd_vec = self.doas_data_vec
            try:
                self.calib_data.cd_vec_err = self.doas_series.fit_errs
            except BaseException:
                pass
            self.calib_data.time_stamps = self.img_stack.time_stamps
        else:
            raise ValueError("Invalid search method...")

    def fov_radius_search(self, cx, cy):
        """Search the FOV disk radius around center coordinate.

        The search varies the radius around the center coordinate and
        extracts image data time series from average values of all pixels
        falling into the current disk. These time series are correlated
        with spectrometer data to find the radius showing highest
        correlation.

        :param int cx: pixel x coordinate of center position
        :param int cy: pixel y coordinate of center position

        """
        stack = self.img_stack
        cd_vec = self.doas_series.values
        if not len(cd_vec) == stack.shape[0]:
            raise ValueError("Mismatch in lengths of input arrays")
        h, w = stack.shape[1:]
        pyrlevel = stack.pyrlevel
        # find maximum radius (around CFOV pos) which still fits into the image
        # shape of the stack used to find the best radius
        max_rad = min([cx, cy, w - cx, h - cy])
        crad = int(self.maxrad * 2**(-pyrlevel))
        if crad < max_rad:
            max_rad_search = crad
        else:
            max_rad_search = max_rad
            self.maxrad = int(max_rad * 2**(pyrlevel))
        # radius array
        radii = arange(1, max_rad_search + 1, 1)
        logger.info("Maximum radius at pyramid level %d: %s"
              % (pyrlevel, max_rad_search))
        # some variable initialisations
        coeffs, coeffs_err = [], []
        max_corr = 0
        tau_vec = None
        mask = zeros((h, w)).astype(float64)
        radius = 0
        # loop over all radii, get tauSeries at each, (merge) and determine
        # correlation coefficient
        for r in radii:

            # now get mean values of all images in stack in circular ROI around
            # CFOV
            tau_series, m = stack.get_time_series(cx, cy, radius=r)
            tau_dat = tau_series.values
            coeff, err = pearsonr(tau_dat, cd_vec)
            logger.info("Rad: {} (R: {:.4f})".format(r, coeff))
            coeffs.append(coeff)
            coeffs_err.append(err)
            # and append correlation coefficient to results
            if coeff > max_corr:
                radius = r
                mask = m.astype(float64)
                max_corr = coeff
                tau_vec = tau_dat
        corr_curve = Series(asarray(coeffs, dtype=float), radii)
        return radius, corr_curve, tau_vec, cd_vec, mask

    # define IFR model function (Super-Gaussian)

    def _fov_gauss_fit(self, corr_img, g2dasym=True, g2dsuper=True,
                       g2dcrop=True, g2dtilt=False, blur=4, **kwargs):
        """Apply 2D gauss fit to correlation image.

        Parameters
        ----------
        corr_img : Img
            correlation image
        g2dasym : bool
            allow for assymetric shape (sigmax != sigmay), True
        g2dsuper: bool
            allow for supergauss fit, True
        g2dcrop : bool
            if True, set outside (1/e amplitude) datapoints = 0, True
        g2dtilt : bool
            allow gauss to be tilted with respect to x/y axis
        blur : int
            width of gaussian smoothing kernel convolved with correlation
            image in order to identify position of maximum

        Returns
        -------
        tuple
            3-element tuple containing

            - array (popt): optimised multi-gauss parameters
            - 2d array (pcov): estimated covariance of popt
            - 2d array: correlation image

        """
        img = corr_img.img
        h, w = img.shape
        xgrid, ygrid = mesh_from_img(img)

        # apply maximum of filtered image to initialise 2D gaussian fit
        (cy, cx) = get_img_maximum(img)
        maxrad = self.maxrad * 2**(-corr_img.pyrlevel)
        mask = make_circular_mask(h, w, cx, cy, maxrad).astype(float)
        img = img * mask
        # constrain fit, if requested
        (popt, pcov, fov_mask) = gauss_fit_2d(img, cx, cy, g2dasym,
                                              g2d_super_gauss=g2dsuper,
                                              g2d_crop=g2dcrop,
                                              g2d_tilt=g2dtilt, **kwargs)

        return (popt, pcov, fov_mask)

    # function convolving the image stack with the obtained FOV distribution
    def convolve_stack_fov(self, fov_mask):
        """Normalize fov image and convolve stack.

        :returns: - stack time series vector within FOV
        """
        # normalize fov_mask
        normsum = fov_mask.sum()
        fov_mask_norm = fov_mask / normsum
        # convolve with image stack
        # stack_data_conv = transpose(self.stac, (2,0,1)) * fov_fitted_norm
        stack_data_conv = self.img_stack.stack * fov_mask_norm
        return stack_data_conv.sum((1, 2))

# OLD STUFF

# =============================================================================
# class DoasCalibDataOLD(object):
#     """Class containing DOAS calibration data
#
#     Parameters
#     ----------
#     tau_vec : ndarray
#         tau data vector for calibration data
#     cd_vec : ndarray
#         DOAS-CD data vector for calibration data
#     cd_vec_err : ndarray
#         Fit errors of DOAS-CDs
#     time_stamps : ndarray
#         array with datetime objects containing time stamps
#         (e.g. start acquisition) of calibration data
#     calib_id : str
#         calibration ID (e.g. "aa", "tau_on", "tau_off")
#     camera : Camera
#         camera object (not necessarily required). A camera can be assigned
#         in order to convert the FOV extend from pixel coordinates into
#         decimal degrees
#
#     """
#     def __init__(self, tau_vec=[], cd_vec=[], cd_vec_err=[],
#                  time_stamps=[], calib_id="", fov=None, camera=None,
#                  polyorder=1):
#
#         #tau data vector within FOV
#         self.tau_vec = asarray(tau_vec).astype(float64)
#         #doas data vector
#         self.cd_vec = asarray(cd_vec).astype(float64)
#         self.cd_vec_err = asarray(cd_vec_err).astype(float64)
#
#         self._calib_funs = CalibFuns()
#         self.time_stamps = time_stamps
#         self.calib_id = calib_id
#
#         self.camera = None
#
#         if not isinstance(fov, DoasFOV):
#             fov = DoasFOV(camera)
#         self.fov = fov
#
#         self._poly = None
#         self._cov = None
#         self._polyorder = None
#         self._allowed_polyorders = [1,2,3]
#         self.polyorder = polyorder
#
#         if isinstance(camera, Camera):
#             self.camera = Camera
#
#     @property
#     def start(self):
#         """Start time of calibration data (datetime)"""
#         try:
#             return self.time_stamps[0]
#         except TypeError:
#             return self.fov.start_search
#
#     @property
#     def stop(self):
#         """Stop time of calibration data (datetime)"""
#         try:
#             return self.time_stamps[-1]
#         except TypeError:
#             return self.fov.stop_search
#
#     @property
#     def calib_id_str(self):
#         """String for calibration ID"""
#         idx=0
#         try:
#             if self.calib_id.split("_")[1].lower() == "aa":
#                 idx=1
#             try:
#                 return CALIB_ID_STRINGS[self.calib_id.split("_")[idx]]
#             except:
#                 return self.calib_id.split("_")[idx]
#         except:
#             return ""
#
#     @property
#     def polyorder(self):
#         """Current order of fit polynomial"""
#         return self._polyorder
#
#     @polyorder.setter
#     def polyorder(self, val):
#         if not val in self._allowed_polyorders:
#             raise ValueError("Invalid value for polyorder: %.1f. "
#                              "Choose from %s"
#                              % (val, self._allowed_polyorders))
#         self._polyorder = val
#         if isinstance(self._poly, poly1d):
#             logger.warning("Polynomial order was changed and changes were not yet "
#                  "applied. Please call "
#                  "fit_calib_polynomial to retrieve the calibration "
#                  "polynomial for the new settings")
#
#     @property
#     def poly(self):
#         """Calibration polynomial"""
#         if not isinstance(self._poly, poly1d):
#             self.fit_calib_polynomial()
#         return self._poly
#
#     @poly.setter
#     def poly(self, value):
#         if not isinstance(value, poly1d):
#             raise ValueError("Need numpy poly1d object...")
#         self._poly=value
#
#     @property
#     def cov(self):
#         """Covariance matriy of calibration polynomial"""
#         if not isinstance(self._cov, ndarray):
#             self.fit_calib_polynomial()
#         return self._cov
#
#     @cov.setter
#     def cov(self, value):
#         raise IOError("Covariance matrix of calibration polynomial cannot "
#                       "be set manually, please call function "
#                       "fit_calib_polynomial")
#
#     @property
#     def coeffs(self):
#         """Coefficients of current calibration polynomial"""
#         return self.poly.coeffs
#
#     @property
#     def slope(self):
#         """Slope of current calib curve"""
#         if self.polyorder > 1:
#             logger.warning("Order of calibration polynomial > 1: use value of slope "
#                  "with care (i.e. also check curvature coefficients of "
#                  "polynomial")
#
#         return self.coeffs[-2]
#
#     @property
#     def slope_err(self):
#         """Slope error of current calib curve"""
#         if self.polyorder > 1:
#             logger.warning("Order of calibration polynomial > 1: use slope error with "
#                  "care")
#         return sqrt(self.cov[-2][-2])
#
#     @property
#     def y_offset(self):
#         """Y-axis offset of calib curve"""
#         return self.coeffs[-1]
#
#     @property
#     def y_offset_err(self):
#         """Error of y axis offset of calib curve"""
#         return sqrt(self.cov[-1][-1])
#
#     @property
#     def cd_tseries(self):
#         """Pandas Series object of doas data"""
#         return Series(self.cd_vec, self.time_stamps)
#
#     @property
#     def tau_tseries(self):
#         """Pandas Series object of tau data"""
#         return Series(self.tau_vec, self.time_stamps)
#
#     @property
#     def tau_range(self):
#         """Range of tau values extended by 5%
#
#         Returns
#         -------
#         tuple
#             2-element tuple, containing
#
#             - float, tau_min: lower end of tau range
#             - float, tau_max: upper end of tau range
#         """
#         tau = self.tau_vec
#         taumin, taumax = tau.min(), tau.max()
#         if taumin > 0:
#             taumin = 0
#         add = (taumax - taumin) * 0.05
#         return taumin - add, taumax + add
#
#     @property
#     def cd_range(self):
#         """Range of DOAS cd values extended by 5%"""
#         cds = self.cd_vec
#         cdmin, cdmax = cds.min(), cds.max()
#         if cdmin > 0:
#             cdmin = 0
#         add = (cdmax - cdmin) * 0.05
#         return cdmin - add, cdmax + add
#
#     @property
#     def residual(self):
#         """Residual of calibration curve"""
#         return self.poly(self.tau_vec) - self.tau_vec
#
#     def has_calib_data(self):
#         """Checks if calibration data is available"""
#         if not all([len(x) > 0 for x in [self.cd_vec, self.tau_vec]]):
#             return False
#         if not len(self.tau_vec) == len(self.cd_vec):
#             return False
#         return True
#
#     def fit_calib_polynomial(self, polyorder=None, weighted=True,
#                              weights_how="abs",
#                              through_origin=False,
#                              plot=False):
#         """Fit calibration polynomial to current data
#
#         Parameters
#         ----------
#         polyorder : :obj:`int`, optional
#             update current polyorder
#         weighted : bool
#             performs weighted fit based on DOAS errors in ``cd_vec_err``
#             (if available), defaults to True
#         weights_how : str
#             use "rel" if relative errors are supposed to be used (i.e.
#             w=CD_sigma / CD) or "abs" if absolute error is supposed to be
#             used (i.e. w=CD_sigma).
#         through_origin : bool
#             if True, the fit is forced to cross the coordinate origin (
#             done by adding data points)
#         plot : bool
#             If True, the calibration curve and the polynomial are plotted
#
#         Returns
#         -------
#         poly1d
#             calibration polynomial
#         """
#         if not weights_how in ["rel", "abs"]:
#             raise IOError("Invalid input for parameter weights_how:"
#                           "Use rel for relative errors or abs for absolute"
#                           "errors for calculation of weights")
#         if not self.has_calib_data():
#             raise ValueError("Calibration data is not available")
#         try:
#             self.polyorder = polyorder
#         except:
#             pass
# # ======================================================================
# #         if polyorder is None:
# #             polyorder = self.polyorder
# #
# # ======================================================================
#         if sum(isnan(self.tau_vec)) + sum(isnan(self.cd_vec)) > 0:
#             raise ValueError("Encountered nans in data")
#
#         exp = exponent(self.cd_vec.max())
#         yerr = ones(len(self.cd_vec))
#         yerr_abs = True
#         if weighted:
#             if not len(self.cd_vec) == len(self.cd_vec_err):
#                 logger.warning("Could not perform weighted calibration fit: "
#                      "Length mismatch between DOAS data vector"
#                      " and corresponding error vector")
#             elif sum(self.cd_vec_err) == 0:
#                 logger.warning("Could not performed weighted calibration fit: "
#                      "Values of DOAS fit errors are 0. Do you have pydoas "
#                      "installed?")
#             else:
#                 try:
#                     if weights_how == "abs":
#                         yerr = self.cd_vec_err / 10**exp
#                     else:
#                         yerr = self.cd_vec_err / self.cd_vec
#                         yerr_abs = False
#                     #ws = ws / max(ws)
#                 except:
#                     logger.warning("Failed to calculate weights")
#         tau_vals = self.tau_vec
#         cds = self.cd_vec / 10**exp
#
#         fun = self._calib_funs.get_poly(self.polyorder, through_origin)
#
#         coeffs, cov = curve_fit(fun, tau_vals.astype(float64),
#                                 cds.astype(float64),
#                                 sigma=yerr.astype(float64),
#                                 absolute_sigma=yerr_abs)
#         if through_origin:
#             coeffs = append(coeffs, 0.0)
# # ======================================================================
# #         if through_origin:
# #             num = len(tau_vals)
# #             tau_vals = concatenate([tau_vals, zeros(num)])
# #             cds = concatenate([cds, zeros(num)])
# #             ws = concatenate([ws, ones(num)])
# #
# # ======================================================================
# # ======================================================================
# #         coeffs, cov = polyfit(tau_vals, cds,
# #                               polyorder, w=ws, cov=True)
# # ======================================================================
#         #self.polyorder = polyorder
#         #return (fun, coeffs, cov, tau_vals, cds, yerr, yerr_abs)
#         self.poly = poly1d(coeffs * 10**exp)
#         self._cov = cov * 10**(2*exp)
#         if plot:
#             self.plot()
#         return self.poly
#
#     def save_as_fits(self, save_dir=None, save_name=None):
#         """Save calibration data as FITS file
#
#         Parameters
#         ----------
#         save_dir : str
#             save directory, if None, the current working directory is used
#         save_name : str
#             filename of the FITS file (if None, use pyplis default naming)
#         """
#         if not len(self.cd_vec) == len(self.tau_vec):
#             raise ValueError("Could not save calibration data, mismatch in "
#                 " lengths of data arrays")
#         if not len(self.time_stamps) == len(self.cd_vec):
#             self.time_stamps = asarray([datetime(1900,1,1)] *\
#                                                 len(self.cd_vec))
#         #returns abspath of current wkdir if None
#         save_dir = abspath(save_dir)
#         if not isdir(save_dir): #save_dir is a file path
#             save_name = basename(save_dir)
#             save_dir = dirname(save_dir)
#         if save_name is None:
#             save_name = "doascalib_id_%s_%s_%s_%s.fts" %(\
#                 self.calib_id, self.start.strftime("%Y%m%d"),\
#                 self.start.strftime("%H%M"), self.stop.strftime("%H%M"))
#         else:
#             save_name = save_name.split(".")[0] + ".fts"
#         fov_mask = fits.PrimaryHDU()
#         fov_mask.data = self.fov.fov_mask_rel
#         fov_mask.header.update(self.fov.img_prep)
#         fov_mask.header.update(self.fov.search_settings)
#         fov_mask.header["calib_id"] = self.calib_id
#         fov_mask.header.append()
#
#         ifr_res = []
#         if self.fov.method == "pearson":
#             rd = self.fov.result_pearson
#             try:
#                 fov_mask.header.update(cx_rel=rd["cx_rel"],
#                                        cy_rel=rd["cy_rel"],
#                                        rad_rel=rd["rad_rel"])
#             except:
#                 logger.warning("Position of FOV (pearson method) not available")
#
#         elif self.fov.method == "ifr":
#             ifr_res = self.fov.result_ifr["popt"]
#
#         try:
#             hdu_cim = fits.ImageHDU(data = self.fov.corr_img.img)
#         except:
#             hdu_cim = fits.ImageHDU()
#             logger.warning("FOV search correlation image not available")
#
#         tstamps = [x.strftime("%Y%m%d%H%M%S%f") for x in self.time_stamps]
#         col1 = fits.Column(name="time_stamps", format="25A", array=tstamps)
#         col2 = fits.Column(name="tau_vec", format="D", array=self.tau_vec)
#         col3 = fits.Column(name="cd_vec", format="D", array=self.cd_vec)
#         col4 = fits.Column(name="cd_vec_err", format="D",
#                            array=self.cd_vec_err)
#
#
#         cols = fits.ColDefs([col1, col2, col3, col4])
#         arrays = fits.BinTableHDU.from_columns(cols)
#
#         roi = fits.BinTableHDU.from_columns([fits.Column(name="roi",
#                                                          format="I",
#                                                          array=self.fov.roi_abs)])
#         col_ifr = fits.Column(name="ifr_res", format="D", array=ifr_res)
#         res_ifr = fits.BinTableHDU.from_columns([col_ifr])
#         #==============================================================================
#         # col1 = fits.Column(name = 'target', format = '20A', array=a1)
#         # col2 = fits.Column(name = 'V_mag', format = 'E', array=a2)
#         #==============================================================================
#
#         hdulist = fits.HDUList([fov_mask, hdu_cim, arrays, roi, res_ifr])
#         fpath = join(save_dir, save_name)
#         try:
#             remove(fpath)
#         except:
#             pass
#         hdulist.writeto(fpath)
#
#     def load_from_fits(self, file_path):
#         """Load stack object (fits)
#
#         Parameters
#         ----------
#         file_path : str
#             file path of calibration data
#         """
#         if not exists(file_path):
#             raise IOError("DoasCalibData object could not be loaded, "
#                 "path does not exist")
#         hdu = fits.open(file_path)
#         try:
#             self.fov.fov_mask_rel = hdu[0].data.byteswap().newbyteorder()
#         except:
#             print ("(Warning loading DOAS calib data): FOV mask not "
#                 "available")
#
#         prep_keys = Img().edit_log.keys()
#         search_keys = DoasFOVEngine()._settings.keys()
#         self.calib_id = hdu[0].header["calib_id"]
#         for key, val in hdu[0].header.iteritems():
#             k = key.lower()
#             if k in prep_keys:
#                 self.fov.img_prep[k] = val
#             elif k in search_keys:
#                 self.fov.search_settings[k] = val
#             elif k in self.fov.result_pearson.keys():
#                 self.fov.result_pearson[k] = val
#
#         try:
#             self.fov.corr_img = Img(hdu[1].data.byteswap().newbyteorder())
#         except:
#             print ("(Warning loading DOAS calib data): FOV search "
#                 "correlation image not available")
#         try:
#             times = hdu[2].data["time_stamps"].byteswap().newbyteorder()
#             self.time_stamps = [datetime.strptime(x, "%Y%m%d%H%M%S%f")
#                                 for x in times]
#         except:
#             print ("(Warning loading DOAS calib data): Failed to import "
#                         "time stamps")
#         try:
#             self.tau_vec = hdu[2].data["tau_vec"].byteswap().newbyteorder()
#         except:
#             print "Failed to import calibration tau data vector"
#         try:
#             self.cd_vec = hdu[2].data["cd_vec"].byteswap().newbyteorder()
#         except:
#             print "Failed to import calibration doas data vector"
#         try:
#             self.cd_vec_err =\
#                 hdu[2].data["cd_vec_err"].byteswap().newbyteorder()
#         except:
#             print "Failed to import DOAS fit error information in calib data"
#         try:
#             self.fov.result_ifr["popt"] =\
#                 hdu[4].data["ifr_res"].byteswap().newbyteorder()
#         except:
#             print ("Failed to import array containing IFR optimisation "
#                     " results from FOV search")
#         self.fov.roi_abs = hdu[3].data["roi"].byteswap().newbyteorder()
#
#     @property
#     def _poly_str(self):
#         """Return custom string representation of polynomial"""
#         exp = exponent(self.poly.coeffs[0])
#         p = poly1d(round(self.poly / 10**(exp - 2))/10**2)
#         s = "(%s)E%+d" %(p, exp)
#         return s.replace("x", r"$\tau$")
#
#     def plot(self, add_label_str="", shift_yoffset=False, ax=None,
#              **kwargs):
#         """Plot calibration data and fit result
#
#         Parameters
#         ----------
#         add_label_str : str
#             additional string added to label of plots for legend
#         shift_yoffset : bool
#             if True, the data is plotted without y-offset
#         ax :
#             matplotlib axes object, if None, a new one is created
#         """
#         if not "color" in kwargs:
#             kwargs["color"] = "b"
#
#         if ax is None:
#             fig, ax = subplots(1,1, figsize=(10,8))
#
#         taumin, taumax = self.tau_range
#         x = linspace(taumin, taumax, 100)
#
#         cds = self.cd_vec
#         cds_poly = self.poly(x)
#         if shift_yoffset:
#             try:
#                 cds -= self.y_offset
#                 cds_poly -= self.y_offset
#             except:
#                 logger.warning("Failed to subtract y offset")
#
#         ax.plot(self.tau_vec, cds, ls="", marker=".",
#                 label="Data %s" %add_label_str, **kwargs)
#         try:
#             ax.errorbar(self.tau_vec, cds, yerr=self.cd_vec_err,
#                         marker="None", ls=" ", c="#b3b3b3")
#         except:
#             logger.warning("No DOAS-CD errors available")
#         try:
#             ax.plot(x, cds_poly, ls="-", marker="",
#                     label="Fit result", **kwargs)
#
#         except TypeError:
#             print "Calibration poly probably not fitted"
#
#         ax.set_title("DOAS calibration data, ID: %s" %self.calib_id_str)
#         ax.set_ylabel(r"$S_{%s}$ [cm$^{-2}$]" %SPECIES_ID)
#         ax.set_xlabel(r"$\tau_{%s}$" %self.calib_id_str)
#         ax.grid()
#         ax.legend(loc='best', fancybox=True, framealpha=0.7)
#         return ax
#
#     def plot_poly(self, add_label_str="", shift_yoffset=False, ax=None,
#                   **kwargs):
#         """Plot calibration fit result
#
#         Parameters
#         ----------
#         add_label_str : str
#             additional string added to label of plots for legend
#         shift_yoffset : bool
#             if True, the data is plotted without y-offset
#         ax :
#             matplotlib axes object, if None, a new one is created
#         """
#         if not "color" in kwargs:
#             kwargs["color"] = "b"
#
#         if ax is None:
#             fig, ax = subplots(1,1, figsize=(10,8))
#
#         taumin, taumax = self.tau_range
#         x = linspace(taumin, taumax, 100)
#
#         cds_poly = self.poly(x)
#         if shift_yoffset:
#             try:
#                 cds_poly -= self.y_offset
#             except:
#                 logger.warning("Failed to subtract y offset")
#
#         try:
#             ax.plot(x, cds_poly, ls="-", marker="",
#                     label="Fit result %s" %add_label_str, **kwargs)
#
#         except TypeError:
#             print "Calibration poly probably not fitted"
#
#         ax.grid()
#         ax.legend(loc='best', fancybox=True, framealpha=0.7)
#         return ax
#
#     def plot_data_tseries_overlay(self, date_fmt=None, ax=None):
#         """Plot overlay of tau and DOAS time series"""
#         if ax is None:
#             fig, ax = subplots(1,1)
#         s1 = self.tau_tseries
#         s2 = self.cd_tseries
#         p1 = ax.plot(s1.index.to_pydatetime(), s1.values, "--xb",
#                      label = r"$\tau$")
#         ax.set_ylabel("tau")
#         ax2 = ax.twinx()
#
#         p2 = ax2.plot(s2.index.to_pydatetime(), s2.values,"--xr",
#                       label="DOAS CDs")
#         ax2.set_ylabel(r"$S_{%s}$ [cm$^{-2}$]" %SPECIES_ID)
#         ax.set_title("Time series overlay DOAS calib data")
#
#         try:
#             if date_fmt is not None:
#                 ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
#         except:
#             pass
#
#         ps = p1 + p2
#         labs = [l.get_label() for l in ps]
#         ax.legend(ps, labs, loc="best",fancybox=True, framealpha=0.5)
#         ax.grid()
#         rotate_xtick_labels(ax)
#         return (ax, ax2)
#
#     def err(self, value):
#         """Returns measurement error of tau value based on slope error"""
#         val = self(value)
#         r = self.slope_err / self.slope
#         return val * r
#
#     def __call__(self, value, **kwargs):
#         """Define call function to apply calibration
#
#         :param float value: tau or AA value
#         :return: corresponding column density
#         """
#         if not isinstance(self.poly, poly1d):
#             self.fit_calib_polynomial()
#         if isinstance(value, Img):
#             calib_im = value.duplicate()
#             calib_im.img = self.poly(calib_im.img) - self.y_offset
#             calib_im.edit_log["gascalib"] = True
#             return calib_im
#         elif isinstance(value, ImgStack):
#             try:
#                 value = value.duplicate()
#             except MemoryError:
#                 logger.warning("Stack cannot be duplicated, applying calibration to "
#                 "input stack")
#             value.stack = self.poly(value.stack) - self.y_offset
#             value.img_prep["gascalib"] = True
#             return value
#         return self.poly(value) - self.y_offset
# =============================================================================
