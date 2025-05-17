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
"""Pyplis module for image based correction of the signal dilution effect."""
from typing import Optional
import warnings
from numpy import asarray, linspace, exp, ndarray, ones, nan
from dataclasses import dataclass
from scipy.ndimage import median_filter
from scipy.optimize import OptimizeResult
from matplotlib.pyplot import subplots, rcParams

from pandas import Series, DataFrame

from pyplis import logger, print_log
from pyplis.utils import LineOnImage
from pyplis.image import Img
from pyplis.optimisation import dilution_corr_fit
from pyplis.model_functions import dilutioncorr_model
from pyplis.geometry import MeasGeometry
from pyplis.helpers import check_roi, isnum
from pyplis.imagelists import ImgList
from pyplis.exceptions import ImgModifiedError
LABEL_SIZE = rcParams["font.size"] + 2


@dataclass
class DilutionCorrFitResult:
    """Class to bundle the results of the dilution fit."""
    ext: float
    """Extinction coefficient."""
    i0: float
    """Retrieved undiluted intensity of terrain"""
    dists: ndarray
    """Distances to topographic features used as input to the fit."""
    rads: ndarray
    """Radiances of topographic features used as input to the fit"""
    input_img: Img
    """Input image used to retrieve the measured radiances."""
    fit_result: OptimizeResult
    """Result of the fit."""
    
class DilutionCorr:
    """Class for management of dilution correction.

    The class provides functionality to retrieve topographic distances from
    meas geometry, to manage lines in the image used for the retrieval, to
    perform the actual dilution fit (i.e. retrieval of atmospheric scattering
    coefficients) and to apply the dilution correction.

    This class does not store any results related to individual images.

    Parameters
    ----------
    lines : list
        optional, list containing :class:`LineOnImage` objects used to
        retrieve terrain distances for the dilution fit
    meas_geometry : MeasGeometry
        optional, measurement geometry (required for terrain distance
        retrieval)
    **settings :
        settings for terrain distance retrieval:

            - skip_pix: specify pixel step on line for which topo \
                intersections are searched
            - min_slope_angle: minimum slope of topography in order to be \
                considered for topo distance retrieval
            - topo_res_m: interpolation resolution applied to \
                :class:`ElevationProfile` objects used to find intersections \
                of pixel viewing direction with topography

    """

    def __init__(self, lines=None, meas_geometry=None, **settings):
        if lines is None:
            lines = []
        if not isinstance(meas_geometry, MeasGeometry):
            meas_geometry = MeasGeometry()
        self.meas_geometry = meas_geometry
        self.lines = {}

        self.settings = {"skip_pix": 5,
                         "min_slope_angle": 5.0,
                         "topo_res_m": 5.0}

        self._masks_lines = {}
        self._dists_lines = {}
        # additional retrieval points that were added manually using
        # method add_retrieval_point
        self._add_points = []
        self._skip_pix = {}
        self._geopoints = {}
        self._geopoints["add_points"] = []

        for line in lines:
            self.lines[line.line_id] = line

        self.update_settings(**settings)

    @property
    def line_ids(self):
        """Get IDs of all :class:`LineOnImage` objects for distance retrieval.
        """
        return list(self.lines.keys())

    def update_settings(self, **settings):
        """Update settings dict for topo distance retrieval."""
        for k, v in settings.items():
            if k in self.settings:
                self.settings[k] = v

    def add_retrieval_line(self, line):
        """Add one topography retrieval line."""
        if not isinstance(line, LineOnImage):
            raise TypeError("Need LineOnImage object")
        if line.line_id in self.line_ids:
            raise KeyError("A line with ID %s is already assigned to Dilution "
                           "correction engine" % line.line_id)
        self.lines[line.line_id] = line

    def add_retrieval_point(self, pos_x_abs, pos_y_abs, dist=None):
        """Add a distinct pixel with known distance to image.

        Parameters
        ----------
        pos_x_abs : int
            x-pixel position of point in image in absolute coordinate (i.e.
            pyramid level 0 and not cropped)
        pos_y_abs : int
            y-pixel position of point in image in absolute coordinate (i.e.
            pyramid level 0 and not cropped)
        dist : :obj:`float`, optional
            distance to feature in image in m. If None (default), the distance
            will be estimated

        """
        if not isnum(dist):
            logger.info("Input distance for point unspecified, trying automatic access")
            (dist,_, p) = self.meas_geometry.get_topo_distance_pix(pos_x_abs, pos_y_abs)
            self._geopoints["add_points"].append(p)
            dist *= 1000.0
        self._add_points.append((pos_x_abs, pos_y_abs, dist))

    def det_topo_dists_all_lines(self, **settings):
        """Estimate distances to topo distances to all assigned lines.

        Parameters
        ----------
        **settings
            keyword args passed to update search settings (:attr:`settings`)
            and passed to
            :func:`get_topo_distances_line` in :class:`MeasGeometry`

        """
        for lid in self.lines.keys():
            self.det_topo_dists_line(lid, **settings)

    def det_topo_dists_line(self, line_id, **settings):
        """Estimate distances to pixels on current lines.

        Retrieves distances to all :class:`LineOnImage` objects  in
        ``self.lines`` using ``self.meas_geometry`` (i.e. camera position
        and viewing direction).

        Parameters
        ----------
        line_id : str
            ID of line
        **settings :
            additional key word args used to update search settings (passed to
            :func:`get_topo_distances_line` in :class:`MeasGeometry`)

        Returns
        -------
        array
            retrieved distances
        """
        if line_id not in self.lines.keys():
            raise KeyError("No line with ID %s available" % line_id)
        logger.info(f"Searching topo distances for pixels on line {line_id}")
        self.update_settings(**settings)

        l = self.lines[line_id]
        res = self.meas_geometry.get_topo_distances_line(l, **self.settings)
        dists = res["dists"] * 1000.  # convert to m
        self._geopoints[line_id] = res["geo_points"]
        self._dists_lines[line_id] = dists
        self._masks_lines[line_id] = res["ok"]
        self._skip_pix[line_id] = self.settings["skip_pix"]
        return dists

    def get_radiances(self, img, line_ids=None):
        """Get radiances for dilution fit along terrain lines.

        The data is only extracted along specified input lines. The terrain
        distance retrieval :func:`det_topo_dists_lines_line` must have been
        performed for that.

        Parameters
        ----------
        img : Img
            vignetting corrected plume image from which the radiances are
            extracted
        line_ids : list
            if desired, the data can also be accessed for specified line ids,
            which have to be provided in a list. If empty (default), all lines
            assigned to this class are considered

        """
        if line_ids is None:
            line_ids = []
        if not isinstance(img, Img) or not img.edit_log["vigncorr"]:
            raise ValueError("Invalid input, need Img class and Img needs to "
                             "be corrected for vignetting")
        if img.is_cropped or img.is_resized:
            raise ImgModifiedError("Image must not be cropped or rescaled")
        if len(line_ids) == 0:
            line_ids = self.line_ids

        dists, rads = [], []
        for line_id in line_ids:
            if line_id in self._dists_lines:
                skip = int(self._skip_pix[line_id])
                l = self.lines[line_id]
                mask = self._masks_lines[line_id]
                dists.extend(self._dists_lines[line_id][mask])
                rads.extend(l.get_line_profile(img)[::skip][mask])
            else:
                print_log.warning("Distances to line %s not available, please apply "
                     "distance retrieval first using class method "
                     "det_topo_dists_line")
        for x, y, dist in self._add_points:
            dists.append(dist)
            rads.append(img.img[y, x])
        return asarray(dists), asarray(rads)

    def fit(
        self, 
        img: Img, 
        rad_ambient: float, 
        i0_guess: Optional[float] = None,
        i0_min: float = 0.0, 
        i0_max: Optional[float] = None, 
        ext_guess: float = 1e-4, 
        ext_min: float = 0.0,
        ext_max: float = 1e-3,
        line_ids=None
        ) -> DilutionCorrFitResult:
        r"""Perform dilution correction fit to retrieve extinction coefficient.

        Uses :func:`dilution_corr_fit` of :mod:`optimisation` which is a
        bounded least square fit based on the following model function

        .. math::

            I_{meas}(\lambda) = I_0(\lambda)e^{-\epsilon(\lambda)d} +
            I_A(\lambda)(1-e^{-\epsilon(\lambda)d})

        Parameters
        ----------
        img : Img
            vignetting corrected image for radiance extraction
        rad_ambient : float
            ambient intensity (:math:`I_A` in model)
        i0_guess : float
            optional: guess value for initial intensity of topographic
            features, i.e. the reflected radiation before entering scattering
            medium (:math:`I_0` in model, if None, then it is set 5% of the
            ambient intensity ``rad_ambient``)
        i0_min : float
            optional: minimum initial intensity of topographic features
        i0_max : float
            optional: maximum initial intensity of topographic features
        ext_guess : float
            guess value for atm. extinction coefficient
            (:math:`\epsilon` in model)
        ext_min : float
            minimum value for atm. extinction coefficient
        ext_max : float
            maximum value for atm. extinction coefficient
        line_ids : list
            if desired, the data can also be accessed for specified line ids,
            which have to be provided in a list. If empty (default), all lines
            are considered
        plot : bool
            if True, the result is plotted
        **kwargs :
            additional keyword args passed to plotting function (e.g. to
            pass an axes object)

        Returns
        -------
        tuple
            4-element tuple containing

            - retrieved extinction coefficient
            - retrieved initial intensity
            - fit result object
            - axes instance or None (dependent on :param:`plot`)

        """
        if line_ids is None:
            line_ids = []
        dists, rads = self.get_radiances(img, line_ids)
        fit_res = dilution_corr_fit(rads, dists, rad_ambient, i0_guess,
                                    i0_min, i0_max, ext_guess,
                                    ext_min, ext_max)
        
        i0, ext = fit_res.x
        result = DilutionCorrFitResult(
            ext=ext, i0=i0, dists=dists, rads=rads,
            input_img=img, fit_result=fit_res
        )
        return result
    
    def apply_dilution_fit(self, img, rad_ambient, i0_guess=None,
                           i0_min=0, i0_max=None, ext_guess=1e-4, ext_min=0,
                           ext_max=1e-3, line_ids=None, plot=True, **kwargs):
        r"""Perform dilution correction fit to retrieve extinction coefficient.

        Note:
            DEPRECATED: please use method :func:`fit` instead.
            
        Uses :func:`dilution_corr_fit` of :mod:`optimisation` which is a
        bounded least square fit based on the following model function

        .. math::

            I_{meas}(\lambda) = I_0(\lambda)e^{-\epsilon(\lambda)d} +
            I_A(\lambda)(1-e^{-\epsilon(\lambda)d})

        Parameters
        ----------
        img : Img
            vignetting corrected image for radiance extraction
        rad_ambient : float
            ambient intensity (:math:`I_A` in model)
        i0_guess : float
            optional: guess value for initial intensity of topographic
            features, i.e. the reflected radiation before entering scattering
            medium (:math:`I_0` in model, if None, then it is set 5% of the
            ambient intensity ``rad_ambient``)
        i0_min : float
            optional: minimum initial intensity of topographic features
        i0_max : float
            optional: maximum initial intensity of topographic features
        ext_guess : float
            guess value for atm. extinction coefficient
            (:math:`\epsilon` in model)
        ext_min : float
            minimum value for atm. extinction coefficient
        ext_max : float
            maximum value for atm. extinction coefficient
        line_ids : list
            if desired, the data can also be accessed for specified line ids,
            which have to be provided in a list. If empty (default), all lines
            are considered
        plot : bool
            if True, the result is plotted
        **kwargs :
            additional keyword args passed to plotting function (e.g. to
            pass an axes object)

        Returns
        -------
        tuple
            4-element tuple containing

            - retrieved extinction coefficient
            - retrieved initial intensity
            - fit result object
            - axes instance or None (dependent on :param:`plot`)

        """
        warnings.warn("This method is deprecated, please use method 'fit', (note slight change in signature)", DeprecationWarning, stacklevel=2)
        result = self.fit(
            img=img, rad_ambient=rad_ambient, i0_guess=i0_guess,
            i0_min=i0_min, i0_max=i0_max, ext_guess=ext_guess,
            ext_min=ext_min, ext_max=ext_max, line_ids=line_ids
        )
        ax = None
        if plot:
            ax = self.plot_fit_result(result.dists, result.rads, rad_ambient, result.i0, result.ext, **kwargs)
        return result.ext, result.i0, result.fit_result, ax

    def get_ext_coeffs_imglist(self, lst, roi_ambient=None, apply_median=5,
                               **kwargs):
        """Apply dilution fit to all images in an :class:`ImgList`.

        Parameters
        ----------
        lst : ImgList
            image list for which the coefficients are supposed to be retrieved
        roi_ambient : list
            region of interest used to estimage ambient intensity, if None
            (default), usd :attr:`scale_rect` of :class:`PlumeBackgroundModel`
            of the input list
        apply_median : int
            if > 0, then a median filter of provided width is applied to
            the result time series (ext. coeffs and initial intensities)
        **kwargs :
            additional keyword args passed to dilution fit method
            :func:`apply_dilution_fit`.

        Returns
        -------
        DataFrame
            pandas data frame containing time series of retrieved extinction
            coefficients and initial intensities as well as the ambient
            intensities used, access keys are:

                - ``coeffs``: retrieved extinction coefficients
                - ``i0``: retrieved initial intensities
                - ``ia``: retrieved ambient intensities

        """
        if not isinstance(lst, ImgList):
            raise ValueError("Invalid input type for param lst, need ImgList")
        lst.vigncorr_mode = True

        if not check_roi(roi_ambient):
            try:
                roi_ambient = lst.bg_model.scale_rect
            except BaseException:
                pass
            if not check_roi(roi_ambient):
                raise ValueError("Input parameter roi_ambient is not a valied"
                                 "ROI and neither is scale_rect in background "
                                 "model of input image list...")
        cfn = lst.cfn
        lst.goto_img(0)
        nof = lst.nof
        times = lst.acq_times
        coeffs = []
        i0s = []
        ias = []
        for k in range(nof):
            img = lst.current_img()
            try:
                ia = img.crop(roi_ambient, True).mean()

                ext, i0, _, _ = self.apply_dilution_fit(img=img,
                                                        rad_ambient=ia,
                                                        plot=False,
                                                        **kwargs)
                coeffs.append(ext)
                i0s.append(i0)
                ias.append(ia)
            except BaseException:
                coeffs.append(nan)
                i0s.append(nan)
                ias.append(nan)
            lst.goto_next()
        lst.goto_img(cfn)
        if apply_median > 0:
            coeffs = median_filter(coeffs, apply_median)
            i0s = median_filter(i0s, apply_median)
            ias = median_filter(ias, apply_median)
        return DataFrame(dict(coeffs=coeffs, i0=i0s, ia=ias), index=times)

    def correct_img(self, plume_img, ext, plume_bg_img, plume_dists,
                    plume_pix_mask):
        """Perform dilution correction for a plume image.

        Note
        -----

        See :func:`correct_img` for description

        Returns
        -------
        Img
            dilution corrected image

        """
        return correct_img(plume_img, ext, plume_bg_img, plume_dists,
                           plume_pix_mask)

    def plot_fit_result(self, dists, rads, rad_ambient, i0, ext, ax=None):
        """Plot result of dilution fit."""
        if ax is None:
            fig, ax = subplots(1, 1)
        x = linspace(0, dists.max(), 100)
        ints = dilutioncorr_model(x, rad_ambient, i0, ext)
        ax.plot(dists / 1000.0, rads, " x", label="Data")
        ext_perkm = ext * 1000
        lbl_fit = (r"Fit: $I_0$=%.1f DN, $\epsilon$ = %.4f km$^{-1}$"
                   % (i0, ext_perkm))
        ax.plot(x / 1000.0, ints, "--c", label=lbl_fit)
        ax.set_xlabel("Distance [km]", fontsize=LABEL_SIZE)
        ax.set_ylabel("Radiances [DN]", fontsize=LABEL_SIZE)
        ax.set_title(r"$I_A$ = %.1f" % rad_ambient, fontsize=LABEL_SIZE + 2)
        ax.grid()
        # ax = rotate_ytick_labels(ax, deg=45, va="center")
        ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=13)
        return ax

    def get_extinction_coeffs_imglist(self, imglist, ambient_roi_abs,
                                      darkcorr=True, line_ids=None,
                                      **fit_settings):
        """Retrieve extinction coefficients for all imags in list.

        .. note::

            Alpha version: not yet tested

        """
        if line_ids is None:
            line_ids = []
        imglist.aa_mode = False
        imglist.tau_mode = False
        imglist.auto_reload = False
        imglist.darkcorr_mode = True
        if imglist.gaussian_blurring and imglist.pyrlevel == 0:
            logger.info("Adding gaussian blurring of 2 for topographic radiance "
                  "retrieval")
            imglist.gaussian_blurring = 2
        if imglist.pyrlevel != list(self.lines.values())[0].pyrlevel:
            raise ValueError("Mismatch in pyramid level of lines and imglist")
        if len(line_ids) == 0:
            line_ids = self.line_ids
        imglist.vigncorr_mode = True
        imglist.goto_img(0)
        imglist.auto_reload = True
        num = imglist.nof
        i0s, exts, acq_times = ones(num) * nan, ones(num) * nan, [nan] * num
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
                          draw_fov=0, cmap_topo="Oranges",
                          contour_color="#708090", contour_antialiased=True,
                          contour_lw=0.2, axis_off=True, line_ids=None,
                          **kwargs):
        """Draw 3D map of scene including geopoints of distance retrievals.

        Parameters
        ----------
        draw_cam : bool
            insert camera position into map
        draw_source : bool
            insert source position into map
        draw_plume : bool
            insert plume vector into map
        draw_fov : bool
            insert camera FOV (az range) into map
        cmap_topo : str
            string specifying colormap for topography surface plot defaults to
            "Oranges"
        contour_color : str
            string specifying color of contour lines colors of topo contour
            lines (default: "#708090")
        contour_antialiased : bool
            apply antialiasing to surface plot of topography, defaults to False
        contour_lw :
            width of drawn contour lines, defaults to 0.5, use 0 if you do not
            want contour lines inserted
        axis_off : bool
            if True, then the rendering of axes is excluded
        line_ids : list
            if desired, the data can also be accessed for specified line ids,
            which have to be provided in a list. If empty (default), all topo
            lines are drawn

        Returns
        -------
        Map
            plotted map instance (is of type Basemap)

        """
        if line_ids is None:
            line_ids = []
        map3d = self.meas_geometry.draw_map_3d(
            draw_cam, draw_source,
            draw_plume, draw_fov,
            cmap_topo,
            contour_color=contour_color,
            contour_antialiased=contour_antialiased,
            contour_lw=contour_lw)
        if len(line_ids) == 0:
            line_ids = self.line_ids
        for line_id in self.line_ids:
            if line_id in self._dists_lines:
                line = self.lines[line_id]
                mask = self._masks_lines[line_id]
                pts = self._geopoints[line_id][mask]
                map3d.add_geo_points_3d(pts, color=line.color, **kwargs)
        for pt in self._geopoints["add_points"]:
            map3d.draw_geo_point_3d(pt, color="r")

        if axis_off:
            map3d.ax.set_axis_off()
        return map3d


def correct_img(plume_img, ext, plume_bg_img, plume_dists, plume_pix_mask):
    """Perform dilution correction for a plume image.

    Corresponds to Eq. 4 in in `Campion et al., 2015 <http://
    www.sciencedirect.com/science/article/pii/S0377027315000189>`_.

    Parameters
    ----------
    plume_img : Img
        vignetting corrected plume image
    ext : float
        atmospheric extinction coefficient
    plume_bg_img : Img
        vignetting corrected plume background image (can be, for instance,
        retrieved using :mod:`plumebackground`)
    plume_dists : :obj:`array`, :obj:`Img`, :obj:`float`
        plume distance(s) in m. If input is numpy array or :class:`Img` then,
        it must have the same shape as :param:`plume_img`
    plume_pix_mask : ndarray
        mask specifying plume pixels (only those are corrected), can also be
        type :class:`Img`

    Returns
    -------
    Img
        dilution corrected image

    """
    for im in [plume_img, plume_bg_img]:
        if not isinstance(im, Img) or im.edit_log["vigncorr"] is False:
            raise ValueError("Plume and background image need to be Img "
                             "objects and vignetting corrected")

    try:
        plume_dists = plume_dists.img
    except BaseException:
        pass
    try:
        plume_pix_mask = plume_pix_mask.img
    except BaseException:
        pass

    dists = plume_pix_mask * plume_dists
    plume_img.img = ((plume_img.img - plume_bg_img.img *
                     (1 - exp(-ext * dists))) / exp(-ext * dists))
    plume_img.edit_log["dilcorr"] = True
    return plume_img


def get_topo_dists_lines(lines, geom, img=None, skip_pix=5, topo_res_m=5.0,
                         min_slope_angle=5.0, plot=False, line_color="lime"):

    if isinstance(lines, LineOnImage):
        lines = [lines]

    ax = None
    map3d = None

    pts, dists, mask = [], [], []
    for line in lines:
        l = line.to_list()  # line coords as list
        res = geom.get_topo_distances_line(l, skip_pix, topo_res_m,
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
                line.plot_line_on_grid(ax=ax, color=line_color, marker="")
            ax.set_xlim([0, w - 1])
            ax.set_ylim([h - 1, 0])

        map3d = geom.draw_map_3d(0, 0, 0, 0, cmap_topo="gray")
        # insert camera position into 3D map
        map3d.add_geo_points_3d(pts, color=line_color)
        geom.cam_pos.plot_3d(map=map3d, add_name=True, dz_text=40)
        map3d.ax.set_axis_off()

    return dists, asarray(mask), map3d, ax


def perform_dilution_correction(plume_img, ext, plume_bg_img, plume_dist_img,
                                plume_pix_mask):

    dists = plume_pix_mask * plume_dist_img
    return ((plume_img - plume_bg_img *
             (1 - exp(-ext * dists))) / exp(-ext * dists))


def get_extinction_coeff(rads, dists, rad_ambient, plot=True, **kwargs):
    """Perform dilution correction fit to retrieve extinction coefficient.

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
        fig, ax = subplots(1, 1)
        ax.plot(dists, rads, " x", label="Data")
        lbl_fit = r"Fit result: $I_0$=%.1f DN, $\epsilon$ = %.2e" % (i0, ext)
        ax.plot(x, ints, "--c", label=lbl_fit)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Radiances [DN]")
        ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=12)
    return ext, i0, fit_res, ax
