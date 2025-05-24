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
"""Pyplis module containing features related to plume background analysis."""
from numpy import (polyfit, poly1d, linspace, logical_and, log, argmin,
                   gradient, nan, ndarray, arange, ones, finfo, asarray)
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure, subplots, setp
import matplotlib.colors as colors
from collections import OrderedDict as od
from scipy.ndimage import gaussian_filter

from pyplis import logger, print_log
from pyplis.image import Img
from pyplis.utils import LineOnImage
from pyplis.optimisation import PolySurfaceFit
from pyplis.helpers import shifted_color_map, _roi_coordinates
from pyplis.plumespeed import find_movement


class PlumeBackgroundModel(object):
    """Class for plume background modelling and tau image determination.

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

    def __init__(self, bg_raw=None, plume_init=None,
                 init_surf_fit_mask=True, **kwargs):

        if isinstance(bg_raw, ndarray):
            bg_raw = Img(bg_raw)
        if isinstance(plume_init, ndarray):
            plume_init = Img(plume_init)

        self.last_tau_img = None
        self._last_surffit = None
        #: Correction mode
        self._mode = 0

        #: settings for poly surface fit (corr mode: 0)
        self._surface_fit_mask = None
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
        self.ygrad_line_colnum = None  # detector column of vertical line
        self.ygrad_line_polyorder = 2
        self.ygrad_line_startrow = 0  # start row for profile fit
        self.ygrad_line_stoprow = None  # stop row for profile fit
        self.ygrad_line_mask = None  # mask specifying rows for profile fit

        #: Rectangle for linear based correction of horizontal gradient
        #: (applied before ygradient correction is performed)
        #: corr modes: 4, 5
        self.xgrad_rect = None

        #: Settings for quadratic correction of horizontal gradient
        #: (along line) corr modes: 6
        self.xgrad_line_rownum = None
        self.xgrad_line_polyorder = 2
        self.xgrad_line_startcol = 0
        self.xgrad_line_stopcol = None
        self.xgrad_line_mask = None

        # initialisations
        self.update(**kwargs)

        if isinstance(plume_init, Img):
            self.set_missing_ref_areas(plume_init)
            self._init_bgsurf_mask(plume_init)
            if isinstance(bg_raw, Img):
                self.mode = 1
            self.last_tau_img = self.get_tau_image(plume_init, bg_raw)

        self.last_settings = self.settings_dict()

    @property
    def all_modes(self):
        """List containing valid modelling modes."""
        return list(self.mode_info_dict.keys())

    @property
    def mode(self):
        """Return current modelling mode."""
        return self._mode

    @mode.setter
    def mode(self, val):
        if val not in self.all_modes:
            raise ValueError("Invalid mode %d, choose from %s"
                             % (val, self.all_modes))
        self._mode = val

    @property
    def CORR_MODE(self):
        """Return current background modelling mode."""
        raise AttributeError("Deprecated attribute name, please use mode")

    @CORR_MODE.setter
    def CORR_MODE(self, val):
        raise AttributeError("Deprecated attribute name, please use mode")

    @property
    def surface_fit_mask(self):
        """Mask for retrieval mode 0: fit 2D polynomial."""
        return self._surface_fit_mask

    @surface_fit_mask.setter
    def surface_fit_mask(self, val):
        if isinstance(val, Img):
            val = val.img
        self._surface_fit_mask = val

    def check_settings(self):
        """Check if any of the modelling settings is not specified."""
        for value in self.__dict__.values():
            if value is None:
                return False
        return True

    def mean_in_rects(self, img):
        """Determine ``(mean, min, max)`` intensity in reference rectangles.

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
        except BaseException:
            pass
        a = []
        a.append(_mean_in_rect(img, self.scale_rect)[0])
        a.append(_mean_in_rect(img, self.ygrad_rect)[0])
        a.append(_mean_in_rect(img, self.xgrad_rect)[0])
        a = asarray(a)
        return (a.mean(), a.min(), a.max())

    def update(self, **kwargs):
        """Update class attributes.

        :param **kwargs:
        """
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def set_missing_ref_areas(self, plume_img: Img):
        """Find and set missing default sky reference areas for modelling.

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

        Args:
            plume_img (Img): the plume image for which the sky background
                areas are retrieved.
        """
        plume = plume_img.img
        if self.check_settings():
            return
        if self.surface_fit_mask is None:
            self.surface_fit_mask = ones(plume.shape)
        
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

    def calc_sky_background_mask(self, plume_img, next_img=None,
                                 lower_thresh=None,
                                 apply_movement_search=True,
                                 **settings_movement_search):
        """Retrieve and set background mask for 2D poly surface fit.

        Wrapper for method :func:`find_sky_background`

        Calculates mask specifying sky radiance pixels for modelling mode
        0 (plume background retrieval directly from plume image without
        an additional I0-image, using a 2D polynomial surface fit). The
        mask is stored in :attr:`surface_fit_mask`.

        Parameters
        ----------
        plume_img : Img
            the plume image for which the sky background pixels are supposed
            to be detected
        next_img : :obj:`Img`, optional
            second image used to compute the optical flow in :param:`plume_img`
        lower_thresh : :obj:`float`, optional
            lower intensity threshold. If provided, this value is used,
            else, the minimum value is derived from the minimum intensity
            in the plume image within the current 3 sky reference
            rectangles
        **settings_movement_search
            additional keyword arguments passed to :func:`find_movement`.
            Note that these may include settings for the optical flow
            calculation which are further passed to the
            initiation of the :class:`FarnebackSettings` class

        Returns
        -------
        array
            2D-numpy boolean numpy array specifying sky background pixels

        """
        mask = find_sky_background(plume_img, next_img,
                                   self.settings_dict(),
                                   lower_thresh,
                                   apply_movement_search,
                                   **settings_movement_search)
        self.surface_fit_mask = mask
        return mask

    def _init_bgsurf_mask(self, plume):
        logger.info("Initiating BG surface mask in PlumeBackgroundModel")
        mask = ones(plume.shape)
        self.surface_fit_mask = mask
        return mask

    def bg_from_poly_surface_fit(self, plume, mask=None, polyorder=2,
                                 pyrlevel=4):
        """Apply poly surface fit to plume image for bg retrieval.

        Parameters
        ----------
        plume : Img
            plume image
        mask : ndarray
            mask specifying gas free areas (if None, use all pixels). Note that
            the mask needs to be computed within the same ROI and at the same
            pyrlevel as the input plume image
        polyorder : int
            order of polynomial used for fit, defaults to 4
        pyrlevel : int
            pyramid level in which fit is performed
            (e.g. 4 => image size for fit is reduced by factor 2^4 = 16). Note
            that the
        :return tuple: 1st entry: fitted background image
            second: ``PolySurfaceFit`` object

        Returns
        -------
        ndarray
            fitted sky background

        Note
        ----
        The :class:`PolySurfaceFit` object used to retrieve the background
        is stored in the :attr:`_last_surffit`.

        """
        if not isinstance(plume, Img):
            raise TypeError("Need instance of pyplis Img class")
        if mask is None:
            mask = self.surface_fit_mask
        if not isinstance(mask, ndarray):
            try:
                mask = mask.img
                if not mask.shape == plume.shape:
                    raise AttributeError("Shape mismatch between mask and "
                                         "plume image")
            except:
                mask = self._init_bgsurf_mask(plume)
        pyrlevel_rel = pyrlevel - plume.pyrlevel
        if pyrlevel_rel < 0:
            print_log.warning("Pyramid level of input image (%d) is larger than desired "
                 "pyramid level for computation of surface fit (%d). Using "
                 "the current pyrlevel %d of input image" % (plume.pyrlevel,
                                                             pyrlevel))
            pyrlevel_rel = 0
        # update settings from input keyword arg

        fit = PolySurfaceFit(plume.img, mask,
                             polyorder=polyorder,
                             pyrlevel=pyrlevel_rel)
        if not fit.model.shape == plume.shape:
            raise ValueError("Mismatch in shape between input plume image and "
                             "fit result of PolySurfaceFit. Check pyramid "
                             "level of input image")
        self._last_surffit = fit
        return fit.model

    def get_tau_image(self, plume_img, bg_img=None,
                      check_state=True, **kwargs):
        """Determine current tau image for input plume image.

        Parameters
        ----------
        plume_img : Img
            plume image in intensity space
        bg_img : :obj:`Img`, optional
            sky radiance image (for ``self.CORR_MODE = 1 - 6``)
        check_state : bool
            if True and current mode != 0, it is checked whether the input
            images (plume and bg) have the same darkcorrection and vignetting
            state
        **kwargs :
            additional keyword arguments for updating current settings
            (valid input keywords (strings): mode, ygrad_rect,
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
            with regard to vignetting or dark correction.

        """
        if not isinstance(plume_img, Img):
            raise TypeError("Invalid, input type: need Img object...")
        # update current settings
        for k, v in kwargs.items():
            self.__setitem__(k, v)

        if not plume_img.is_darkcorr:
            logger.warning("plume image is not corrected for dark current")
        if plume_img.is_tau:
            raise AttributeError("Input image is already tau image")
        tau = None
        if self.mode == 0:  # no sky radiance image, poly surface fit
            bg = self.bg_from_poly_surface_fit(plume_img,
                                               self.surface_fit_mask,
                                               self.surface_fit_polyorder,
                                               self.surface_fit_pyrlevel)
            r = bg / plume_img.img
            # make sure no 0 values or neg. numbers are in the image
            r[r <= 0] = finfo(float).eps
            tau = log(r)

        else:
            if check_state:
                self._check_img_states(plume_img, bg_img)

            r = bg_img.img / plume_img.img
            # make sure no 0 values or neg. numbers are in the image
            r[r <= 0] = finfo(float).eps
            tau = log(r)
            if self.mode != 99:
                tau = self.correct_tau_curvature_ref_areas(tau)

        tau_img = Img(tau, **plume_img.meta)
        tau_img.edit_log.update(plume_img.edit_log)
        tau_img.meta["bit_depth"] = nan
        tau_img.edit_log["is_tau"] = True

        self.last_tau_img = tau_img

        return tau_img

    def get_aa_image(self, plume_on, plume_off, bg_on=None, bg_off=None,
                     check_state=True, **kwargs):
        """Retrieve apparent absorbance image from on and off imgs.

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
        check_state : bool
            if True and current mode != 0, it is checked whether the input
            images (plume and bg) have the same darkcorrection and vignetting
            state
        **kwargs :
            additional keyword arguments for updating current settings
            (valid input keywords (strings), e.g. ``surface_fit_mask`` if
            ``mode == 0``

        Returns
        -------
        Img
            plume AA image

        """
        if not isinstance(plume_on, Img) or not isinstance(plume_off, Img):
            raise TypeError("Need Img objects for background modelling")
        for k, v in kwargs.items():
            self.__setitem__(k, v)

        if self.mode == 0:
            mask = self.surface_fit_mask
            po = self.surface_fit_polyorder
            pyr = self.surface_fit_pyrlevel
            bg_on = self.bg_from_poly_surface_fit(plume_on, mask, po, pyr)
            bg_off = self.bg_from_poly_surface_fit(plume_off, mask, po, pyr)
            r_on = bg_on / plume_on.img
            # make sure no 0 values or neg. numbers are in the image
            r_on[r_on <= 0] = finfo(float).eps

            r_off = bg_off / plume_off.img
            # make sure no 0 values or neg. numbers are in the image
            r_off[r_off <= 0] = finfo(float).eps
            aa = log(r_on) - log(r_off)
        else:
            if check_state:
                self._check_img_states(plume_on, plume_off, bg_on, bg_off)

            r1 = bg_on.img / plume_on.img
            r1[r1 <= 0] = finfo(float).eps
            r2 = bg_off.img / plume_off.img
            r2[r2 <= 0] = finfo(float).eps
            aa = log(r1) - log(r2)
            if self.mode != 99:
                aa = self.correct_tau_curvature_ref_areas(aa)

        aa_img = Img(aa, **plume_on.meta)
        aa_img.edit_log.update(plume_on.edit_log)
        aa_img.meta["bit_depth"] = nan
        aa_img.edit_log["is_tau"] = True
        aa_img.edit_log["is_aa"] = True

        self.last_tau_img = aa_img
        return aa_img

    def correct_tau_curvature_ref_areas(self, tau_init):
        """Scale and correct curvature in initial tau image.

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
        mode = self.mode

        if not 1 <= mode <= 6:
            raise ValueError("This method only works for background model"
                             "modes (param CORR_MODE) 1-6")
        try:
            tau_init = tau_init.img
        except BaseException:
            pass
        if mode == 1:
            return scale_tau_img(tau_init, self.scale_rect)
        elif mode == 2:
            return corr_tau_curvature_vert_two_rects(tau_init,
                                                     self.scale_rect,
                                                     self.ygrad_rect)
        elif mode == 3:
            return corr_tau_curvature_vert_line(tau_init,
                                                self.ygrad_line_colnum,
                                                self.ygrad_line_startrow,
                                                self.ygrad_line_stoprow,
                                                self.ygrad_line_mask,
                                                self.ygrad_line_polyorder)[0]

        elif mode == 4:
            tau_init = corr_tau_curvature_vert_two_rects(tau_init,
                                                         self.scale_rect,
                                                         self.ygrad_rect)
            return corr_tau_curvature_hor_two_rects(tau_init,
                                                    self.scale_rect,
                                                    self.xgrad_rect)

        elif mode == 5:
            tau_init = corr_tau_curvature_vert_line(tau_init,
                                                    self.ygrad_line_colnum,
                                                    self.ygrad_line_startrow,
                                                    self.ygrad_line_stoprow,
                                                    self.ygrad_line_mask,
                                                    self.ygrad_line_polyorder
                                                    )[0]

            return corr_tau_curvature_hor_two_rects(tau_init,
                                                    self.scale_rect,
                                                    self.xgrad_rect)
        elif mode == 6:
            tau_init = corr_tau_curvature_vert_line(tau_init,
                                                    self.ygrad_line_colnum,
                                                    self.ygrad_line_startrow,
                                                    self.ygrad_line_stoprow,
                                                    self.ygrad_line_mask,
                                                    self.ygrad_line_polyorder
                                                    )[0]

            return corr_tau_curvature_hor_line(tau_init,
                                               self.xgrad_line_rownum,
                                               self.xgrad_line_startcol,
                                               self.xgrad_line_stopcol,
                                               self.xgrad_line_mask,
                                               self.xgrad_line_polyorder)[0]

    """Plotting"""

    def plot_sky_reference_areas(self, plume):
        """Plot the current sky ref areas into a plume image."""
        d = self.settings_dict()
        try:
            return plot_sky_reference_areas(plume, d)
        except:
            raise ValueError("Could not plot sky reference areas. Please check"
                             " if all relevant sky reference areas are set and"
                             " if not, set them manually or use class method"
                             " set_missing_ref_areas")

    def plot_tau_result(self, tau_img=None, tau_min=None, tau_max=None,
                        edit_profile_labels=True, legend_loc=3,
                        figheight=8, add_mode_info=False,
                        fsize_legend=12, fsize_labels=16,
                        **add_lines):
        """Plot current tau image including all reference areas.

        Parameters
        ----------
        tau_img : Img
            the tau image to be displayed
        tau_min : :obj:`float`, optional
            lower tau boundary to be displayed
        tau_max : :obj:`float`, optional
            upper tau boundary for colormap
        edit_profile_labels : bool
            beta version of smart layout for axis labels from profile
            subplots
        legend_loc : int
            number ID for specifying legend position
        figheight : int
            figure height in inches (dpi=matplotlib default)
        add_mode_info : bool
            if True, information about the used correction mode is
            included in the plot
        **kwargs:
            additional lines to be plotted, e.g.::
                pcs = [300, 400, 500, 600]

        """
        tau = tau_img
        if not isinstance(tau, Img):
            tau = self.last_tau_img
        if not isinstance(tau, Img):
            raise AttributeError("No tau image available in background model")
        tau = tau.duplicate().to_pyrlevel(0)
        tmin = tau_min
        tmax = tau_max
        if tau_max is None:
            tau_max = tau.max()
        if tau_min is None:
            tau_min = - tau_max
        cmap = shifted_color_map(tau_min, tau_max)
        ax = []
        figheight = 8
        # margins (left, bottom, top, inner)
        lm, bm, tm, im = 0.10, 0.10, 0.02, 0.01
        tau_frac = 0.7
        h0, w0 = tau.shape
        R = w0 / float(h0)

        d_panels = 1 - tm - bm - im - tau_frac

        fig = figure(figsize=(R * figheight, figheight))

        ax.append(fig.add_axes([lm, bm,
                                tau_frac, tau_frac]))

        ax.append(fig.add_axes([lm + tau_frac + im, bm,
                                d_panels, tau_frac]))

        ax.append(fig.add_axes([lm, bm + tau_frac + im,
                                tau_frac, d_panels]))

        if self.mode == 0:
            ax.append(fig.add_axes([lm + tau_frac + im,
                                    bm + tau_frac + im,
                                    d_panels, d_panels]))
            palette = colors.ListedColormap(['white', 'lime'])
            norm = colors.BoundaryNorm([0, .5, 1], palette.N)

            ax[3].imshow(self.surface_fit_mask, cmap=palette, norm=norm,
                         alpha=.7)

            ax[3].set_xticklabels([])
            ax[3].set_yticklabels([])

        ax[0].imshow(tau.img, cmap=cmap, vmin=tau_min, vmax=tau_max)

        ax[0].plot([self.ygrad_line_colnum, self.ygrad_line_colnum],
                   [0, h0], "-b", label="vert profile")
        ax[0].plot([0, w0], [self.xgrad_line_rownum, self.xgrad_line_rownum],
                   "-c", label="hor profile")
        for k, l in add_lines.items():
            try:
                x0, y0, x1, y1 = l.to_list()
                c = l.color
            except:
                x0, y0, x1, y1 = l
                c = "g"

            ax[0].plot([x0, x1], [y0, y1], "-", lw=2, c=c, label=k)

        ax[0].set_xlim([0, w0 - 1])
        ax[0].set_ylim([h0 - 1, 0])

        xs, ys, ws, hs = _roi_coordinates(self.scale_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="lime", fc="lime",
                        label="scale_rect", alpha=0.3))

        xs, ys, ws, hs = _roi_coordinates(self.ygrad_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="b", fc="b",
                        label="ygrad_rect", alpha=0.3))

        xs, ys, ws, hs = _roi_coordinates(self.xgrad_rect)
        ax[0].add_patch(Rectangle((xs, ys), ws, hs, ec="c", fc="c",
                        label="xgrad_rect", alpha=0.3))

        ax[2].set_xticklabels([])
        ax[1].set_yticklabels([])

        # plot vertical profile
        lvert = LineOnImage(self.ygrad_line_colnum, 0, self.ygrad_line_colnum,
                            h0 - 1, line_id="vert")
        p_vert = lvert.get_line_profile(tau.img)

        ax[1].plot(p_vert, arange(0, len(p_vert), 1), "-b",
                   label="vert profile")
        ax[1].yaxis.tick_right()
        ax[1].set_ylim([h0 - 1, 0])
        setp(ax[1].xaxis.get_majorticklabels(), rotation=15)
        ax[1].yaxis.tick_right()

        # plot horizontal profile
        line_hor = LineOnImage(0, self.xgrad_line_rownum, w0 - 1,
                               self.xgrad_line_rownum, line_id="hor")
        p_hor = line_hor.get_line_profile(tau.img)
        ax[2].plot(arange(0, len(p_hor), 1), p_hor, "-c",
                   label="hor profile")
        # ax[2].get_yaxis().set_ticks(horYLabels)
        # ax[2].set_ylim([-.05,.25])
        ax[2].set_xlim([0, w0 - 1])

        # subplots_adjust(wspace=0.02, hspace=0.02)
        ax[2].axhline(0, ls="--", color="k")
        ax[1].axvline(0, ls="--", color="k")

        if edit_profile_labels:
            low, high = tmin, tmax
            if low is None:
                low = p_vert.min()
            if high is None:
                high = p_vert.max()
            _range = high - low
            lbls = [0]
            if high > 0 and high / _range > 0.2:
                lbls.append(high - _range * .1)
            if low < 0 and abs(low) / high > 0.5:
                lbls.insert(0, low + _range * .1)
            ax[1].get_xaxis().set_ticks(lbls)
            lbl_str = ["%.2f" % lbl for lbl in lbls]
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
                lbls.append(high - _range * .1)
            if low < 0 and abs(low) / high > 0.5:
                lbls.insert(0, low + _range * .1)
            ax[2].get_yaxis().set_ticks(lbls)
            lbl_str = ["%.2f" % lbl for lbl in lbls]
            ax[2].set_ylim([low, high])
            ax[2].set_yticklabels(lbl_str)

        ax[1].set_xlabel(r"$\tau$", fontsize=fsize_labels)
        ax[2].set_ylabel(r"$\tau$", fontsize=fsize_labels)
        if add_mode_info:
            ax[0].set_xlabel("CORR_MODE: %s" % self.CORR_MODE,
                             fontsize=fsize_labels)
        ax[0].legend(loc=legend_loc, fancybox=True, framealpha=0.7,
                     fontsize=fsize_legend)
        return fig

    """Helpers"""

    def settings_dict(self):
        """Write current sky reference areas and masks into dictionary."""
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

    @property
    def mode_info_dict(self):
        """Return information on available bg modelling modes."""
        return od([[0, "No additional BG image: poly surface fit using plume"
                       " image pixels specified with mask"],
                   [1, "Scaling of bg image in rect scale_rect"],
                   [2, "Scaling (mode 1, scale_rect) and linear y gradient "
                       "correction using rects scale_rect and ygrad_rect"],
                   [3, "Scaling (mode 1, scale_rect) and quadratic y "
                       "gradient correction using vertical profile line"],
                   [4, "Like 2, including linear x gradient correction using "
                       "rect xgrad_rect"],
                   [5, "Like 3, including linear x gradient correction using "
                       "rect xgrad_rect"],
                   [6, "Like 3, including quadratic x gradient correction "
                       "using horizontal profile line"],
                   [99, "USE AS IS: no background modelling performed"]])

    def mode_info(self, mode_num):
        """Return short information about one of the available modelling modes.

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
        """Print information about the different correction modes."""
        print_log.info("Available modes for automatic plume background retrieval")
        for k, v in self.mode_info_dict.items():
            print_log.info("Mode %s: %s" % (k, v))

    def _check_rect(self, rect, img):
        """Check if rect is not None and if it is within image borders.

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

    def _check_img_states(self, img, *more_images):
        """Check if other images have the same edit state relative to image."""
        if not all([isinstance(x, Img) for x in more_images]):
            raise TypeError("All provided images need to be of type Img")
        if not all([x.is_vigncorr == img.is_vigncorr for x in more_images]):
            raise AttributeError("Cannot model tau image: mismatch with "
                                 "regard to vignetting correction state "
                                 "between plume image(s) and sky radiance "
                                 "image(s)")
        elif not all([x.is_darkcorr == img.is_darkcorr for x in more_images]):
            raise AttributeError("Cannot model tau image: mismatch with "
                                 "regard to dark correction state "
                                 "between plume image(s) and sky radiance "
                                 "image(s)")

    def __setitem__(self, key, value):
        """Update class item."""
        if key in self.__dict__:
            logger.info("Updating %s in background model" % key)
            self.__dict__[key] = value
        elif key == "mode":
            "Updating %s in background model" % key
            self.mode = value
        elif key == "surface_fit_mask":
            self.surface_fit_mask = value
        elif key == "CORR_MODE":
            logger.warning("Got input key CORR_MODE which is out-dated in versions 0.10+"
                 ". Updated background modelling mode accordingly")
            self.mode = value

    def __call__(self, plume, bg, **kwargs):
        return self.get_tau_image(plume, bg, **kwargs)


def _mean_in_rect(img_array, rect=None):
    """Get mean and standard deviation of pixels within rectangle.

    :param ndarray imgarray: the image data
    :param rect: rectanglular area ``[x0, y0, x1, y1]` where x0 < x1, y0 < y1
    """
    if rect is None:
        sub = img_array
    else:
        sub = img_array[rect[1]: rect[3], rect[0]: rect[2]]
    return sub.mean(), sub.std()


def scale_tau_img(tau, rect):
    """Scale tau image such that it fulfills tau==0 in reference area."""
    avg, _ = _mean_in_rect(tau, rect)
    return tau - avg


def scale_bg_img(bg, plume, rect):
    """Normalise background image to plume image intensity in input rect.

    Parameters
    ----------
    bg : ndarray
        background image
    plume : ndarray
        plume image
    rect : list
        list containing rectangle coordinates

    Returns
    -------
    ndarray
        the scaled background image

    """
    # bg, plume = [x.img for x in [bg, plume] if isinstance(x, Img)]
    mean_bg, _ = _mean_in_rect(bg, rect)
    mean_img, _ = _mean_in_rect(plume, rect)
    del_rad = mean_img / float(mean_bg)
    return bg * del_rad


def corr_tau_curvature_vert_two_rects(tau0, r0, r1):
    """Apply vertical linear background curvature correction to tau img.

    Retrieves pixel mean value from two rectangular areas and determines
    linear offset function based on the vertical positions of the rectangle
    center coordinates. The corresponding offset for each image row is then
    subtracted from the input tau image

    :param (ndarray, Img) tau0: inital tau image
    :param list r0: 1st rectanglular area ``[x0, y0, x1, y1]`
    :param list r1: 2nd rectanglular area ``[x0, y0, x1, y1]`
    :return ndarray: modified tau image

    """
    try:
        tau0 = tau0.img
    except BaseException:
        pass
    y0, y1 = 0.5 * (r0[1] + r0[3]), 0.5 * (r1[1] + r1[3])
    max_y = tau0.shape[0]

    mean_r0, _ = _mean_in_rect(tau0, r0)
    mean_r1, _ = _mean_in_rect(tau0, r1)

    slope = float(mean_r0 - mean_r1) / float(y0 - y1)
    offs = mean_r1 - slope * y1

    ygrid = linspace(0, max_y - 1, max_y, dtype=int)
    poly_vals = offs + slope * ygrid
    tau_mod = (tau0.T - poly_vals).T
    return tau_mod  # , vert_poly


def corr_tau_curvature_hor_two_rects(tau0, r0, r1):
    """Apply horizonzal linear background curvature correction to tau img.

    Retrieves pixel mean values from two rectangular areas and determines
    linear offset function based on the horizontal positions of the
    rectangle center coordinates. The corresponding offset for each image
    row is then subtracted from the input tau image.

    :param (ndarray, Img) tau0: inital tau image
    :param list r0: 1st rectanglular area ``[x0, y0, x1, y1]`
    :param list r1: 2nd rectanglular area ``[x0, y0, x1, y1]`
    :return ndarray: modified tau image
    """
    try:
        tau0 = tau0.img
    except BaseException:
        pass
    x0, x1 = 0.5 * (r0[0] + r0[2]), 0.5 * (r1[0] + r1[2])
    max_x = tau0.shape[1]

    mean_r0, _ = _mean_in_rect(tau0, r0)
    mean_r1, _ = _mean_in_rect(tau0, r1)

    slope = float(mean_r0 - mean_r1) / float(x0 - x1)
    offs = mean_r1 - slope * x1

    xgrid = linspace(0, max_x - 1, max_x, dtype=int)
    poly_vals = offs + slope * xgrid
    tau_mod = tau0 - poly_vals
    return tau_mod  # , vert_poly


def corr_tau_curvature_vert_line(tau0, pos_x, start_y=0, stop_y=None,
                                 row_mask=None, polyorder=2):
    """Correct vertical tau curvature using selected row indices of vertical line.

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
    except BaseException:
        pass
    max_y = tau0.shape[0]

    line_vert = LineOnImage(pos_x, 0, pos_x, max_y)
    vert_profile = line_vert.get_line_profile(tau0)

    if stop_y is None:
        stop_y = max_y

    ygrid = linspace(0, max_y - 1, max_y, dtype=int)
    try:
        if len(row_mask) == max_y:
            mask = row_mask
    except BaseException:
        mask = logical_and(ygrid >= start_y, ygrid <= stop_y)

    vert_poly = poly1d(polyfit(ygrid[mask], vert_profile[mask], polyorder))

    poly_vals = vert_poly(ygrid)
    tau_mod = (tau0.T - poly_vals).T
    return (tau_mod, vert_poly)


def corr_tau_curvature_hor_line(tau0, pos_y, start_x=0, stop_x=None,
                                col_mask=None, polyorder=2):
    #  fixme: this doc is propably wrong
    """Correct vertical tau curvature using selected row indices of vertical line.

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
    except BaseException:
        pass
    max_x = tau0.shape[1]
    line_hor = LineOnImage(0, pos_y, max_x, pos_y)
    hor_profile = line_hor.get_line_profile(tau0)

    if stop_x is None:
        stop_x = max_x

    xgrid = linspace(0, max_x - 1, max_x, dtype=int)
    try:
        if len(col_mask) == max_x:
            mask = col_mask
    except BaseException:
        mask = logical_and(xgrid >= start_x, xgrid <= stop_x)

    hor_poly = poly1d(polyfit(xgrid[mask], hor_profile[mask], polyorder))

    poly_vals = hor_poly(xgrid)
    tau_mod = tau0 - poly_vals
    return (tau_mod, hor_poly)


def find_sky_reference_areas(plume_img, sigma_blur=2, plot=False):
    """Take an input plume image and identify suitable sky reference areas."""
    try:
        plume = plume_img.img
    except BaseException:
        plume = plume_img
    plume = gaussian_filter(plume, sigma_blur)
    h, w = plume.shape
    results = {}
    vert_mag, hor_mag = int(h * 0.005) + 1, int(w * 0.005) + 1

    # estimate mean intensity in left image part (without flank pixels)
    y0_left = argmin(gradient(plume[vert_mag: h - vert_mag, hor_mag]))

    avg_left = plume[vert_mag: y0_left - vert_mag, hor_mag:hor_mag * 2].mean()
    # estimate mean intensity in right image part (without flank pixels
    grad = gradient(plume[vert_mag: h - vert_mag, w - hor_mag])
    y0_right = argmin(grad)
    avg_right = plume[vert_mag: y0_right - vert_mag,
                      w - 2 * hor_mag: w - hor_mag].mean()
    results["xgrad_line_rownum"] = vert_mag
    # brighter on the right image side (assume this is clear sky)
    if avg_right > avg_left:
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
    """Plot provided sky reference areas into a plume image.

    :param (ndarray, Img) plume_img: plume image data
    :param dict settings_dict: dictionary containing settings (e.g. retrieved
        using :func:`find_sky_reference_areas`)
    """
    try:
        plume = plume_img.img
    except BaseException:
        plume = plume_img
    if ax is None:
        fig, ax = subplots(1, 1)
    r = settings_dict
    h0, w0 = plume.shape[:2]

    ax.imshow(plume, cmap="gray")
    ax.plot([r["ygrad_line_colnum"], r["ygrad_line_colnum"]],
            [r["ygrad_line_startrow"], r["ygrad_line_stoprow"]],
            "-", c="lime", label="vert profile")
    ax.plot([r["xgrad_line_startcol"], r["xgrad_line_stopcol"]],
            [r["xgrad_line_rownum"], r["xgrad_line_rownum"]],
            "-", c="lime", label="hor profile")
    ax.set_xlim([0, w0 - 1])
    ax.set_ylim([h0 - 1, 0])

    xs, ys, ws, hs = _roi_coordinates(r["scale_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="lime", fc="lime", alpha=0.3,
                           label="scale_rect"))

    xs, ys, ws, hs = _roi_coordinates(r["ygrad_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="b", fc="b", alpha=0.3,
                           label="ygrad_rect"))

    xs, ys, ws, hs = _roi_coordinates(r["xgrad_rect"])
    ax.add_patch(Rectangle((xs, ys), ws, hs, ec="c", fc="c", alpha=0.3,
                           label="xgrad_rect"))
    ax.legend(loc="best", fancybox=True, framealpha=0.5, fontsize=10)

    return ax


def find_sky_background(plume_img, next_img=None,
                        bgmodel_settings_dict=None,
                        lower_thresh=None,
                        apply_movement_search=True,
                        **settings_movement_search):
    """Prepare mask for background fit based on analysis of current image.

    The mask is determined by applying an intensity threshold to a plume
    image based on the intensities in 3 sky reference rectangles of an
    instance of the :class:`PlumeBackgroundModel` object. If not specified
    on input from an exisiting instance of the class (using
    :param:`bgmodel_settings_dict`), the 3 required reference areas are
    calculated automatically using :func:`find_sky_reference_areas`.
    Alternatively, the lower threshold can be provided on input using
    :param:`lower_thresh`.
    Furthermore, pixels showing movement between the input image
    :param:`plume_img`and a second provided image (i.e. :param:`next_img`)
    can be excluded. The detection of movement between the two frames is
    performed using the movement detection algorithm :func:`find_movement`
    (see :mod:`plumespeed`).

    Note
    ----
    1. This is a Beta version
    2. The method requires images in radiances space (i.e. plume and
        terrain pixels appear darker than the sky background).
    3. The input plume image should not contain clouds. If it does, it is
        highly recommended to make use of the movement detection algorithm

    Parameters
    ----------
    plume_img : Img
        the plume image for which the sky background pixels are supposed
        to be detected
    next_img : :obj:`Img`, optional
        second image used to compute the optical flow in :param:`plume_img`
    bgmodel_settings_dict : dict
        dictionary containing information about sky reference areas (e.g.
        created using :func:`settings_dict` from an existing instance of
        the :class:`PlumeBackgroundModel`). If not specified, the required
        reference rectangles (``scale_rect``, ``ygrad_rect``,
        ``xgrad_rect``) are determined automatically using
        :func:`find_sky_reference_areas`.
    lower_thresh : :obj:`float`, optional
        lower intensity threshold. If provided, this value is used rather
        than the value derived from the 3 sky reference rectangles (see
        :param:`bgmodel_settings_dict`)
    **settings_movement_search
        additional keyword arguments passed to :func:`find_movement`.
        Note that these may include settings for the optical flow
        calculation which are further passed to the
        initiation of the :class:`FarnebackSettings` class

    Returns
    -------
    array
        2D-numpy boolean numpy array specifying sky background pixels

    """
    if bgmodel_settings_dict is None:
        bgmodel_settings_dict = {}
    if not isinstance(plume_img, Img):
        raise ValueError("Invalid input for parameter plume_img: need"
                         "Img object, got %s" % type(plume_img))
    if plume_img.is_tau:
        raise AttributeError("Input plume_img is an optical density image. "
                             "This method only works for images in "
                             "intensity mode where pixel values are gray "
                             "values!")
    if lower_thresh is None:
        bg_model = PlumeBackgroundModel(**bgmodel_settings_dict)
        bg_model.set_missing_ref_areas(plume_img)
        mean, low, high = bg_model.mean_in_rects(plume_img)
        lower_thresh = mean * 0.9

    mask = (plume_img.img > lower_thresh)
    if apply_movement_search:
        if not isinstance(next_img, Img):
            raise ValueError("Invalid input for parameter next_img: need"
                             "Img object, got %s" % type(next_img))
        no_movement = ~find_movement(plume_img, next_img,
                                     **settings_movement_search)
        mask = mask * no_movement

    return mask
