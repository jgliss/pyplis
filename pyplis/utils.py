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
"""Pyplis module containing low level utilitiy methods and classes."""
import os
from numpy import (vstack, asarray, ndim, round, hypot, linspace, sum, zeros,
                   angle, array, cos, sin, arctan, dot, int32, pi,
                   isnan, nan, mean, ndarray)

from numpy.linalg import norm
from scipy.ndimage import map_coordinates


from matplotlib.pyplot import subplot, subplots, tight_layout, draw
from matplotlib.patches import Polygon, Rectangle

from pandas import Series
from cv2 import cvtColor, COLOR_BGR2GRAY, fillPoly

from pyplis import logger
from pyplis.helpers import map_coordinates_sub_img, same_roi, map_roi, roi2rect
from pyplis.inout import get_cam_ids
from pyplis.glob import DEFAULT_ROI

def identify_camera_from_filename(filepath):
    """Identify camera based on image filepath convention.

    Parameters
    ----------
    filepath : str
        valid image file path

    Returns
    -------
    str
       ID of Camera that matches best

    Raises
    ------
    IOError
        Exception is raised if no match can be found

    """
    from pyplis.camera_base_info import CameraBaseInfo
    if not os.path.exists(filepath):
        logger.warning("Invalid file path")
    cam_id = None
    all_ids = get_cam_ids()
    max_match_num = 0
    for cid in all_ids:
        cam = CameraBaseInfo(cid)
        cam.get_img_meta_from_filename(filepath)
        matches = sum(list(cam._fname_access_flags.values()))
        if matches > max_match_num:
            max_match_num = matches
            cam_id = cid
    if max_match_num == 0:
        raise IOError("Camera type could not be identified based on input"
                      "file name {}".format(os.path.basename(filepath)))
    return cam_id


class LineOnImage(object):
    """Class representing a line on an image

    Main purpose is data extraction along this line on a discrete image grid.
    This is done using spline interpolation.

    Parameters
    ----------
    x0 : int
        start x coordinate
    y0 : int
        start y coordinate
    x1 : int
        stop x coordinate
    y1 : int
        stop y coordinate
    normal_orientation : str
        orientation of normal vector, choose from left or right (left means in
        negative x direction for a vertical line)
    roi_abs_def : list
        ROI specifying image sub coordinate system in which the line
        coordinates are defined (is used to convert to other image shape
        settings)
    pyrlevel_def : int
        pyramid level of image for which start /stop coordinates are defined
    line_id : str
        string for identification (optional)

    Note
    ----
    The input coordinates correspond to relative image coordinates
    with respect to the input ROI (``roi_def``) and pyramid level
    (``pyrlevel_def``)
    """
    def __init__(self, x0=0, y0=0, x1=1, y1=1, normal_orientation="right",
                 roi_abs_def=DEFAULT_ROI, pyrlevel_def=0, line_id="",
                 color="lime", linestyle="-"):

        self.line_id = line_id  # string ID of line
        self.color = color
        self.linestyle = linestyle
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        elif x0 == x1 and y0 > y1:
            y0, y1 = y1, y0
        self.x0 = x0  # start x coordinate
        self.y0 = y0  # start y coordinate
        self.x1 = x1  # stop x coordinate
        self.y1 = y1  # stop y coordinate

        self._roi_abs_def = roi_abs_def
        self._pyrlevel_def = pyrlevel_def
        self._rect_roi_rot = None
        self._line_roi_abs = DEFAULT_ROI
        self._last_rot_roi_mask = None

        self.profile_coords = None

        self._dir_idx = {"left": 0,
                         "right": 1}

        self.normal_vecs = [None, None]

        self._velo_glob = nan
        self._velo_glob_err = nan
        self._plume_props = None

        self.check_coordinates()
        self.normal_orientation = normal_orientation

        self.prepare_coords()

    @property
    def start(self):
        """x, y coordinates of start point (``[x0, y0]``)."""
        return [self.x0, self.y0]

    @start.setter
    def start(self, val):
        try:
            if len(val) == 2:
                self.x0 = val[0]
                self.y0 = val[1]
        except BaseException:
            logger.warning("Start coordinates could not be set")

    @property
    def stop(self):
        """x, y coordinates of stop point (``[x1, y1]``)."""
        return [self.x1, self.y1]

    @stop.setter
    def stop(self, val):
        try:
            if len(val) == 2:
                self.x1 = val[0]
                self.y1 = val[1]
        except BaseException:
            logger.warning("Stop coordinates could not be set")

    @property
    def center_pix(self):
        """Return coordinate of center pixel."""
        dx, dy = self._delx_dely()
        xm = self.x0 + dx / 2.
        ym = self.y0 + dy / 2.
        return xm, ym

    @property
    def normal_orientation(self):
        """Get / set value for orientation of normal vector."""
        return self._normal_orientation

    @normal_orientation.setter
    def normal_orientation(self, val):
        if val not in ["left", "right"]:
            raise ValueError("Invalid input for attribute orientation, please"
                             " choose from left or right")
        dx, dy = self._delx_dely()
        if dx * dy < 0:
            self._dir_idx["left"] = 1
            self._dir_idx["right"] = 0
        self._normal_orientation = val

    @property
    def line_frame(self):
        """ROI framing the line (in line coordinate system)."""
        return map_roi(self._line_roi_abs, self.pyrlevel_def)

    @property
    def line_frame_abs(self):
        """ROI framing the line (in absolute coordinate system)."""
        return self._line_roi_abs

    @property
    def roi_def(self):
        """ROI in which line is defined (at current ``pyrlevel``)."""
        return map_roi(self.roi_abs_def, pyrlevel_rel=self.pyrlevel_def)

    @property
    def roi_abs_def(self):
        """Return current ROI (in absolute detector coordinates)."""
        return self._roi_abs_def

    @roi_abs_def.setter
    def roi_abs_def(self):
        raise AttributeError("This attribute is not supposed to be changed, "
                             "please use method convert() to create a new "
                             "LineOnImage object "
                             "corresponding to other image shape settings")

    # Redundancy (after renaming attribute in v0.10)
    @property
    def pyrlevel(self):
        """Pyramid level at which line coords are defined."""
        logger.warning("This method was renamed in version 0.10. "
             "Please use pyrlevel_def")
        return self._pyrlevel_def

    @pyrlevel.setter
    def pyrlevel(self):
        raise AttributeError("This attribute is not supposed to be changed, "
                             "please use method convert() to create a new "
                             "LineOnImage object "
                             "corresponding to other image shape settings")

    @property
    def roi_abs(self):
        """Return current ROI (in absolute detector coordinates)."""
        logger.warning("This method was renamed in version 0.10. Please use roi_abs_def")
        return self._roi_abs_def

    @roi_abs.setter
    def roi_abs(self):
        raise AttributeError("This attribute is not supposed to be changed, "
                             "please use method convert() to create a new "
                             "LineOnImage object "
                             "corresponding to other image shape settings")

    @property
    def pyrlevel_def(self):
        """Pyramid level at which line coords are defined."""
        return self._pyrlevel_def

    @pyrlevel_def.setter
    def pyrlevel_def(self):
        """Raise AttributeError."""
        raise AttributeError("This attribute is not supposed to be changed, "
                             "please use method convert() to create a new "
                             "LineOnImage object "
                             "corresponding to other image shape settings")

    @property
    def coords(self):
        """Return coordinates as ROI list."""
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def rect_roi_rot(self):
        """Rectangle specifying coordinates of ROI aligned with line normal."""
        try:
            if not self._rect_roi_rot.shape == (5, 2):
                raise Exception
        except BaseException:
            logger.info("Rectangle for rotated ROI was not set and is not being "
                  "set to default depth of +/- 30 pix around line. Use "
                  "method set_rect_roi_rot to change the rectangle")
            self.set_rect_roi_rot()
        return self._rect_roi_rot

    @property
    def velo_glob(self):
        """Global velocity in m/s, assigned to this line.

        Raises
        ------
        AttributeError
            if current value is not of type float

        """
        if not isinstance(self._velo_glob, float) or isnan(self._velo_glob):
            raise AttributeError("Global velocity not assigned to line")
        return self._velo_glob

    @velo_glob.setter
    def velo_glob(self, val):
        try:
            val = float(val)
            if isnan(val):
                raise Exception
        except BaseException:
            raise ValueError("Invalid input, need float or int...")
        if val < 0:
            raise ValueError("Velocity must be larger than 0")
        elif val > 40:
            logger.warning("Large value warning: input velocity exceeds 40 m/s")
        self._velo_glob = val
        if self._velo_glob_err is None or isnan(self._velo_glob_err):
            logger.warning("Global velocity error not assigned, assuming 50% of "
                 "velocity")
            self.velo_glob_err = val * 0.50

    @property
    def velo_glob_err(self):
        """Error of global velocity in m/s, assigned to this line.

        Raises
        ------
        AttributeError
            if current value is not of type float

        """
        if not isinstance(self._velo_glob_err, float) or\
                isnan(self._velo_glob_err):
            raise AttributeError("Global velocity error not assigned to line")
        return self._velo_glob_err

    @velo_glob_err.setter
    def velo_glob_err(self, val):
        try:
            val = float(val)
            if isnan(val):
                raise Exception
        except BaseException:
            raise ValueError("Invalid input, need float or int...")
        self._velo_glob_err = val

    @property
    def plume_props(self):
        """:class:`LocalPlumeProperties` object assigned to this list."""
        from pyplis import LocalPlumeProperties
        if not isinstance(self._plume_props, LocalPlumeProperties):
            raise AttributeError("Local plume properties not assigned to line")
        return self._plume_props

    @plume_props.setter
    def plume_props(self, val):
        from pyplis import LocalPlumeProperties
        if not isinstance(val, LocalPlumeProperties):
            raise ValueError("Invalid input, need class LocalPlumeProperties")
        self._plume_props = val

    def dist_other(self, other):
        """Determine the distance to another line.

        Note
        ----
            1. The offset is applied in relative coordinates, i.e. it does not
            consider the pyramide level or ROI.

            #. The two lines need to be parallel

        Parameters
        ----------
        other : LineOnImage
            the line to which the distance is retrieved

        Returns
        -------
        float
            retrieved distance in pixel coordinates

        Raises
        ------
        ValueError
            if the two lines are not parallel

        """
        dx0, dy0 = other.x0 - self.x0, other.y0 - self.y0
        dx1, dy1 = other.x1 - self.x1, other.y1 - self.y1
        if dx1 != dx0 or dy1 != dy0:
            logger.warning("Lines are not parallel...")
        return mean([norm([dx0, dy0]), norm([dx1, dy1])])

    def offset(self, pixel_num=20, line_id=None):
        """Return a new line shifted within normal direction.

        Note
        ----

            1. The offset is applied in relative coordinates, i.e. it does not
                consider the pyramide level or ROI

            2. The determined required displacement (dx, dy) is converted into
                integers

        Parameters
        ----------
        pixel_num : int
            shift length in pixels
        line_id : str
            string ID of new line, if None (default) it is set automatically

        Returns
        -------
        LineOnImage
            shifted line

        """
        if line_id is None:
            line_id = self.line_id + "_shifted_%dpix" % pixel_num
        dx, dy = self.normal_vector * pixel_num
        x0, x1 = self.x0 + int(dx), self.x1 + int(dx)
        y0, y1 = self.y0 + int(dy), self.y1 + int(dy)
        return LineOnImage(x0, y0, x1, y1,
                           normal_orientation=self.normal_orientation,
                           line_id=line_id,
                           color=self.color,
                           linestyle=self.linestyle,
                           pyrlevel_def=self.pyrlevel_def)

    def convert(self, to_pyrlevel=0, to_roi_abs=DEFAULT_ROI):
        """Convert to other image preparation settings."""
        if to_pyrlevel == self.pyrlevel_def and same_roi(self.roi_abs_def,
                                                         to_roi_abs):
            logger.info("Same shape settings, returning current line object""")
            return self
        # first convert to absolute coordinates
        ((x0, x1),
         (y0, y1)) = map_coordinates_sub_img([self.x0, self.x1],
                                             [self.y0, self.y1],
                                             roi_abs=self._roi_abs_def,
                                             pyrlevel=self._pyrlevel_def,
                                             inverse=True)
        # now convert from absolute into specified coords
        (x0, x1), (y0, y1) = map_coordinates_sub_img([x0, x1], [y0, y1],
                                                     roi_abs=to_roi_abs,
                                                     pyrlevel=to_pyrlevel,
                                                     inverse=False)

        new_line = LineOnImage(x0, y0, x1, y1, roi_abs_def=to_roi_abs,
                               pyrlevel_def=to_pyrlevel,
                               normal_orientation=self.normal_orientation,
                               line_id=self.line_id,
                               color=self.color, linestyle=self.linestyle)
        try:
            new_line.velo_glob = self.velo_glob
        except BaseException:
            pass
        try:
            new_line.velo_glob_err = self.velo_glob_err
        except BaseException:
            pass
        try:
            new_line.plume_props = self.plume_props
        except BaseException:
            pass

        return new_line

    def check_coordinates(self):
        """Check line coordinates.

        Checks if coordinates are in the right order and exchanges start / stop
        points if not

        Raises
        ------
        ValueError
            if any of the current coordinates is smaller than zero

        """
        if any([x < 0 for x in self.coords]):
            raise ValueError("Invalid value encountered, all coordinates of "
                             "line must exceed zero, current coords: %s"
                             % self.coords)
        if self.start[0] > self.stop[0]:
            logger.info("x coordinate of start point is larger than of stop point: "
                  "start and stop will be exchanged")
            self.start, self.stop = self.stop, self.start

    def in_image(self, img_array):
        """Check if this line is within the coordinates of an image array.

        Parameters
        ----------
        img_array : array
            image data

        Returns
        -------
        bool
            True if point is in image, False if not

        """
        if not all(self.point_in_image(p, img_array)
                   for p in [self.start, self.stop]):
            return False
        return True

    def point_in_image(self, x, y, img_array):
        """Check if a given coordinate is within image.

        Parameters
        ----------
        x : int
            x coordinate of point
        y : int
            y coordinate of point
        img_array : array
            image data

        Returns
        -------
        bool
            True if point is in image, False if not

        """
        h, w = img_array.shape[:2]
        if not 0 < x < w:
            logger.info("x coordinate out of image")
            return False
        if not 0 < y < h:
            logger.info("y coordinate out of image")
            return False
        return True

    def get_roi_abs_coords(self, img_array, add_left=5, add_right=5,
                           add_bottom=5, add_top=5):
        """Get a rectangular ROI covering this line.

        Parameters
        ----------
        add_left : int
            expand range to left of line (in pix)
        add_right : int
            expand range to right of line (in pix)
        add_bottom : int
            expand range to bottom of line (in pix)
        add_top : int
            expand range to top of line (in pix)

        Returns
        -------
        list
            ROI around this line

        """
        x0, x1 = self.start[0] - add_left, self.stop[0] + add_right
        # y start must not be smaller than y stop
        y_arr = [self.start[1], self.stop[1]]
        y_min, y_max = min(y_arr), max(y_arr)
        y0, y1 = y_min - add_top, y_max + add_bottom
        roi = self.check_roi_borders([x0, y0, x1, y1], img_array)
        roi_abs = map_roi(roi, pyrlevel_rel=-self.pyrlevel_def)
        self._line_roi_abs = roi_abs
        return roi_abs

    def integrate_profile(self, input_img, pix_step_length=None):
        """Integrate the line profile on input image.

        Parameters
        ----------
        input_img : Img
            input image data for

        """
        try:
            # in case input is an Img
            input_img = input_img.img
        except:
            pass
        vals = self.get_line_profile(input_img)
        if pix_step_length is None:
            logger.warning("No information about integration step lengths provided "
                 "Integration is performed in units of pixels")
            return sum(vals)
        try:
            pix_step_length = pix_step_length.img
        except:
            pass
        if isinstance(pix_step_length, ndarray):
            if not pix_step_length.shape == input_img.shape:
                raise ValueError("Shape mismatch between input image and "
                                 "pixel")
            pix_step_length = self.get_line_profile(pix_step_length)
        return sum(vals * pix_step_length)

    def _roi_from_rot_rect(self):
        """Set current ROI from current rotated rectangle coords."""
        r = self._rect_roi_rot
        xc = asarray([x[0] for x in r])
        xc[xc < 0] = 0
        yc = asarray([x[1] for x in r])
        yc[yc < 0] = 0
        roi = [xc.min(), yc.min(), xc.max(), yc.max()]
        self._line_roi_abs = map_roi(roi, pyrlevel_rel=-self.pyrlevel_def)
        return roi

    def set_rect_roi_rot(self, depth=None):
        """Get rectangle for rotated ROI based on current tilting.

        Note
        ----
        This function also changes the current ``roi_abs`` attribute

        Parameters
        ----------
        depth : int
            depth of rotated ROI (in normal direction of line) in pixels

        Returns
        -------
        list
            rectangle coordinates

        """
        dx, dy = self._delx_dely()
        if depth is None:
            depth = norm((dx, dy)) * 0.10

        n = self.normal_vecs[1]
        dx0, dy0 = n * depth / 2.0

# ==============================================================================
#
#         if sign(dx0) == sign(dy0):
#             dx0 = -dx0
#             dy0 = -dy0
# ==============================================================================
        x0 = self.x0 + int(dx0)
        y0 = self.y0 + int(dy0)
        offs = array([x0, y0])

        w = self.length()
        r = array([(0, 0), (w, 0), (w, depth), (0, depth), (0, 0)])

        dx, dy = self._delx_dely()
        try:
            theta = arctan(dy / dx)
        except ZeroDivisionError:
            theta = pi / 2
        # rotation matrix (account for neg. y direction)
        m_rot = array([[cos(theta), sin(theta)],
                       [-sin(theta), cos(theta)]])
        r = dot(r, m_rot) + offs
        self._rect_roi_rot = r
        self._roi_from_rot_rect()
        return r

# ==============================================================================
#     def set_rect_roi_rot_v0(self, depth=None):
#         """Get rectangle for rotated ROI based on current tilting
#
#         Note
#         ----
#         This function also changes the current ``roi_abs`` attribute
#
#         Parameters
#         ----------
#         depth : int
#             depth of rotated ROI (in normal direction of line) in pixels
#
#         Returns
#         -------
#         list
#             rectangle coordinates
#         """
#         if depth is None:
#             depth  = self.length() * 0.10
#         dx0, dy0 = self.normal_vector * depth / 2.0
#
#         if sign(dx0) == sign(dy0):
#             dx0 = -dx0
#             dy0 = -dy0
#         x0 = self.x0 + int(dx0)
#         y0 = self.y0 + int(dy0)
#         offs = array([x0, y0])
#
#         w = self.length()
#         r = array([(0, 0), (w, 0), (w, depth), (0, depth), (0, 0)])
#
#         dx, dy = self._delx_dely()
#         try:
#             theta = arctan(dy / dx)
#         except ZeroDivisionError:
#             theta = pi / 2
#         #rotation matrix (account for neg. y direction)
#         m_rot = array([[cos(theta), sin(theta)],
#                         [-sin(theta), cos(theta)]])
#         r = dot(r, m_rot) + offs
#         self._rect_roi_rot = r
#         self._roi_from_rot_rect()
#         return r
# ==============================================================================

    def get_rotated_roi_mask(self, shape):
        """Return pixel access mask for rotated ROI.

        Parameters
        ----------
        shape : tuple
            shape of image for which the mask is supposed to be used

        Returns
        -------
        array
            bool array that can be used to access pixels within the ROI

        """
        try:
            if not self._last_rot_roi_mask.shape == shape:
                raise Exception
            mask = self._last_rot_roi_mask
        except BaseException:
            mask = zeros(shape)
            rect = self.rect_roi_rot
            poly = array([rect], dtype=int32)
            fillPoly(mask, poly, 1)
            mask = mask.astype(bool)
        self._last_rot_roi_mask = mask
        return mask

    def check_roi_borders(self, roi, img_array):
        """Check if all points of ROI are within image borders.

        Parameters
        ----------
        roi : list
            ROI rectangle ``[x0,y0,x1,y1]``
        img_array : array
            exemplary image data for which the ROI is checked

        Returns
        -------
        list
            roi within image coordinates (unchanged, if input is ok, else image
            borders)

        """
        x0, y0 = roi[0], roi[1]
        x1, y1 = roi[2], roi[3]
        h, w = img_array.shape
        if not x0 >= 0:
            x0 = 1
        if not x1 < w:
            x1 = w - 2
        if not y0 >= 0:
            y0 = 1
        if not y1 < h:
            y1 = h - 2
        return [x0, y0, x1, y1]

    def prepare_coords(self):
        """Prepare the analysis mesh.

        Note
        ----

        The number of analysis points on this object correponds to the physical
        length of this line in pixel coordinates.
        """
        length = self.length()
        x0, y0 = self.start
        x1, y1 = self.stop

        x = linspace(x0, x1, length)
        y = linspace(y0, y1, length)
        self.profile_coords = vstack((y, x))
        self.det_normal_vecs()
        self.set_rect_roi_rot()

    def length(self):
        """Determine the length in pixel coordinates."""
        return int(round(hypot(*self._delx_dely())))

    def get_line_profile(self, array, order=1, **kwargs):
        """Retrieve the line profile along pixels in input array.

        Parameters
        ----------
        array : array
            2D data array (e.g. image data). Color images are converted into
            gray scale using :func:`cv2.cvtColor`.
        order : int
            order of spline interpolation used to retrieve the values along
            input coordinates (passed to :func:`map_coordinates`)
        **kwargs
            additional keword args passed to interpolation method
            :func:`map_coordinates`

        Returns
        -------
        array
            profile

        """
        try:
            array = array.img  # if input is Img object
        except BaseException:
            pass
        if ndim(array) != 2:
            if ndim(array) != 3:
                logger.info("Error retrieving line profile, invalid dimension of "
                      "input array: %s" % (ndim(array)))
                return
            if array.shape[2] != 3:
                logger.info("Error retrieving line profile, invalid dimension of "
                      "input array: %s" % (ndim(array)))
                return
            "Input in BGR, conversion into gray image"
            array = cvtColor(array, COLOR_BGR2GRAY)

        # Extract the values along the line, using interpolation
        zi = map_coordinates(array, self.profile_coords, order=order, **kwargs)
        if sum(isnan(zi)) != 0:
            logger.warning("Retrieved NaN for one or more pixels along line on input "
                 "array")
        return zi

    """Plotting / visualisation etc...
    """

    def plot_line_on_grid(self, img_arr=None, ax=None, include_normal=False,
                          include_roi_rot=False, include_roi=False,
                          annotate_normal=False, **kwargs):
        """Draw this line on the image.

        Parameters
        ----------
        img_arr : ndarray
            if specified, the array is plotted using :func:`imshow` and onto
            that axes, the line is drawn
        ax :
            matplotlib axes object. Is created if unspecified. Leave
            :param:`img_arr` empty if you want the line to be drawn onto an
            already existing image (plotted in ax)
        include_normal : bool
            if True, the line normal vector is drawn
        include_roi_rot : bool
            if True, a line-orientation specific ROI is drawn
        include_roi : bool
            if True, an ROI is drawn which spans the i,j range of the image
            covered by the line
        annotate_normal : bool
            if True, the normal vector is annotated (only if include_normal is
            set True)
        **kwargs :
            additional keyword arguments for plotting of line (please use
            following keys: marker for marker style, mec for marker
            edge color, c for line color and ls for line style)

        Returns
        -------
        Axes
            matplotlib axes instance

        """
        new_ax = False
        keys = kwargs.keys()
        if "mec" not in keys:
            kwargs["mec"] = "none"
        if "color" not in keys:
            kwargs["color"] = self.color
        if "ls" not in keys:
            kwargs["ls"] = self.linestyle
        if "marker" not in keys:
            kwargs["marker"] = "o"
        if "label" not in keys:
            kwargs["label"] = self.line_id
        if ax is None:
            new_ax = True
            ax = subplot(111)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        c = kwargs["color"]
        if img_arr is not None:
            ax.imshow(img_arr, cmap="gray")
        p = ax.plot([self.start[0], self.stop[0]], [self.start[1],
                                                    self.stop[1]], **kwargs)
        if img_arr is not None:
            ax.set_xlim([0, img_arr.shape[1]])
            ax.set_ylim([img_arr.shape[0], 0])
        if include_normal:
            mag = self.norm * 0.06  # 3 % of length of line
            n = self.normal_vector * mag
            xm, ym = self.center_pix
            epx, epy = n[0], n[1]
            c = p[0].get_color()

            ax.arrow(xm, ym, epx, epy, head_width=mag / 2, head_length=mag,
                     fc=c, ec=c)
            if annotate_normal:
                ax.text(xm + epx * 2, ym + epy * 3, r'$\hat{n}$',
                        color=c,
                        fontweight='bold',
                        fontsize=18)

        if include_roi:
            x0, y0, w, h = roi2rect(self.roi)
            ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=c))
        if include_roi_rot:
            self.plot_rotated_roi(color=c, ax=ax)
        # axis('image')
        if new_ax:
            ax.set_title("Line " + str(self.line_id))
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        draw()
        return ax

    def plot_rotated_roi(self, color=None, ax=None):
        """Plot current rotated ROI into axes.

        Parameters
        ----------
        color
            optional, color information. If None (default) then the current
            line color is used
        ax : :obj:`Axes`, optional
            matplotlib axes object, if None, a figure with one subplot will
            be created

        Returns
        -------
        Axes
            axes instance

        """
        if ax is None:
            ax = subplot(111)
        if color is None:
            color = self.color
        r = self.rect_roi_rot
        p = Polygon(r, fc=color, alpha=0.2)
        ax.add_patch(p)
        return ax

    def plot_line_profile(self, img_arr, ax=None):
        """Plot the line profile."""
        if ax is None:
            ax = subplot(111)
        p = self.get_line_profile(img_arr)
        ax.set_xlim([0, self.length()])
        ax.grid()
        ax.plot(p, label=self.line_id)
        ax.set_title("Profile")
        return ax

    def plot(self, img_arr):
        """Create two subplots showing line on image and corresponding profile.

        Parameters
        ----------
        img_arr : array
            the image data

        Returns
        -------
        Figure
            figure containing the supblots

        """
        fig, axes = subplots(1, 2)
        self.plot_line_on_grid(img_arr, axes[0])
        self.plot_line_profile(img_arr, axes[1])
        tight_layout()
        return fig

    def _delx_dely(self):
        """Length of x and y range covered by line."""
        return float(self.x1) - float(self.x0), float(self.y1) - float(self.y0)

    @property
    def norm(self):
        """Return length of line in pixels."""
        dx, dy = self._delx_dely()
        return norm([dx, dy])

    def det_normal_vecs(self):
        """Get both normal vectors."""
        dx, dy = self._delx_dely()
        v1, v2 = array([-dy, dx]), array([dy, -dx])
        self.normal_vecs = [v1 / norm(v1), v2 / norm(v2)]
        return self.normal_vecs

    @property
    def normal_vector(self):
        """Get normal vector corresponding to current orientation setting."""
        return self.normal_vecs[self._dir_idx[self.normal_orientation]]

    @property
    def complex_normal(self):
        """Return current normal vector as complex number."""
        dx, dy = self.normal_vector
        return complex(-dy, dx)

    @property
    def normal_theta(self):
        """Return orientation of normal vector in degrees.

        The angles correspond to:

            1. 0    =>  to the top (neg. y direction)
            2. 90   =>  to the right (pos. x direction)
            3. 180  =>  to the bottom (pos. y direction)
            4. 270  =>  to the left (neg. x direction)
        """
        return angle(self.complex_normal, True) % 360

    def to_list(self):
        """Return line coordinates as 4-list."""
        return [self.x0, self.y0, self.x1, self.y1]

    def to_dict(self):
        """Write relevant parameters to dictionary."""
        return {"class": "LineOnImage",
                "line_id": self.line_id,
                "color": self.color,
                "linestyle": self.linestyle,
                "x0": self.x0,
                "y0": self.y0,
                "x1": self.x1,
                "y1": self.y1,
                "_normal_orientation": self._normal_orientation,
                "_pyrlevel_def": self._pyrlevel_def,
                "_roi_abs_def": self._roi_abs_def}

    def from_dict(self, settings_dict):
        """Load line parameters from dictionary.

        Parameters
        ----------
        settings_dict : dict
            dictionary containing line parameters (cf. :func:`to_dict`)

        """
        for k, v in settings_dict.items():
            if k in self.__dict__:
                self.__dict__[k] = v

        self.check_coordinates()
        self.prepare_coords()

    @property
    def orientation_info(self):
        """Return string about orientation of line and normal."""
        dx, dy = self._delx_dely()
        s = "delx, dely = %s, %s\n" % (dx, dy)
        s += "normal orientation: %s\n" % self.normal_orientation
        s += "normal vector: %s\n" % self.normal_vector
        s += "Theta normal: %s\n" % self.normal_theta
        return s

    def __str__(self):
        s = ("Line %s: [%d, %d, %d, %d], @pyrlevel %d, @ROI: %s"
             % (self.line_id, self.x0, self.y0, self.x1, self.y1,
                self.pyrlevel_def, self.roi_abs_def))
        return s


class Filter(object):
    """Object representing an interference filter.

    A low level helper class to store information of interference filters.
    """

    def __init__(self, id=None, type="on", acronym="default",
                 meas_type_acro=None, center_wavelength=nan):
        """Initialize of object.

        :param str id ("on"): string identification of this object for
            working environment
        :param str type ("on"): Type of object (choose from "on" and "off")
        :param str acronym (""): acronym for identification in filename
        :param str meas_type_acro (""): acronym for meastype identification in
            filename
        :param str center_wavelength (nan): center transmission wavelength
            of filter
        """
        if type not in ["on", "off"]:
            raise ValueError("Invalid type specification for filter: %s, "
                             "please use on or off as type")
        if id is None:
            id = type

        if meas_type_acro is None:
            meas_type_acro = acronym

        self.id = id
        self.type = type

        # filter acronym (e.g. F01, i.e. as used in filename)
        self.acronym = acronym
        self.meas_type_acro = meas_type_acro

        # filter central wavelength
        self.center_wavelength = center_wavelength
        self.trans_curve = None
        # filter peak transmission
        if self.id is None:
            self.id = self.type

    def to_list(self):
        """Return filter info as list."""
        return [self.id, self.type, self.acronym, self.meas_type_acro,
                self.center_wavelength]

    def set_trans_curve(self, data, wavelengths=None):
        """Assign transmission curve to this filter.

        :param ndarray data: transmission data
        :param ndarray wavelengths: corresponding wavelength array
        :returns: :class:`pandas.Series` object

        .. note::

            Also accepts :class:`pandas.Series` as input using input param
            data and leaving wavelength empty, in this case, the Series index
            is assumed to be the wavelenght data

        """
        if isinstance(data, Series):
            self.trans_curve = data
        else:
            try:
                self.trans_curve = Series(data, wavelengths)
            except BaseException:
                logger.info("Failed to set transmission curve in Filter %s" %
                      self.id)

    def __str__(self):
        s = ("\nFilter\n-----------\n"
             "id: {}\n"
             "type: {}\n"
             "acronym: {}\n"
             "meas_type_acro: {}\n"
             "center_wavelength: {}\n"
             .format(self.id, self.type, self.acronym, self.meas_type_acro,
                     self.center_wavelength))
        return s

    def __repr__(self):
        s = ("Filter {}; type: {}; acronym: {}; meas_type_acro: {}; "
             "center_wavelength: {}"
             .format(self.id, self.type, self.acronym, self.meas_type_acro,
                     self.center_wavelength))
        return s

    def print_specs(self):
        """Print __str__."""
        logger.info(self.__str__())


class DarkOffsetInfo(object):
    """Base class for storage of dark offset information.

    Similar to :class:`Filter`. This object can be used to store relevant
    information of different types of dark and offset images. The attribute
    "read_gain" is set 0 by default. For some camera types (e.g. Hamamatsu
    c8484 16c as used in the ECII SO2 camera), the signal can be enhancened
    with an electronic read_gain (measured in dB) on read. This can be helpful
    in low light conditions. However, it significantly increases the noise in
    the images and therefore also the dark image signal.
    """

    def __init__(self, id="dark", type="dark", acronym="", meas_type_acro=None,
                 read_gain=0):
        """Initialize object.

        :param str id: string identification of this object for
            working environment  (default: "dark")
        :param str type: Type of object (e.g. dark or offset, default: "dark")
        :param str acronym: acronym for identification in filename
        :param str meas_type_acro: acronym for meastype identification in
            filename
        :param str read_gain: string specifying read_gain mode of this object
            (use 0 or 1, default is 0)

        """
        if type not in ["dark", "offset"]:
            raise ValueError("Invalid type specification for "
                             "DarkOffsetInfo: %s, please use dark or offset "
                             "as type")
        self.id = id
        self.type = type
        self.acronym = acronym
        if meas_type_acro is None:
            meas_type_acro = acronym
        self.meas_type_acro = meas_type_acro
        self.read_gain = read_gain

    def to_list(self):
        """Return parameters as list."""
        return [self.id, self.type, self.acronym, self.meas_type_acro,
                self.read_gain]

    def __str__(self):
        """Get string representation."""
        s = ("\nDarkOffsetInfo\n---------------------------------\n"
             "ID: %s\n"
             "Type: %s\n"
             "Acronym: %s\n"
             "Meas type acronym: %s\n"
             "Read gain: %s\n" % (self.id, self.type, self.acronym,
                                  self.meas_type_acro, self.read_gain))
        return s
