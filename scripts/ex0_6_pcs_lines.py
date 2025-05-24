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
"""Pyplis introduction script no 6 - LineOnImage objects and their orientation.

This script introduces how to create LineOnImage objects within the image plane
and specify their orientation (i.e. the direction into which the normal vector
of the line points). This is mainly important for emission rate retrievals,
where, for instance velcoity displacement vectors (e.g. from an optical flow
algorithm) have to be multiplied with the normal vector of such a line (using
the dot product).
"""
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER
import pathlib
import matplotlib.pyplot as plt
from pyplis import LineOnImage
from matplotlib.cm import get_cmap

def create_example_lines():
    """Create some exemplary lines."""
    lines_r = []  # lines with orientation "right" are stored within this list
    lines_l = []  # lines with orientation "left" are stored within this list

    cmap = get_cmap("jet")
    # horizontal line, normal orientation to the top (0 deg)
    lines_r.append(LineOnImage(x0=10, y0=10, x1=90, y1=10,
                               normal_orientation="right",
                               color=cmap(0),
                               line_id="Line 1"))

    # horizontal line, normal orientation to the bottom (180 deg)
    lines_l.append(LineOnImage(10, 10, 90, 10, normal_orientation="left",
                               color=cmap(0),
                               line_id="Line 1"))

    # Vertical line normal to the right (90 deg)
    lines_r.append(LineOnImage(10, 10, 10, 90, normal_orientation="right",
                               color=cmap(50),
                               line_id="Line 2"))

    # Vertical line normal to the left (270 deg)
    lines_l.append(LineOnImage(10, 10, 10, 90, normal_orientation="left",
                               color=cmap(50),
                               line_id="Line 2"))

    # Slanted line 45 degrees
    lines_r.append(LineOnImage(20, 30, 50, 60, normal_orientation="right",
                               color=cmap(100),
                               line_id="Line 3"))

    # Slanted line 45 degrees
    lines_l.append(LineOnImage(20, 30, 50, 60, normal_orientation="left",
                               color=cmap(100),
                               line_id="Line 3"))

    # Slanted line 45 degrees
    lines_r.append(LineOnImage(90, 10, 10, 90, normal_orientation="right",
                               color=cmap(150),
                               line_id="Line 4"))

    # Slanted line 45 degrees
    lines_l.append(LineOnImage(90, 10, 10, 90, normal_orientation="left",
                               color=cmap(150),
                               line_id="Line 4"))

    lines_r.append(LineOnImage(60, 20, 80, 90, normal_orientation="right",
                               color=cmap(200),
                               line_id="Line 5"))

    lines_l.append(LineOnImage(60, 20, 80, 90, normal_orientation="left",
                               color=cmap(200),
                               line_id="Line 5"))

    lines_r.append(LineOnImage(40, 20, 30, 90, normal_orientation="right",
                               color=cmap(250),
                               line_id="Line 6"))
    lines_l.append(LineOnImage(40, 20, 30, 90, normal_orientation="left",
                               color=cmap(250),
                               line_id="Line 6"))

    return lines_r, lines_l

def main():
    plt.close("all")
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))

    lines_r, lines_l = create_example_lines()

    for k in range(len(lines_r)):
        line = lines_r[k]
        lbl = "%s" % line.line_id
        line.plot_line_on_grid(ax=ax[0], include_normal=1,
                               include_roi_rot=True, label=lbl)
    for k in range(len(lines_l)):
        line = lines_l[k]
        lbl = "%s" % line.line_id
        line.plot_line_on_grid(ax=ax[1], include_normal=1,
                               include_roi_rot=True, label=lbl)

    ax[0].set_title("Orientation right")
    ax[0].legend(loc="best", fontsize=8, framealpha=0.5)
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([100, 0])

    ax[1].set_title("Orientation left")
    ax[1].legend(loc="best", fontsize=8, framealpha=0.5)
    ax[1].set_xlim([0, 100])
    ax[1].set_ylim([100, 0])
    # ## IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        outfile = SAVE_DIR / f"ex0_6_out_1.{FORMAT}"
        fig.savefig(outfile, format=FORMAT, dpi=DPI)

    # Import script options
    options= ARGPARSER.parse_args()

    # If applicable, do some tests. This is done only if TESTMODE is active:
    # testmode can be activated globally (see SETTINGS.py) or can also be
    # activated from the command line when executing the script using the
    # option --test 1
    if int(options.test):
        import numpy.testing as npt

        npt.assert_array_equal([sum([x.length() for x in lines_l]),
                                sum([x.length() for x in lines_r]),
                                sum([sum(x.coords) for x in lines_r]),
                                int(sum([sum(x.normal_vector + y.normal_vector)
                                    for x, y in zip(lines_l, lines_r)])),
                                int(sum([sum(x.center_pix) for x in lines_l])),
                                ],
                               [459, 459, 1030, 0, 515])

        npt.assert_allclose(
            actual=[sum([sum(x) for x in lines_l[2].rect_roi_rot])],
            desired=[368.79393923],
            rtol=1e-7)

        print(f"All tests passed in script: {pathlib.Path(__file__).name}")
    try:
        if int(options.show) == 1:
            plt.show()
    except Exception:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    main()