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
"""example script 002 - Combine background correction modes.

This script illustrates how to efficiently apply the plume background analysis
in case no sky reference image is available.

"""
from os.path import join
from matplotlib.pyplot import close
import numpy as np
import sys
import pyplis

from ex01_analysis_setup import create_dataset

from matplotlib import rcParams
rcParams.update({'font.size': 15})

sys.path.append(join(".."))

# SCRIPT MAIN FUNCTION
if __name__ == "__main__":
    close("all")
    ds = create_dataset()

    on = ds.get_list("on")
    on.darkcorr_mode = True
    on.gaussian_blurring = 2

    # Find and plot sky reference areas
    on.bg_model.set_missing_ref_areas(on.current_img())
    ax = on.bg_model.plot_sky_reference_areas(on.current_img())

    # this is a beta version
    kernel = np.ones((90, 90), dtype=np.uint8)
    mask = on.prepare_bg_fit_mask(dilation=True, dilate_kernel=kernel,
                                  optflow_blur=0, optflow_median=10,
                                  i_min=1500, i_max=2600, plot_masks=True)
    mask = pyplis.Img(mask)
    ax2 = mask.show(tit="Input mask for surface fit")
    on.set_bg_img_from_polyfit(mask.img)
    ax3 = on.bg_img.show(tit="Surface fit result")

    on.bg_model.mode = 5
    on.tau_mode = True
    on.show_current()
