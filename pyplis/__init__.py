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
"""Package initialisation."""
from __future__ import (absolute_import, division)
from ._init_utils import (check_requirements, _init_logger,
                          _init_supplemental, change_loglevel,
                          get_loglevel)

PYDOASAVAILABLE, GEONUMAVAILABLE  = check_requirements()
logger, print_log = _init_logger()

__version__, __dir__ = _init_supplemental()
_LIBDIR = __dir__  # from older version

URL_TESTDATA = ("https://folk.nilu.no/~arve/pyplis/"
                "pyplis_etna_testdata.zip")

from .setupclasses import MeasSetup, Camera, Source
from .geometry import MeasGeometry
from .utils import Filter, DarkOffsetInfo, LineOnImage
from .image import Img, ProfileTimeSeriesImg
from .dataset import Dataset
from .imagelists import ImgList, CellImgList, DarkImgList
from .plumebackground import PlumeBackgroundModel
from .cellcalib import CellCalibData, CellCalibEngine
from .calib_base import CalibData
from .doascalib import DoasCalibData, DoasFOV, DoasFOVEngine
from .plumespeed import (find_signal_correlation, OptflowFarneback,
                         FarnebackSettings, LocalPlumeProperties,
                         VeloCrossCorrEngine)
from .processing import ImgStack, PixelMeanTimeSeries
from .dilutioncorr import DilutionCorr
from .fluxcalc import (EmissionRateAnalysis, EmissionRates,
                       EmissionRateSettings)
from .optimisation import PolySurfaceFit, MultiGaussFit
from . import custom_image_import
from .inout import download_test_data, find_test_data
