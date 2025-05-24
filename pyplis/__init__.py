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
from pyplis._init_utils import (check_requirements, _init_logger,
                          _init_supplemental, change_loglevel,
                          get_loglevel)

PYDOASAVAILABLE = check_requirements()
logger, print_log = _init_logger()

__version__, __dir__ = _init_supplemental()

URL_TESTDATA = ("https://folk.nilu.no/~arve/pyplis/"
                "pyplis_etna_testdata.zip")

from pyplis.setupclasses import MeasSetup, Camera, Source
from pyplis.geometry import MeasGeometry
from pyplis.utils import Filter, DarkOffsetInfo, LineOnImage
from pyplis.image import Img, ProfileTimeSeriesImg
from pyplis.dataset import Dataset
from pyplis.imagelists import ImgList, CellImgList, DarkImgList
from pyplis.plumebackground import PlumeBackgroundModel
from pyplis.cellcalib import CellCalibData, CellCalibEngine
from pyplis.calib_base import CalibData
from pyplis.doascalib import DoasCalibData, DoasFOV, DoasFOVEngine
from pyplis.plumespeed import (find_signal_correlation, OptflowFarneback,
                         FarnebackSettings, LocalPlumeProperties,
                         VeloCrossCorrEngine)
from pyplis.processing import ImgStack, PixelMeanTimeSeries
from pyplis.dilutioncorr import DilutionCorr
from pyplis.fluxcalc import (EmissionRateAnalysis, EmissionRates,
                       EmissionRateSettings)
from pyplis.optimisation import PolySurfaceFit, MultiGaussFit
from pyplis import custom_image_import
from pyplis.inout import download_test_data, find_test_data
