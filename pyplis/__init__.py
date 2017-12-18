# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliß (jonasgliss@gmail.com)
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

from os.path import abspath, dirname
from pkg_resources import get_distribution
from matplotlib import rcParams

rcParams["mathtext.default"] = u"regular"

__dir__ = abspath(dirname(__file__))
__version__ = get_distribution('pyplis').version
_LIBDIR = __dir__ #from older version

URL_TESTDATA = ("https://folk.nilu.no/~gliss/pyplis_testdata/"
                "pyplis_etna_testdata.zip")

try:
    import pydoas
    PYDOASAVAILABLE =True
except:
    PYDOASAVAILABLE = False

try:
    import geonum
    GEONUMAVAILABLE = 1
except:
    GEONUMAVAILABLE = 0

from inout import download_test_data, find_test_data
from geometry import MeasGeometry
from utils import Filter, DarkOffsetInfo
from image import Img
import custom_image_import
from dataset import Dataset
from imagelists import ImgList, CellImgList
from plumebackground import PlumeBackgroundModel
from cellcalib import CellCalibData, CellCalibEngine
from doascalib import DoasCalibData, DoasFOV, DoasFOVEngine
from plumespeed import find_signal_correlation, OptflowFarneback,\
    FarnebackSettings, LocalPlumeProperties
from processing import LineOnImage, ImgStack, ProfileTimeSeriesImg,\
    PixelMeanTimeSeries
from dilutioncorr import DilutionCorr
from fluxcalc import EmissionRateAnalysis, EmissionRates,\
    EmissionRateSettings
from optimisation import PolySurfaceFit, MultiGaussFit

from setupclasses import MeasSetup, Camera, Source


#==============================================================================
# import model_functions 
# import helpers
# import exceptions
# import glob
#==============================================================================




#import gui_features as gui_features