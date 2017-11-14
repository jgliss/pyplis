# -*- coding: utf-8 -*-
"""
Author: Jonas Gliß
Email: jonasgliss@gmail.com
License: GPLv3+
"""
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