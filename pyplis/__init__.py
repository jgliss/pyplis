# -*- coding: utf-8 -*-
from os.path import abspath, dirname
from pkg_resources import get_distribution

__version__ = get_distribution('pyplis').version

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

_LIBDIR = abspath(dirname(__file__))

from inout import download_test_data, find_test_data
from geometry import MeasGeometry
from utils import Filter, DarkOffsetInfo
from image import Img
import custom_image_import
from dataset import Dataset
from plumebackground import PlumeBackgroundModel
from cellcalib import CellCalibData, CellCalibEngine
from doascalib import DoasCalibData, DoasFOV, DoasFOVEngine
from plumespeed import find_signal_correlation, OpticalFlowFarneback,\
    OpticalFlowFarnebackSettings
from processing import LineOnImage, ImgStack, ProfileTimeSeriesImg,\
    PixelMeanTimeSeries
from dilutioncorr import DilutionCorr
from fluxcalc import EmissionRateAnalysis, EmissionRateResults,\
    EmissionRateSettings
import optimisation
import model_functions
from setupclasses import MeasSetup, Camera, Source
 
import helpers
import exceptions
import glob

from matplotlib import rcParams

rcParams["mathtext.default"] = u"regular"


#import gui_features as gui_features