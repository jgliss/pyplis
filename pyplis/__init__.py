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

import inout
import geometry
import utils
from image import Img
import dataset
import plumebackground
import cellcalib
import doascalib
import plumespeed
import processing
import dilutioncorr
import fluxcalc
import optimisation
import model_functions
import setupclasses  
 
import helpers
import exceptions

SPECIES_ID = r"SO2"

#map of internal calibration access keys to string repr. for plots
_CALIB_ID_STRINGS = {"on" :  "On",
                     "off":  "Off",
                     "aa" :  "AA"}

from matplotlib import rcParams

rcParams["mathtext.default"] = u"regular"

#import gui_features as gui_features