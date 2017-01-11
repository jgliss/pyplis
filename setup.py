# -*- coding: utf-8 -*-
from setuptools import setup
from codecs import open
from os.path import join, abspath, dirname

here = abspath(dirname(__file__))

with open(join(here,'README.rst'), encoding = 'utf-8') as file:
    readme = file.read()

with open("VERSION.rst") as f:
    version = f.readline()
    f.close()
    
with open(join("piscope","data", "_paths.txt"), 'w'): pass
    
setup(
    name        =   'piscope',
    version     =   version,
    author      =   'Jonas Gliss',
    author_email=   'jg@nilu.no',
    license     =   'GPLv3+',
    url         =   'https://github.com/jgliss/piscope',
    description = ('A Python library for the analysis UV SO2 camera data'),
    long_description = readme,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.,
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='sample setuptools development',
    #packages    =   ['piscope'],
    package_dir =   {'piscope'     :    'piscope',
                     'piscope.gui_features' :    'piscope/gui_features'},
    packages =  ['piscope', 'piscope.gui_features'],
                 
    package_data=   {'piscope'     :   ['data/*.txt',
                                        'data/*.rst'
                                        'data/*.png',
                                        'data/icons/axialis/k/*.png',
                                        'data/icons/axialis/r/*.png',
                                        'data/icons/axialis/g/*.png',
                                        'data/icons/axialis/ReadMe.txt',
                                        'data/icons/myIcons/k/*.png',
                                        'data/icons/myIcons/r/*.png',
                                        'data/icons/myIcons/g/*.png']},
#==============================================================================
#     install_requires    =   ["scipy>=0.17.0",
#                              "dill",
#                              "PyQt4",
#                              "peakutils"],
#==============================================================================
    install_requires    =   [],
    dependency_links    =   [],   
    entry_points = {},#'console_scripts': ['sample=sample:main',],},
)