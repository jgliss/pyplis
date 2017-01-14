# -*- coding: utf-8 -*-
"""
piSCOPE example script 0

This first example script is supposed to get familiar with all necessary 
input parameters for a basic plume data setup. It creates a 
:class:`piscope.Setup.BaseSetup` object and opens the graphical editing 
window. 

The dialog is split into 3 tabs.

    1. The first tab is for editing image base path, start / stop time stamps
        of the measurement and to provide meteorological informaiton. Click
        "confirm" after edit.
    2. The source can be specified in the 2nd tab 
    3. The camera and filename convention can be specified in the 3rd tab. The
        default cameras (listed in a drop down menu) can be specified in the
        text file "cam_info.txt" located in the data package directory 
        piscope\piscope\data    
    
"""

import piscope as piscope

win = piscope.gui.open_app()