# -*- coding: utf-8 -*-
"""
piscope introduction script 5 - Optical flow Farneback live view using webcam

Create an OpticalFlowFarneback object and activate live view (requires webcam)
"""

import piscope

flow = piscope.plumespeed.OpticalFlowFarneback()
flow.live_example()
