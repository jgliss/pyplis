# -*- coding: utf-8 -*-
"""
pyplis introduction script 5 - Optical flow Farneback live view using webcam

Create an OpticalFlowFarneback object and activate live view (requires webcam)
"""

import pyplis

flow = pyplis.plumespeed.OpticalFlowFarneback()
flow.live_example()
