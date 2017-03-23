# -*- coding: utf-8 -*-
"""
pyplis introduction script 5 - Optical flow Farneback live view using webcam

Create an OpticalFlowFarneback object and activate live view (requires webcam)
"""
from SETTINGS import check_version
# Raises Exception if conflict occurs
check_version()

import pyplis

flow = pyplis.plumespeed.OptflowFarneback()
flow.live_example()
