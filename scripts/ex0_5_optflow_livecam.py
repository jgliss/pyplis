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
"""Pyplis introduction script 5 - Optical flow Farneback liveview using webcam.

Create an OpticalFlowFarneback object and activate live view (requires webcam)
"""
import signal
import sys
import pyplis



flow = pyplis.plumespeed.OptflowFarneback()

def signal_handler(sig, frame):
        print("\nExiting gracefully...")
        sys.exit(0)  # Exit without error

def main():    
    signal.signal(signal.SIGINT, signal_handler)
    try:
        flow.live_example()
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()