# -*- coding: utf-8 -*-
"""
Created on Thu May 04 11:39:39 2017

@author: Jonas
"""

from os import listdir

paths = [f  for f in listdir(".") if f[:4] == "ex0_" and f[4]!="5" and f.endswith("py")]

for path in paths:
    print path
    execfile(path)