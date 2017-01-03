# -*- coding: utf-8 -*-
"""
A GUI environment for the Python library "piSCOPE"
"""
from PyQt4.QtGui import QApplication
from sys import argv
#import os

import object_widgets
import edit_widgets
import imgviewer
import plot_widgets
import setup_widgets
    
"""
Space for high level functions
"""
def open_img_viewer():
    app = QApplication(argv)
    win = imgviewer.ImgViewer()
    win.show()
    app.exec_() #run main loop  d
    return win
    
    