# -*- coding: utf-8 -*-
"""
Pyplis test module for imagelists.py base module of Pyplis
Author: Solvejg Dinger
"""

import pyplis
import pyplis.imagelists as il
#from pyplis.setupclasses import Camera
from os.path import join
import pytest

BASE_DIR = join(pyplis.__dir__, "data", "testdata_minimal")
IMG_DIR = join(BASE_DIR, "images")

PLUME_FILE = join(IMG_DIR, 
                  'EC2_1106307_1R02_2015091607110434_F01_Etna.fts')
PLUME_FILE_NEXT = join(IMG_DIR, 
                       'EC2_1106307_1R02_2015091607113241_F01_Etna.fts')

@pytest.fixture(scope="module")
def ec2_cam():
    """Load and return camera object"""
    return pyplis.Camera(cam_id='ecII')

@pytest.fixture(scope="module")
def comtessa_cam():
    """Load and return camera object"""
    return pyplis.Camera(cam_id='comtessa')

@pytest.fixture(scope="module")
def plume_files():
    """Load and return a list of files"""
    return [PLUME_FILE, PLUME_FILE_NEXT]

@pytest.fixture(scope="function")
def BaseImgList_ec2(ec2_cam, plume_files):
    return il.BaseImgList(files=plume_files, camera=ec2_cam)

# Create empty class objects and fill them manually
def test_empty_BaseImgList(ec2_cam, plume_files):
    baseimglist = il.BaseImgList()
    baseimglist.files = plume_files
    baseimglist.camera = ec2_cam
    baseimglist.load()
    assert baseimglist.nof == 2

def test_empty_DarkImgList():
    darkimglist = il.DarkImgList()
    assert darkimglist.nof == 0
    
def test_empty_ImgList(ec2_cam, plume_files):
    imglist = il.ImgList()
    imglist.files = plume_files
    imglist.camera = ec2_cam
    imglist.load()
    
def test_empty_CellImgList():
    cellimglist = il.CellImgList()
    assert cellimglist.nof == 0
    
def test_empty_ImgListLayered(comtessa_cam):
    imglistlayered = il.ImgListLayered()
    imglistlayered.camera = comtessa_cam
    assert imglistlayered.nof == 0

# Create class objects
def test_BaseImgList_ec2(BaseImgList_ec2):
    BaseImgList_ec2.goto_next()
    assert BaseImgList_ec2.nof == 2
    
def test_BaseImgList_ec2_str(plume_files):
    baseimglist = il.BaseImgList(files=plume_files, camera='ecII')