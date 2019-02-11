# -*- coding: utf-8 -*-
"""
Pyplis test module for imagelists.py base module of Pyplis
Author: Solvejg Dinger
"""

# general packages
from os.path import join
import pytest
import datetime as dt

# pyplis imports
import pyplis
import pyplis.imagelists as il
#from pyplis.setupclasses import Camera



BASE_DIR = join(pyplis.__dir__, "data", "testdata_minimal")
IMG_DIR = join(BASE_DIR, "images")

PLUME_FILE = join(IMG_DIR, 
                  'EC2_1106307_1R02_2015091607110434_F01_Etna.fts')
PLUME_FILE_NEXT = join(IMG_DIR, 
                       'EC2_1106307_1R02_2015091607113241_F01_Etna.fts')

BG_FILE = join(IMG_DIR, 'EC2_1106307_1R02_2015091607022602_F01_Etna.fts')
OFFSET_FILE = join(IMG_DIR, "EC2_1106307_1R02_2015091607064723_D0L_Etna.fts")
DARK_FILE = join(IMG_DIR, "EC2_1106307_1R02_2015091607064865_D1L_Etna.fts")

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

@pytest.fixture(scope="function")
def imglist(ec2_cam, plume_files):
    return il.ImgList(files=plume_files, camera=ec2_cam, list_id='test')

@pytest.fixture(scope="function")
def bg_imglist(ec2_cam):
    return il.ImgList(files=[BG_FILE], camera=ec2_cam)
    
# Create empty class objects and fill them manually
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_empty_BaseImgList(ec2_cam, plume_files):
    baseimglist = il.BaseImgList()
    baseimglist.files = plume_files
    baseimglist.camera = ec2_cam
    baseimglist.load()
    assert baseimglist.nof == 2
    assert baseimglist.start_acq[0] == dt.datetime(2015, 9, 16, 7, 11, 4, 340000)

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_empty_DarkImgList():
    darkimglist = il.DarkImgList()
    assert darkimglist.nof == 0

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_empty_ImgList(ec2_cam, plume_files):
    imglist = il.ImgList()
    imglist.files = plume_files
    imglist.camera = ec2_cam
    imglist.load()

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_empty_CellImgList():
    cellimglist = il.CellImgList()
    assert cellimglist.nof == 0

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_empty_ImgListLayered(comtessa_cam):
    imglistlayered = il.ImgListLayered()
    imglistlayered.camera = comtessa_cam
    assert imglistlayered.nof == 0

# Create class objects
def test_BaseImgList_ec2(BaseImgList_ec2):
    BaseImgList_ec2.goto_next()
    assert BaseImgList_ec2.current_time() == \
                    dt.datetime(2015, 9, 16, 7, 11, 32, 410000)
    
def test_BaseImgList_ec2_str(plume_files):
    baseimglist = il.BaseImgList(files=plume_files, camera='ecII')
    assert baseimglist.start_acq[0] == \
                dt.datetime(2015, 9, 16, 7, 11, 4, 340000)

# pops warning that list has only one entry
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_list_linking(imglist, bg_imglist):
    imglist.set_bg_list(bg_imglist)
    
    
# Test specific class methods
@pytest.mark.parametrize("method, index", [("nearest", 1), ("pad", 0),
                                           ("backfill", 1)])
def test_closest_index(BaseImgList_ec2, method, index):
    timestamp = dt.datetime(2015, 9, 16, 7, 11, 20)
    assert BaseImgList_ec2.timestamp_to_index(timestamp, method) == index

from pandas import DataFrame
from pyplis.processing import PixelMeanTimeSeries

@pytest.mark.parametrize("return_val", [True, False])
def test_mean_timeseries(BaseImgList_ec2, return_val):
    rect = [50, 50, 100, 100]
    mean_timeseries = BaseImgList_ec2.get_mean_value(roi=rect,
                                                     return_dataframe=return_val)
    if return_val:
        assert isinstance(mean_timeseries, DataFrame)
        assert mean_timeseries['mean'].values == pytest.approx([146.2516, 146.9500])
        assert mean_timeseries['std'].values == pytest.approx([1.823485, 1.679375])
    else:
        assert isinstance(mean_timeseries, PixelMeanTimeSeries)
        assert mean_timeseries.values == pytest.approx([146.2516, 146.9500])
        assert mean_timeseries.std == pytest.approx([1.823485, 1.679375])
    
if __name__ == "__main__":
    camera = ec2_cam()
    files = plume_files()
    baseil = BaseImgList_ec2(camera, files)
    #baseil.goto_img(0)
    #print(baseil.index)
    #baseil.goto_next()
    #print(baseil.index)
    series = baseil.get_mean_value(0,1,roi=[50, 50, 100, 100], return_dataframe=True)
    print(baseil.start_acq)